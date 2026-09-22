"""
compute_full_distance_matrix.py
================================
Computes road distances (km) for ALL pairwise market combinations
(1,984 × 1,984 = 3,936,256 pairs) using OpenRouteService Matrix API.

Strategy
--------
  - Load unique markets from the hourly cold-loads CSV (deduplicated on market_id)
  - Split markets into batches of BATCH_SIZE (default 50)
  - Each ORS request: batch_i (sources) × batch_j (destinations)
  - 40 × 40 = 1,600 requests total
  - Estimated runtime: ~43 minutes

Checkpoint / resume
-------------------
  A checkpoint file (road_distance_matrix_checkpoint.pkl) is saved every
  CHECKPOINT_EVERY batches.  Re-running the script automatically resumes
  from the last checkpoint — already-completed batches are skipped and
  pairs that previously fell back to Haversine are re-queried with ORS.

Output
------
  road_distance_matrix_full.pkl   — dict {(market_id_a, market_id_b): road_km}
  road_distance_matrix_full.csv   — long-format CSV for inspection

  Update road_distances_path in config.py to point at the new pkl.

Usage
-----
  pip install requests pandas tqdm
  python compute_full_distance_matrix.py \\
      --markets "pipeline_runs/full/market_hourly_cold_loads_2019.csv" \\
      --api-key YOUR_ORS_KEY

  Or:  export ORS_API_KEY=YOUR_KEY  then run without --api-key
"""

import os, sys, time, math, pickle, argparse, requests
import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
ORS_URL          = "https://api.heigit.org/openrouteservice/v2/matrix/driving-car"
BATCH_SIZE       = 50        # ORS Collaborative plan: 2500 daily, 40/min for Matrix V2
RATE_LIMIT_S     = 1.6       # 60s / 40 req/min
MAX_RETRIES      = 3
OUT_PKL          = "road_distance_matrix_full.pkl"
OUT_CSV          = "road_distance_matrix_full.csv"
CHECKPOINT_FILE  = "road_distance_matrix_checkpoint.pkl"
CHECKPOINT_EVERY = 20        # save checkpoint every N batch-pairs

LON_COL = "x"
LAT_COL = "y"
ID_COL  = "market_id"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def ors_matrix(src_coords, dst_coords, api_key):
    """
    POST to ORS matrix API.
    src_coords, dst_coords: lists of [lon, lat]
    Returns 2D list of distances in metres (None where no route found).
    """
    n_src = len(src_coords)
    all_coords = src_coords + dst_coords
    src_idx    = list(range(n_src))
    dst_idx    = list(range(n_src, len(all_coords)))

    body = {
        "locations":    all_coords,
        "sources":      src_idx,
        "destinations": dst_idx,
        "metrics":      ["distance"],
        "units":        "m",
    }
    headers = {
        "Authorization": api_key,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(ORS_URL, json=body, headers=headers, timeout=60)
            if r.status_code == 200:
                return r.json()["distances"]
            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                tqdm.write(f"  ⚠ Rate limited — waiting {wait}s …")
                time.sleep(wait)
            else:
                tqdm.write(f"  ✗ ORS error {r.status_code}: {r.text[:200]}")
                return None
        except requests.RequestException as e:
            tqdm.write(f"  ✗ Request failed (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def save_checkpoint(all_dists, fallback_pairs, completed_batches):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({
            "all_dists":         all_dists,
            "fallback_pairs":    fallback_pairs,
            "completed_batches": completed_batches,
        }, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {}, set(), set()
    with open(CHECKPOINT_FILE, "rb") as f:
        cp = pickle.load(f)
    n_pairs    = len(cp["all_dists"])
    n_fallback = len(cp["fallback_pairs"])
    n_batches  = len(cp["completed_batches"])
    print(f"  Resuming from checkpoint: {n_batches} batches done, "
          f"{n_pairs:,} pairs loaded ({n_fallback:,} Haversine fallbacks to retry).")
    return cp["all_dists"], cp["fallback_pairs"], cp["completed_batches"]


def main(markets_csv, api_key):
    if not api_key or api_key == "YOUR_KEY_HERE":
        print("ERROR: ORS API key not set.")
        print("  --api-key YOUR_KEY  or  export ORS_API_KEY=YOUR_KEY")
        print("  Free signup: https://openrouteservice.org/dev/#/signup")
        sys.exit(1)

    # ── Load markets ──────────────────────────────────────────────────────────
    df = pd.read_csv(markets_csv, usecols=[ID_COL, LON_COL, LAT_COL])
    df = df.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    # Index by market_id for fast coord lookup
    df_idx = df.set_index(ID_COL)
    ids    = df[ID_COL].tolist()
    coords = [[row[LON_COL], row[LAT_COL]] for _, row in df.iterrows()]
    n      = len(ids)

    # Split into batches
    batches = []
    for i in range(0, n, BATCH_SIZE):
        batches.append(list(range(i, min(i + BATCH_SIZE, n))))

    n_batches  = len(batches)
    total_reqs = n_batches * n_batches
    est_min    = total_reqs * RATE_LIMIT_S / 60

    print(f"Loaded {n} markets.")
    print(f"Batches: {n_batches} × {n_batches} = {total_reqs} ORS requests")
    print(f"Estimated time: ~{est_min:.1f} minutes\n")

    # ── Load checkpoint if available ──────────────────────────────────────────
    all_dists, fallback_pairs, completed_batches = load_checkpoint()
    fallback_cnt = len(fallback_pairs)

    pbar = tqdm(total=total_reqs, desc="Requests",
                initial=len(completed_batches))

    batch_counter = 0

    for bi, src_batch in enumerate(batches):
        for bj, dst_batch in enumerate(batches):

            batch_key = (bi, bj)

            # Skip already-completed batches (unless they had fallbacks to retry)
            if batch_key in completed_batches:
                # Check if any pair in this batch still needs ORS retry
                src_ids = [ids[k] for k in src_batch]
                dst_ids = [ids[k] for k in dst_batch]
                batch_fallbacks = fallback_pairs & {
                    (int(id_a), int(id_b))
                    for id_a in src_ids for id_b in dst_ids
                }
                if not batch_fallbacks:
                    pbar.update(1)
                    continue

            src_ids    = [ids[k]    for k in src_batch]
            dst_ids    = [ids[k]    for k in dst_batch]
            src_coords = [coords[k] for k in src_batch]
            dst_coords = [coords[k] for k in dst_batch]

            time.sleep(RATE_LIMIT_S)
            mat = ors_matrix(src_coords, dst_coords, api_key)

            for si, id_a in enumerate(src_ids):
                for di, id_b in enumerate(dst_ids):
                    pair = (int(id_a), int(id_b))
                    if mat is not None and mat[si][di] is not None:
                        km = mat[si][di] / 1000.0
                        # Remove from fallback set if it was previously a fallback
                        fallback_pairs.discard(pair)
                        if pair in all_dists:
                            fallback_cnt -= 1
                    else:
                        if pair not in fallback_pairs:
                            # Fallback: Haversine × mean Nigeria circuity factor
                            row_a = df_idx.loc[id_a]
                            row_b = df_idx.loc[id_b]
                            km = haversine_km(
                                row_a[LAT_COL], row_a[LON_COL],
                                row_b[LAT_COL], row_b[LON_COL]
                            ) * 1.437
                            fallback_pairs.add(pair)
                            fallback_cnt += 1
                        else:
                            # Keep existing Haversine value unchanged
                            km = all_dists.get(pair)
                            if km is None:
                                row_a = df_idx.loc[id_a]
                                row_b = df_idx.loc[id_b]
                                km = haversine_km(
                                    row_a[LAT_COL], row_a[LON_COL],
                                    row_b[LAT_COL], row_b[LON_COL]
                                ) * 1.437
                    all_dists[pair] = round(km, 3)

            completed_batches.add(batch_key)
            pbar.update(1)
            batch_counter += 1

            if batch_counter % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_dists, fallback_pairs, completed_batches)
                tqdm.write(f"  ✓ Checkpoint saved ({len(all_dists):,} pairs, "
                           f"{len(fallback_pairs):,} fallbacks remaining)")

    pbar.close()

    # ── Final save ────────────────────────────────────────────────────────────
    print(f"\nSaving {OUT_PKL} …")
    with open(OUT_PKL, "wb") as f:
        pickle.dump(all_dists, f)

    print(f"Saving {OUT_CSV} …")
    rows = [{"market_id_a": a, "market_id_b": b, "road_km": km}
            for (a, b), km in all_dists.items()]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n✓ Done.")
    print(f"  Total pairs saved : {len(all_dists):,}")
    print(f"  Haversine fallbacks remaining: {len(fallback_pairs):,}")
    if fallback_pairs:
        print(f"  Re-run after quota reset to replace fallbacks with road distances.")
    print(f"\n  → Update config.py:")
    print(f"    road_distances_path = '{OUT_PKL}'")
    print(f"\n  → Upload {OUT_PKL} to CREATE cluster.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", required=True,
                        help="Path to market_hourly_cold_loads_2019.csv (pipeline_runs/full)")
    parser.add_argument("--api-key",
                        default=os.getenv("ORS_API_KEY", "YOUR_KEY_HERE"),
                        help="ORS API key")
    args = parser.parse_args()
    main(args.markets, args.api_key)
