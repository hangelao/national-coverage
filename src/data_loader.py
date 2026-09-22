"""
data_loader.py
--------------
Loads, validates, aligns, and optionally down-samples the two input files:

  • market_hourly_cold_loads_2019.csv  — pre-computed hourly cold loads
  • ninja-weather … pvlib.csv          — state-level hourly solar PV capacity factors

Public API
----------
    load_cold_loads(path)            -> pd.DataFrame  (long format, full 8760-h index)
    load_solar_cf(path)              -> pd.DataFrame  (8760 × 37 states, DatetimeIndex)
    load_market_locations(df)        -> pd.DataFrame  (one row per market)
    pivot_cold_loads(df)             -> pd.DataFrame  (8760 × n_markets, DatetimeIndex)
    align_solar_to_markets(solar_df, locations_df)
                                     -> pd.DataFrame  (8760 × n_markets, DatetimeIndex)
    select_representative_periods(load_pivot, n_weeks, seed)
                                     -> (DatetimeIndex, np.ndarray weights)
    load_road_distances(path)        -> dict  {(id_a, id_b): road_km}
    load_all(cfg)                    -> dict  (convenience wrapper)
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def normalise_id(mid) -> str:
    """
    Convert any market_id representation to a clean string.

    Handles int (110081), float (110081.0), and str inputs.
    Strips a trailing ".0" that Pandas sometimes adds when a numeric
    column is loaded with NaNs present.
    """
    s = str(mid)
    if s.endswith(".0"):
        s = s[:-2]
    return s


# Map of non-standard state spellings to their canonical form in the
# solar-CF CSV header.  The only mismatch found in the data is Fct → FCT.
_STATE_ALIASES: dict[str, str] = {
    "fct": "FCT",
}


def _normalise_state(state: str) -> str:
    """Return the canonical state name that matches the solar-CF column header."""
    stripped = state.strip()
    lower = stripped.lower()
    if lower in _STATE_ALIASES:
        return _STATE_ALIASES[lower]
    return stripped


# ---------------------------------------------------------------------------
# Full-year hourly time index for 2019
# ---------------------------------------------------------------------------

_YEAR_INDEX_2019: pd.DatetimeIndex = pd.date_range(
    "2019-01-01", periods=8760, freq="h"
)


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_cold_loads(path: str) -> pd.DataFrame:
    """
    Load market_hourly_cold_loads_2019.csv and return a tidy long-format
    DataFrame with a complete 8 760-hour time index for every market.

    Columns returned
    ----------------
    market_id      : str      (normalised string ID)
    timestamp      : datetime (built from "date" + "hour")
    state          : str      (canonical, matches solar-CF header)
    urban_class    : str
    cold_load_kW   : float    (0.0 where the market has no data for that hour)
    hourly_meat_kg : float    (0.0 for zero-filled hours; omitted if not in CSV)

    Note: x/y coordinates are NOT returned here — they come from the
    separate market locations file via load_market_locations().

    Strategy for missing hours
    --------------------------
    Some markets have fewer than 8 760 rows (markets that operate only on
    certain days of the year).  Missing hours are filled with cold_load_kW = 0
    so the optimizer sees a complete, zero-demand window rather than a gap.
    """
    df = pd.read_csv(path, dtype={"market_id": "Int64"})

    # --- normalise market_id to str ---
    df["market_id"] = df["market_id"].apply(normalise_id)

    # --- build timestamp ---
    if "date" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["date"])
            + pd.to_timedelta(df["hour"], unit="h")
        )
    else:
        df["timestamp"] = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                columns={"Year": "year", "Month": "month", "Day": "day",
                         "Hour": "hour", "Minute": "minute"}
            )
        )

    # --- canonical state names ---
    df["state"] = df["state"].apply(_normalise_state)

    # --- keep only the columns needed downstream ---
    # x/y coordinates are NOT in this CSV — they come from the markets
    # locations file loaded separately via load_market_locations().
    # hourly_meat_kg is optional; keep it when available.
    base_keep = ["market_id", "timestamp", "state", "urban_class", "cold_load_kW"]
    has_meat  = "hourly_meat_kg" in df.columns
    keep      = base_keep + (["hourly_meat_kg"] if has_meat else [])
    df = df[keep].copy()

    # --- zero-fill missing hours for every market ---
    # Extract static metadata (first occurrence is sufficient)
    meta_cols = ["state", "urban_class"]
    meta = (
        df.drop_duplicates("market_id")
        .set_index("market_id")[meta_cols]
    )

    filled_parts: list[pd.DataFrame] = []
    for mid, grp in df.groupby("market_id", sort=False):
        ts_grp = grp.set_index("timestamp")
        # Reindex cold load (zero-fill)
        grp_indexed = (
            ts_grp["cold_load_kW"]
            .reindex(_YEAR_INDEX_2019, fill_value=0.0)
            .rename("cold_load_kW")
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        grp_indexed["market_id"] = mid
        # Re-attach static metadata
        for col in meta_cols:
            grp_indexed[col] = meta.loc[mid, col]
        # Reindex hourly_meat_kg (zero-fill) if present
        if has_meat:
            meat_series = (
                ts_grp["hourly_meat_kg"]
                .reindex(_YEAR_INDEX_2019, fill_value=0.0)
            )
            grp_indexed["hourly_meat_kg"] = meat_series.values
        filled_parts.append(grp_indexed)

    result = pd.concat(filled_parts, ignore_index=True)
    result = result[keep]          # enforce column order
    result = result.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    return result


def load_solar_cf(path: str) -> pd.DataFrame:
    """
    Load the Renewables.ninja state-level PV capacity-factor CSV.

    Returns a DataFrame with
      • DatetimeIndex  — 8 760 hourly timestamps for 2019
      • Columns        — one per state (37 states), all float [0, 1]

    The "Nigeria" national-average column and the raw "time" column are
    dropped; only the 37 per-state columns are kept.
    """
    raw = pd.read_csv(path, comment="#")

    # Build DatetimeIndex — column is "local_time" in some file variants,
    # "time" in others; dayfirst handles the dd/mm/YYYY format used here.
    time_col = "local_time" if "local_time" in raw.columns else "time"
    time_index = pd.to_datetime(raw[time_col], dayfirst=True)

    # Drop auxiliary columns; keep per-state CF only
    drop_cols = {time_col, "Nigeria"}
    cf_cols = [c for c in raw.columns if c not in drop_cols]
    cf = raw[cf_cols].copy()
    cf.index = time_index
    cf.index.name = "timestamp"
    cf = cf.astype(float)

    if len(cf) != 8760:
        warnings.warn(
            f"Solar CF file has {len(cf)} rows; expected 8 760. "
            "Proceeding, but check the file.",
            stacklevel=2,
        )

    return cf


def load_market_locations(
    cold_loads_df: pd.DataFrame,
    markets_csv: str = None,
) -> pd.DataFrame:
    """
    Return one row per market with coordinates and state.

    Two data sources are needed:
      • cold_loads_df  — provides the set of active market IDs and their state
                         names (already normalised by _normalise_state).
      • markets_csv    — GIS file (e.g. markets_with_urban_class_3km.csv) that
                         contains x (longitude) and y (latitude) keyed on FID.

    If markets_csv is provided the function reads x/y from there and joins on
    market_id == FID.  Otherwise it falls back to looking for x/y columns
    directly in cold_loads_df (legacy path).

    Returns a DataFrame with columns:
        market_id  (str), x (longitude), y (latitude), state (str)

    Rows are sorted by state then market_id.
    """
    # Unique market metadata from cold-loads (already has normalised state)
    base = (
        cold_loads_df
        .drop_duplicates("market_id")[["market_id", "state"]]
        .copy()
    )

    if markets_csv is not None:
        gis = pd.read_csv(markets_csv, dtype={"uniq_id": "Int64"})
        gis["market_id"] = gis["uniq_id"].apply(normalise_id)
        gis = gis[["market_id", "x", "y"]].copy()
        locs = base.merge(gis, on="market_id", how="left")
    else:
        # Legacy: x/y expected directly in cold_loads_df
        base2 = (
            cold_loads_df
            .drop_duplicates("market_id")[["market_id", "x", "y", "state"]]
            .copy()
        )
        locs = base2

    locs = (
        locs
        .sort_values(["state", "market_id"])
        .reset_index(drop=True)
    )
    return locs


# ---------------------------------------------------------------------------
# Road distance matrix loader
# ---------------------------------------------------------------------------

def load_road_distances(path: str) -> dict:
    """
    Load pre-computed road distance matrix.
    Returns dict {(uniq_id_a, uniq_id_b): road_km}.
    Computed via OpenRouteService API (OpenStreetMap data).
    """
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Pivot / alignment helpers
# ---------------------------------------------------------------------------

def group_markets_by_state(locations: pd.DataFrame) -> dict:
    """Return a dict mapping state_name → sorted list of market_ids."""
    return (
        locations.groupby("state")["market_id"]
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )


def pivot_cold_loads(cold_loads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-format cold-loads DataFrame into a wide matrix.

    Returns
    -------
    pd.DataFrame
        Shape  : (8 760, n_markets)
        Index  : DatetimeIndex — full hourly 2019 timestamps
        Columns: str market_ids

    Missing (timestamp, market) pairs are filled with 0.
    """
    pivot = cold_loads_df.pivot_table(
        index="timestamp",
        columns="market_id",
        values="cold_load_kW",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot.index.name = "timestamp"
    # Ensure the full year index is present (reindex fills any gaps with 0)
    pivot = pivot.reindex(_YEAR_INDEX_2019, fill_value=0.0)
    pivot.columns.name = None
    return pivot


def align_solar_to_markets(
    solar_df: pd.DataFrame,
    locations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map the state-level solar capacity-factor series to each individual market.

    Parameters
    ----------
    solar_df     : output of load_solar_cf()      — (8760 × 37 states)
    locations_df : output of load_market_locations() — (n_markets × 4)

    Returns
    -------
    pd.DataFrame
        Shape  : (8 760, n_markets)
        Index  : DatetimeIndex
        Columns: str market_ids
        Values : hourly solar PV capacity factors [0, 1]

    State-name matching is case-insensitive so "Fct" / "FCT" mismatches
    are handled transparently.
    """
    # Build a case-normalised lookup: lower → actual solar column name
    solar_col_map: dict[str, str] = {c.lower(): c for c in solar_df.columns}

    missing_states: set[str] = set()
    series_dict: dict[str, np.ndarray] = {}

    for _, row in locations_df.iterrows():
        mid = row["market_id"]
        state = row["state"]
        col = solar_col_map.get(state.lower())
        if col is None:
            missing_states.add(state)
            series_dict[mid] = np.zeros(len(solar_df))
        else:
            series_dict[mid] = solar_df[col].values

    if missing_states:
        warnings.warn(
            f"No solar CF column found for states: {sorted(missing_states)}. "
            "Their markets will have zero solar generation.",
            stacklevel=2,
        )

    result = pd.DataFrame(series_dict, index=solar_df.index)
    result.index.name = "timestamp"
    result.columns.name = None
    return result


# ---------------------------------------------------------------------------
# Representative-period selection
# ---------------------------------------------------------------------------

def select_representative_periods(
    load_pivot: pd.DataFrame,
    n_weeks: int = 12,
    seed: int = 42,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Select n_weeks representative weeks from the full 8 760-h load matrix
    using k-means clustering on aggregate weekly load profiles.

    Each of the 52 complete weeks in 2019 is described by a 168-element
    feature vector (aggregate cold load across all markets).  K-means
    groups the 52 weeks into n_weeks clusters; one week per cluster is
    chosen as the representative.

    Parameters
    ----------
    load_pivot : (8 760 × n_markets) DataFrame, output of pivot_cold_loads()
    n_weeks    : number of representative weeks to select
    seed       : random state for k-means

    Returns
    -------
    selected_timestamps : DatetimeIndex
        All hourly timestamps belonging to the selected representative weeks
        (length = n_weeks × 168).
    weights : np.ndarray, shape (n_weeks × 168,)
        Each representative hour is weighted by the number of original weeks
        mapped to its cluster, so that costs computed over the representative
        subset scale correctly to a full year.
    """
    HOURS_PER_WEEK = 168
    N_COMPLETE_WEEKS = 52                # 52 × 168 = 8 736 h  (first 364 days)

    # Aggregate demand across all markets for fast clustering
    total_load = load_pivot.values.sum(axis=1)   # (8760,)

    # Reshape into (52, 168) — use only the first 8 736 hours
    weekly_profiles = total_load[: N_COMPLETE_WEEKS * HOURS_PER_WEEK].reshape(
        N_COMPLETE_WEEKS, HOURS_PER_WEEK
    )

    if n_weeks >= N_COMPLETE_WEEKS:
        # Use all weeks; weight = 1 everywhere
        selected_indices = list(range(N_COMPLETE_WEEKS))
        cluster_counts = {i: 1 for i in range(N_COMPLETE_WEEKS)}
    else:
        km = KMeans(n_clusters=n_weeks, random_state=seed, n_init=20)
        labels = km.fit_predict(weekly_profiles)   # (52,)

        # For each cluster pick the week closest to its centroid
        selected_indices: list[int] = []
        cluster_counts: dict[int, int] = {}
        for c in range(n_weeks):
            mask = labels == c
            cluster_counts[c] = int(mask.sum())
            dists = np.linalg.norm(
                weekly_profiles[mask] - km.cluster_centers_[c], axis=1
            )
            # Index within the cluster → global week index
            global_idx = np.where(mask)[0][np.argmin(dists)]
            selected_indices.append(int(global_idx))

    # Build the output timestamp index and weights array
    base_index = load_pivot.index                 # full DatetimeIndex (8 760 h)
    ts_list: list[pd.DatetimeIndex] = []
    wt_list: list[np.ndarray] = []

    for cluster_id, week_idx in enumerate(selected_indices):
        start = week_idx * HOURS_PER_WEEK
        end   = start + HOURS_PER_WEEK
        ts_list.append(base_index[start:end])
        weight = cluster_counts.get(cluster_id, 1)
        wt_list.append(np.full(HOURS_PER_WEEK, float(weight)))

    selected_timestamps = pd.DatetimeIndex(
        np.concatenate([t.values for t in ts_list])
    )
    weights = np.concatenate(wt_list)

    return selected_timestamps, weights


def prepare_optimization_snapshots(
    load_pivot: pd.DataFrame,
    config,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Build snapshots and objective weights from the shared Config settings."""
    if config.use_representative_weeks:
        snapshots, weights = select_representative_periods(
            load_pivot,
            n_weeks=config.n_representative_weeks,
            seed=config.representative_period_seed,
        )
        # The clustering covers 52 complete weeks (8,736 h). Normalise its
        # objective weights to the configured economic year (normally 8,760 h).
        weights = weights * (config.hours_per_year / weights.sum())
    else:
        snapshots = load_pivot.index
        weights = np.ones(len(snapshots), dtype=float)

    step = config.time_resolution_hours
    if not isinstance(step, int) or step < 1:
        raise ValueError("time_resolution_hours must be a positive integer")
    if step > 1:
        keep = np.arange(0, len(snapshots), step)
        snapshots = snapshots[keep]
        weights = weights[keep] * step

    return snapshots, weights


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def load_all(cfg, road_distances_path: str = None) -> dict:
    """
    Load and prepare all inputs described by cfg.

    Parameters
    ----------
    cfg                  : Config instance from config.py
    road_distances_path  : Optional path to road_distance_matrix.pkl.
                           If provided, the dict is loaded and returned
                           under the key 'road_distances'.  Falls back to
                           cfg.road_distances_path when not supplied.

    Returns
    -------
    dict with keys:
        cold_loads      — long-format DataFrame (full year, zero-filled)
        solar_cf        — (8760 × 37) state-level capacity factors
        locations       — (n_markets × 4) market metadata
        load_pivot      — (8760 × n_markets) cold loads wide matrix
        solar_pivot     — (8760 × n_markets) per-market solar CF matrix
        timestamps      — DatetimeIndex used for the optimisation
        weights         — np.ndarray snapshot weights (all 1 s if full year)
        n_snapshots     — int length of timestamps
        road_distances  — dict {(id_a, id_b): road_km} or None if not loaded
    """
    print("Loading cold loads …")
    cold_loads = load_cold_loads(cfg.hourly_loads_csv)

    print("Loading solar capacity factors …")
    solar_cf = load_solar_cf(cfg.solar_csv)

    print("Extracting market locations …")
    locations = load_market_locations(cold_loads)

    print("Pivoting cold loads …")
    load_pivot = pivot_cold_loads(cold_loads)

    print("Aligning solar CFs to markets …")
    solar_pivot = align_solar_to_markets(solar_cf, locations)

    # Time index and snapshot weights
    if cfg.use_representative_weeks:
        print(f"Selecting {cfg.n_representative_weeks} representative weeks …")
        timestamps, weights = prepare_optimization_snapshots(load_pivot, cfg)
        load_pivot  = load_pivot.loc[timestamps]
        solar_pivot = solar_pivot.loc[timestamps]
    else:
        timestamps, weights = prepare_optimization_snapshots(load_pivot, cfg)
        load_pivot  = load_pivot.loc[timestamps]
        solar_pivot = solar_pivot.loc[timestamps]

    # Road distance matrix
    _rd_path = road_distances_path or getattr(cfg, "road_distances_path", None)
    road_distances = None
    if _rd_path:
        print(f"Loading road distances from {_rd_path} …")
        road_distances = load_road_distances(_rd_path)
        print(f"  {len(road_distances):,} market pairs loaded.")

    print(
        f"Data ready: {len(locations)} markets, "
        f"{locations['state'].nunique()} states, "
        f"{len(timestamps)} snapshots."
    )

    return {
        "cold_loads"     : cold_loads,
        "solar_cf"       : solar_cf,
        "locations"      : locations,
        "load_pivot"     : load_pivot,
        "solar_pivot"    : solar_pivot,
        "timestamps"     : timestamps,
        "weights"        : weights,
        "n_snapshots"    : len(timestamps),
        "road_distances" : road_distances,
    }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
