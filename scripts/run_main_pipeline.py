"""
run.py
------
Single entry-point for the solar cold-chain optimisation pipeline.

Usage
-----
    python run.py          # runs the built-in debug case

    or import and call run() programmatically:

        from scripts.run_main_pipeline import run
        results = run(
            hourly_loads_csv="...",
            solar_csv="...",
            output_dir="...",
        )
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import Config
from data_loader import (
    load_cold_loads,
    load_road_distances,
    load_solar_cf,
    load_market_locations,
    normalise_id,
    pivot_cold_loads,
    align_solar_to_markets,
    prepare_optimization_snapshots,
    group_markets_by_state,
)
from optimizer import run_standalone, run_intrastate, run_national_connected
from results import (
    compile_state_summary,
    compile_technology_mix,
    compile_transport_breakdown,
    save_all_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.1f} s"


# ---------------------------------------------------------------------------
# Top-level worker functions for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------

def _run_standalone_state(args):
    state, mids, sc, cl, snapshots, weights, cfg = args
    return state, run_standalone(state, mids, sc, cl, snapshots, weights, cfg)


def _run_intrastate_state(args):
    state, mids, sc, cl, locs_sub, snapshots, weights, cfg = args
    return state, run_intrastate(state, mids, sc, cl,
                                 locs_sub, snapshots, weights, cfg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    hourly_loads_csv: str,
    solar_csv: str,
    output_dir: str,
    road_distances_path: str,
    markets_csv: str = None,
    states: list[str] = None,
    solver: str = "gurobi",
    use_representative_weeks: Optional[bool] = None,
    n_representative_weeks: Optional[int] = None,
    seed: Optional[int] = None,
    debug: bool = False,
    debug_markets_per_state: int = 3,
    debug_states: list[str] = None,
    n_workers: int = 4,
    solver_threads: int = 4,
    solver_threads_c3: Optional[int] = None,
    time_resolution_hours: Optional[int] = None,
) -> dict:
    """
    Run the full three-configuration cold-chain optimisation pipeline.

    Parameters
    ----------
    hourly_loads_csv      : Path to market hourly cold loads CSV.
    solar_csv             : Path to solar capacity factor CSV.
    output_dir            : Directory for all output files.
    road_distances_path   : Path to road_distance_matrix_full.pkl — pre-computed
                            road distances (km) via OpenRouteService API.
                            The pipeline aborts immediately if the file is
                            not found.
    markets_csv           : Path to market GIS file with x/y coordinates
                            (e.g. markets_with_urban_class_3km.csv).
                            Required — x/y are not in hourly_loads_csv.
    states                : States to include (None = all states).
    solver                : LP solver — must be "gurobi".
    use_representative_weeks: Optional override of the shared Config setting.
    n_representative_weeks  : Optional override of the shared Config setting.
    seed                    : Optional override of the shared Config seed.
    debug                   : If True, restrict to a small subset for
                              fast end-to-end testing.
    debug_markets_per_state : Markets per state kept in debug mode.
    debug_states            : States to run in debug mode
                              (default ["Abia"] when debug=True).
    n_workers               : Number of parallel worker processes for
                              Config 1 and Config 2 (default 4).
    solver_threads_c3      : Gurobi threads for the Config 3 solve
                              specifically (one single national LP, no
                              competing processes at that point — safe to
                              use most/all of your allocated CPUs). If
                              None, falls back to cfg.solver_threads (same
                              value used for Config 1/2's many parallel
                              worker processes, which should stay LOW so
                              n_workers * cfg.solver_threads doesn't
                              oversubscribe your CPU allocation).

    Returns
    -------
    dict with keys:
        config, standalone_results, intrastate_results,
        interstate_result, state_summary, technology_mix,
        transport_breakdown, locations
    """
    t_pipeline = time.time()

    # ── Load road distance matrix (fail fast) ─────────────────────────────
    if not os.path.exists(road_distances_path):
        raise FileNotFoundError(
            f"Road distance matrix not found: '{road_distances_path}'. "
            "Generate it with compute_road_distances.py before running the pipeline."
        )
    print(f"Loading road distances from '{road_distances_path}' …")
    road_distances = load_road_distances(road_distances_path)
    print(f"  {len(road_distances):,} market pairs loaded.")

    # ── Build Config ──────────────────────────────────────────────────────
    cfg = Config(
        include_pcm        = False,
        hourly_loads_csv  = hourly_loads_csv,
        solar_csv         = solar_csv,
        output_dir        = output_dir,
        road_distances_path = road_distances_path,
        road_distances    = road_distances,
        solver            = solver,
        solver_threads    = solver_threads,
        use_representative_weeks=(Config.use_representative_weeks
                                  if use_representative_weeks is None
                                  else use_representative_weeks),
        n_representative_weeks=(Config.n_representative_weeks
                                if n_representative_weeks is None
                                else n_representative_weeks),
        representative_period_seed=(Config.representative_period_seed
                                    if seed is None else seed),
        time_resolution_hours=(Config.time_resolution_hours
                               if time_resolution_hours is None
                               else time_resolution_hours),
    )
    # Research-design invariant: all three main configurations use the same
    # battery-only technology set. PCM is supplementary and never launched
    # by the standard run.py workflow.
    cfg_c1 = replace(cfg, include_pcm=False)
    cfg_c2 = replace(cfg, include_pcm=False)

    print("=" * 64)
    print("  Solar Cold-Chain National Network Optimisation Pipeline")
    print("=" * 64)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    t0 = time.time()

    cold_loads = load_cold_loads(hourly_loads_csv)
    solar_wide = load_solar_cf(solar_csv)
    locations  = load_market_locations(cold_loads, markets_csv=markets_csv)
    load_pivot = pivot_cold_loads(cold_loads)
    solar_pivot = align_solar_to_markets(solar_wide, locations)

    print(f"  {len(locations)} markets, "
          f"{locations['state'].nunique()} states  [{_elapsed(t0)}]")

    # ── Step 2: Select snapshots ──────────────────────────────────────────
    snapshots, weights = prepare_optimization_snapshots(load_pivot, cfg)
    period_label = (f"{cfg.n_representative_weeks} representative weeks"
                    if cfg.use_representative_weeks else "full year")
    print(f"  {period_label}, {cfg.time_resolution_hours}h resolution "
          f"({len(snapshots)} snapshots, weight sum={weights.sum():.0f})")

    # ── Apply debug constraints ───────────────────────────────────────────
    if debug:
        _states = debug_states if debug_states else ["Abia"]
        # Trim each debug state to the first N markets
        keep_ids: list[str] = []
        for st in _states:
            mids = locations.loc[locations["state"] == st, "market_id"].tolist()
            keep_ids.extend(mids[:debug_markets_per_state])
        locations  = locations[locations["market_id"].isin(keep_ids)].copy()
        cold_loads = cold_loads[cold_loads["market_id"].isin(keep_ids)].copy()
        states     = _states
        print(f"  [DEBUG] {len(_states)} states, "
              f"{len(keep_ids)} markets, "
              f"{len(snapshots)} snapshots")

    # ── Filter to requested states ────────────────────────────────────────
    if states is not None:
        locations  = locations[locations["state"].isin(states)].copy()
        cold_loads = cold_loads[cold_loads["market_id"].isin(
            locations["market_id"]
        )].copy()

    # ── Build per-market arrays (snapshot-aligned) ────────────────────────
    all_mids = locations["market_id"].tolist()

    # Restrict pivots to selected snapshots and markets
    lp_r = load_pivot.loc[snapshots, [m for m in all_mids if m in load_pivot.columns]]
    sp_r = solar_pivot.loc[snapshots, [m for m in all_mids if m in solar_pivot.columns]]

    solar_cfs: dict[str, np.ndarray] = {
        m: sp_r[m].values for m in all_mids if m in sp_r.columns
    }
    cold_load_arrays: dict[str, np.ndarray] = {
        m: lp_r[m].values for m in all_mids if m in lp_r.columns
    }

    # ── Group markets by state ────────────────────────────────────────────
    state_groups = group_markets_by_state(locations)
    active_states = sorted(state_groups.keys())
    n_states = len(active_states)

    print(f"  Running {n_states} states, "
          f"{len(all_mids)} markets total")

    # ── Checkpoint directory ──────────────────────────────────────────────
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Step 3: Config 1 — Standalone ────────────────────────────────────
    print(f"\n[2/5] Config 1 — Standalone")
    t0 = time.time()

    standalone_results: dict[str, dict] = {}
    _n_workers = min(n_workers, len(active_states))

    for state in active_states:
        ckpt = os.path.join(checkpoint_dir, f"c1_{state}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, "rb") as f:
                cached = pickle.load(f)
            cached_mids = {normalise_id(m) for m in cached.get("market_ids", [])}
            current_mids = {normalise_id(m) for m in state_groups[state]}
            if (cached_mids == current_mids
                    and cached.get("include_pcm") is False
                    and cached.get("checkpoint_compatibility_marker")
                    == cfg_c1.checkpoint_compatibility_marker):
                standalone_results[state] = cached
                print(f"  Config 1 | {state}: loaded from checkpoint")
            else:
                print(
                    f"  Config 1 | {state}: stale checkpoint ignored "
                    f"({len(cached_mids)} cached vs "
                    f"{len(current_mids)} current markets)"
                )

    pending_c1 = [s for s in active_states if s not in standalone_results]
    tasks = [
        (state,
         state_groups[state],
         {m: solar_cfs[m] for m in state_groups[state] if m in solar_cfs},
         {m: cold_load_arrays[m] for m in state_groups[state]
          if m in cold_load_arrays},
         snapshots, weights, cfg_c1)
        for state in pending_c1
    ]
    if tasks:
        with ProcessPoolExecutor(max_workers=min(_n_workers, len(tasks)),
                                 max_tasks_per_child=1) as executor:
            futures = {executor.submit(_run_standalone_state, t): t[0]
                       for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    state, result = future.result()
                    standalone_results[state] = result
                    with open(os.path.join(checkpoint_dir, f"c1_{state}.pkl"), "wb") as f:
                        pickle.dump(result, f)
                    print(f"  Config 1 | {i}/{len(pending_c1)}: {state} done")
                except Exception as exc:
                    state = futures[future]
                    print(f"  Config 1 | {state} ERROR: {exc}")

    c1_total = sum(r["total_cost"] for r in standalone_results.values())
    print(f"  Config 1 national total: {c1_total:,.0f} USD/year  [{_elapsed(t0)}]")

    # ── Step 4: Config 2 — Intra-state ───────────────────────────────────
    print(f"\n[3/5] Config 2 — Intra-state connected")
    t0 = time.time()

    intrastate_results: dict[str, dict] = {}

    for state in active_states:
        ckpt = os.path.join(checkpoint_dir, f"c2_{state}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, "rb") as f:
                cached = pickle.load(f)
            cached_mids = {normalise_id(m) for m in cached.get("market_ids", [])}
            current_mids = {normalise_id(m) for m in state_groups[state]}
            if (cached_mids == current_mids
                    and cached.get("include_pcm") is False
                    and cached.get("checkpoint_compatibility_marker")
                    == cfg_c2.checkpoint_compatibility_marker):
                intrastate_results[state] = cached
                print(f"  Config 2 | {state}: loaded from checkpoint")
            else:
                print(
                    f"  Config 2 | {state}: stale checkpoint ignored "
                    f"({len(cached_mids)} cached vs "
                    f"{len(current_mids)} current markets)"
                )

    pending_c2 = [s for s in active_states if s not in intrastate_results]
    tasks = [
        (state,
         state_groups[state],
         {m: solar_cfs[m] for m in state_groups[state] if m in solar_cfs},
         {m: cold_load_arrays[m] for m in state_groups[state]
          if m in cold_load_arrays},
         locations[locations["market_id"].isin(
             state_groups[state])].copy(),
         snapshots, weights, cfg_c2)
        for state in pending_c2
    ]
    if tasks:
        with ProcessPoolExecutor(max_workers=min(_n_workers, len(tasks)),
                                 max_tasks_per_child=1) as executor:
            futures = {executor.submit(_run_intrastate_state, t): t[0]
                       for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    state, result = future.result()
                    intrastate_results[state] = result
                    with open(os.path.join(checkpoint_dir, f"c2_{state}.pkl"), "wb") as f:
                        pickle.dump(result, f)
                    print(f"  Config 2 | {i}/{len(pending_c2)}: {state} done")
                except Exception as exc:
                    state = futures[future]
                    print(f"  Config 2 | {state} ERROR: {exc}")

    c2_total = sum(r["total_cost"] for r in intrastate_results.values())
    saving_c1_c2 = (c1_total - c2_total) / c1_total * 100 if c1_total > 0 else 0.0
    print(f"  Config 2 national total: {c2_total:,.0f} USD/year  "
          f"(saving C1→C2: {saving_c1_c2:.1f}%)  [{_elapsed(t0)}]")

    # ── Step 5: Config 3 — Full national mesh LP ─────────────────────────
    print("\n[4/5] Config 3 — Full national mesh LP")
    t0 = time.time()

    ckpt_c3 = os.path.join(checkpoint_dir, "c3_national.pkl")
    interstate_result = None

    if os.path.exists(ckpt_c3):
        with open(ckpt_c3, "rb") as f:
            cached = pickle.load(f)
        cached_mids = {normalise_id(m) for m in cached.get("market_ids", [])}
        current_mids = {normalise_id(m) for m in all_mids}
        if (cached_mids == current_mids
                and cached.get("n_snapshots") == len(snapshots)
                and cached.get("include_pcm") is False
                and cached.get("checkpoint_compatibility_marker")
                == cfg.checkpoint_compatibility_marker):
            interstate_result = cached
            c3_total = interstate_result.get("total_cost", float("nan"))
            inter_links = interstate_result.get("interstate_links_built", 0)
            print(f"  Config 3 | loaded from checkpoint")
            print(f"  Config 3 total cost    : {c3_total:,.0f} USD/year")
            print(f"  Inter-state links built: {inter_links}  [{_elapsed(t0)}]")
        else:
            print(
                f"  Config 3 | stale checkpoint ignored "
                f"({len(cached_mids)} cached vs {len(current_mids)} current markets, "
                f"{cached.get('n_snapshots')} cached vs {len(snapshots)} current snapshots)"
            )

    if interstate_result is None:
        try:
            print(f"  Config 3 | Full national mesh (all markets, all states, "
                  f"no threshold, no pruning floor)")
            cfg_c3 = replace(
                cfg,
                include_pcm=False,
                solver_threads=solver_threads_c3 if solver_threads_c3 is not None
                               else cfg.solver_threads,
            )
            interstate_result = run_national_connected(
                intrastate_results = intrastate_results,
                state_groups       = state_groups,
                solar_cfs          = solar_cfs,
                cold_loads         = cold_load_arrays,
                locations          = locations,
                snapshots          = snapshots,
                weights            = weights,
                config             = cfg_c3,
            )
            interstate_result["market_ids"]  = all_mids
            interstate_result["n_snapshots"] = len(snapshots)
            with open(ckpt_c3, "wb") as f:
                pickle.dump(interstate_result, f)
            inter_links = interstate_result.get("interstate_links_built", 0)
            c3_total    = interstate_result.get("total_cost", float("nan"))
            print(f"  Config 3 total cost    : {c3_total:,.0f} USD/year")
            print(f"  Inter-state links built: {inter_links}  [{_elapsed(t0)}]")
        except Exception as exc:
            print(f"  Config 3 FAILED: {exc}")
            interstate_result = {
                "total_cost"            : float("nan"),
                "total_cost_incl_spokes": float("nan"),
                "spoke_cost"            : 0.0,
                "hub_count"             : 0,
                "interstate_links_built": 0,
                "transport"             : None,
                "capacities"            : None,
                "include_pcm"           : False,
                "checkpoint_compatibility_marker": cfg.checkpoint_compatibility_marker,
            }

    # ── Step 6: Compile and save results ──────────────────────────────────
    print("\n[5/5] Compiling and saving results …")
    t0 = time.time()

    n_markets_per_state = {s: len(state_groups[s]) for s in active_states}

    c3_fair = interstate_result.get(
        "total_cost_incl_spokes", interstate_result.get("total_cost", float("nan"))
    )
    state_summary = compile_state_summary(
        standalone_results, intrastate_results,
        c3_fair, n_markets_per_state,
        hub_capacity_threshold_kw=cfg.hub_capacity_threshold_kw,
    )
    technology_mix = compile_technology_mix(
        standalone_results, intrastate_results
    )
    transport_breakdown = compile_transport_breakdown(
        intrastate_results, interstate_result, locations=locations
    )
    save_all_results(
        state_summary, technology_mix, transport_breakdown,
        interstate_result, locations, cfg, output_dir,
        standalone_results=standalone_results,
        intrastate_results=intrastate_results,
        cold_loads_df=cold_loads,
        cold_load_arrays=cold_load_arrays,
    )
    print(f"  Results saved to: {output_dir}  [{_elapsed(t0)}]")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  PIPELINE COMPLETE")
    print(f"  Total runtime          : {_elapsed(t_pipeline)}")
    print(f"  Config 1 (Standalone)  : {c1_total:>12,.0f} USD/year")
    print(f"  Config 2 (Intra-state) : {c2_total:>12,.0f} USD/year  "
          f"({saving_c1_c2:.1f}% saving)")
    if c3_fair == c3_fair:   # not nan
        saving_c1_c3 = (c1_total - c3_fair) / c1_total * 100 if c1_total > 0 else 0.0
        saving_c2_c3 = (c2_total - c3_fair) / c2_total * 100 if c2_total > 0 else 0.0
        print(f"  Config 3 (Hub-only)    : {c3_fair:>12,.0f} USD/year  "
              f"(C1→C3: {saving_c1_c3:.1f}%, C2→C3: {saving_c2_c3:.1f}%)")
    print("=" * 64)

    return {
        "config"               : cfg,
        "standalone_results"   : standalone_results,
        "intrastate_results"   : intrastate_results,
        "interstate_result"    : interstate_result,
        "state_summary"        : state_summary,
        "technology_mix"       : technology_mix,
        "transport_breakdown"  : transport_breakdown,
        "locations"            : locations,
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the final C1/C2/C3 model.")
    parser.add_argument("--hourly-loads", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("--markets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--solver-threads-c3", type=int, default=20)
    args = parser.parse_args()
    run(
        hourly_loads_csv=args.hourly_loads,
        solar_csv=args.solar,
        road_distances_path=args.roads,
        markets_csv=args.markets,
        output_dir=args.output_dir,
        debug=False,
        n_workers=args.n_workers,
        solver_threads=args.solver_threads,
        solver_threads_c3=args.solver_threads_c3,
    )
