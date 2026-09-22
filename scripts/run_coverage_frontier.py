"""
run_budget_frontier.py
----------------------
Entry point for the budget-constrained access frontier.

Loads exactly the same inputs as run.py, then sweeps a national capital budget
across the requested configurations and writes budget_frontier.csv.

Usage
-----
    python run_budget_frontier.py                       # C1 + C2, full national
    python run_budget_frontier.py --configs C1 C2 C3    # add the national mesh
    python run_budget_frontier.py --debug               # 3 states, 5 markets each

C3 is opt-in: the full mesh LP must be re-solved at every budget point, which
is roughly an order of magnitude more expensive than C1/C2.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import argparse
import os
import time
from dataclasses import replace

import numpy as np

from budget_frontier import (
    DEFAULT_VOLL,
    check_voll_dominance,
    run_budget_frontier,
    run_coverage_frontier,
)
from config import Config
from input_paths import CORRECTED_LOAD, MARKETS, ROAD_DISTANCES, SOLAR_CF
from data_loader import (
    align_solar_to_markets,
    group_markets_by_state,
    load_cold_loads,
    load_market_locations,
    load_road_distances,
    load_solar_cf,
    pivot_cold_loads,
    prepare_optimization_snapshots,
)


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=["C1", "C2"],
                    choices=["C1", "C2", "C3"])
    ap.add_argument("--output-dir",
                    default=os.path.join(_ROOT, "outputs", "coverage_frontier"))
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
                    help="budget levels as fractions of unconstrained C1 capex")
    ap.add_argument("--voll", type=float, default=DEFAULT_VOLL)
    ap.add_argument("--supply", choices=["solar", "diesel"], default="solar",
                    help="generation technology at every market")
    ap.add_argument("--mode", choices=["budget", "coverage"], default="budget",
                    help="budget: served demand at capped capex (Fig. 2b). "
                         "coverage: least total cost to serve each fraction — "
                         "the deployment-path formulation; --fractions are "
                         "then coverage levels, not budget levels")
    ap.add_argument("--reference-capex", type=float, default=None,
                    help="pin the budget grid to this reference (USD/yr) "
                         "instead of this run's own C1 anchor — use the solar "
                         "C1 full-service capex when sweeping diesel so both "
                         "frontiers share an x-axis")
    ap.add_argument("--solver-threads", type=int, default=8)
    ap.add_argument(
        "--max-transport-hours", type=float, default=None,
        help="restrict transport links to journeys completable within this "
             "many hours (physical consistency: links deliver same-snapshot). "
             "Set to the dispatch interval, e.g. 3.")
    ap.add_argument(
        "--rep-weeks", type=int, default=None,
        help="override Config.n_representative_weeks (robustness checks)")
    ap.add_argument(
        "--time-res-hours", type=int, default=None,
        help="override Config.time_resolution_hours (robustness checks)")
    ap.add_argument(
        "--bar-conv-tol", type=float, default=None,
        help="override Gurobi BarConvTol for the C3 barrier solve (default "
             "1e-6 from config). Use a tighter value such as 1e-8 to test "
             "whether reported served fractions are convergence-limited.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-states", nargs="+",
                    default=["Taraba", "Plateau", "Katsina"])
    ap.add_argument("--debug-markets-per-state", type=int, default=5)
    args = ap.parse_args()

    hourly_loads_csv = str(CORRECTED_LOAD)
    solar_csv = str(SOLAR_CF)
    road_distances_path = str(ROAD_DISTANCES)
    markets_csv = str(MARKETS)

    cfg = replace(Config(), solver_threads=args.solver_threads)
    if args.max_transport_hours is not None:
        cfg = replace(cfg, max_transport_link_hours=args.max_transport_hours)
        print(f"  [physical] transport links capped at "
              f"{cfg.van_speed_kmh * args.max_transport_hours:.0f} km "
              f"({args.max_transport_hours:g} h at {cfg.van_speed_kmh:g} km/h)")
    if args.rep_weeks is not None:
        cfg = replace(cfg, n_representative_weeks=args.rep_weeks)
        print(f"  [robustness] n_representative_weeks={args.rep_weeks}")
    if args.time_res_hours is not None:
        cfg = replace(cfg, time_resolution_hours=args.time_res_hours)
        print(f"  [robustness] time_resolution_hours={args.time_res_hours}")
    if args.bar_conv_tol is not None:
        cfg = replace(cfg, c3_barrier_options={
            **cfg.c3_barrier_options, "BarConvTol": args.bar_conv_tol})
        print(f"  [solver] BarConvTol overridden to {args.bar_conv_tol:g}")
    cfg.road_distances = load_road_distances(road_distances_path)
    check_voll_dominance(cfg, args.voll)

    print("=" * 64)
    print("  Budget-Constrained Cold-Chain Access Frontier")
    print("=" * 64)
    t0 = time.time()

    cold_loads = load_cold_loads(hourly_loads_csv)
    solar_wide = load_solar_cf(solar_csv)
    locations = load_market_locations(cold_loads, markets_csv=markets_csv)
    load_pivot = pivot_cold_loads(cold_loads)
    solar_pivot = align_solar_to_markets(solar_wide, locations)
    print(f"  {len(locations)} markets, {locations['state'].nunique()} states "
          f"[{time.time() - t0:.1f}s]")

    snapshots, weights = prepare_optimization_snapshots(load_pivot, cfg)
    print(f"  {len(snapshots)} snapshots, weight sum={weights.sum():.0f}")

    if args.debug:
        keep: list = []
        for st in args.debug_states:
            mids = locations.loc[locations["state"] == st, "market_id"].tolist()
            keep.extend(mids[: args.debug_markets_per_state])
        locations = locations[locations["market_id"].isin(keep)].copy()
        print(f"  [DEBUG] {len(args.debug_states)} states, {len(keep)} markets")

    all_mids = locations["market_id"].tolist()
    lp_r = load_pivot.loc[snapshots, [m for m in all_mids if m in load_pivot.columns]]
    sp_r = solar_pivot.loc[snapshots, [m for m in all_mids if m in solar_pivot.columns]]

    solar_cfs = {m: sp_r[m].values for m in all_mids if m in sp_r.columns}
    cold_load_arrays = {m: lp_r[m].values for m in all_mids if m in lp_r.columns}
    state_groups = group_markets_by_state(locations)

    total_demand_kwh = float(
        (lp_r.to_numpy(dtype=float) * weights[:, None]).sum()
    )
    print(f"  annual cooling demand in scope: {total_demand_kwh / 1e6:.2f} GWh_cold "
          f"({total_demand_kwh / (3.14 * 36 / 3600) / 1e6:,.1f} kt meat-equivalent)")

    if args.mode == "coverage":
        run_coverage_frontier(
            state_groups=state_groups,
            solar_cfs=solar_cfs,
            cold_loads=cold_load_arrays,
            locations=locations,
            snapshots=snapshots,
            weights=weights,
            config=cfg,
            output_dir=args.output_dir,
            configs=tuple(args.configs),
            coverage_levels=tuple(args.fractions),
            supply=args.supply,
        )
        print(f"\nTotal runtime: {time.time() - t0:.1f}s")
        return

    run_budget_frontier(
        state_groups=state_groups,
        solar_cfs=solar_cfs,
        cold_loads=cold_load_arrays,
        locations=locations,
        snapshots=snapshots,
        weights=weights,
        config=cfg,
        output_dir=args.output_dir,
        configs=tuple(args.configs),
        budget_fractions=tuple(args.fractions),
        voll=args.voll,
        supply=args.supply,
        reference_capex_override=args.reference_capex,
    )

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
