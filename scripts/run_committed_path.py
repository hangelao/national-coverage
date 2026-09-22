"""Corrected fixed-July equivalent of the original committed-build workflow."""
from __future__ import annotations
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
import pandas as pd
from budget_frontier import solve_budget_point
from corrected_fixed_july_common import (ROOT, SNAPSHOTS, WEIGHTS,
    assert_fixed_july, fixed_july_config, load_fixed_july_inputs)
from committed_path import make_lockin_mutator

COVERAGE = tuple(i / 100 for i in range(5, 101, 5))
FREE_RESULTS = ROOT / "solar_coverage_corrected_four_hour_july_20260902" / "coverage_frontier.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs" / "committed_path")
    parser.add_argument("--free-frontier", required=True, type=Path,
        help="coverage_frontier.csv from run_coverage_frontier.py")
    args = parser.parse_args()
    assert_fixed_july()
    free = pd.read_csv(args.free_frontier)
    free = free[free["coverage_target"].notna()].copy()
    free["coverage_key"] = free["coverage_target"].round(8)
    free_cost = free.set_index(["config", "coverage_key"])["objective_usd"]
    required = {(a, round(c, 8)) for a in ("C1", "C2") for c in COVERAGE}
    missing = required.difference(free_cost.index)
    if missing:
        raise RuntimeError(f"missing unrestricted comparators: {sorted(missing)}")
    locations, solar, cold, states = load_fixed_july_inputs()
    cfg = fixed_july_config()
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap_dir = args.output_dir / "capacities"
    for architecture in ("C1", "C2"):
        previous = None
        for coverage in COVERAGE:
            row = solve_budget_point(
                architecture, states, solar, cold, locations, SNAPSHOTS,
                WEIGHTS, cfg, budget=None, coverage=coverage,
                capacity_export_dir=str(cap_dir),
                network_mutator=(make_lockin_mutator(previous)
                                 if previous is not None else None))
            comparator = float(free_cost.loc[(architecture, round(coverage, 8))])
            row["unrestricted_objective_usd"] = comparator
            row["committed_penalty_percent"] = 100.0 * (
                row["objective_usd"] / comparator - 1.0)
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                args.output_dir / "committed_path_with_comparator.csv",
                index=False)
            previous = pd.read_csv(
                cap_dir / f"capacities_{architecture}_solar_coverage_{coverage:.3f}.csv",
                index_col=0).fillna(0.0)

if __name__ == "__main__":
    main()
