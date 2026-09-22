"""Run the six corrected fixed-July cluster coverage cases for Panel c."""
from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from budget_frontier import solve_budget_point
from cluster_sweep import build_clusters, cluster_statistics
from corrected_fixed_july_common import (
    LOAD,
    ROOT,
    SNAPSHOTS,
    WEIGHTS,
    assert_fixed_july,
    fixed_july_config,
    load_fixed_july_inputs,
)


CASES = (
    ("size<=5_km<=50", 5.0, 50.0),
    ("size<=10_km<=100", 10.0, 100.0),
)
COVERAGES = (0.75, 0.85, 0.95)


def reference_costs(frontier: pd.DataFrame, coverage: float) -> tuple[float, float]:
    rows = frontier[frontier["coverage_target"].eq(coverage)]
    c1 = rows.loc[rows["config"].eq("C1"), "objective_usd"]
    c2 = rows.loc[rows["config"].eq("C2"), "objective_usd"]
    if len(c1) != 1 or len(c2) != 1:
        raise RuntimeError(
            f"Expected one corrected C1 and C2 reference at {coverage:.0%}; "
            f"found C1={len(c1)}, C2={len(c2)}"
        )
    return float(c1.iloc[0]), float(c2.iloc[0])


def benefit_share(c1: float, cluster: float, c2: float) -> float:
    return (c1 - cluster) / (c1 - c2) * 100.0


def write_rows(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-frontier", required=True, type=Path)
    parser.add_argument("--existing-cluster-sweep", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [p for p in args.output_dir.iterdir() if p.name != "run.log"]
    if unexpected:
        raise RuntimeError(
            "Output directory is not empty; refusing to overwrite: "
            + ", ".join(str(p) for p in unexpected)
        )

    assert_fixed_july()
    cfg = fixed_july_config()
    locations, solar_cfs, cold_loads, state_groups = load_fixed_july_inputs()
    frontier = pd.read_csv(args.reference_frontier)

    manifest = {
        "workflow": "corrected cluster coverage cases for Panel c",
        "corrected_load": str(LOAD),
        "cluster_logic": "cluster_sweep.build_clusters (complete linkage)",
        "cluster_cases": [
            {"grouping": g, "max_cluster_size": s, "max_cluster_km": d}
            for g, s, d in CASES
        ],
        "coverage_targets": list(COVERAGES),
        "snapshots": [str(SNAPSHOTS[0]), str(SNAPSHOTS[-1])],
        "snapshot_count": len(SNAPSHOTS),
        "snapshot_weight": float(WEIGHTS[0]),
        "snapshot_weight_sum": float(WEIGHTS.sum()),
        "solver": cfg.solver,
        "solver_threads": cfg.solver_threads,
        "reference_frontier": str(args.reference_frontier),
        "existing_full_coverage_cluster_sweep": str(args.existing_cluster_sweep),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    rows: list[dict] = []
    result_path = args.output_dir / "cluster_coverage_results.csv"
    for coverage in COVERAGES:
        c1_cost, c2_cost = reference_costs(frontier, coverage)
        for grouping, max_size, max_km in CASES:
            case_dir = args.output_dir / f"coverage_{coverage:.2f}" / grouping
            capacity_dir = case_dir / "capacities"
            capacity_dir.mkdir(parents=True, exist_ok=False)
            groups = build_clusters(state_groups, cfg, max_size, max_km)
            stats = cluster_statistics(groups, state_groups, cfg)
            print(
                f"[panelc-cluster] {coverage:.0%} {grouping}: "
                f"{len(groups)} clusters",
                flush=True,
            )
            started = time.time()
            base = {
                "coverage_target": coverage,
                "cluster_configuration": grouping,
                "max_cluster_size": max_size,
                "max_cluster_km": max_km,
                "c1_annual_system_cost_usd": c1_cost,
                "unrestricted_c2_annual_system_cost_usd": c2_cost,
                "capacity_output_dir": str(capacity_dir),
                "reference_source": str(args.reference_frontier),
                **stats,
            }
            try:
                point = solve_budget_point(
                    "C2",
                    groups,
                    solar_cfs,
                    cold_loads,
                    locations,
                    SNAPSHOTS,
                    WEIGHTS,
                    cfg,
                    budget=None,
                    coverage=coverage,
                    capacity_export_dir=str(capacity_dir),
                )
                row = {
                    **base,
                    "achieved_coverage": float(point["served_fraction"]),
                    "cluster_annual_system_cost_usd": float(point["objective_usd"]),
                    "annualised_capital_cost_usd_per_year": float(
                        point["capex_built_usd_per_year"]
                    ),
                    "n_links": int(point["n_links"]),
                    "solve_seconds": float(point["solve_seconds"]),
                    "solver_status": "ok",
                    "termination_condition": "optimal",
                    "benefit_captured_percent": benefit_share(
                        c1_cost, float(point["objective_usd"]), c2_cost
                    ),
                    "error": "",
                }
            except Exception as exc:
                row = {
                    **base,
                    "achieved_coverage": float("nan"),
                    "cluster_annual_system_cost_usd": float("nan"),
                    "annualised_capital_cost_usd_per_year": float("nan"),
                    "n_links": float("nan"),
                    "solve_seconds": time.time() - started,
                    "solver_status": "failed",
                    "termination_condition": "failed",
                    "benefit_captured_percent": float("nan"),
                    "error": repr(exc),
                }
                print(f"[panelc-cluster] FAILED: {exc!r}", flush=True)
            rows.append(row)
            write_rows(rows, result_path)
            write_rows([row], case_dir / "result.csv")
            print(
                f"[panelc-cluster] saved {coverage:.0%} {grouping} "
                f"status={row['solver_status']}",
                flush=True,
            )
            gc.collect()

    # Add the two existing valid full-coverage rows only to the final reporting
    # table; they are not rerun by this workflow.
    full_c1, full_c2 = reference_costs(frontier, 1.0)
    existing = pd.read_csv(args.existing_cluster_sweep)
    summary = list(rows)
    for grouping, max_size, max_km in CASES:
        match = existing[existing["grouping"].eq(grouping)]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one existing full-coverage row for {grouping}; "
                f"found {len(match)}"
            )
        old = match.iloc[0]
        cost = float(old["total_cost_usd_per_year"])
        summary.append(
            {
                "coverage_target": 1.0,
                "cluster_configuration": grouping,
                "max_cluster_size": max_size,
                "max_cluster_km": max_km,
                "achieved_coverage": 1.0,
                "cluster_annual_system_cost_usd": cost,
                "annualised_capital_cost_usd_per_year": float(
                    old["capex_usd_per_year"]
                ),
                "n_links": int(old["n_links"]),
                "solve_seconds": float("nan"),
                "solver_status": "ok_existing",
                "termination_condition": "optimal_existing",
                "c1_annual_system_cost_usd": full_c1,
                "unrestricted_c2_annual_system_cost_usd": full_c2,
                "benefit_captured_percent": benefit_share(
                    full_c1, cost, full_c2
                ),
                "capacity_output_dir": "",
                "reference_source": str(args.reference_frontier),
                "cluster_result_source": str(args.existing_cluster_sweep),
                "error": "",
            }
        )

    summary_path = args.output_dir / "cluster_coverage_summary_with_existing_100.csv"
    write_rows(summary, summary_path)
    failures = [r for r in rows if r["solver_status"] != "ok"]
    print(
        f"[panelc-cluster] COMPLETE new_cases={len(rows)} failures={len(failures)} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
