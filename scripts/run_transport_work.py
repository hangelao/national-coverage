"""Corrected C3 80/100% fixed-July transport-work workflow.

Transport work follows the original flow accounting: annual weighted source
flow on each conventional transport link, multiplied by road distance.
"""
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

COVERAGE = (0.80, 1.00)

def make_transport_exporter(output_dir, coverage, cfg):
    def export(network, row):
        label = f"coverage_{coverage:.2f}"
        links = network.links.loc[network.links.index.str.endswith("_conv")].copy()
        p0 = network.links_t.p0.loc[:, links.index].clip(lower=0.0)
        active = p0.columns[p0.max(axis=0) > 1e-8]
        links = links.loc[active]
        p0 = p0.loc[:, active]
        weights = network.snapshot_weightings["objective"]
        summary_rows = []
        dispatch_parts = []
        for name in active:
            source = str(links.at[name, "bus0"]).removesuffix("_cold")
            destination = str(links.at[name, "bus1"]).removesuffix("_cold")
            distance = cfg.road_distances.get((int(source), int(destination)))
            if distance is None:
                distance = cfg.road_distances[(int(destination), int(source))]
            flow = p0[name]
            weighted_flow = flow * weights
            annual_flow = float(weighted_flow.sum())
            summary_rows.append({
                "link": name, "from_market": source, "to_market": destination,
                "road_distance_km": distance,
                "capacity_kw": float(links.at[name, "p_nom_opt"]),
                "annual_flow_kwh": annual_flow,
                "transport_work_kwh_km": annual_flow * distance,
                "marginal_cost_per_kwh": float(links.at[name, "marginal_cost"])})
            dispatch_parts.append(pd.DataFrame({
                "timestamp": network.snapshots, "link": name,
                "from_market": source, "to_market": destination,
                "road_distance_km": distance, "flow_kw": flow.to_numpy(),
                "snapshot_weight_hours": weights.to_numpy(),
                "weighted_flow_kwh": weighted_flow.to_numpy(),
                "transport_work_kwh_km": weighted_flow.to_numpy() * distance}))
        detail_dir = output_dir / label
        detail_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(detail_dir / "transport_link_summary.csv", index=False)
        dispatch = (pd.concat(dispatch_parts, ignore_index=True)
                    if dispatch_parts else pd.DataFrame(columns=[
                        "timestamp", "link", "from_market", "to_market",
                        "road_distance_km", "flow_kw", "snapshot_weight_hours",
                        "weighted_flow_kwh", "transport_work_kwh_km"]))
        dispatch.to_csv(detail_dir / "transport_link_dispatch.csv.gz",
                        index=False, compression="gzip")
        row["transport_work_kwh_km"] = (
            float(summary["transport_work_kwh_km"].sum())
            if not summary.empty else 0.0)
        row["transport_flow_kwh"] = (
            float(summary["annual_flow_kwh"].sum())
            if not summary.empty else 0.0)
    return export

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs" / "transport_work")
    args = parser.parse_args()
    assert_fixed_july()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    locations, solar, cold, states = load_fixed_july_inputs()
    cfg = fixed_july_config()
    rows = []
    for coverage in COVERAGE:
        row = solve_budget_point(
            "C3", states, solar, cold, locations, SNAPSHOTS, WEIGHTS, cfg,
            budget=None, coverage=coverage,
            capacity_export_dir=str(args.output_dir / "capacities"),
            solution_exporter=make_transport_exporter(
                args.output_dir, coverage, cfg))
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            args.output_dir / "coverage_frontier.csv", index=False)
    work = {round(r["coverage_target"], 2): r["transport_work_kwh_km"]
            for r in rows}
    pd.DataFrame([{
        "transport_work_80_kwh_km": work[0.8],
        "transport_work_100_kwh_km": work[1.0],
        "final_20_increment_kwh_km": work[1.0] - work[0.8],
        "final_20_share_percent":
            100.0 * (work[1.0] - work[0.8]) / work[1.0],
    }]).to_csv(
        args.output_dir / "final_20_transport_work_summary.csv", index=False)

if __name__ == "__main__":
    main()
