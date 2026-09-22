"""Run only the remaining corrected coarse continuous-diesel manuscript cases."""
from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_frontier import solve_budget_point
from corrected_fixed_july_common import (
    LOAD,
    ROOT,
    SNAPSHOTS,
    WEIGHTS,
    assert_fixed_july,
    fixed_july_config,
    load_fixed_july_inputs,
)
from diesel_base_case import emissions_breakdown


C1_MISSING = (
    0.05, 0.10, 0.15, 0.20,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.80, 0.85, 0.90, 0.95,
)
C1_REUSED = (0.25, 0.75, 1.00)
C2_CASES = (0.25, 0.75, 1.00)
C3_CASES = (1.00,)



def installed_capital_usd_from_capacities(frame: pd.DataFrame, cfg) -> float:
    """Installed equipment cost; fridge export is Link electrical-input kW."""
    return float(
        frame.get("solar_kw", 0.0).sum() * cfg.pv_capex_per_kw
        + frame.get("diesel_kw", 0.0).sum() * cfg.diesel_genset_capex_per_kw
        + frame.get("battery_kwh", 0.0).sum()
        * cfg.battery_energy_capex_per_kwh
        + frame.get("batt_charge_kw", 0.0).sum()
        * cfg.battery_power_capex_per_kw
        + frame.get("fridge_kw_cold", 0.0).sum()
        * cfg.fridge_cop * cfg.fridge_capex_per_kw_cold
    )


def installed_capital_usd_from_network(network, cfg) -> float:
    gen = network.generators
    links = network.links
    stores = network.stores
    solar_kw = float(gen.loc[gen.index.str.endswith("_solar"), "p_nom_opt"].sum())
    diesel_kw = float(gen.loc[gen.index.str.endswith("_diesel"), "p_nom_opt"].sum())
    battery_kwh = float(stores.loc[stores.index.str.endswith("_battery"), "e_nom_opt"].sum())
    converter_kw = float(links.loc[links.index.str.endswith("_batt_charge"), "p_nom_opt"].sum())
    fridge_input_kw = float(links.loc[links.index.str.endswith("_fridge"), "p_nom_opt"].sum())
    return float(
        solar_kw * cfg.pv_capex_per_kw
        + diesel_kw * cfg.diesel_genset_capex_per_kw
        + battery_kwh * cfg.battery_energy_capex_per_kwh
        + converter_kw * cfg.battery_power_capex_per_kw
        + fridge_input_kw * cfg.fridge_cop * cfg.fridge_capex_per_kw_cold
    )


def add_standard_fields(row: dict, *, origin: str) -> dict:
    row["architecture"] = row["config"]
    row["target_coverage"] = float(row["coverage_target"])
    row["achieved_coverage"] = float(row["served_fraction"])
    row["annual_system_cost_usd_per_year"] = float(row["objective_usd"])
    row["annualised_capital_cost_usd_per_year"] = float(
        row["capex_built_usd_per_year"]
    )
    row["result_origin"] = origin
    return row


def reused_c1_rows(cfg) -> list[dict]:
    frontier = pd.read_csv(EXISTING_C1 / "coverage_frontier.csv")
    rows: list[dict] = []
    for coverage in C1_REUSED:
        selected = frontier[
            frontier["config"].eq("C1")
            & frontier["supply"].eq("diesel")
            & np.isclose(frontier["coverage_target"], coverage)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"expected one reusable C1 diesel row at {coverage:.0%}, "
                f"found {len(selected)}"
            )
        row = selected.iloc[0].to_dict()
        caps = pd.read_csv(
            EXISTING_C1 / "capacities"
            / f"capacities_C1_diesel_coverage_{coverage:.3f}.csv"
        )
        row["genset_kw_built"] = float(caps["diesel_kw"].sum())
        row["upfront_capital_cost_usd"] = installed_capital_usd_from_capacities(
            caps, cfg
        )
        # C1 has no transport links, so objective minus annualised equipment
        # cost is exactly genset fuel expenditure.
        row["genset_kwh_per_year"] = float(
            max(row["objective_usd"] - row["capex_built_usd_per_year"], 0.0)
            / cfg.diesel_marginal_cost_per_kwh
        )
        row["genset_litres_per_year"] = (
            row["genset_kwh_per_year"] / cfg.diesel_genset_kwh_per_litre
        )
        row["transport_litres_per_year"] = 0.0
        rows.append(add_standard_fields(row, origin="reused_existing_C1"))
    return rows


def solve_case(
    architecture,
    coverage,
    state_groups,
    solar_cfs,
    cold_loads,
    locations,
    cfg,
    capacity_dir,
):
    def exporter(network, row):
        diesel_names = [
            str(name) for name in network.generators.index
            if str(name).endswith("_diesel")
        ]
        row.update(emissions_breakdown(network, cfg, diesel_names))
        row["upfront_capital_cost_usd"] = installed_capital_usd_from_network(
            network, cfg
        )

    row = solve_budget_point(
        architecture,
        state_groups,
        solar_cfs,
        cold_loads,
        locations,
        SNAPSHOTS,
        WEIGHTS,
        cfg,
        budget=None,
        supply="diesel",
        coverage=coverage,
        capacity_export_dir=str(capacity_dir),
        solution_exporter=exporter,
    )
    return add_standard_fields(row, origin="new_solve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "continuous_diesel",
    )
    parser.add_argument("--reusable-c1-dir", required=True, type=Path,
                        help="C1 diesel output from run_diesel_c1_check.py")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    capacity_dir = args.output_dir / "capacities"
    capacity_dir.mkdir()

    assert_fixed_july()
    locations, solar_cfs, cold_loads, state_groups = load_fixed_july_inputs()
    cfg = fixed_july_config()

    manifest = {
        "formulation": "coarse continuous diesel LP; no minimum genset",
        "min_genset_kw_enforced": 0.0,
        "corrected_load": str(LOAD),
        "snapshots": [str(SNAPSHOTS[0]), str(SNAPSHOTS[-1])],
        "snapshot_count": len(SNAPSHOTS),
        "snapshot_weight": float(WEIGHTS[0]),
        "solver_threads": cfg.solver_threads,
        "c3_solver_options": cfg.c3_barrier_options,
        "new_cases": {
            "C1": list(C1_MISSING),
            "C2": list(C2_CASES),
            "C3": list(C3_CASES),
        },
        "reused_cases": {"C1": list(C1_REUSED)},
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    global EXISTING_C1
    EXISTING_C1 = args.reusable_c1_dir
    rows = reused_c1_rows(cfg)
    result_path = args.output_dir / "continuous_diesel_results.csv"
    pd.DataFrame(rows).sort_values(
        ["config", "coverage_target"]
    ).to_csv(result_path, index=False)

    cases = (
        [("C1", c) for c in C1_MISSING]
        + [("C2", c) for c in C2_CASES]
        + [("C3", c) for c in C3_CASES]
    )
    for architecture, coverage in cases:
        print(
            f"[final-diesel] {architecture} continuous diesel @ {coverage:.0%}",
            flush=True,
        )
        rows.append(
            solve_case(
                architecture,
                coverage,
                state_groups,
                solar_cfs,
                cold_loads,
                locations,
                cfg,
                capacity_dir,
            )
        )
        pd.DataFrame(rows).sort_values(
            ["config", "coverage_target"]
        ).to_csv(result_path, index=False)

    print(f"[final-diesel] wrote {result_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
