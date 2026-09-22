#!/usr/bin/env python3
"""Corrected four-hour demand-profile synchronisation sweep."""
from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import load_cold_loads  # noqa: E402
from input_paths import CORRECTED_LOAD, MARKETS as MARKETS_PATH, ROAD_DISTANCES, SOLAR_CF

LOAD = CORRECTED_LOAD
SOLAR = SOLAR_CF
MARKETS = MARKETS_PATH
ROADS = ROAD_DISTANCES
OUT = ROOT / "outputs" / "demand_synchronisation"
PROFILE_ROOT = OUT / "profiles"
OPT_ROOT = OUT / "optimization"
SNAPS = pd.date_range("2019-07-02", periods=56, freq="3h")
WEIGHTS = np.full(56, 8760.0 / 56.0)
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_profiles() -> dict:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    PROFILE_ROOT.mkdir()
    needed = {"market_id", "timestamp", "state", "urban_class", "cold_load_kW"}
    parts = []
    for chunk in pd.read_csv(LOAD, chunksize=250_000):
        chunk["market_id"] = chunk["market_id"].astype("Int64").astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                columns={"Year":"year","Month":"month","Day":"day","Hour":"hour","Minute":"minute"}
            )
        )
        selected = chunk[chunk["timestamp"].isin(set(SNAPS))]
        if not selected.empty:
            parts.append(selected[["market_id", "timestamp", "state", "urban_class", "cold_load_kW"]])
    base = pd.concat(parts, ignore_index=True)
    base["state"] = base["state"].replace({"Fct": "FCT", "fct": "FCT"})
    base = base.drop_duplicates(["market_id", "timestamp"], keep="first")
    wide = base.pivot(index="market_id", columns="timestamp", values="cold_load_kW").reindex(columns=SNAPS).fillna(0.0)
    meta = base.drop_duplicates("market_id").set_index("market_id").reindex(wide.index)
    loads = wide.to_numpy(float)
    states = meta["state"].to_numpy()
    energy = loads @ WEIGHTS
    common = {}
    for state in sorted(pd.unique(states)):
        mask = states == state
        aggregate = loads[mask].sum(axis=0)
        common[state] = aggregate / float(aggregate @ WEIGHTS)
    for alpha in ALPHAS:
        modified = np.empty_like(loads)
        for state in sorted(pd.unique(states)):
            mask = states == state
            modified[mask] = ((1.0 - alpha) * loads[mask] +
                              alpha * energy[mask, None] * common[state][None, :])
        frame = pd.DataFrame({
            "market_id": np.repeat(wide.index.to_numpy(), len(SNAPS)),
            "timestamp": np.tile(SNAPS.to_numpy(), len(wide)),
            "Year": np.tile(SNAPS.year, len(wide)),
            "Month": np.tile(SNAPS.month, len(wide)),
            "Day": np.tile(SNAPS.day, len(wide)),
            "Hour": np.tile(SNAPS.hour, len(wide)),
            "Minute": np.tile(SNAPS.minute, len(wide)),
            "state": np.repeat(states, len(SNAPS)),
            "urban_class": np.repeat(meta["urban_class"].astype(str).to_numpy(), len(SNAPS)),
            "cold_load_kW": modified.reshape(-1),
            "objective_weight": np.tile(WEIGHTS, len(wide)),
            "generator_weight": np.tile(WEIGHTS, len(wide)),
            "store_elapsed_hours": 3.0,
            "alpha": alpha,
        })
        path = PROFILE_ROOT / f"alpha_{alpha:.2f}" / "modified_loads.csv"
        path.parent.mkdir()
        frame.to_csv(path, index=False)
    manifest = {
        "workflow": "corrected four-hour demand-profile synchronisation sensitivity",
        "alpha_values": list(ALPHAS),
        "formulation": "L_i,t(alpha)=(1-alpha)L_i,t + alpha E_i g_s,t",
        "energy_definition": "E_i=sum_t(w_t L_i,t)",
        "state_profile_definition": "g_s,t=sum_(j in s)L_j,t / sum_t(w_t sum_(j in s)L_j,t)",
        "coordination_cases": ["C1", "C2", "C3"],
        "coverage": "unrestricted full-service optimisation (same as original run workflow)",
        "representative_week": "2019-07-02 through 2019-07-08",
        "snapshots": 56,
        "resolution_hours": 3,
        "snapshot_weight": 8760.0 / 56.0,
        "input_load": str(LOAD),
        "input_load_sha256": digest(LOAD),
        "solar_input": str(SOLAR),
        "markets_input": str(MARKETS),
        "roads_input": str(ROADS),
        "solver": {"name": "gurobi", "c1_c2_workers": 4, "c1_c2_threads": 4, "c3_threads": 20},
        "profiles": {f"{a:.2f}": str(PROFILE_ROOT / f"alpha_{a:.2f}" / "modified_loads.csv") for a in ALPHAS},
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_all(manifest: dict) -> None:
    from run import run
    from data_loader import prepare_optimization_snapshots
    import run as main_run

    fixed = pd.DatetimeIndex(SNAPS)
    weights = WEIGHTS.copy()
    main_run.prepare_optimization_snapshots = lambda _load_pivot, _config: (fixed.copy(), weights.copy())
    OPT_ROOT.mkdir()
    for alpha in ALPHAS:
        label = f"alpha_{alpha:.2f}"
        case_out = OPT_ROOT / label
        case_out.mkdir()
        print(f"START_OPTIMIZATION alpha={label}", flush=True)
        run(
            hourly_loads_csv=str(PROFILE_ROOT / label / "modified_loads.csv"),
            solar_csv=str(SOLAR),
            output_dir=str(case_out),
            road_distances_path=str(ROADS),
            markets_csv=str(MARKETS),
            debug=False,
            n_workers=4,
            solver_threads=4,
            solver_threads_c3=20,
            use_representative_weeks=True,
            n_representative_weeks=1,
            time_resolution_hours=3,
        )
        print(f"COMPLETE_OPTIMIZATION alpha={label}", flush=True)


if __name__ == "__main__":
    m = write_profiles()
    print(f"PROFILES_READY alphas={','.join(f'{a:.2f}' for a in ALPHAS)} snapshots=56 weight_sum={WEIGHTS.sum():.0f}", flush=True)
    run_all(m)

