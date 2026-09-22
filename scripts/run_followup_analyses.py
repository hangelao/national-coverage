from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import gc
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "sensitivity"))

from budget_frontier import run_budget_frontier, solve_budget_point
from cluster_sweep import DEFAULT_RADII_KM, DEFAULT_SIZES, build_clusters, cluster_statistics
from config import Config
from data_loader import (align_solar_to_markets, group_markets_by_state,
                         load_cold_loads, load_market_locations,
                         load_road_distances, load_solar_cf, pivot_cold_loads)
from diesel_base_case import SCENARIOS, solve_scenario
from input_paths import CORRECTED_LOAD, MARKETS, ROAD_DISTANCES, SOLAR_CF
from sensitivity_technoeconomic import CASES, SensitivityConfig

LOAD = str(CORRECTED_LOAD)
SOLAR = str(SOLAR_CF)
MARKETS = str(MARKETS)
ROADS = str(ROAD_DISTANCES)
SNAPS = pd.date_range("2019-07-02", periods=56, freq="3h")
WEIGHTS = np.full(56, 8760.0 / 56.0)
OUT = os.path.join(ROOT, "outputs", "followup_analyses")


def inputs(cfg):
    cold = load_cold_loads(LOAD)
    solar = load_solar_cf(SOLAR)
    loc = load_market_locations(cold, markets_csv=MARKETS)
    lp = pivot_cold_loads(cold)
    sp = align_solar_to_markets(solar, loc)
    mids = loc["market_id"].tolist()
    mids = [m for m in mids if m in lp.columns and m in sp.columns]
    lp = lp.loc[SNAPS, mids]
    sp = sp.loc[SNAPS, mids]
    return (loc, {m: sp[m].to_numpy() for m in mids},
            {m: lp[m].to_numpy() for m in mids}, group_markets_by_state(loc))


def c3_coverage(loc, solar, cold, states):
    out = os.path.join(OUT, "solar_c3_coverage")
    os.makedirs(out, exist_ok=True)
    cfg = replace(Config(), solver_threads=16)
    cfg.road_distances = load_road_distances(ROADS)
    rows = []
    for level in (.50, .75, .85, .95, 1.0):
        print(f"[C3] {level:.0%}", flush=True)
        row = solve_budget_point("C3", states, solar, cold, loc, SNAPS, WEIGHTS,
                                 cfg, budget=None, coverage=level,
                                 capacity_export_dir=os.path.join(out, "capacities"))
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out, "coverage_frontier.csv"), index=False)
        gc.collect()


def diesel(loc, solar, cold, states):
    out = os.path.join(OUT, "diesel_base_case")
    os.makedirs(out, exist_ok=True)
    cfg = replace(Config(), solver_threads=16)
    cfg.road_distances = load_road_distances(ROADS)
    rows = []
    for kind in ("C1", "C2", "C3"):
        for scenario in SCENARIOS:
            print(f"[diesel] {kind}/{scenario}", flush=True)
            rows.append(solve_scenario(kind, scenario, states, solar, cold, loc,
                                       SNAPS, WEIGHTS, cfg, min_genset_kw=cfg.min_genset_kw))
            pd.DataFrame(rows).to_csv(os.path.join(out, "diesel_base_case.csv"), index=False)
            gc.collect()


def clusters(loc, solar, cold, states):
    out = os.path.join(OUT, "cluster_scale")
    os.makedirs(out, exist_ok=True)
    cfg = replace(Config(), solver_threads=16)
    cfg.road_distances = load_road_distances(ROADS)
    rows = []
    for radius in DEFAULT_RADII_KM:
        for size in DEFAULT_SIZES:
            if size == 1 or (not math.isfinite(size) and not math.isfinite(radius)):
                continue
            groups = build_clusters(states, cfg, size, radius)
            stats = cluster_statistics(groups, states, cfg)
            label = f"size<={size:g}_km<={radius:g}"
            print(f"[cluster] {label}", flush=True)
            p = solve_budget_point("C2", groups, solar, cold, loc, SNAPS, WEIGHTS,
                                   cfg, budget=None)
            rows.append({"grouping": label, "total_cost_usd_per_year": p["objective_usd"],
                         "capex_usd_per_year": p["capex_built_usd_per_year"],
                         "n_links": p["n_links"], **stats,
                         "max_cluster_size": size, "max_cluster_km": radius})
            pd.DataFrame(rows).to_csv(os.path.join(out, "cluster_sweep.csv"), index=False)
            gc.collect()


def sensitivities(loc, solar, cold, states):
    root = os.path.join(OUT, "technoeconomic_sensitivity")
    os.makedirs(root, exist_ok=True)
    base = replace(SensitivityConfig(), solver_threads=16)
    base.road_distances = load_road_distances(ROADS)
    budgets = (.1, .2, .3, .4, .5, .6, .7, .85, 1.0)
    for case in CASES:
        cfg = replace(base, **case.change_dict)
        cfg.road_distances = base.road_distances
        bdir = os.path.join(root, "fig2b_budget", case.name)
        print(f"[sens-budget] {case.name}", flush=True)
        run_budget_frontier(states, solar, cold, loc, SNAPS, WEIGHTS, cfg, bdir,
                            configs=("C1", "C2"), budget_fractions=budgets,
                            supply="solar")
        pdir = os.path.join(root, "trigger_range", case.name)
        rows = []
        for kind in ("C1", "C2"):
            for level in (.6, .7, .75, .8, .85, .9, .95, 1.0):
                print(f"[sens-trigger] {case.name} {kind} {level:.0%}", flush=True)
                row = solve_budget_point(kind, states, solar, cold, loc, SNAPS, WEIGHTS,
                                         cfg, budget=None, coverage=level)
                row["case"] = case.name
                rows.append(row)
                os.makedirs(pdir, exist_ok=True)
                pd.DataFrame(rows).to_csv(os.path.join(pdir, "path_sensitivity.csv"), index=False)
                gc.collect()


def main():
    t = time.time()
    print("CORRECTED_FOLLOWUP_START", flush=True)
    cfg = Config()
    cfg.road_distances = load_road_distances(ROADS)
    loc, solar, cold, states = inputs(cfg)
    print(f"inputs {len(loc)} markets, {len(SNAPS)} snapshots, weight_sum={WEIGHTS.sum():.0f}", flush=True)
    clusters(loc, solar, cold, states)
    sensitivities(loc, solar, cold, states)
    print(f"CORRECTED_FOLLOWUP_COMPLETE seconds={time.time()-t:.1f}", flush=True)


if __name__ == "__main__":
    main()
