"""
run_committed_path.py
---------------------
The committed (continuous-building) deployment path — R18.

The coverage frontier (R15/R16a) solves each coverage level independently:
nothing forces the 50% build to contain the 25% build. This driver solves the
levels IN SEQUENCE, fixing every capex-bearing capacity built at one level as
a lower bound for the next. The result is a feasible incremental investment
programme by construction, and the premium of its cost over the unconstrained
frontier is the price of committing early — near zero if the nestedness
result (retention ≥93–99.9%) is what it appears to be.

Locked components per market: solar generator p_nom, battery store e_nom,
fridge / battery charge / battery discharge link p_nom. Transport links carry
no capital cost and are left free.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import pandas as pd

from budget_frontier import solve_budget_point
from config import Config
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

# capacity-export column -> (component frame, bound column, name suffix)
LOCK_MAP = {
    "solar_kw": ("generators", "p_nom_min", "_solar"),
    "battery_kwh": ("stores", "e_nom_min", "_battery"),
    "fridge_kw_cold": ("links", "p_nom_min", "_fridge"),
    "batt_charge_kw": ("links", "p_nom_min", "_batt_charge"),
    "batt_discharge_kw": ("links", "p_nom_min", "_batt_discharge"),
}


def make_lockin_mutator(caps: pd.DataFrame):
    """Return a network mutator imposing ``caps`` as lower bounds."""

    def _apply(network) -> None:
        for col, (comp, bound, suffix) in LOCK_MAP.items():
            if col not in caps.columns:
                continue
            frame = getattr(network, comp)
            names = caps.index.astype(str) + suffix
            mask = names.isin(frame.index)
            frame.loc[names[mask], bound] = caps.loc[mask, col].to_numpy()

    return _apply
