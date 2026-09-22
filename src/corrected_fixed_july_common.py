"""Shared isolated inputs for corrected fixed-July follow-up workflows."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from config import Config
from data_loader import (align_solar_to_markets, group_markets_by_state,
    load_cold_loads, load_market_locations, load_road_distances, load_solar_cf,
    pivot_cold_loads)
from input_paths import CORRECTED_LOAD, ROAD_DISTANCES, SOLAR_CF, MARKETS as MARKETS_PATH

ROOT = Path(__file__).resolve().parents[1]
LOAD = CORRECTED_LOAD
SOLAR = SOLAR_CF
MARKETS = MARKETS_PATH
ROADS = ROAD_DISTANCES
SNAPSHOTS = pd.date_range("2019-07-02", periods=56, freq="3h")
WEIGHTS = np.full(56, 8760.0 / 56.0)

def fixed_july_config(*, max_transport_hours=None):
    cfg = replace(Config(), solver_threads=16,
                  max_transport_link_hours=max_transport_hours)
    cfg.road_distances = load_road_distances(str(ROADS))
    return cfg

def load_fixed_july_inputs():
    cold = load_cold_loads(str(LOAD))
    solar = load_solar_cf(str(SOLAR))
    locations = load_market_locations(cold, markets_csv=str(MARKETS))
    load_pivot = pivot_cold_loads(cold)
    solar_pivot = align_solar_to_markets(solar, locations)
    mids = [m for m in locations["market_id"].tolist()
            if m in load_pivot.columns and m in solar_pivot.columns]
    loads = load_pivot.loc[SNAPSHOTS, mids]
    solar_cf = solar_pivot.loc[SNAPSHOTS, mids]
    return (locations, {m: solar_cf[m].to_numpy() for m in mids},
            {m: loads[m].to_numpy() for m in mids},
            group_markets_by_state(locations))

def assert_fixed_july():
    assert len(SNAPSHOTS) == 56
    assert SNAPSHOTS[0] == pd.Timestamp("2019-07-02 00:00:00")
    assert SNAPSHOTS[-1] == pd.Timestamp("2019-07-08 21:00:00")
    assert np.allclose(WEIGHTS, 8760.0 / 56.0)
    assert np.isclose(WEIGHTS.sum(), 8760.0)
    assert LOAD.exists()
