"""Portable input-path configuration for the Paper 2 workflows.

The public repository deliberately contains no data. Set the environment
variables below, or pass paths directly to the command-line workflows.
"""

from __future__ import annotations

import os
from pathlib import Path


def configured_path(variable: str, default_name: str) -> Path:
    """Return an input path from an environment variable or a local default."""

    return Path(os.environ.get(variable, default_name)).expanduser()


CORRECTED_LOAD = configured_path(
    "PAPER2_CORRECTED_LOAD",
    "market_hourly_cold_loads_2019_four_hour_spread.csv.gz",
)
SOLAR_CF = configured_path(
    "PAPER2_SOLAR_CF",
    "ninja-weather-country-NG-pv_cf_surface_area_wtd-merra2-2019-pvlib.csv",
)
MARKETS = configured_path(
    "PAPER2_MARKETS",
    "markets_with_urban_class_3km.csv",
)
ROAD_DISTANCES = configured_path(
    "PAPER2_ROAD_DISTANCES",
    "road_distance_matrix_full.pkl",
)

# Used only by the optional demand-generation workflow.
POPULATION = configured_path("PAPER2_POPULATION", "NIMC_population_by_state.xlsx")


def require_file(path: Path, description: str) -> Path:
    """Fail with a publication-friendly message when an input is absent."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}. Configure the relevant "
            "PAPER2_* environment variable or pass the path on the CLI."
        )
    return path
