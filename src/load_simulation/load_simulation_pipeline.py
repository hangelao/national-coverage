"""
Single entry point: annual meat (kg) → daily kg → hourly kg → hourly cold load (kW).

Replaces running these separately:
  load_simulation_annual.py → load_daily.py → load_hourly.py → cold_load_index.py

Profiles:
  --profile full   : all markets from the classified markets CSV (~13k+ when using urban_thresholds output)
  --profile test   : first N markets (default 100), or a dedicated small CSV if present

Examples:
  python load_simulation_pipeline.py --profile full
  python load_simulation_pipeline.py --profile test --test-n 100
  python load_simulation_pipeline.py --profile full --markets /path/to/markets_with_urban_class_3km.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants (aligned with the original scripts)
# -----------------------------------------------------------------------------

MODELLED_MARKET_SHARE = 0.80

NATIONAL_BEEF_SCENARIOS_KG = {
    "baseline_2015_470kt": 470_000_000,
    "sensitivity_2050_1393kt": 1_393_000_000,
}

URBAN_CLASS_WEIGHTS = {
    "Urban": 1.8,
    "Peri-urban": 1.0,
    "Rural": 0.6,
}

TARGET_YEAR = 2019

WEEKDAY_WEIGHTS = {
    "Monday": 1.0,
    "Tuesday": 1.0,
    "Wednesday": 1.0,
    "Thursday": 1.0,
    "Friday": 1.1,
    "Saturday": 1.2,
    "Sunday": 1.1,
}

EID_AL_FITR_START = pd.Timestamp("2019-05-28")
EID_AL_FITR_END = pd.Timestamp("2019-06-06")
EID_AL_FITR_FACTORS = {"Urban": 1.4, "Peri-urban": 1.2, "Rural": 1.1}

EID_AL_ADHA_START = pd.Timestamp("2019-08-01")
EID_AL_ADHA_END = pd.Timestamp("2019-08-14")
EID_AL_ADHA_FACTORS = {"Urban": 1.5, "Peri-urban": 1.6, "Rural": 1.8}

HOURLY_PATTERN_RAW = {
    0: 0.005,
    1: 0.003,
    2: 0.002,
    3: 0.003,
    4: 0.010,
    5: 0.025,
    6: 0.050,
    7: 0.080,
    8: 0.100,
    9: 0.090,
    10: 0.085,
    11: 0.080,
    12: 0.075,
    13: 0.070,
    14: 0.065,
    15: 0.060,
    16: 0.055,
    17: 0.050,
    18: 0.040,
    19: 0.030,
    20: 0.020,
    21: 0.015,
    22: 0.010,
    23: 0.008,
}
_pattern_sum = sum(HOURLY_PATTERN_RAW.values())
HOURLY_PATTERN = {h: HOURLY_PATTERN_RAW[h] / _pattern_sum for h in range(24)}

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FULL_MARKETS_CANDIDATES: Sequence[str] = (
    "markets_with_urban_class_3km.csv",
    "markets_with_urban_class.csv",
)

OPENING_COLS = [
    "mrkt_mon",
    "mrkt_tue",
    "mrkt_wed",
    "mrkt_thur",
    "mrkt_fri",
    "mrkt_sat",
    "mrkt_sun",
]

POPULATION_FILE_OPTIONS = [
    "NIMC_population_by_state.xlsx",
    "NIMC_population_by_state.csv",
    "../original dataset/NIMC Registration 2019-2022.xlsx",
    "../original dataset/NIMC_population_by_state.csv",
]


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------


def find_first_existing(base: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = base / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find any of: {[str(base / c) for c in candidates]}"
    )


def find_population_file(base: Path) -> str:
    for name in POPULATION_FILE_OPTIONS:
        p = base / name
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Could not find population file. Tried: {[str(base / n) for n in POPULATION_FILE_OPTIONS]}"
    )


def _read_population_dataframe(population_file: str) -> pd.DataFrame:
    file_ext = Path(population_file).suffix.lower()
    if file_ext in (".xlsx", ".xls"):
        if file_ext == ".xlsx":
            population_df = pd.read_excel(
                population_file, sheet_name=0, engine="openpyxl", header=1
            )
            if population_df.empty or len(population_df.columns) < 2:
                population_df = pd.read_excel(
                    population_file, sheet_name=0, engine="openpyxl", header=None
                )
                header_row = None
                for idx in range(min(5, len(population_df))):
                    row_str = " ".join(
                        [
                            str(x).upper()
                            for x in population_df.iloc[idx].values
                            if pd.notna(x)
                        ]
                    )
                    if "STATE" in row_str and ("TOTAL" in row_str or "POP" in row_str):
                        header_row = idx
                        break
                if header_row is not None:
                    population_df.columns = population_df.iloc[header_row]
                    population_df = population_df.iloc[header_row + 1 :].reset_index(
                        drop=True
                    )
        else:
            population_df = pd.read_excel(
                population_file, sheet_name=0, engine="xlrd", header=1
            )
    elif file_ext == ".csv":
        population_df = pd.read_csv(population_file)
    else:
        raise ValueError(f"Unsupported population format: {file_ext}")
    return population_df


def normalize_population_columns(population_df: pd.DataFrame) -> pd.DataFrame:
    state_cols: List[str] = []
    pop_cols: List[str] = []
    for col in population_df.columns:
        col_str = str(col).upper().strip()
        if "STATE" in col_str and col_str != "S/N":
            state_cols.append(col)
        if "POPULATION" in col_str or "POP" in col_str or "TOTAL" in col_str:
            if "TOTAL" in col_str or "POPULATION" in col_str:
                pop_cols.insert(0, col)
            else:
                pop_cols.append(col)
    if not state_cols:
        for col in population_df.columns:
            col_str = str(col).upper().strip()
            if col_str not in ["S/N", "SN", "ZONE", "REGION"] and len(col_str) > 2:
                if not col_str.replace(" ", "").isdigit():
                    state_cols.append(col)
    if not pop_cols:
        for col in population_df.columns:
            col_str = str(col).upper().strip()
            if col_str not in ["S/N", "SN", "STATE", "STATES", "ZONE", "REGION"]:
                sample_values = population_df[col].dropna().head(10)
                if len(sample_values) > 0:
                    numeric_count = sum(
                        pd.to_numeric(sample_values, errors="coerce").notna()
                    )
                    if numeric_count / len(sample_values) > 0.7:
                        pop_cols.append(col)
    if not state_cols or not pop_cols:
        raise ValueError(
            f"Could not detect state/pop columns. Columns: {population_df.columns.tolist()}"
        )
    out = population_df[[state_cols[0], pop_cols[0]]].copy()
    out.columns = ["state", "population"]
    out["state"] = out["state"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["state", "population"])
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out = out.dropna(subset=["population"])
    out["state_normalized"] = out["state"]
    return out


# -----------------------------------------------------------------------------
# Market ID / state columns (same logic as original)
# -----------------------------------------------------------------------------


def detect_market_columns(markets_df: pd.DataFrame) -> Tuple[str, str]:
    if "market_id" in markets_df.columns:
        market_id_col = "market_id"
    elif "uniq_id" in markets_df.columns:
        market_id_col = "uniq_id"
    else:
        raise ValueError(
            f"No market_id/uniq_id. Columns: {markets_df.columns.tolist()}"
        )
    if "state" in markets_df.columns:
        state_col = "state"
    elif "statename" in markets_df.columns:
        state_col = "statename"
    else:
        raise ValueError(
            f"No state/statename. Columns: {markets_df.columns.tolist()}"
        )
    if "urban_class" not in markets_df.columns:
        raise ValueError(
            f"No urban_class. Columns: {markets_df.columns.tolist()}"
        )
    return market_id_col, state_col


def market_coords_from_locations(
    markets_df: pd.DataFrame, market_id_col: str
) -> Optional[pd.DataFrame]:
    """Return two-column market_id + x + y if present; else None."""
    if "x" not in markets_df.columns or "y" not in markets_df.columns:
        return None
    c = markets_df[[market_id_col, "x", "y"]].copy()
    c = c.rename(columns={market_id_col: "market_id"})
    return c.drop_duplicates(subset=["market_id"], keep="first")


def merge_coords_onto(
    df: pd.DataFrame, coords: Optional[pd.DataFrame], how: str = "left"
) -> pd.DataFrame:
    if coords is None or coords.empty:
        return df
    out = df.merge(coords, on="market_id", how=how)
    return out


# -----------------------------------------------------------------------------
# Step 1 — Annual allocation
# -----------------------------------------------------------------------------


def run_annual_allocation(
    markets_csv: Path,
    out_annual_csv: Path,
    modelled_annual_meat_kg: float,
    population_file: Optional[str] = None,
) -> pd.DataFrame:
    print(f"[annual] Loading markets: {markets_csv}")
    markets_full = pd.read_csv(markets_csv)
    market_id_col, state_col = detect_market_columns(markets_full)

    mdf = markets_full[[market_id_col, state_col, "urban_class"]].copy()
    mdf.columns = ["market_id", "state", "urban_class"]
    mdf["state_original"] = mdf["state"].astype(str).str.strip()
    mdf["state"] = mdf["state_original"].str.upper()
    mdf = mdf.dropna(subset=["market_id", "state", "urban_class"])

    pop_path = population_file or find_population_file(SCRIPT_DIR)
    print(f"[annual] Population: {pop_path}")
    raw_pop = _read_population_dataframe(pop_path)
    population_df = normalize_population_columns(raw_pop)

    national_population = population_df["population"].sum()
    if national_population == 0:
        raise ValueError("Total national population is zero")
    population_df["state_annual_meat_kg"] = (
        modelled_annual_meat_kg
        * (population_df["population"] / national_population)
    )

    mdf["base_weight"] = mdf["urban_class"].map(URBAN_CLASS_WEIGHTS)
    unmapped = mdf[mdf["base_weight"].isna()]
    if not unmapped.empty:
        print(
            f"[annual] Warning: dropping unmapped urban_class: {unmapped['urban_class'].unique()}"
        )
        mdf = mdf[mdf["base_weight"].notna()].copy()
    if len(mdf) == 0:
        raise ValueError("No valid markets after urban_class mapping")

    mdf["state_base_weight_sum"] = mdf.groupby("state")["base_weight"].transform("sum")
    mdf = mdf[mdf["state_base_weight_sum"] > 0].copy()
    mdf["market_share"] = mdf["base_weight"] / mdf["state_base_weight_sum"]

    mdf = mdf.merge(
        population_df[["state_normalized", "state_annual_meat_kg"]].rename(
            columns={"state_normalized": "state"}
        ),
        on="state",
        how="left",
    )
    mdf["state"] = mdf["state_original"]
    mdf = mdf.drop(columns=["state_original"], errors="ignore")

    missing = mdf[mdf["state_annual_meat_kg"].isna()]
    if not missing.empty:
        print(
            f"[annual] Warning: dropping markets in unknown states: {missing['state'].unique()}"
        )
        mdf = mdf[mdf["state_annual_meat_kg"].notna()].copy()

    mdf["market_annual_meat_kg"] = (
        mdf["market_share"] * mdf["state_annual_meat_kg"]
    )

    out = mdf[
        [
            "market_id",
            "state",
            "urban_class",
            "base_weight",
            "market_share",
            "market_annual_meat_kg",
        ]
    ].copy()
    coords = market_coords_from_locations(markets_full, market_id_col)
    if coords is not None:
        out = merge_coords_onto(out, coords, how="left")
    else:
        print("[annual] Note: markets CSV has no x/y; coordinate columns omitted.")

    annual_order = [
        "market_id",
        "x",
        "y",
        "state",
        "urban_class",
        "base_weight",
        "market_share",
        "market_annual_meat_kg",
    ]
    out = out[[c for c in annual_order if c in out.columns]].copy()
    out_annual_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_annual_csv, index=False)
    print(
        f"[annual] Saved {len(out):,} markets → {out_annual_csv} "
        f"(total kg ≈ {out['market_annual_meat_kg'].sum():,.0f})"
    )
    return out


# -----------------------------------------------------------------------------
# Step 2 — Daily disaggregation
# -----------------------------------------------------------------------------


def convert_opening_indicators(series: pd.Series) -> pd.Series:
    series_str = series.astype(str).str.strip()
    result = series_str.map(
        {
            "1": 1,
            "1.0": 1,
            "yes": 1,
            "Yes": 1,
            "YES": 1,
            "True": 1,
            "true": 1,
            "TRUE": 1,
            "0": 0,
            "0.0": 0,
            "no": 0,
            "No": 0,
            "NO": 0,
            "False": 0,
            "false": 0,
            "FALSE": 0,
            "nan": 0,
            "None": 0,
            "": 0,
        }
    )
    return result.fillna(0).astype(int)


def get_holiday_factor(date: pd.Timestamp, urban_class: str) -> float:
    if EID_AL_FITR_START <= date <= EID_AL_FITR_END:
        return EID_AL_FITR_FACTORS.get(urban_class, 1.0)
    if EID_AL_ADHA_START <= date <= EID_AL_ADHA_END:
        return EID_AL_ADHA_FACTORS.get(urban_class, 1.0)
    return 1.0


def create_daily_calendar(year: int) -> pd.DataFrame:
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    calendar_df = pd.DataFrame({"date": dates})
    calendar_df["weekday_name"] = calendar_df["date"].dt.day_name()
    return calendar_df


def run_daily_disaggregation(
    annual_csv: Path,
    markets_csv: Path,
    out_daily_csv: Path,
    year: int = TARGET_YEAR,
) -> pd.DataFrame:
    print(f"[daily] Loading annual: {annual_csv}")
    markets_df = pd.read_csv(annual_csv)
    core_cols = ["market_id", "state", "urban_class", "market_annual_meat_kg"]
    missing_core = [c for c in core_cols if c not in markets_df.columns]
    if missing_core:
        raise ValueError(f"Annual file missing columns {missing_core}")

    markets_locations = pd.read_csv(markets_csv)
    market_id_col, _ = detect_market_columns(markets_locations)

    if "x" not in markets_df.columns or "y" not in markets_df.columns:
        coords = market_coords_from_locations(markets_locations, market_id_col)
        if coords is not None:
            markets_df = merge_coords_onto(markets_df, coords, how="left")
        else:
            print("[daily] Note: no x/y in annual file and markets CSV; coordinates omitted.")
    merge_cols = [market_id_col] + OPENING_COLS
    for c in merge_cols:
        if c not in markets_locations.columns:
            raise ValueError(
                f"Markets file missing '{c}' (need opening days). Columns: "
                f"{markets_locations.columns.tolist()}"
            )
    opening_df = markets_locations[merge_cols].copy()
    opening_df = opening_df.rename(columns={market_id_col: "market_id"})
    for col in OPENING_COLS:
        opening_df[col] = convert_opening_indicators(opening_df[col])

    markets_df = markets_df.merge(opening_df, on="market_id", how="left")
    for col in OPENING_COLS:
        markets_df[col] = markets_df[col].fillna(0).astype(int)
    all_zero = markets_df[OPENING_COLS].sum(axis=1) == 0
    if all_zero.any():
        print(
            f"[daily] Warning: {all_zero.sum()} markets have no opening data; assuming open all week."
        )
        for col in OPENING_COLS:
            markets_df.loc[all_zero, col] = 1

    print(f"[daily] Building calendar {year}…")
    calendar_df = create_daily_calendar(year)
    weekday_to_col = {
        "Monday": "mrkt_mon",
        "Tuesday": "mrkt_tue",
        "Wednesday": "mrkt_wed",
        "Thursday": "mrkt_thur",
        "Friday": "mrkt_fri",
        "Saturday": "mrkt_sat",
        "Sunday": "mrkt_sun",
    }

    markets_df["key"] = 1
    calendar_df["key"] = 1
    market_date_df = markets_df.merge(calendar_df, on="key").drop("key", axis=1)

    market_date_df["is_open"] = market_date_df.apply(
        lambda row: row[weekday_to_col[row["weekday_name"]]] == 1,
        axis=1,
    )
    market_date_df["weekday_weight"] = market_date_df["weekday_name"].map(
        WEEKDAY_WEIGHTS
    )
    market_date_df["holiday_factor"] = market_date_df.apply(
        lambda row: get_holiday_factor(row["date"], row["urban_class"]),
        axis=1,
    )
    market_date_df["daily_weight"] = (
        market_date_df["is_open"].astype(int)
        * market_date_df["weekday_weight"]
        * market_date_df["holiday_factor"]
    )
    market_date_df["weight_sum"] = market_date_df.groupby("market_id")[
        "daily_weight"
    ].transform("sum")
    market_date_df["daily_share"] = np.where(
        market_date_df["weight_sum"] > 0,
        market_date_df["daily_weight"] / market_date_df["weight_sum"],
        0.0,
    )
    market_date_df["daily_meat_kg"] = (
        market_date_df["market_annual_meat_kg"] * market_date_df["daily_share"]
    )

    daily_cols = ["market_id"]
    if "x" in market_date_df.columns and "y" in market_date_df.columns:
        daily_cols.extend(["x", "y"])
    daily_cols.extend(["date", "state", "urban_class", "daily_meat_kg"])
    out = market_date_df[daily_cols].copy()
    out = out.sort_values(by=["market_id", "date"]).reset_index(drop=True)
    out_daily_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_daily_csv, index=False)
    print(f"[daily] Saved {len(out):,} rows → {out_daily_csv}")
    return out


# -----------------------------------------------------------------------------
# Step 3 — Hourly meat
# -----------------------------------------------------------------------------


def generate_hourly_shares(market_id, date, seed=None) -> np.ndarray:
    if seed is None:
        date_str = pd.Timestamp(date).strftime("%Y%m%d")
        key = f"{market_id}_{date_str}".encode()
        seed = int(hashlib.md5(key).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    hourly_shares = np.array([HOURLY_PATTERN[h] for h in range(24)])
    variation = rng.uniform(0.95, 1.05, size=24)
    hourly_shares = hourly_shares * variation
    return hourly_shares / hourly_shares.sum()


def run_hourly_meat(daily_csv: Path, out_hourly_csv: Path) -> pd.DataFrame:
    print(f"[hourly meat] Loading {daily_csv}")
    daily_df = pd.read_csv(daily_csv)
    required = ["market_id", "date", "state", "urban_class", "daily_meat_kg"]
    miss = [c for c in required if c not in daily_df.columns]
    if miss:
        raise ValueError(f"Daily file missing {miss}")
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    hourly_records = []
    for _, row in daily_df.iterrows():
        if row["daily_meat_kg"] == 0:
            continue
        shares = generate_hourly_shares(row["market_id"], row["date"])
        for hour in range(24):
            rec = {
                "market_id": row["market_id"],
                "date": row["date"].strftime("%Y-%m-%d"),
                "hour": hour,
                "state": row["state"],
                "urban_class": row["urban_class"],
                "hourly_meat_kg": row["daily_meat_kg"] * shares[hour],
            }
            if "x" in daily_df.columns and "y" in daily_df.columns:
                rec["x"] = row["x"]
                rec["y"] = row["y"]
            hourly_records.append(rec)
    hourly_df = pd.DataFrame(hourly_records)
    h_order = ["market_id"]
    if "x" in hourly_df.columns and "y" in hourly_df.columns:
        h_order.extend(["x", "y"])
    h_order.extend(["date", "hour", "state", "urban_class", "hourly_meat_kg"])
    hourly_df = hourly_df[[c for c in h_order if c in hourly_df.columns]]
    hourly_df = hourly_df.sort_values(
        by=["market_id", "date", "hour"]
    ).reset_index(drop=True)
    out_hourly_csv.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(out_hourly_csv, index=False)
    print(f"[hourly meat] Saved {len(hourly_df):,} rows → {out_hourly_csv}")
    return hourly_df


# -----------------------------------------------------------------------------
# Step 4 — Hourly cold load (kW)
# -----------------------------------------------------------------------------


def meat_cooling_energy_kwh(
    meat_mass_kg: np.ndarray,
    specific_heat_kj_per_kgK: float,
    delta_T_K: float,
) -> np.ndarray:
    energy_kJ = meat_mass_kg * specific_heat_kj_per_kgK * np.abs(delta_T_K)
    return energy_kJ / 3600.0


def spread_batch_cooling_over_hours(
    batch_energy_kwh: np.ndarray,
    cooldown_hours: int,
    *,
    cyclic: bool = True,
) -> np.ndarray:
    """Convert hourly batch energies to overlapping cooling-power demand.

    Each batch contributes Q / cooldown_hours kW in its arrival hour and in
    the following cooldown_hours - 1 hours. For a full-year model, cyclic=True
    wraps batches at the end of the year into the first hours of the same
    model year. This preserves every batch's full cooling energy while
    retaining an 8,760-hour periodic input suitable for the downstream
    representative-period model.

    batch_energy_kwh may be one-dimensional or may have time on its final
    axis. A non-cyclic option is available for unit tests and truncated data,
    but the full-year generator below deliberately uses cyclic boundaries.
    """
    if not isinstance(cooldown_hours, int) or isinstance(cooldown_hours, bool):
        raise TypeError("cooldown_hours must be an integer number of hours")
    if cooldown_hours < 1:
        raise ValueError("cooldown_hours must be at least one hour")

    energy = np.asarray(batch_energy_kwh, dtype=float)
    if energy.ndim < 1:
        raise ValueError("batch_energy_kwh must have a time axis")
    if cooldown_hours > energy.shape[-1]:
        raise ValueError("cooldown_hours cannot exceed the time-series length")

    power = np.zeros_like(energy, dtype=float)
    contribution = energy / float(cooldown_hours)
    if cyclic:
        for offset in range(cooldown_hours):
            power += np.roll(contribution, shift=offset, axis=-1)
    else:
        for offset in range(cooldown_hours):
            if offset == 0:
                power += contribution
            else:
                power[..., offset:] += contribution[..., :-offset]
    return power


def run_hourly_cold_load(
    hourly_meat_csv: Path,
    out_cold_csv: Path,
    cooldown_hours: float = 4.0,
) -> pd.DataFrame:
    print(f"[cold load] Loading {hourly_meat_csv}")
    df = pd.read_csv(hourly_meat_csv)
    if "hourly_meat_kg" not in df.columns:
        raise ValueError("Expected column hourly_meat_kg")
    if not float(cooldown_hours).is_integer():
        raise ValueError(
            "Four-hour spreading requires an integer cooldown duration"
        )
    cooldown_steps = int(cooldown_hours)

    required_time_columns = {"date", "hour"}
    missing_time_columns = required_time_columns.difference(df.columns)
    if missing_time_columns:
        raise ValueError(
            f"Hourly meat file missing time columns {sorted(missing_time_columns)}"
        )

    timestamps = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(df["hour"], unit="h")
    )
    years = timestamps.dt.year.unique()
    if len(years) != 1:
        raise ValueError(
            "Full-year cooling generation requires exactly one calendar year"
        )
    year = int(years[0])
    full_hours = pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h",
    )
    market_ids = pd.Index(df["market_id"].drop_duplicates(), name="market_id")

    working = df.copy()
    working["timestamp"] = timestamps
    if working.duplicated(["market_id", "timestamp"]).any():
        raise ValueError("Duplicate market-hour rows found in hourly meat input")

    static_columns = [
        c for c in ("x", "y", "state", "urban_class") if c in working.columns
    ]
    static = (
        working.groupby("market_id", sort=False)[static_columns].first()
        if static_columns else pd.DataFrame(index=market_ids)
    )
    full_index = pd.MultiIndex.from_product(
        [market_ids, full_hours], names=["market_id", "timestamp"]
    )
    meat = (
        working.set_index(["market_id", "timestamp"])["hourly_meat_kg"]
        .reindex(full_index, fill_value=0.0)
        .to_numpy(dtype=float)
        .reshape(len(market_ids), len(full_hours))
    )

    specific_heat = 3.14
    delta_T = 4.0 - 40.0
    batch_energy_kwh = meat_cooling_energy_kwh(
        meat,
        specific_heat,
        delta_T,
    )
    cold_load_kw = spread_batch_cooling_over_hours(
        batch_energy_kwh,
        cooldown_steps,
        cyclic=True,
    )

    output = pd.DataFrame(
        {
            "market_id": full_index.get_level_values("market_id"),
            "timestamp": full_index.get_level_values("timestamp"),
            "hourly_meat_kg": meat.ravel(),
            "cold_load_kW": cold_load_kw.ravel(),
            "cooling_energy_kWh": batch_energy_kwh.ravel(),
        }
    )
    for column in static_columns:
        output[column] = output["market_id"].map(static[column])

    dt = output["timestamp"]
    output["Year"] = dt.dt.year
    output["Month"] = dt.dt.month
    output["Day"] = dt.dt.day
    output["Hour"] = dt.dt.hour
    output["Minute"] = 0

    cold_front = ["market_id"]
    if "x" in output.columns and "y" in output.columns:
        cold_front.extend(["x", "y"])
    cold_front.extend(
        ["Year", "Month", "Day", "Hour", "Minute",
         "state", "urban_class", "hourly_meat_kg", "cold_load_kW",
         "cooling_energy_kWh"]
    )
    output = output[[c for c in cold_front if c in output.columns]].copy()
    out_cold_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_cold_csv, index=False)
    print(f"[cold load] Saved {len(output):,} rows → {out_cold_csv}")
    return output


# -----------------------------------------------------------------------------
# Resolve markets path + optional subset
# -----------------------------------------------------------------------------


def test_markets_candidate_name(n: int) -> str:
    return f"markets_with_urban_class_3km_test_{n}.csv"

def _pick_existing_test_markets_file(min_n: int) -> Optional[Path]:
    """
    Pick the smallest existing markets_with_urban_class_3km_test_{N}.csv with N >= min_n.
    Returns None if no such file exists.
    """
    candidates: List[Tuple[int, Path]] = []
    for p in SCRIPT_DIR.glob("markets_with_urban_class_3km_test_*.csv"):
        name = p.name
        prefix = "markets_with_urban_class_3km_test_"
        if not name.startswith(prefix) or not name.endswith(".csv"):
            continue
        n_str = name[len(prefix) : -len(".csv")]
        try:
            n_val = int(n_str)
        except ValueError:
            continue
        if n_val >= min_n:
            candidates.append((n_val, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]

def prepare_markets_with_urban_class_from_source(
    *,
    max_markets: Optional[int],
    output_csv: Path,
) -> Path:
    """
    Generate a markets CSV with x/y + built-up stats + urban_class starting from the
    original markets dataset and GHSL rasters (same logic as urban_thresholds.py).
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.is_file():
        print(f"[setup] Using existing prepared markets CSV: {output_csv}")
        return output_csv

    try:
        import urban_thresholds as ut  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import urban_thresholds.py. Run this script from the "
            "'load simulation' directory, or ensure it is on PYTHONPATH."
        ) from e

    print(
        f"[setup] Preparing markets from source (max_markets={max_markets}) "
        f"→ {output_csv}"
    )

    # Step 1: load original markets
    markets_gdf = ut.load_markets(ut.MARKETS_CSV)
    markets_gdf = markets_gdf.head(max_markets).copy()

    # Step 2: raster CRS
    raster_crs = ut.get_raster_crs(ut.RASTER_DIR)

    # Step 3: reproject to raster CRS
    invalid = ~markets_gdf.geometry.is_valid
    if invalid.sum() > 0:
        markets_gdf.loc[invalid, "geometry"] = markets_gdf.loc[invalid].geometry.buffer(0)
    try:
        markets_proj = markets_gdf.to_crs(raster_crs)
    except Exception:
        from pyproj import Transformer
        from shapely.geometry import Point

        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        coords = [(geom.x, geom.y) for geom in markets_gdf.geometry]
        transformed_coords = transformer.transform(
            [c[0] for c in coords], [c[1] for c in coords]
        )
        new_geometry = [Point(xy) for xy in zip(transformed_coords[0], transformed_coords[1])]
        markets_proj = markets_gdf.copy()
        markets_proj.geometry = new_geometry
        markets_proj.crs = raster_crs

    # Step 4: per-market built-up stats
    results = []
    total_markets = len(markets_proj)
    progress_interval = max(1, total_markets // 10)
    print(f"[setup] Computing built-up stats for {total_markets} markets…")
    for idx, row in markets_proj.iterrows():
        market_num = len(results) + 1
        if market_num % progress_interval == 0 or market_num == total_markets:
            print(f"  [setup] {market_num}/{total_markets} markets…")

        buffer_geom = row.geometry.buffer(ut.BUFFER_SIZE_M)
        raster_paths = ut.get_intersecting_rasters(
            buffer_geom, ut.RASTER_DIR, raster_crs
        )
        if not raster_paths:
            stats = {
                "builtup_mean_3km": 0.0,
                "builtup_p90_3km": 0.0,
                "builtup_fraction_3km": 0.0,
            }
        else:
            values = ut.extract_builtup_values(buffer_geom, raster_paths, raster_crs)
            stats = ut.compute_statistics(values)
        results.append({"index": idx, **stats})

    results_df = pd.DataFrame(results).set_index("index")
    markets_proj = markets_proj.join(results_df)

    # Step 5-7: thresholds + classify
    urban_threshold = markets_proj["builtup_mean_3km"].quantile(ut.URBAN_QUANTILE)
    peri_urban_threshold = markets_proj["builtup_mean_3km"].quantile(ut.PERI_URBAN_QUANTILE)
    markets_proj = ut.classify_markets(markets_proj, urban_threshold, peri_urban_threshold)

    # Step 8: back to EPSG:4326 and save (drop geometry, keep x/y columns)
    try:
        markets_output = markets_proj.to_crs("EPSG:4326")
    except Exception:
        from pyproj import Transformer
        from shapely.geometry import Point

        transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        coords = [(geom.x, geom.y) for geom in markets_proj.geometry]
        transformed_coords = transformer.transform(
            [c[0] for c in coords], [c[1] for c in coords]
        )
        new_geometry = [Point(xy) for xy in zip(transformed_coords[0], transformed_coords[1])]
        markets_output = markets_proj.copy()
        markets_output.geometry = new_geometry
        markets_output.crs = "EPSG:4326"

    output_df = markets_output.drop(columns=["geometry"])
    output_df.to_csv(output_csv, index=False)
    print(f"[setup] Prepared markets saved → {output_csv}")
    return output_csv


def resolve_markets_for_run(
    profile: str,
    test_n: int,
    markets_arg: Optional[str],
    subset_out: Path,
) -> Path:
    """Return path to the markets CSV used for annual + daily opening merge."""
    if markets_arg:
        p = Path(markets_arg).expanduser()
        if not p.is_file():
            raise FileNotFoundError(markets_arg)
        return p

    if profile == "full":
        try:
            return find_first_existing(SCRIPT_DIR, DEFAULT_FULL_MARKETS_CANDIDATES)
        except FileNotFoundError:
            return prepare_markets_with_urban_class_from_source(
                max_markets=None,
                output_csv=SCRIPT_DIR / DEFAULT_FULL_MARKETS_CANDIDATES[0],
            )

    # TEST profile: generate markets (with urban_class) from source dataset + rasters.
    # This avoids depending on any pre-made markets_with_urban_class_3km_test_*.csv.
    return prepare_markets_with_urban_class_from_source(
        max_markets=test_n,
        output_csv=subset_out,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annual→daily→hourly meat→hourly cold load (single script)."
    )
    p.add_argument(
        "--scenario",
        choices=tuple(NATIONAL_BEEF_SCENARIOS_KG),
        default="baseline_2015_470kt",
        help="National annual beef-consumption scenario",
    )
    p.add_argument(
        "--profile",
        choices=("full", "test"),
        default="full",
        help="full = all markets in classified CSV; test = N markets (see --test-n)",
    )
    p.add_argument(
        "--test-n",
        type=int,
        default=100,
        help="Number of markets when --profile test (default 100)",
    )
    p.add_argument(
        "--markets",
        default=None,
        help="Explicit markets CSV (overrides automatic lookup)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: pipeline_runs/<profile>[_<n>])",
    )
    p.add_argument(
        "--population",
        default=None,
        help="Explicit population xlsx/csv (optional)",
    )
    p.add_argument(
        "--cooldown-hours",
        type=float,
        default=4.0,
        help="Hours window for kWh→kW (same as original cold_load_index)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "test" and args.test_n < 1:
        raise SystemExit("--test-n must be >= 1")

    label = "full" if args.profile == "full" else f"test_{args.test_n}"
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else SCRIPT_DIR / "pipeline_runs" / label
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    national_annual_beef_kg = NATIONAL_BEEF_SCENARIOS_KG[args.scenario]
    modelled_annual_beef_kg = national_annual_beef_kg * MODELLED_MARKET_SHARE

    markets_path = resolve_markets_for_run(
        args.profile,
        args.test_n,
        args.markets,
        subset_out=out_dir / "markets_subset.csv",
    )

    n_mk = len(pd.read_csv(markets_path))
    manifest = out_dir / "run_manifest.txt"
    manifest.write_text(
        f"scenario={args.scenario}\n"
        f"national_annual_beef_kg={national_annual_beef_kg}\n"
        f"modelled_market_share={MODELLED_MARKET_SHARE}\n"
        f"modelled_annual_beef_kg={modelled_annual_beef_kg}\n"
        f"profile={args.profile}\n"
        f"markets_csv={markets_path}\n"
        f"n_markets={n_mk}\n",
        encoding="utf-8",
    )

    annual_p = out_dir / "market_annual_meat_allocation.csv"
    daily_p = out_dir / "market_daily_meat_2019.csv"
    hourly_meat_p = out_dir / "market_hourly_meat_2019.csv"
    cold_p = out_dir / "market_hourly_cold_loads_2019.csv"

    run_annual_allocation(
        markets_path,
        annual_p,
        modelled_annual_meat_kg=modelled_annual_beef_kg,
        population_file=args.population,
    )
    run_daily_disaggregation(annual_p, markets_path, daily_p, year=TARGET_YEAR)
    run_hourly_meat(daily_p, hourly_meat_p)
    run_hourly_cold_load(
        hourly_meat_p, cold_p, cooldown_hours=args.cooldown_hours
    )

    print("\nDone.")
    print(f"Outputs under: {out_dir}")
    print(f"  {annual_p.name}")
    print(f"  {daily_p.name}")
    print(f"  {hourly_meat_p.name}")
    print(f"  {cold_p.name}")


if __name__ == "__main__":
    main()
