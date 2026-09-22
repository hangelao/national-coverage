"""
Independent one-at-a-time techno-economic sensitivity workflow.

Static commands (never optimize and never create result directories):

    python sensitivity/sensitivity_technoeconomic.py --list-cases
    python sensitivity/sensitivity_technoeconomic.py --validate-only

Future, explicit optimization commands:

    python sensitivity/sensitivity_technoeconomic.py --case pv_capex_low
    python sensitivity/sensitivity_technoeconomic.py --all

The optimization entry points use the existing run.run pipeline unchanged.
Each explicit run starts with a new case directory and an empty checkpoint
directory. No checkpoint is ever intentionally reused.
"""
from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import inspect
import json
import math
import pickle
import sys
import traceback
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd


SENSITIVITY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SENSITIVITY_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import Config  # noqa: E402
from input_paths import CORRECTED_LOAD, MARKETS, ROAD_DISTANCES, SOLAR_CF


FORMAL_BASELINE_RESULTS = Path("outputs/formal_baseline")
FORMAL_BASELINE_MANIFEST = FORMAL_BASELINE_RESULTS / "run_manifest.txt"
DEFAULT_OUTPUT_ROOT = Path("outputs/technoeconomic_sensitivity")

DEMAND_CSV = CORRECTED_LOAD
SOLAR_CSV = SOLAR_CF
MARKETS_CSV = MARKETS
ROAD_DISTANCES = ROAD_DISTANCES

EXPECTED_STATES = 37
EXPECTED_MARKETS = 1984
EXPECTED_SNAPSHOTS = 56
N_WORKERS = 4
C1_C2_SOLVER_THREADS = 4
C3_SOLVER_THREADS = 20

# The requested transport-cost sensitivity coefficient is a normalization
# control. At 0.063 USD/km the active Config.conv_marginal_cost formula is
# unchanged; other values scale that complete formula proportionally.
BASE_TRANSPORT_COST_COEFFICIENT_USD_PER_KM = 0.063

PROTECTED_SOURCE_FILES = (
    "config.py",
    "network_builder.py",
    "optimizer.py",
    "run.py",
    "data_loader.py",
)

STAGES = ("c1", "c2", "c3")
QUEUE_MARKERS = (
    "ALL_CASES_COMPLETE",
    "ALL_CASES_FINISHED_WITH_FAILURES",
    "QUEUE_ABORTED",
)


class QueueFatalError(RuntimeError):
    """Shared failure after which attempting another case would be unsafe."""


class CaseStageError(RuntimeError):
    """Failure confined to one configuration of one sensitivity case."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class CaseOutcome:
    """Final status and summary row for one independently attempted case."""

    case: str
    status: str
    stage_status: dict[str, str]
    completed_configurations: list[str]
    failed_configuration: str | None
    skipped_configurations: list[str]
    process_exit_code: int
    error: str
    started_at: str
    finished_at: str
    output_path: str
    summary: pd.DataFrame
    checkpoint_path: str
    stdout_path: str
    stderr_path: str

@dataclass(frozen=True)
class CaseSpec:
    """Definition of one isolated sensitivity case."""

    name: str
    changes: tuple[tuple[str, float], ...]
    description: str

    @property
    def change_dict(self) -> dict[str, float]:
        return dict(self.changes)

    @property
    def changed_fields(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.changes)


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "pv_capex_low",
        (("pv_capex_per_kw", 464.0),),
        "PV CAPEX 20% below baseline",
    ),
    CaseSpec(
        "pv_capex_high",
        (("pv_capex_per_kw", 696.0),),
        "PV CAPEX 20% above baseline",
    ),
    CaseSpec(
        "battery_capex_low",
        (
            ("battery_energy_capex_per_kwh", 192.8),
            ("battery_power_capex_per_kw", 297.6),
        ),
        "Battery energy and inverter CAPEX group 20% below baseline",
    ),
    CaseSpec(
        "battery_capex_high",
        (
            ("battery_energy_capex_per_kwh", 289.2),
            ("battery_power_capex_per_kw", 446.4),
        ),
        "Battery energy and inverter CAPEX group 20% above baseline",
    ),
    CaseSpec(
        "fridge_capex_low",
        (("fridge_capex_per_kw_cold", 1005.059383),),
        "Refrigerator CAPEX 20% below formal baseline",
    ),
    CaseSpec(
        "fridge_capex_high",
        (("fridge_capex_per_kw_cold", 1507.589075),),
        "Refrigerator CAPEX 20% above formal baseline",
    ),
    CaseSpec(
        "fridge_cop_low",
        (("fridge_cop", 2.2),),
        "Refrigerator COP reduced to 2.2",
    ),
    CaseSpec(
        "fridge_cop_high",
        (("fridge_cop", 3.2),),
        "Refrigerator COP increased to 3.2",
    ),
    CaseSpec(
        "transport_cost_low",
        (("transport_cost_coefficient_usd_per_km", 0.0315),),
        "Transport-cost coefficient reduced from 0.063 to 0.0315 USD/km",
    ),
    CaseSpec(
        "transport_cost_high",
        (("transport_cost_coefficient_usd_per_km", 0.0945),),
        "Transport-cost coefficient increased from 0.063 to 0.0945 USD/km",
    ),
    CaseSpec(
        "thermal_tau_1h",
        (("transit_thermal_tau_hr", 1.0),),
        "Transport thermal-retention time constant reduced to 1 hour",
    ),
    CaseSpec(
        "thermal_tau_3h",
        (("transit_thermal_tau_hr", 3.0),),
        "Transport thermal-retention time constant increased to 3 hours",
    ),
)
CASE_BY_NAME = {case.name: case for case in CASES}


@dataclass
class SensitivityConfig(Config):
    """
    Config with one sensitivity-only transport normalization field.

    The baseline value exactly reproduces Config.conv_marginal_cost. This
    avoids editing shared Config while making the requested 0.063 USD/km
    coefficient independently replaceable and picklable for worker processes.
    """

    transport_cost_coefficient_usd_per_km: float = (
        BASE_TRANSPORT_COST_COEFFICIENT_USD_PER_KM
    )

    def conv_marginal_cost(self, d: float) -> float:
        scale = (
            self.transport_cost_coefficient_usd_per_km
            / BASE_TRANSPORT_COST_COEFFICIENT_USD_PER_KM
        )
        return super().conv_marginal_cost(d) * scale


@dataclass
class _InjectedCaseConfig(SensitivityConfig):
    """
    Runtime Config class temporarily injected into run.py.

    run.run constructs Config internally. Its existing API has no config
    argument, so this picklable subclass applies only the active case changes
    after every construction/dataclasses.replace call. The run module's
    original Config class is restored in a finally block.
    """

    active_changes: ClassVar[dict[str, float]] = {}

    def __post_init__(self) -> None:
        for field_name, value in self.active_changes.items():
            setattr(self, field_name, value)
        self.validate_transport_operating_window()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_key_value_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
            values[key] = value
    return values


def _jsonable_config(config: SensitivityConfig) -> dict:
    """Serialize Config without embedding the large road-distance dictionary."""
    values = asdict(config)
    road_distances = values.pop("road_distances", None)
    values["road_distances_loaded"] = road_distances is not None
    values["road_distance_pair_count"] = (
        len(config.road_distances) if config.road_distances is not None else 0
    )
    return values


def make_baseline_config() -> SensitivityConfig:
    """Return a new formal-baseline Config object on every call."""
    return SensitivityConfig(
        include_pcm=False,
        hourly_loads_csv=str(DEMAND_CSV),
        solar_csv=str(SOLAR_CSV),
        output_dir="",
        road_distances_path=str(ROAD_DISTANCES),
        road_distances=None,
        solver="gurobi",
        solver_threads=C1_C2_SOLVER_THREADS,
    )


def make_case_config(case: CaseSpec) -> SensitivityConfig:
    """Create a fresh baseline object and replace only this case's fields."""
    baseline = make_baseline_config()
    return replace(baseline, **case.change_dict)


def _field_differences(
    baseline: SensitivityConfig,
    candidate: SensitivityConfig,
) -> dict[str, tuple[object, object]]:
    differences: dict[str, tuple[object, object]] = {}
    for definition in fields(SensitivityConfig):
        name = definition.name
        before = getattr(baseline, name)
        after = getattr(candidate, name)
        if before != after:
            differences[name] = (before, after)
    return differences


def _assert_baseline_contract() -> dict[str, str]:
    """Confirm active Config and source hashes still match the formal run."""
    if not FORMAL_BASELINE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Formal baseline manifest missing: {FORMAL_BASELINE_MANIFEST}"
        )
    manifest = _read_key_value_manifest(FORMAL_BASELINE_MANIFEST)
    baseline = make_baseline_config()

    expected_values = {
        "pv_capex_per_kw": 580.0,
        "battery_energy_capex_per_kwh": 241.0,
        "battery_power_capex_per_kw": 372.0,
        "fridge_capex_per_kw_cold": 1256.324229,
        "fridge_cop": 2.7,
        "fridge_lifetime_years": 20,
        "discount_rate": 0.08,
        "transit_thermal_tau_hr": 2.0,
        "van_speed_kmh": 40.0,
        "include_pcm": False,
        "solver_threads": 4,
        "use_representative_weeks": True,
        "n_representative_weeks": 1,
        "representative_period_seed": 42,
        "time_resolution_hours": 3,
        "transport_operating_start_hour": 4,
        "transport_operating_end_hour": 19,
        "checkpoint_compatibility_marker": (
            "battery_external_ac_power_v1_transport_window_v1"
        ),
    }
    for field_name, expected in expected_values.items():
        actual = getattr(baseline, field_name)
        if actual != expected:
            raise AssertionError(
                f"Baseline Config mismatch: {field_name}={actual!r}, "
                f"expected {expected!r}"
            )

    manifest_contract = {
        "input_demand_path": str(DEMAND_CSV),
        "solar_input_path": str(SOLAR_CSV),
        "markets_input_path": str(MARKETS_CSV),
        "road_distances_path": str(ROAD_DISTANCES),
        "number_of_markets": str(EXPECTED_MARKETS),
        "number_of_states": str(EXPECTED_STATES),
        "number_of_snapshots": str(EXPECTED_SNAPSHOTS),
        "snapshot_weight_sum": "8760",
        "representative_weeks": "true",
        "number_of_representative_weeks": "1",
        "representative_period_seed": "42",
        "time_resolution_hours": "3",
        "transport_operating_start_hour": "4",
        "transport_operating_end_hour": "19",
        "checkpoint_compatibility_marker": (
            "battery_external_ac_power_v1_transport_window_v1"
        ),
        "c1_c2_workers": str(N_WORKERS),
        "c1_c2_solver_threads_per_worker": str(C1_C2_SOLVER_THREADS),
        "c3_solver_threads": str(C3_SOLVER_THREADS),
        "include_pcm": "false for C1, C2, and C3",
    }
    for key, expected in manifest_contract.items():
        actual = manifest.get(key)
        if actual != expected:
            raise AssertionError(
                f"Formal manifest mismatch: {key}={actual!r}, expected {expected!r}"
            )

    for filename in PROTECTED_SOURCE_FILES:
        manifest_key = f"{filename}_sha256"
        actual_hash = _sha256(PROJECT_DIR / filename)
        expected_hash = manifest.get(manifest_key)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"Protected source differs from formal baseline: {filename}"
            )

    for path in (DEMAND_CSV, SOLAR_CSV, MARKETS_CSV, ROAD_DISTANCES):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_method_cost = Config.conv_marginal_cost(baseline, 100.0)
    if not math.isclose(
        baseline.conv_marginal_cost(100.0),
        base_method_cost,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError(
            "SensitivityConfig baseline does not reproduce Config transport cost"
        )
    return manifest


def _eligible_interstate_pairs_by_tau(
    tau_values: Iterable[float],
) -> dict[float, int]:
    """
    Recalculate national-mesh interstate eligibility without building a model.

    Returns undirected pair counts. The optimization creates two directed
    links per retained pair.
    """
    markets = pd.read_csv(
        MARKETS_CSV,
        usecols=["uniq_id", "statename"],
        dtype={"uniq_id": "Int64"},
    )
    if len(markets) != EXPECTED_MARKETS:
        raise AssertionError(
            f"Market count {len(markets)} != formal baseline {EXPECTED_MARKETS}"
        )
    market_state = {
        int(row.uniq_id): str(row.statename)
        for row in markets.itertuples(index=False)
    }
    with ROAD_DISTANCES.open("rb") as handle:
        road_distances = pickle.load(handle)

    expected_matrix_entries = EXPECTED_MARKETS * EXPECTED_MARKETS
    if len(road_distances) != expected_matrix_entries:
        raise AssertionError(
            f"Road-distance entry count {len(road_distances)} != "
            f"{expected_matrix_entries}"
        )

    taus = sorted(set(float(value) for value in tau_values))
    counts = {tau: 0 for tau in taus}
    baseline = make_baseline_config()
    maximum_distance = {
        tau: (
            -baseline.van_speed_kmh
            * tau
            * math.log(baseline.transit_efficiency_floor)
        )
        for tau in taus
    }
    for (market_a, market_b), distance in road_distances.items():
        # The input is a complete directed matrix including the diagonal;
        # the national builder enumerates each unordered i<j pair once.
        if int(market_a) >= int(market_b):
            continue
        state_a = market_state.get(int(market_a))
        state_b = market_state.get(int(market_b))
        if state_a is None or state_b is None:
            raise AssertionError(
                f"Road-distance market missing from market set: "
                f"{market_a}, {market_b}"
            )
        if state_a == state_b:
            continue
        for tau in taus:
            if float(distance) <= maximum_distance[tau]:
                counts[tau] += 1
    return counts


def validate_cases(*, include_link_counts: bool = True) -> pd.DataFrame:
    """
    Validate one-at-a-time isolation and all case-derived properties.

    This function performs no writes, imports no optimizer, and opens no
    checkpoint. It may read the road-distance input for eligibility counts.
    """
    _assert_baseline_contract()
    untouched_baseline = make_baseline_config()
    baseline_snapshot = copy.deepcopy(asdict(untouched_baseline))
    baseline_cost_100km = untouched_baseline.conv_marginal_cost(100.0)
    baseline_efficiencies = {
        distance: untouched_baseline.transit_efficiency(distance)
        for distance in (0.0, 100.0, 250.0)
    }

    tau_counts: dict[float, int] = {}
    if include_link_counts:
        tau_counts = _eligible_interstate_pairs_by_tau((1.0, 2.0, 3.0))

    rows: list[dict] = []
    case_objects: list[SensitivityConfig] = []
    for case in CASES:
        baseline = make_baseline_config()
        candidate = make_case_config(case)
        if any(candidate is previous for previous in case_objects):
            raise AssertionError("Case Config object was reused")
        case_objects.append(candidate)

        differences = _field_differences(baseline, candidate)
        if set(differences) != set(case.changed_fields):
            raise AssertionError(
                f"{case.name}: changed fields {sorted(differences)} != "
                f"expected {sorted(case.changed_fields)}"
            )
        for field_name, expected in case.changes:
            actual = getattr(candidate, field_name)
            if actual != expected:
                raise AssertionError(
                    f"{case.name}: {field_name}={actual!r}, expected {expected!r}"
                )
        if candidate.include_pcm:
            raise AssertionError(f"{case.name}: PCM must remain disabled")

        expected_fridge_annual = (
            candidate.fridge_capex_per_kw_cold
            * candidate.fridge_cop
            * candidate.capital_recovery_factor(
                candidate.fridge_lifetime_years
            )
        )
        if not math.isclose(
            candidate.fridge_annualised_capex_per_kw_electric,
            expected_fridge_annual,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{case.name}: refrigerator annualisation failed")

        expected_battery_energy = (
            candidate.battery_energy_capex_per_kwh
            * candidate.capital_recovery_factor(
                candidate.battery_lifetime_years
            )
        )
        expected_battery_power = (
            candidate.battery_power_capex_per_kw
            * candidate.capital_recovery_factor(
                candidate.battery_lifetime_years
            )
        )
        if not math.isclose(
            candidate.battery_annualised_energy_capex_per_kwh,
            expected_battery_energy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{case.name}: battery-energy annualisation failed")
        if not math.isclose(
            candidate.battery_annualised_power_capex_per_kw,
            expected_battery_power,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{case.name}: battery-power annualisation failed")

        expected_transport = baseline_cost_100km * (
            candidate.transport_cost_coefficient_usd_per_km
            / BASE_TRANSPORT_COST_COEFFICIENT_USD_PER_KM
        )
        if not math.isclose(
            candidate.conv_marginal_cost(100.0),
            expected_transport,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise AssertionError(f"{case.name}: transport-cost scaling failed")

        for distance in baseline_efficiencies:
            expected_efficiency = math.exp(
                -(distance / candidate.van_speed_kmh)
                / candidate.transit_thermal_tau_hr
            )
            if not math.isclose(
                candidate.transit_efficiency(distance),
                expected_efficiency,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise AssertionError(
                    f"{case.name}: thermal efficiency failed at {distance} km"
                )
        if case.name.startswith("thermal_tau_"):
            if not math.isclose(
                candidate.conv_marginal_cost(100.0),
                baseline_cost_100km,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise AssertionError(
                    f"{case.name}: thermal case changed transport cost"
                )
        if case.name.startswith("transport_cost_"):
            for distance, baseline_efficiency in baseline_efficiencies.items():
                if not math.isclose(
                    candidate.transit_efficiency(distance),
                    baseline_efficiency,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise AssertionError(
                        f"{case.name}: cost case changed thermal efficiency"
                    )

        rows.append(
            {
                "case": case.name,
                "changed_fields": ",".join(case.changed_fields),
                "changed_values": json.dumps(case.change_dict, sort_keys=True),
                "pv_annualised_capex_usd_per_kw_year":
                    candidate.pv_annualised_capex_per_kw,
                "battery_energy_annualised_capex_usd_per_kwh_year":
                    candidate.battery_annualised_energy_capex_per_kwh,
                "battery_power_annualised_capex_usd_per_kw_year":
                    candidate.battery_annualised_power_capex_per_kw,
                "fridge_annualised_capex_usd_per_kw_electric_year":
                    candidate.fridge_annualised_capex_per_kw_electric,
                "transport_marginal_cost_at_100km_usd_per_kwh":
                    candidate.conv_marginal_cost(100.0),
                "transport_efficiency_at_100km":
                    candidate.transit_efficiency(100.0),
                "eligible_interstate_pairs": (
                    tau_counts.get(candidate.transit_thermal_tau_hr, np.nan)
                    if include_link_counts else np.nan
                ),
                "eligible_directed_interstate_links": (
                    2 * tau_counts[candidate.transit_thermal_tau_hr]
                    if include_link_counts else np.nan
                ),
                "validation": "PASS",
            }
        )

    if asdict(untouched_baseline) != baseline_snapshot:
        raise AssertionError("Original baseline Config object was modified")
    return pd.DataFrame(rows)


def list_cases() -> None:
    baseline = make_baseline_config()
    print("Formal-baseline one-at-a-time techno-economic cases")
    print(f"Baseline results: {FORMAL_BASELINE_RESULTS}")
    print(f"Default future output root: {DEFAULT_OUTPUT_ROOT}")
    for index, case in enumerate(CASES, 1):
        values = ", ".join(
            f"{name}={value:g}" for name, value in case.changes
        )
        baseline_values = ", ".join(
            f"{name}={getattr(baseline, name):g}" for name, _ in case.changes
        )
        print(
            f"{index:2d}. {case.name}: {values} "
            f"(baseline: {baseline_values})"
        )
    print("No optimization was run.")


def _assert_safe_output_root(output_root: Path, *, all_cases: bool) -> Path:
    resolved = output_root.expanduser().resolve()
    forbidden = {
        FORMAL_BASELINE_RESULTS.resolve(),
        (
            PROJECT_DIR
            / "results_peak_coincidence_470kt_electric_fridge_20260722"
        ).resolve(),
        (
            PROJECT_DIR
            / "results_technoeconomic_470kt_electric_fridge_20260723"
        ).resolve(),
    }
    if resolved in forbidden:
        raise ValueError(f"Refusing protected result root: {resolved}")
    if resolved == PROJECT_DIR.resolve() or PROJECT_DIR.resolve() not in resolved.parents:
        raise ValueError(
            f"Output root must be a dedicated child of {PROJECT_DIR}: {resolved}"
        )
    if all_cases and resolved.exists():
        raise FileExistsError(
            f"--all requires a new result root; already exists: {resolved}"
        )
    return resolved


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(
        [
            *checkpoint_dir.glob("c1_*.pkl"),
            *checkpoint_dir.glob("c2_*.pkl"),
            *checkpoint_dir.glob("c3_national.pkl"),
        ]
    )


def _require_stage_checkpoints(
    checkpoint_dir: Path,
    pattern: str,
    expected: int,
    stage: str,
) -> None:
    count = sum(1 for _ in checkpoint_dir.glob(pattern))
    if count != expected:
        raise RuntimeError(
            f"{stage} gate failed: expected {expected} {pattern} files in "
            f"{checkpoint_dir}, found {count}"
        )


def _prepare_case_paths(
    output_root: Path,
    case: CaseSpec,
) -> tuple[Path, Path, Path, Path, Path]:
    case_dir = output_root / "cases" / case.name
    output_dir = case_dir / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = case_dir / "logs"
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    if case_dir.exists():
        raise QueueFatalError(
            f"Refusing existing case directory: {case_dir}"
        )
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        logs_dir.mkdir(parents=True, exist_ok=False)
        stdout_path.touch(exist_ok=False)
        stderr_path.touch(exist_ok=False)
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot create isolated case structure for {case.name}: {exc}"
        ) from exc
    if checkpoint_dir.resolve().parent != output_dir.resolve():
        raise QueueFatalError(
            f"Checkpoint isolation failed: {checkpoint_dir}"
        )
    existing = _checkpoint_files(checkpoint_dir)
    if existing:
        raise QueueFatalError(f"Checkpoint directory is not empty: {existing}")
    return case_dir, output_dir, checkpoint_dir, stdout_path, stderr_path


def _component_capex(capacity: pd.DataFrame, config: Config) -> float:
    if capacity is None or capacity.empty:
        return 0.0
    return float(
        capacity.get("solar_kw", pd.Series(dtype=float)).sum()
        * config.pv_annualised_capex_per_kw
        + capacity.get("fridge_kw", pd.Series(dtype=float)).sum()
        * config.fridge_annualised_capex_per_kw_electric
        + capacity.get("battery_kwh", pd.Series(dtype=float)).sum()
        * config.battery_annualised_energy_capex_per_kwh
        + capacity.get("battery_power_kw", pd.Series(dtype=float)).sum()
        * config.battery_annualised_power_capex_per_kw
        + capacity.get("pcm_kwh", pd.Series(dtype=float)).sum()
        * config.pcm_annualised_capex_per_kwh
    )


def _transport_totals(frames: Iterable[pd.DataFrame]) -> dict[str, float]:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return {"capacity_kw": 0.0, "flow_kwh": 0.0, "cost_usd": 0.0}
    combined = pd.concat(usable, ignore_index=True)
    if "fuel_opex_usd" in combined:
        cost = float(combined["fuel_opex_usd"].sum())
    else:
        cost = float(
            (
                combined["marginal_cost_per_kwh"]
                * combined["total_flow_kwh"]
            ).sum()
        )
    return {
        "capacity_kw": float(combined["capacity_kw"].sum()),
        "flow_kwh": float(combined["total_flow_kwh"].sum()),
        "cost_usd": cost,
    }


def _capacity_totals(capacity: pd.DataFrame, config: Config) -> dict[str, float]:
    def total(column: str) -> float:
        if capacity is None or capacity.empty or column not in capacity:
            return 0.0
        return float(capacity[column].sum())

    fridge_input = total("fridge_kw")
    return {
        "pv_capacity_kw": total("solar_kw"),
        "fridge_input_capacity_kw_electric": fridge_input,
        "fridge_output_capacity_kw_cold": fridge_input * config.fridge_cop,
        "battery_energy_capacity_kwh": total("battery_kwh"),
        "battery_inverter_capacity_kw": total("battery_power_kw"),
        "pcm_capacity_kwh": total("pcm_kwh"),
    }


def _stage_payloads_from_pipeline(pipeline_result: dict) -> dict[str, dict]:
    """Normalize successful pipeline results for stage-aware reporting."""
    standalone = pipeline_result["standalone_results"]
    intrastate = pipeline_result["intrastate_results"]
    national = pipeline_result["interstate_result"]
    return {
        "c1": {
            "capacity": pd.concat(
                [result["capacities"] for result in standalone.values()],
                axis=0,
            ),
            "transport": _transport_totals([]),
            "cost": float(
                sum(result["total_cost"] for result in standalone.values())
            ),
        },
        "c2": {
            "capacity": pd.concat(
                [result["capacities"] for result in intrastate.values()],
                axis=0,
            ),
            "transport": _transport_totals(
                result.get("transport") for result in intrastate.values()
            ),
            "cost": float(
                sum(result["total_cost"] for result in intrastate.values())
            ),
        },
        "c3": {
            "capacity": national["capacities"],
            "transport": _transport_totals([national.get("transport")]),
            "cost": float(national["total_cost"]),
        },
    }


def _load_completed_stage_payloads(
    checkpoint_dir: Path,
    stage_status: Mapping[str, str],
) -> dict[str, dict]:
    """Read only this case's own checkpoints to retain completed-stage results."""
    payloads: dict[str, dict] = {}
    for stage, pattern in (("c1", "c1_*.pkl"), ("c2", "c2_*.pkl")):
        if stage_status[stage] != "complete":
            continue
        paths = sorted(checkpoint_dir.glob(pattern))
        if len(paths) != EXPECTED_STATES:
            raise CaseStageError(
                stage,
                f"Cannot report completed {stage.upper()}: expected "
                f"{EXPECTED_STATES} case-local checkpoints, found {len(paths)}",
            )
        results = []
        for path in paths:
            with path.open("rb") as handle:
                results.append(pickle.load(handle))
        payloads[stage] = {
            "capacity": pd.concat(
                [result["capacities"] for result in results], axis=0
            ),
            "transport": _transport_totals(
                [] if stage == "c1" else
                (result.get("transport") for result in results)
            ),
            "cost": float(sum(result["total_cost"] for result in results)),
        }

    if stage_status["c3"] == "complete":
        path = checkpoint_dir / "c3_national.pkl"
        if not path.is_file():
            raise CaseStageError("c3", "Completed C3 checkpoint is missing")
        with path.open("rb") as handle:
            result = pickle.load(handle)
        payloads["c3"] = {
            "capacity": result["capacities"],
            "transport": _transport_totals([result.get("transport")]),
            "cost": float(result["total_cost"]),
        }
    return payloads


def _case_summary(
    case: CaseSpec,
    config: SensitivityConfig,
    payloads: Mapping[str, dict],
    stage_status: Mapping[str, str],
    *,
    status: str,
    failed_stage: str | None,
    error: str,
    error_log_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build one summary row, leaving unavailable failed-stage values as NaN."""
    row: dict[str, object] = {
        "case": case.name,
        "status": status,
        "failed_stage": failed_stage or "",
        "c1_status": stage_status["c1"],
        "c2_status": stage_status["c2"],
        "c3_status": stage_status["c3"],
        "changed_fields": ",".join(case.changed_fields),
        "changed_values": json.dumps(case.change_dict, sort_keys=True),
        "error_message": error,
        "error_log_path": str(error_log_path),
    }
    metric_names = (
        "total_annual_system_cost_usd",
        "equipment_capex_usd_per_year",
        "transport_cost_usd_per_year",
        "pv_capacity_kw",
        "fridge_input_capacity_kw_electric",
        "fridge_output_capacity_kw_cold",
        "battery_energy_capacity_kwh",
        "battery_inverter_capacity_kw",
        "pcm_capacity_kwh",
        "transport_capacity_kw",
        "annual_transport_flow_kwh",
    )
    stage_frames: dict[str, pd.DataFrame] = {}
    costs: dict[str, float] = {}
    for stage in STAGES:
        for metric in metric_names:
            row[f"{stage}_{metric}"] = np.nan
        if stage_status[stage] != "complete" or stage not in payloads:
            continue
        payload = payloads[stage]
        capacity = payload["capacity"]
        transport = payload["transport"]
        costs[stage] = float(payload["cost"])
        stage_row = {
            "case": case.name,
            "configuration": stage.upper(),
            "status": "complete",
            "total_annual_system_cost_usd": costs[stage],
            "equipment_capex_usd_per_year":
                _component_capex(capacity, config),
            "transport_cost_usd_per_year": transport["cost_usd"],
            **_capacity_totals(capacity, config),
            "transport_capacity_kw": transport["capacity_kw"],
            "annual_transport_flow_kwh": transport["flow_kwh"],
        }
        stage_frames[stage] = pd.DataFrame([stage_row])
        for key, value in stage_row.items():
            if key not in {"case", "configuration", "status"}:
                row[f"{stage}_{key}"] = value

    for before, after, label in (
        ("c1", "c2", "c1_to_c2"),
        ("c2", "c3", "c2_to_c3"),
        ("c1", "c3", "c1_to_c3"),
    ):
        row[f"{label}_saving_usd"] = np.nan
        row[f"{label}_saving_pct"] = np.nan
        if before not in costs or after not in costs:
            continue
        saving = costs[before] - costs[after]
        row[f"{label}_saving_usd"] = saving
        row[f"{label}_saving_pct"] = (
            saving / costs[before] * 100.0 if costs[before] else np.nan
        )
    return pd.DataFrame([row]), stage_frames


def _case_manifest(
    case: CaseSpec,
    config: SensitivityConfig,
    case_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    status: str,
) -> dict:
    baseline = make_baseline_config()
    return {
        "created_at": _now_iso(),
        "started_at": _now_iso(),
        "status": status,
        "case": case.name,
        "description": case.description,
        "baseline_results": str(FORMAL_BASELINE_RESULTS),
        "baseline_config": _jsonable_config(baseline),
        "case_config": _jsonable_config(config),
        "changed_parameters": {
            name: {"baseline": getattr(baseline, name), "case": value}
            for name, value in case.changes
        },
        "stage_status": {stage: "pending" for stage in STAGES},
        "completed_configurations": [],
        "failed_configuration": None,
        "skipped_configurations": [],
        "process_exit_code": None,
        "error": "",
        "paths": {
            "case_root": str(case_dir),
            "combined_output_directory": str(output_dir),
            "checkpoint_directory": str(checkpoint_dir),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "c1_output": str(case_dir / "C1_summary.csv"),
            "c2_output": str(case_dir / "C2_summary.csv"),
            "c3_output": str(case_dir / "C3_summary.csv"),
            "case_summary": str(case_dir / "case_summary.csv"),
        },
        "inputs": {
            "demand_csv": str(DEMAND_CSV),
            "solar_csv": str(SOLAR_CSV),
            "markets_csv": str(MARKETS_CSV),
            "road_distances": str(ROAD_DISTANCES),
        },
        "execution": {
            "order": ["C1 standalone", "C2 intrastate", "C3 national"],
            "cases_sequential": True,
            "case_failures_are_independent": True,
            "c1_c2_workers": N_WORKERS,
            "c1_c2_solver_threads_per_worker": C1_C2_SOLVER_THREADS,
            "c3_solver_threads": C3_SOLVER_THREADS,
            "include_pcm": False,
            "checkpoint_loading_policy": (
                "empty case-local directory only; any pipeline checkpoint "
                "load aborts the queue"
            ),
        },
        "source_hashes": _current_protected_hashes(),
    }


def _write_json(path: Path, value: Mapping) -> None:
    def _default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")

    try:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise QueueFatalError(f"Cannot write {path}: {exc}") from exc


def _current_protected_hashes() -> dict[str, str]:
    return {
        filename: _sha256(PROJECT_DIR / filename)
        for filename in PROTECTED_SOURCE_FILES
    }


def _assert_protected_sources_unchanged(
    expected_hashes: Mapping[str, str],
) -> None:
    current = _current_protected_hashes()
    changed = [name for name in PROTECTED_SOURCE_FILES
               if current[name] != expected_hashes[name]]
    if changed:
        raise QueueFatalError(
            "Protected shared source changed during queue: "
            + ", ".join(changed)
        )


def _validate_shared_run_interface() -> None:
    try:
        import run as main_run  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise QueueFatalError(f"Cannot import shared run module: {exc}") from exc
    source = inspect.getsource(main_run.run)
    contract = 'checkpoint_dir = os.path.join(output_dir, "checkpoints")'
    if contract not in source:
        raise QueueFatalError(
            "run.py checkpoint routing changed; refusing to launch"
        )


def _write_case_marker(
    case_dir: Path,
    marker: str,
    case: str,
    error: str = "",
) -> None:
    existing = [name for name in ("CASE_COMPLETE", "CASE_FAILED")
                if (case_dir / name).exists()]
    if existing:
        raise QueueFatalError(
            f"Case {case} already has final marker(s): {existing}"
        )
    body = f"case={case}\nstatus={'complete' if marker == 'CASE_COMPLETE' else 'failed'}\n"
    if error:
        body += f"error={error}\n"
    try:
        (case_dir / marker).write_text(body, encoding="utf-8")
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot write final marker for {case}: {exc}"
        ) from exc


def _finalize_case(
    case: CaseSpec,
    config: SensitivityConfig,
    case_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    manifest_path: Path,
    manifest: dict,
    stage_status: dict[str, str],
    started_at: str,
    pipeline_result: dict | None,
    failure: Exception | None,
) -> CaseOutcome:
    status = "failed" if failure else "complete"
    failed_stage = getattr(failure, "stage", None) if failure else None
    error = f"{type(failure).__name__}: {failure}" if failure else ""
    finished_at = _now_iso()

    if failure:
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write("".join(traceback.format_exception(failure)))

    try:
        payloads = (
            _stage_payloads_from_pipeline(pipeline_result)
            if status == "complete" and pipeline_result is not None
            else _load_completed_stage_payloads(checkpoint_dir, stage_status)
        )
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot read case-local completed results for {case.name}: {exc}"
        ) from exc
    except Exception as reporting_exc:
        payloads = {}
        reporting_error = (
            f"Partial-result reporting error: {type(reporting_exc).__name__}: "
            f"{reporting_exc}"
        )
        error = f"{error}; {reporting_error}" if error else reporting_error
        status = "failed"
        if failed_stage is None:
            failed_stage = getattr(reporting_exc, "stage", "c3")
            stage_status[failed_stage] = "failed"

    completed = [stage.upper() for stage in STAGES
                 if stage_status[stage] == "complete"]
    skipped = [stage.upper() for stage in STAGES
               if stage_status[stage] == "skipped"]

    summary, stage_frames = _case_summary(
        case,
        config,
        payloads,
        stage_status,
        status=status,
        failed_stage=failed_stage,
        error=error,
        error_log_path=stderr_path,
    )
    exit_code = 0 if status == "complete" else 1
    metadata = {
        "completed_configurations": ",".join(completed),
        "failed_configuration": failed_stage.upper() if failed_stage else "",
        "skipped_configurations": ",".join(skipped),
        "process_exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_path": str(output_dir),
        "checkpoint_path": str(checkpoint_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    for key, value in metadata.items():
        summary[key] = value

    try:
        summary.to_csv(case_dir / "case_summary.csv", index=False)
        for stage, frame in stage_frames.items():
            frame.to_csv(case_dir / f"{stage.upper()}_summary.csv", index=False)
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot write derived summaries for {case.name}: {exc}"
        ) from exc

    manifest.update({
        "status": status,
        "finished_at": finished_at,
        "stage_status": stage_status,
        "completed_configurations": completed,
        "failed_configuration": failed_stage.upper() if failed_stage else None,
        "skipped_configurations": skipped,
        "process_exit_code": exit_code,
        "error": error,
        "summary": summary.iloc[0].to_dict(),
    })
    _write_json(manifest_path, manifest)
    marker = "CASE_COMPLETE" if status == "complete" else "CASE_FAILED"
    _write_case_marker(case_dir, marker, case.name, error)
    return CaseOutcome(
        case=case.name,
        status=status,
        stage_status=dict(stage_status),
        completed_configurations=completed,
        failed_configuration=failed_stage,
        skipped_configurations=skipped,
        process_exit_code=exit_code,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        output_path=str(output_dir),
        checkpoint_path=str(checkpoint_dir),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        summary=summary,
    )


def _stage_status_after_failure(
    current: Mapping[str, str],
    failed_stage: str,
) -> dict[str, str]:
    """Mark one failed stage and skip only its later stages."""
    if failed_stage not in STAGES:
        raise ValueError(f"Unknown configuration stage: {failed_stage}")
    updated = dict(current)
    updated[failed_stage] = "failed"
    failed_index = STAGES.index(failed_stage)
    for later_stage in STAGES[failed_index + 1:]:
        if updated[later_stage] == "pending":
            updated[later_stage] = "skipped"
    return updated


def _run_one_case(case: CaseSpec, output_root: Path) -> CaseOutcome:
    """Run one isolated C1 -> C2 -> C3 case and return its final outcome."""
    started_at = _now_iso()
    (
        case_dir,
        output_dir,
        checkpoint_dir,
        stdout_path,
        stderr_path,
    ) = _prepare_case_paths(output_root, case)
    case_config = make_case_config(case)
    manifest_path = case_dir / "case_manifest.json"
    manifest = _case_manifest(
        case,
        case_config,
        case_dir,
        output_dir,
        checkpoint_dir,
        stdout_path,
        stderr_path,
        status="running",
    )
    manifest["started_at"] = started_at
    _write_json(manifest_path, manifest)

    try:
        import run as main_run  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise QueueFatalError(f"Cannot import shared run module: {exc}") from exc

    original_config_class = main_run.Config
    original_print = builtins.print
    completion_seen = False
    active_stage = "c1"
    stage_status = {stage: "pending" for stage in STAGES}
    pipeline_result: dict | None = None
    failure: Exception | None = None

    with stdout_path.open("a", encoding="utf-8") as stdout_handle:
        def guarded_print(*args, **kwargs):
            nonlocal completion_seen, active_stage
            message = " ".join(str(arg) for arg in args)
            original_print(*args, **kwargs)
            original_print(
                *args,
                file=stdout_handle,
                sep=kwargs.get("sep", " "),
                end=kwargs.get("end", "\n"),
                flush=True,
            )
            if "loaded from checkpoint" in message:
                raise QueueFatalError(
                    "Checkpoint load detected in isolated sensitivity case"
                )
            if "Config 1 |" in message and " ERROR:" in message:
                raise CaseStageError("c1", message.strip())
            if "Config 2 |" in message and " ERROR:" in message:
                raise CaseStageError("c2", message.strip())
            if "Config 3 FAILED:" in message:
                raise CaseStageError("c3", message.strip())
            marker = message.strip()
            if marker.startswith("[3/5] Config 2"):
                try:
                    _require_stage_checkpoints(
                        checkpoint_dir, "c1_*.pkl", EXPECTED_STATES, "C1"
                    )
                except RuntimeError as exc:
                    raise CaseStageError("c1", str(exc)) from exc
                stage_status["c1"] = "complete"
                active_stage = "c2"
            elif marker.startswith("[4/5] Config 3"):
                try:
                    _require_stage_checkpoints(
                        checkpoint_dir, "c1_*.pkl", EXPECTED_STATES, "C1"
                    )
                    _require_stage_checkpoints(
                        checkpoint_dir, "c2_*.pkl", EXPECTED_STATES, "C2"
                    )
                except RuntimeError as exc:
                    raise CaseStageError("c2", str(exc)) from exc
                stage_status["c1"] = "complete"
                stage_status["c2"] = "complete"
                active_stage = "c3"
            elif "PIPELINE COMPLETE" in marker:
                completion_seen = True

        _InjectedCaseConfig.active_changes = case.change_dict
        main_run.Config = _InjectedCaseConfig
        builtins.print = guarded_print
        try:
            pipeline_result = main_run.run(
                hourly_loads_csv=str(DEMAND_CSV),
                solar_csv=str(SOLAR_CSV),
                output_dir=str(output_dir),
                road_distances_path=str(ROAD_DISTANCES),
                markets_csv=str(MARKETS_CSV),
                debug=False,
                n_workers=N_WORKERS,
                solver_threads=C1_C2_SOLVER_THREADS,
                solver_threads_c3=C3_SOLVER_THREADS,
            )
        except QueueFatalError:
            raise
        except Exception as exc:
            failure = (
                exc if isinstance(exc, CaseStageError)
                else CaseStageError(
                    active_stage, f"{type(exc).__name__}: {exc}"
                )
            )
        finally:
            builtins.print = original_print
            main_run.Config = original_config_class
            _InjectedCaseConfig.active_changes = {}

    if failure is None:
        try:
            if not completion_seen:
                raise CaseStageError(
                    active_stage,
                    "Expected PIPELINE COMPLETE marker was not emitted",
                )
            _require_stage_checkpoints(
                checkpoint_dir, "c1_*.pkl", EXPECTED_STATES, "C1"
            )
            _require_stage_checkpoints(
                checkpoint_dir, "c2_*.pkl", EXPECTED_STATES, "C2"
            )
            if not (checkpoint_dir / "c3_national.pkl").is_file():
                raise CaseStageError(
                    "c3", "C3 checkpoint missing after pipeline completion"
                )
            if not (output_dir / "national_summary.csv").is_file():
                raise CaseStageError(
                    "c3", "National summary missing after pipeline completion"
                )
            stage_status = {stage: "complete" for stage in STAGES}
        except Exception as exc:
            failure = (
                exc if isinstance(exc, CaseStageError)
                else CaseStageError(active_stage, str(exc))
            )

    if failure is not None:
        stage_status = _stage_status_after_failure(
            stage_status, failure.stage
        )

    return _finalize_case(
        case,
        case_config,
        case_dir,
        output_dir,
        checkpoint_dir,
        stdout_path,
        stderr_path,
        manifest_path,
        manifest,
        stage_status,
        started_at,
        pipeline_result,
        failure,
    )


def _execute_independent_queue(
    cases: Iterable[CaseSpec],
    case_runner: Callable[[CaseSpec], CaseOutcome],
    shared_guard: Callable[[], None],
    outcome_callback: Callable[[list[CaseOutcome]], None] | None = None,
) -> tuple[list[CaseOutcome], QueueFatalError | None]:
    """Attempt independent cases until exhausted or a shared fatal error occurs."""
    outcomes: list[CaseOutcome] = []
    for case in cases:
        try:
            shared_guard()
            outcome = case_runner(case)
            outcomes.append(outcome)
            if outcome_callback is not None:
                outcome_callback(outcomes)
            shared_guard()
        except QueueFatalError as exc:
            return outcomes, exc
        except BaseException as exc:
            return outcomes, QueueFatalError(
                f"Unexpected queue-level failure: {type(exc).__name__}: {exc}"
            )
    return outcomes, None


def _queue_marker_name(
    outcomes: Iterable[CaseOutcome],
    expected_cases: int,
    fatal_error: QueueFatalError | None,
) -> str:
    outcomes = list(outcomes)
    if fatal_error is not None or len(outcomes) != expected_cases:
        return "QUEUE_ABORTED"
    if all(outcome.status == "complete" for outcome in outcomes):
        return "ALL_CASES_COMPLETE"
    return "ALL_CASES_FINISHED_WITH_FAILURES"


def _write_queue_summary(output_root: Path, outcomes: list[CaseOutcome]) -> None:
    if not outcomes:
        return
    try:
        pd.concat(
            [outcome.summary for outcome in outcomes], ignore_index=True
        ).to_csv(output_root / "technoeconomic_summary.csv", index=False)
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot write technoeconomic_summary.csv: {exc}"
        ) from exc


def _write_queue_marker(
    output_root: Path,
    marker: str,
    outcomes: Iterable[CaseOutcome],
    error: str = "",
) -> None:
    if marker not in QUEUE_MARKERS:
        raise ValueError(marker)
    existing = [name for name in QUEUE_MARKERS
                if (output_root / name).exists()]
    if existing:
        raise QueueFatalError(f"Queue marker already exists: {existing}")
    outcomes = list(outcomes)
    body = (
        f"status={marker.lower()}\n"
        f"attempted_cases={len(outcomes)}\n"
        f"complete_cases={sum(o.status == 'complete' for o in outcomes)}\n"
        f"failed_cases={sum(o.status == 'failed' for o in outcomes)}\n"
    )
    if error:
        body += f"error={error}\n"
    try:
        (output_root / marker).write_text(body, encoding="utf-8")
    except OSError as exc:
        raise QueueFatalError(f"Cannot write queue marker: {exc}") from exc


def run_selected_cases(
    cases: Iterable[CaseSpec],
    output_root: Path,
    *,
    all_cases: bool,
) -> int:
    """Run isolated cases sequentially while continuing after case failures."""
    selected = list(cases)
    if not selected:
        raise ValueError("No cases selected")

    # Shared preflight must pass before any result directory is created.
    _assert_baseline_contract()
    resolved_root = _assert_safe_output_root(
        output_root,
        all_cases=all_cases,
    )
    validate_cases(include_link_counts=True)
    _validate_shared_run_interface()
    protected_hashes = _current_protected_hashes()
    try:
        resolved_root.mkdir(parents=True, exist_ok=not all_cases)
    except OSError as exc:
        raise QueueFatalError(
            f"Cannot create shared result structure: {exc}"
        ) from exc

    def shared_guard() -> None:
        _assert_protected_sources_unchanged(protected_hashes)
        if not resolved_root.is_dir():
            raise QueueFatalError(
                f"Shared output root became unavailable: {resolved_root}"
            )

    def case_runner(case: CaseSpec) -> CaseOutcome:
        index = selected.index(case) + 1
        print(f"[case {index}/{len(selected)}] {case.name}")
        outcome = _run_one_case(case, resolved_root)
        if outcome.status == "failed":
            print(
                f"Case {case.name} failed at "
                f"{outcome.failed_configuration.upper()}; continuing queue."
            )
        return outcome

    outcomes, fatal_error = _execute_independent_queue(
        selected,
        case_runner,
        shared_guard,
        lambda rows: _write_queue_summary(resolved_root, rows),
    )
    marker = _queue_marker_name(outcomes, len(selected), fatal_error)
    if all_cases:
        _write_queue_marker(
            resolved_root,
            marker,
            outcomes,
            str(fatal_error) if fatal_error else "",
        )
    if fatal_error is not None:
        raise fatal_error
    return 0 if marker == "ALL_CASES_COMPLETE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent one-at-a-time techno-economic sensitivity for the "
            "formal 470 kt baseline."
        )
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--list-cases",
        action="store_true",
        help="List the 12 cases without running optimization.",
    )
    actions.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate Config isolation and derived values; never optimize.",
    )
    actions.add_argument(
        "--case",
        choices=tuple(CASE_BY_NAME),
        help="Explicitly run one isolated case in the future.",
    )
    actions.add_argument(
        "--all",
        action="store_true",
        help="Explicitly run all 12 cases sequentially in the future.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Future isolated result root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.list_cases, args.validate_only, args.case, args.all)):
        parser.print_help()
        print("\nNo explicit run option supplied; no optimization was run.")
        return 0
    if args.list_cases:
        list_cases()
        return 0
    if args.validate_only:
        validation = validate_cases(include_link_counts=True)
        print(validation.to_string(index=False))
        print("\nSTATIC VALIDATION PASSED")
        print("No result directory was created and no optimization was run.")
        return 0
    if args.case:
        return run_selected_cases(
            [CASE_BY_NAME[args.case]],
            args.output_root,
            all_cases=False,
        )
    return run_selected_cases(CASES, args.output_root, all_cases=True)


if __name__ == "__main__":
    raise SystemExit(main())
