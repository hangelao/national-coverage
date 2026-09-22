"""
budget_frontier.py
------------------
Budget-constrained cold-chain access frontier.

The main pipeline (run.py) answers "what does it cost to serve all cooling
demand?".  This module answers the complementary question that matters for
deployment: "given a fixed national investment budget, how much of the demand
can actually be served, and does coordination let the same money reach more
markets?"

Formulation
-----------
Two changes to the standard Config 1-3 networks:

  1. Load shedding.  Every ``*_cold`` bus gains a non-extendable Generator
     priced at VOLL.  Demand that the built infrastructure cannot meet is
     supplied by this generator, so the LP stays feasible at any budget and
     the shed dispatch *is* the unserved cooling energy.

  2. A global capital budget.  A single linopy constraint caps the sum of
     annualised capital cost over every extendable Generator / Link / Store:

         sum_i  capital_cost_i * capacity_i  <=  B

     Shedding generators are excluded (they carry zero capital cost anyway).

Minimising (capex + VOLL * unserved) subject to capex <= B therefore serves as
much demand as the budget allows, and sweeping B traces the access frontier.

Why the configurations are rebuilt here
---------------------------------------
Without a budget, Config 1 and Config 2 decompose into independent per-market
and per-state LPs, which is what optimizer.run_standalone/run_intrastate
exploit.  A *shared national budget couples every market*, so all three
configurations must be solved as one national LP.  Rather than duplicate
build logic, all three are produced from network_builder.build_national_network
by varying two inputs:

    C1  state_groups = one group per market   + floor > 1  -> no links at all
    C2  state_groups = the real state groups  + floor > 1  -> intra-state only
    C3  state_groups = the real state groups  + configured floor -> full mesh

``transit_efficiency`` is bounded above by 1, so any floor above 1 prunes every
cross-group link while leaving within-group links untouched.  This reuses the
audited builder rather than adding a parallel one.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from config import Config
from network_builder import build_national_network
from optimizer import _solve

# Sensible cooling energy per kg of meat: cp = 3.14 kJ/kg.K over dT = 36 K,
# i.e. 3.14 * 36 / 3600 kWh/kg.  Matches the constant used by the load
# simulation pipeline ("load simulation/load_simulation_pipeline.py").
KWH_COLD_PER_KG: float = 3.14 * 36.0 / 3600.0

# Value of lost load for unserved cooling (USD per kWh_cold).  Only needs to be
# far above any capital cost per kWh delivered so that the optimiser always
# prefers building capacity while budget remains.  Reported results are
# insensitive to the exact value; see check_voll_dominance().
DEFAULT_VOLL: float = 100.0

# A market counts as "served" when this fraction or more of its annual cooling
# demand is met by built infrastructure rather than shedding.
SERVED_THRESHOLD: float = 0.99


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def _groups_for(kind: str, state_groups: Dict[str, list]) -> Dict[str, list]:
    """Return the state_groups mapping that yields the requested topology."""
    if kind == "C1":
        # One singleton group per market -> every pair is "cross-group" and is
        # therefore pruned by the raised floor, leaving zero transport links.
        return {f"m{mid}": [mid] for mids in state_groups.values() for mid in mids}
    if kind in ("C2", "C3"):
        return state_groups
    raise ValueError(f"kind must be one of C1/C2/C3, got {kind!r}")


def _config_for(kind: str, config: Config) -> Config:
    """Return a Config whose pruning floor produces the requested topology."""
    if kind in ("C1", "C2"):
        # transit_efficiency(d) = exp(-t/tau) <= 1, so a floor of 2.0 removes
        # every cross-group link while intra-group links are added unfiltered.
        return replace(config, transit_efficiency_floor=2.0)
    return config


def add_load_shedding(
    network: pypsa.Network,
    cold_loads: Dict[str, np.ndarray],
    voll: float = DEFAULT_VOLL,
) -> list[str]:
    """
    Attach an unserved-energy generator to every cold bus.

    Capacity is fixed (not extendable) at each market's own peak demand, so the
    shedding generators add no variables to the capital budget and can always
    cover the full load if nothing is built.

    Returns the list of generator names created.
    """
    cold_buses = [b for b in network.buses.index if str(b).endswith("_cold")]
    if not cold_buses:
        raise RuntimeError("no '*_cold' buses found — cannot add load shedding")

    names, p_noms = [], []
    for bus in cold_buses:
        mid = str(bus)[: -len("_cold")]
        arr = cold_loads.get(mid)
        if arr is None:
            arr = cold_loads.get(int(mid)) if str(mid).isdigit() else None
        if arr is None:
            raise KeyError(f"no cold-load array for market {mid!r}")
        names.append(f"{mid}_shed")
        # 1.01 headroom keeps the LP feasible against float rounding in p_set.
        p_noms.append(float(np.max(arr)) * 1.01)

    network.add(
        "Generator",
        names,
        bus=cold_buses,
        carrier="shed",
        p_nom=p_noms,
        p_nom_extendable=False,
        capital_cost=0.0,
        marginal_cost=voll,
    )
    return names


def make_capex_budget_functionality(budget_usd_per_year: float):
    """
    Build an ``extra_functionality`` callable capping total annualised capex.

    The constraint sums capital_cost * capacity over every extendable
    Generator, Link and Store in the model.  Shedding generators are
    non-extendable and therefore never appear.
    """

    def _apply(n: pypsa.Network, snapshots) -> None:
        model = n.model
        terms = []

        for var_name, df in (
            ("Generator-p_nom", n.generators),
            ("Link-p_nom", n.links),
            ("Store-e_nom", n.stores),
        ):
            try:
                var = model.variables[var_name]
            except KeyError:
                continue  # no extendable components of this type

            comp_names = [str(x) for x in var.coords["name"].values]
            costs = df.loc[comp_names, "capital_cost"].to_numpy(dtype=float)
            # Defensive: shedding generators are non-extendable, so they should
            # not be here at all. Zero them if the assumption ever changes.
            costs = np.where(
                [nm.endswith("_shed") for nm in comp_names], 0.0, costs
            )
            coeff = xr.DataArray(
                costs, coords={"name": comp_names}, dims=("name",)
            )
            terms.append((var * coeff).sum())

        if not terms:
            raise RuntimeError("no extendable capacity found to constrain")

        total_capex = terms[0]
        for extra in terms[1:]:
            total_capex = total_capex + extra

        model.add_constraints(
            total_capex <= budget_usd_per_year, name="capex_budget"
        )

    return _apply


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _weighted_energy(series_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Annual energy (kWh) per column, using the objective snapshot weights."""
    return series_df.multiply(weights, axis=0).sum(axis=0)


def summarise_solution(
    network: pypsa.Network,
    shed_names: Sequence[str],
    locations: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Extract served/unserved cooling energy and market coverage from a solved
    budget-constrained network.
    """
    weights = network.snapshot_weightings["objective"].to_numpy(dtype=float)

    # Demand: fixed Loads on the cold buses.
    demand_kwh = _weighted_energy(network.loads_t.p_set, weights)
    demand_kwh.index = [str(i)[: -len("_cold_load")] for i in demand_kwh.index]

    # Unserved: dispatch of the shedding generators.
    shed_cols = [c for c in network.generators_t.p.columns if c in set(shed_names)]
    shed_kwh = _weighted_energy(network.generators_t.p[shed_cols], weights)
    shed_kwh.index = [str(i)[: -len("_shed")] for i in shed_kwh.index]
    shed_kwh = shed_kwh.reindex(demand_kwh.index).fillna(0.0)

    served_kwh = (demand_kwh - shed_kwh).clip(lower=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        served_frac = np.where(demand_kwh > 0, served_kwh / demand_kwh, 1.0)
    served_frac = pd.Series(served_frac, index=demand_kwh.index)

    served_mask = served_frac >= SERVED_THRESHOLD

    out = {
        "total_demand_kwh": float(demand_kwh.sum()),
        "served_kwh": float(served_kwh.sum()),
        "unserved_kwh": float(shed_kwh.sum()),
        "served_fraction": float(served_kwh.sum() / demand_kwh.sum())
        if demand_kwh.sum() > 0 else np.nan,
        "markets_total": int(len(demand_kwh)),
        "markets_served": int(served_mask.sum()),
        "beef_tonnes_served": float(served_kwh.sum() / KWH_COLD_PER_KG / 1000.0),
    }

    # Equity split — requires the urban_class column from the markets file.
    if locations is not None and "urban_class" in locations.columns:
        cls = (
            locations.assign(market_id=locations["market_id"].astype(str))
            .set_index("market_id")["urban_class"]
            .reindex(demand_kwh.index)
        )
        for label in ("Urban", "Peri-urban", "Rural"):
            sel = cls == label
            n_sel = int(sel.sum())
            out[f"markets_total_{label.lower()}"] = n_sel
            out[f"markets_served_{label.lower()}"] = (
                int((served_mask & sel).sum()) if n_sel else 0
            )

    return out


def _realised_capex(network: pypsa.Network) -> float:
    """Total annualised capital cost actually built in a solved network."""
    total = 0.0
    for df, opt_col in (
        (network.generators, "p_nom_opt"),
        (network.links, "p_nom_opt"),
        (network.stores, "e_nom_opt"),
    ):
        if df.empty or opt_col not in df.columns:
            continue
        total += float((df[opt_col].fillna(0.0) * df["capital_cost"]).sum())
    return total


def make_coverage_functionality(
    min_served_fraction: float, total_demand_kwh: float
):
    """
    Build an ``extra_functionality`` enforcing "serve at least this fraction".

    The budget frontier asks "given capital B, how much can be served?"; this
    asks the transposed question, "to serve fraction s, what is the least total
    cost?" — the natural formulation for a deployment path from 0% to 100%
    coverage, and the only one that puts capex-light/opex-heavy diesel and
    capex-heavy/opex-free solar on a comparable axis. (Under a capital budget
    diesel saturates instantly — its full-service capex is ~1% of solar's — so
    budget points cannot locate the diesel-solar branch.)

    Use with ``voll=0``: shedding must be costless in the objective, otherwise
    a VOLL of $100/kWh forces full service regardless of the floor and the
    solve reduces to the unconstrained problem. With free shedding the solver
    serves exactly the cheapest ``s`` of demand, which is the definition of the
    least-cost coverage path.
    """
    max_unserved = (1.0 - min_served_fraction) * total_demand_kwh

    def _apply(n: pypsa.Network, snapshots) -> None:
        model = n.model
        p = model.variables["Generator-p"]
        shed = [str(x) for x in p.coords["name"].values
                if str(x).endswith("_shed")]
        if not shed:
            raise RuntimeError("coverage floor needs shedding generators")
        snap_dim = [d for d in p.dims if d != "name"][0]
        w = xr.DataArray(
            n.snapshot_weightings["generators"].to_numpy(dtype=float),
            coords={snap_dim: p.coords[snap_dim].values}, dims=(snap_dim,))
        total_shed = (p.sel(name=shed) * w).sum()
        model.add_constraints(total_shed <= max_unserved,
                              name="coverage_floor")

    return _apply


def _export_capacities(network: pypsa.Network, out_dir: str) -> str:
    """
    Write per-market built capacities for one solved point.

    Needed to test whether the deployment path is a feasible BUILD SEQUENCE:
    a sequence of static optima is only a construction schedule if the build
    at lower coverage is (near-)nested inside the build at higher coverage.
    Aggregate summaries cannot answer that; per-market capacities can.
    One small CSV per solve, named after the network (config_supply_mode).
    """
    os.makedirs(out_dir, exist_ok=True)
    gen = network.generators
    rows = {}

    def _acc(names, values, col, suffix):
        # Strip the FULL suffix: rsplit("_", 1) mangled multi-part suffixes
        # ("112234_batt_charge" -> mid "112234_batt"), which silently broke
        # the committed-path lock-in for battery power links.
        for n, v in zip(names, values):
            rows.setdefault(str(n)[: -len(suffix)], {})[col] = float(v)

    for suffix, col in (("_solar", "solar_kw"), ("_diesel", "diesel_kw")):
        m = gen.index.str.endswith(suffix)
        if m.any():
            _acc(gen.index[m], gen.loc[m, "p_nom_opt"], col, suffix)
    st = network.stores
    if len(st):
        m = st.index.str.endswith("_battery")
        _acc(st.index[m], st.loc[m, "e_nom_opt"], "battery_kwh", "_battery")
    lk = network.links
    for suffix, col in (("_fridge", "fridge_kw_cold"),
                        ("_batt_charge", "batt_charge_kw"),
                        ("_batt_discharge", "batt_discharge_kw")):
        m = lk.index.str.endswith(suffix)
        if m.any():
            _acc(lk.index[m], lk.loc[m, "p_nom_opt"], col, suffix)

    df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("market_id")
    path = os.path.join(out_dir, f"capacities_{network.name}.csv")
    df.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# Frontier drivers
# ---------------------------------------------------------------------------

def run_coverage_frontier(
    state_groups, solar_cfs, cold_loads, locations, snapshots, weights,
    config, output_dir, configs=("C1",), coverage_levels=(
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    supply: str = "solar",
    export_capacities: bool = True,
) -> "pd.DataFrame":
    """
    Least total annual cost to serve each coverage level.

    One row per (config, coverage) plus the unconstrained anchor. The anchor
    doubles as a cross-check: its objective must match the coverage=1.0 point.
    Written incrementally, like the budget frontier.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows: list[dict] = []
    path = os.path.join(output_dir, "coverage_frontier.csv")

    for kind in configs:
        print(f"\n[coverage] anchor: unconstrained {kind} ({supply}) …")
        rows.append(solve_budget_point(
            kind, state_groups, solar_cfs, cold_loads, locations,
            snapshots, weights, config, budget=None, supply=supply))
        pd.DataFrame(rows).to_csv(path, index=False)
        cap_dir = (os.path.join(output_dir, "capacities")
                   if export_capacities else None)
        for s_level in coverage_levels:
            print(f"[coverage] {kind} ({supply}) @ {s_level:.0%} served …")
            rows.append(solve_budget_point(
                kind, state_groups, solar_cfs, cold_loads, locations,
                snapshots, weights, config, budget=None, supply=supply,
                coverage=s_level, capacity_export_dir=cap_dir))
            pd.DataFrame(rows).to_csv(path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\n[coverage] wrote {path}  ({len(df)} points)")
    return df



def solve_budget_point(
    kind: str,
    state_groups: Dict[str, list],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
    budget: Optional[float],
    voll: float = DEFAULT_VOLL,
    supply: str = "solar",
    coverage: Optional[float] = None,
    capacity_export_dir: Optional[str] = None,
    network_mutator=None,
    solution_exporter=None,
) -> dict:
    """
    Build and solve one (configuration, budget) point.

    ``budget=None`` solves the unconstrained problem, which gives the capex
    required for full service and anchors the top of the frontier.

    ``supply="diesel"`` swaps the generation technology: every market gets a
    dispatchable genset and PV is pinned to zero. Same demand, same links, same
    budget machinery — so the two frontiers are directly comparable and the
    diesel-vs-solar branch of the deployment path can be located.
    """
    if supply not in ("solar", "diesel"):
        raise ValueError(f"unknown supply {supply!r}")
    if coverage is not None and budget is not None:
        raise ValueError("budget and coverage are mutually exclusive")
    cfg = _config_for(kind, config)
    groups = _groups_for(kind, state_groups)

    t0 = time.time()
    network = build_national_network(
        groups, solar_cfs, cold_loads, locations, snapshots, weights, cfg
    )
    if coverage is not None:
        network.name = f"{kind}_{supply}_coverage_{coverage:.3f}"
    else:
        network.name = f"{kind}_{supply}_budget_" \
            f"{'unconstrained' if budget is None else f'{budget:.0f}'}"

    if supply == "diesel":
        # Deferred import: diesel_base_case imports this module at top level.
        from diesel_base_case import add_diesel_gensets, disable_solar
        add_diesel_gensets(network, cfg)
        disable_solar(network)

    if network_mutator is not None:
        # Committed-path hook: lets a caller impose lower bounds carried over
        # from a previous solve (incremental build-out), or any other
        # pre-solve adjustment. Applied before shedding is attached.
        network_mutator(network)

    # Coverage mode needs costless shedding — see make_coverage_functionality.
    shed_names = add_load_shedding(
        network, cold_loads, voll=0.0 if coverage is not None else voll)

    if coverage is not None:
        total_demand = float(sum(
            np.asarray(v, dtype=float) @ np.asarray(weights, dtype=float)
            for v in cold_loads.values()))
        extra = make_coverage_functionality(coverage, total_demand)
    else:
        extra = (make_capex_budget_functionality(budget)
                 if budget is not None else None)
    solver_opts = cfg.c3_barrier_options if kind == "C3" else None
    _solve(network, cfg, extra_solver_options=solver_opts,
           extra_functionality=extra)

    row = {
        "config": kind,
        "supply": supply,
        "coverage_target": np.nan if coverage is None else float(coverage),
        "budget_usd_per_year": np.nan if budget is None else float(budget),
        "objective_usd": float(network.objective),
        "capex_built_usd_per_year": _realised_capex(network),
        "n_links": int(len(network.links)),
        "solve_seconds": round(time.time() - t0, 1),
    }
    if capacity_export_dir:
        _export_capacities(network, capacity_export_dir)
    row.update(summarise_solution(network, shed_names, locations))
    # Cost of the *served* food, the figure that goes on Fig. 2a's second axis.
    row["usd_per_tonne_served"] = (
        row["capex_built_usd_per_year"] / row["beef_tonnes_served"]
        if row["beef_tonnes_served"] > 0 else np.nan
    )
    if solution_exporter is not None:
        solution_exporter(network, row)
    return row


def run_budget_frontier(
    state_groups: Dict[str, list],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
    output_dir: str,
    configs: Sequence[str] = ("C1", "C2"),
    budget_fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
    voll: float = DEFAULT_VOLL,
    supply: str = "solar",
    reference_capex_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Trace the access frontier for each configuration.

    Budgets are expressed as fractions of the *unconstrained C1 capex*, so the
    x-axis is directly interpretable: "at 40% of what standalone provision for
    everyone would cost, how much demand does each configuration serve?"

    C3 is opt-in via ``configs`` because the full national mesh LP is far more
    expensive to solve than C1/C2 and must be re-solved at every budget point.

    ``reference_capex_override`` pins the budget grid to an externally supplied
    reference instead of this run's own C1 anchor. Essential when comparing
    supply technologies: a diesel C1 saturates at a fraction of solar C1's
    capital, so only a *shared* reference (the solar C1 full-service capex)
    puts both frontiers on the same x-axis.

    Returns the frontier table and writes ``budget_frontier.csv`` to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows: list[dict] = []
    path = os.path.join(output_dir, "budget_frontier.csv")

    def flush(reference_capex: float | None = None) -> pd.DataFrame:
        """
        Persist everything solved so far.

        Called after every solve, anchors included: a C3 anchor costs ~15 min
        and a constrained point ~1 h, so nothing solved should ever be held in
        memory across the next solve.
        """
        df = pd.DataFrame(rows)
        df["budget_fraction_of_c1_full"] = (
            df["budget_usd_per_year"] / reference_capex
            if reference_capex else pd.NA
        )
        df.to_csv(path, index=False)
        return df

    print(f"\n[frontier] anchor: unconstrained C1 solve … (supply={supply})")
    anchor = solve_budget_point(
        "C1", state_groups, solar_cfs, cold_loads, locations,
        snapshots, weights, config, budget=None, voll=voll, supply=supply,
    )
    rows.append(anchor)
    reference_capex = (reference_capex_override
                       if reference_capex_override is not None
                       else anchor["capex_built_usd_per_year"])
    print(f"[frontier] full-service C1 capex = "
          f"${anchor['capex_built_usd_per_year']:,.0f}/yr"
          + (f"  (budget grid pinned to external reference "
             f"${reference_capex:,.0f}/yr)"
             if reference_capex_override is not None else ""))
    flush(reference_capex)

    for kind in configs:
        if kind != "C1":
            print(f"\n[frontier] anchor: unconstrained {kind} solve …")
            rows.append(solve_budget_point(
                kind, state_groups, solar_cfs, cold_loads, locations,
                snapshots, weights, config, budget=None, voll=voll,
                supply=supply,
            ))
            flush(reference_capex)
        for frac in budget_fractions:
            budget = reference_capex * frac
            print(f"[frontier] {kind} @ {frac:.0%} of full-service capex "
                  f"(${budget:,.0f}/yr) …")
            rows.append(solve_budget_point(
                kind, state_groups, solar_cfs, cold_loads, locations,
                snapshots, weights, config, budget=budget, voll=voll,
                supply=supply,
            ))
            flush(reference_capex)

    df = flush(reference_capex)
    print(f"\n[frontier] wrote {path}  ({len(df)} points)")
    return df


def check_voll_dominance(config: Config, voll: float = DEFAULT_VOLL) -> None:
    """
    Warn if VOLL is not comfortably above the cheapest way to deliver cooling.

    If VOLL were too low the optimiser would prefer shedding to building even
    with budget left over, and the frontier would understate achievable service.
    """
    # Cheapest conceivable delivered cooling: refrigeration capacity running
    # flat out all year, ignoring PV and storage (a strict lower bound).
    floor = config.fridge_annualised_capex_per_kw_electric / (
        8760.0 * config.fridge_cop
    )
    if voll < 10.0 * floor:
        print(f"  [warn] VOLL={voll} is within 10x of the delivered-cooling "
              f"cost floor ({floor:.4f} USD/kWh_cold) — raise it.")
