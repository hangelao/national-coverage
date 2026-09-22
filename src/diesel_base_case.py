"""
diesel_base_case.py
-------------------
The counterfactual the manuscript's Discussion needs.

Off-grid cold storage in Nigerian markets that exists today runs on small
diesel gensets. Without that base case, "coordinated solar cold chains are
cheaper" has no referent — cheaper than what? This module supplies it by
offering each market a dispatchable diesel genset alongside (or instead of)
solar, and reports cost and combustion CO2 for three supply scenarios:

  solar   — PV + battery only (the manuscript's baseline)
  diesel  — genset only (the incumbent technology)
  hybrid  — both available, least cost decides

crossed with the three coordination configurations (C1/C2/C3).

The economics are deliberately opposed: a genset is cheap to buy and expensive
to run (59.6 USD/kW/yr annualised, 0.307 USD/kWh_e fuel), PV is the reverse
(54.3 USD/kW/yr, zero fuel). That asymmetry is the substantive point — under a
*capital* constraint diesel looks attractive, and only lifetime cost exposes
the difference. It also connects directly to the budget frontier: capital
scarcity is exactly what pushes planners toward the technology with the worse
lifetime cost.
"""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from budget_frontier import KWH_COLD_PER_KG, _config_for, _groups_for
from config import Config
from network_builder import build_national_network
from optimizer import _solve

SCENARIOS: tuple[str, ...] = ("solar", "diesel", "hybrid")


def add_diesel_gensets(network: pypsa.Network, config: Config) -> list[str]:
    """
    Attach a dispatchable diesel genset to every market's electricity bus.

    Extendable capacity, no availability profile: unlike PV a genset runs on
    demand, which is precisely why it remains the incumbent despite its fuel
    bill. Returns the generator names for the emissions accounting.
    """
    elec_buses = [b for b in network.buses.index if str(b).endswith("_elec")]
    names = [f"{b[:-len('_elec')]}_diesel" for b in elec_buses]
    if not names:
        return []
    network.add(
        "Generator", names,
        bus=elec_buses,
        p_nom_extendable=True,
        p_nom_min=0.0,
        marginal_cost=config.diesel_marginal_cost_per_kwh,
        capital_cost=config.diesel_genset_annualised_capex_per_kw,
    )
    return names


def disable_solar(network: pypsa.Network) -> None:
    """
    Remove PV as an option without deleting it.

    Capacity is pinned to zero rather than dropped so that the LP keeps an
    identical variable structure across scenarios — the comparison then differs
    only in what is permitted, not in how the problem is posed.
    """
    solar = [g for g in network.generators.index if str(g).endswith("_solar")]
    network.generators.loc[solar, "p_nom_extendable"] = False
    network.generators.loc[solar, "p_nom"] = 0.0


def make_min_genset_functionality(min_kw: float, caps: Dict[str, float]):
    """
    Build an ``extra_functionality`` enforcing "build nothing, or at least
    ``min_kw``" for every diesel genset.

    Without it the LP builds two-watt gensets at 93% of markets purely to shave
    battery capacity — economically coherent but physically impossible, since
    the smallest unit on the market is around 5 kVA. Encoding the real choice
    needs a binary per market:

        p_nom <= cap * build
        p_nom >= min_kw * build          build in {0, 1}

    ``caps`` supplies a per-market big-M (that market's peak electrical demand).
    A tight, market-specific cap matters: a single loose global big-M would
    weaken the LP relaxation and make the MILP far slower to prove optimal.
    """
    import linopy  # noqa: F401  (imported for its side effect on model typing)

    def _add(network: pypsa.Network, snapshots) -> None:
        model = network.model
        try:
            p_nom = model.variables["Generator-p_nom"]
        except KeyError:
            return  # no extendable generators
        # linopy indexes capacity variables by a plain "name" coordinate.
        available = [str(n) for n in p_nom.coords["name"].values]
        names = [n for n in available if n.endswith("_diesel")]
        if not names:
            return
        index = pd.Index(names, name="name")
        build = model.add_variables(
            binary=True, coords=[index], name="genset-build")
        cap = xr.DataArray(
            [max(caps.get(n, 0.0), min_kw) for n in names],
            coords={"name": names}, dims=("name",))
        selected = p_nom.sel(name=names)
        model.add_constraints(selected - cap * build <= 0, name="genset-max")
        model.add_constraints(selected - min_kw * build >= 0, name="genset-min")

    return _add


def genset_capacity_caps(
    cold_loads: Dict[str, np.ndarray], config: Config
) -> Dict[str, float]:
    """
    Per-market big-M: the largest genset that could ever be useful.

    A genset only ever serves refrigeration, so peak electrical demand is peak
    cold load divided by the COP. Doubling it leaves headroom without loosening
    the relaxation more than necessary.
    """
    return {
        f"{mid}_diesel": 2.0 * float(np.max(arr)) / config.fridge_cop
        for mid, arr in cold_loads.items()
    }


def emissions_breakdown(
    network: pypsa.Network, config: Config, diesel_names: Sequence[str],
) -> dict:
    """
    Full annualised CO2e for one solution, not just genset combustion.

    Comparing "emissions to meet the same food-chain demand" requires every
    source, because the configurations trade them against each other:

      • genset fuel     — combustion, exact from dispatch
      • transport fuel  — combustion, exact; derived from transport opex, which
                          is priced directly off diesel litres. This is the one
                          coordination ADDS, so omitting it flatters C2/C3.
      • PV / battery /  — embodied, annualised over each technology's lifetime.
        fridge / genset   Factors are PLACEHOLDERS (see Config).

    Solar is not zero-emission once manufacturing is counted, and coordination
    is not free once the vans are counted; both matter for an honest comparison.
    """
    weights = network.snapshot_weightings["generators"].to_numpy(dtype=float)

    # --- combustion: gensets ---
    present = [n for n in diesel_names if n in network.generators_t.p.columns]
    genset_kwh = 0.0
    if present:
        dispatch = network.generators_t.p[present].to_numpy(dtype=float)
        genset_kwh = float((dispatch * weights[:, None]).sum())
    genset_litres = genset_kwh / config.diesel_genset_kwh_per_litre

    # --- combustion: road transport ---
    # Transport links carry marginal_cost = conv_marginal_cost(distance), which
    # is (diesel_price / km_per_litre) * d / cold_per_trip. Multiplying flow by
    # that cost and dividing by the fuel price recovers litres exactly, with no
    # need to re-derive distances here.
    transport_usd = 0.0
    conv = [c for c in network.links.index if str(c).endswith("_conv")]
    if conv and not network.links_t.p0.empty:
        cols = [c for c in conv if c in network.links_t.p0.columns]
        if cols:
            flow = network.links_t.p0[cols].to_numpy(dtype=float)
            mc = network.links.loc[cols, "marginal_cost"].to_numpy(dtype=float)
            transport_usd = float((flow * weights[:, None] * mc[None, :]).sum())
    transport_litres = transport_usd / config.diesel_price_usd_per_litre

    combustion_kg = (
        (genset_litres + transport_litres) * config.diesel_co2_kg_per_litre)

    # --- embodied, annualised over each technology's own lifetime ---
    gen = network.generators
    solar_kw = float(gen.loc[gen.index.str.endswith("_solar"), "p_nom_opt"].sum())
    genset_kw = float(gen.loc[gen.index.str.endswith("_diesel"), "p_nom_opt"].sum()) \
        if any(gen.index.str.endswith("_diesel")) else 0.0
    battery_kwh = float(network.stores["e_nom_opt"].sum())
    fridge_kw = float(network.links.loc[
        network.links.index.str.endswith("_fridge"), "p_nom_opt"].sum())

    embodied_kg = (
        solar_kw * config.pv_embodied_kgco2_per_kw / config.pv_lifetime_years
        + battery_kwh * config.battery_embodied_kgco2_per_kwh
        / config.battery_lifetime_years
        + fridge_kw * config.fridge_embodied_kgco2_per_kw
        / config.fridge_lifetime_years
        + genset_kw * config.genset_embodied_kgco2_per_kw
        / config.diesel_genset_lifetime_years
    )

    return {
        "genset_kwh_per_year": genset_kwh,
        "genset_litres_per_year": genset_litres,
        "transport_litres_per_year": transport_litres,
        "transport_opex_usd_per_year": transport_usd,
        "combustion_tco2_per_year": combustion_kg / 1000.0,
        "embodied_tco2_per_year": embodied_kg / 1000.0,
        "total_tco2_per_year": (combustion_kg + embodied_kg) / 1000.0,
        "solar_kw_built": solar_kw,
        "genset_kw_built": genset_kw,
        "battery_kwh_built": battery_kwh,
        "fridge_kw_built": fridge_kw,
        "markets_with_genset": int((gen.loc[
            gen.index.str.endswith("_diesel"), "p_nom_opt"] > 1e-6).sum())
        if any(gen.index.str.endswith("_diesel")) else 0,
    }


def _weighted_dispatch(network: pypsa.Network, names: Sequence[str]) -> float:
    """Annual energy from the named generators [kWh], using objective weights."""
    present = [n for n in names if n in network.generators_t.p.columns]
    if not present:
        return 0.0
    weights = network.snapshot_weightings["generators"].to_numpy(dtype=float)
    dispatch = network.generators_t.p[present].to_numpy(dtype=float)
    return float((dispatch * weights[:, None]).sum())


def solve_scenario(
    kind: str,
    scenario: str,
    state_groups: Dict[str, list],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
    min_genset_kw: float = 0.0,
) -> dict:
    """
    Solve one (configuration, supply scenario) pair and summarise it.

    ``min_genset_kw`` > 0 enforces realistic genset sizing, which turns the
    problem into a MILP — see ``make_min_genset_functionality``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected {SCENARIOS}")

    cfg = _config_for(kind, config)
    groups = _groups_for(kind, state_groups)

    t0 = time.time()
    network = build_national_network(
        groups, solar_cfs, cold_loads, locations, snapshots, weights, cfg)
    network.name = f"{kind}_{scenario}"

    diesel_names: list[str] = []
    extra = None
    if scenario in ("diesel", "hybrid"):
        diesel_names = add_diesel_gensets(network, cfg)
        if min_genset_kw > 0:
            extra = make_min_genset_functionality(
                min_genset_kw, genset_capacity_caps(cold_loads, cfg))
    if scenario == "diesel":
        disable_solar(network)

    solver_opts = dict(cfg.c3_barrier_options) if kind == "C3" else {}
    if extra is not None:
        # Planning-grade MILP tolerance: proving the last 0.1% of optimality on
        # ~2,000 binaries costs far more than the answer is worth here.
        solver_opts["MIPGap"] = 1e-3
    _solve(network, cfg, extra_solver_options=solver_opts or None,
           extra_functionality=extra)

    diesel_kwh = _weighted_dispatch(network, diesel_names)
    solar_names = [g for g in network.generators.index if str(g).endswith("_solar")]
    solar_kwh = _weighted_dispatch(network, solar_names)

    total_cold = float(
        sum(np.asarray(v, dtype=float) @ np.asarray(weights, dtype=float)
            for v in cold_loads.values()))
    beef_kg = total_cold / KWH_COLD_PER_KG
    emissions = emissions_breakdown(network, cfg, diesel_names)

    row = {
        "config": kind,
        "scenario": scenario,
        "min_genset_kw": min_genset_kw,
        "total_cost_usd_per_year": float(network.objective),
        "solar_kwh_per_year": solar_kwh,
        "diesel_kwh_per_year": diesel_kwh,
        "diesel_share_of_generation": (
            diesel_kwh / (diesel_kwh + solar_kwh)
            if (diesel_kwh + solar_kwh) > 0 else 0.0),
        "beef_tonnes_served": beef_kg / 1000.0,
        "n_links": int(len(network.links)),
        "solve_seconds": round(time.time() - t0, 1),
    }
    row.update(emissions)
    for key, label in (("combustion_tco2_per_year", "combustion"),
                       ("embodied_tco2_per_year", "embodied"),
                       ("total_tco2_per_year", "total")):
        row[f"{label}_kgco2_per_kg_beef"] = (
            row[key] * 1000.0 / beef_kg if beef_kg > 0 else 0.0)
    return row
