"""
optimizer.py
------------
Solves the three configurations of the solar cold-chain pipeline.

Public API
----------
    run_standalone(state_name, market_ids, solar_cfs, cold_loads,
                   snapshots, weights, config) -> dict

    run_intrastate(state_name, market_ids, solar_cfs, cold_loads,
                   locations, snapshots, weights, config) -> dict

    run_interstate(intrastate_results, standalone_results, locations,
                   solar_cfs, cold_loads,
                   snapshots, weights, config) -> dict

Solver
------
Gurobi only.  Set Config.solver = "gurobi" and Config.solver_threads
as required.  Any other solver name or a missing Gurobi licence raises
RuntimeError before optimisation is attempted.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from data_loader import normalise_id
from config import Config
from network_builder import (
    build_standalone_network,
    build_intrastate_network,
    build_interstate_network,
    build_national_network,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_id_lookup(d: dict) -> dict:
    """Map both str and int forms of each market_id to the same value."""
    lookup: dict = {}
    for k, v in d.items():
        lookup[str(k)] = v
        try:
            lookup[int(k)] = v
        except (ValueError, TypeError):
            pass
    return lookup


def _lookup_id(lookup: dict, mid):
    for key in (str(mid), mid):
        if key in lookup:
            return lookup[key]
    return None


def _add_storage_power_capacity_constraints(
    network: pypsa.Network, snapshots
) -> None:
    """Tie external AC-side battery charge/discharge inverter capacities."""
    p_nom = network.model["Link-p_nom"]
    link_names = set(network.links.index.astype(str))

    for technology, charge_suffix, discharge_suffix in (
        ("battery", "_batt_charge", "_batt_discharge"),
    ):
        charge_names = sorted(name for name in link_names if name.endswith(charge_suffix))
        discharge_names = [
            name[: -len(charge_suffix)] + discharge_suffix for name in charge_names
        ]
        missing = [name for name in discharge_names if name not in link_names]
        if missing:
            raise RuntimeError(
                f"Missing {technology} discharge Link(s) for shared-power "
                f"constraints: {missing[:5]}"
            )
        if not charge_names:
            continue

        pair_ids = [name[: -len(charge_suffix)] for name in charge_names]
        charge = p_nom.sel(name=charge_names).rename(name="storage_pair")
        discharge = p_nom.sel(name=discharge_names).rename(name="storage_pair")
        charge = charge.assign_coords(storage_pair=pair_ids)
        discharge = discharge.assign_coords(storage_pair=pair_ids)
        discharge_efficiency = xr.DataArray(
            network.links.loc[discharge_names, "efficiency"].to_numpy(dtype=float),
            coords={"storage_pair": pair_ids},
            dims=("storage_pair",),
        )
        network.model.add_constraints(
            charge == discharge_efficiency * discharge,
            name=f"Link-{technology}_shared_power_capacity",
        )


def _solve(network: pypsa.Network, config: Config,
           extra_solver_options: dict = None,
           extra_functionality=None) -> pypsa.Network:
    """
    Solve the network with Gurobi and return the solved network.

    extra_solver_options : optional dict of Gurobi parameters merged on top of
        the default {"Threads": …} — used by the Config 3 national solve to pass
        barrier tuning (Method/Crossover/BarConvTol).

    Raises
    ------
    RuntimeError
        • if config.solver is not 'gurobi'
        • if Gurobi is unavailable or the licence is missing
        • if the solve terminates with a non-optimal status
    """
    if config.solver.lower() != "gurobi":
        raise RuntimeError(
            f"Solver must be 'gurobi', got '{config.solver}'. "
            "Update Config.solver = 'gurobi'."
        )

    n_buses     = len(network.buses)
    n_snapshots = len(network.snapshots)
    net_name    = getattr(network, "name", "?")
    print(f"    [solve] '{net_name}': {n_buses} buses, "
          f"{len(network.links)} links, {n_snapshots} snapshots")

    solver_options = {"Threads": config.solver_threads}
    if extra_solver_options:
        solver_options.update(extra_solver_options)

    try:
        def combined_extra_functionality(n, snapshots):
            if extra_functionality is not None:
                extra_functionality(n, snapshots)
            _add_storage_power_capacity_constraints(n, snapshots)

        result = network.optimize(
            solver_name=config.solver,
            solver_options=solver_options,
            extra_functionality=combined_extra_functionality,
        )
    except Exception as exc:
        # Wrap linopy / Gurobi installation errors with a clear message
        raise RuntimeError(
            "Gurobi solve failed. Ensure Gurobi is installed and a valid "
            f"licence is active.\nOriginal error: {exc}"
        ) from exc

    # PyPSA 0.25+ returns (status, termination_condition)
    if isinstance(result, tuple):
        status, termination = result
    else:
        status, termination = str(result), "unknown"

    obj = float(network.objective) if hasattr(network, "objective") else float("nan")
    print(f"    [solve] '{net_name}': status={status!r}, "
          f"termination={termination!r}, objective={obj:,.2f}")

    if status != "ok" or termination != "optimal":
        raise RuntimeError(
            f"Optimisation did not converge: status='{status}', "
            f"termination='{termination}'."
        )

    return network


def _extract_capacities(
    network: pypsa.Network,
    market_ids: List[str],
) -> pd.DataFrame:
    """
    Pull optimised capacities for every market from a solved network.

    Returns
    -------
    pd.DataFrame, indexed by market_id, with columns:
        solar_kw    — PV capacity (kWp)
        fridge_kw   — Fridge electrical input capacity (kW)
        battery_kwh — Battery energy capacity (kWh)
        pcm_kwh     — PCM thermal storage capacity (kWh)
        battery_power_kw — Shared external AC-side inverter rating, reported
                           from the costed charge Link p_nom_opt (kW)
        pcm_power_kw     — NaN compatibility field; PCM has no power capacity
    Missing components default to 0.0.
    """
    rows: list[dict] = []
    for mid in market_ids:
        solar_kw    = _safe_get(network.generators, f"{mid}_solar",   "p_nom_opt")
        fridge_kw   = _safe_get(network.links,      f"{mid}_fridge",  "p_nom_opt")
        battery_kwh = _safe_get(network.stores,     f"{mid}_battery", "e_nom_opt")
        pcm_kwh     = _safe_get(network.stores,     f"{mid}_pcm",     "e_nom_opt")
        battery_power_kw = _safe_get(
            network.links, f"{mid}_batt_charge", "p_nom_opt"
        )
        pcm_power_kw = float("nan")
        rows.append({
            "market_id"  : mid,
            "solar_kw"   : solar_kw,
            "fridge_kw"  : fridge_kw,
            "battery_kwh": battery_kwh,
            "pcm_kwh"    : pcm_kwh,
            "battery_power_kw": battery_power_kw,
            "pcm_power_kw"    : pcm_power_kw,
        })
    return pd.DataFrame(rows).set_index("market_id")


def _safe_get(component_df: pd.DataFrame, name: str, attr: str) -> float:
    """Return component attribute as float, or 0.0 if not present."""
    if name in component_df.index:
        return float(component_df.loc[name, attr])
    return 0.0


def _extract_transport_summary(network: pypsa.Network) -> pd.DataFrame:
    """
    Summarise all built transport links in a solved network.

    A link is considered a transport link if its name contains "to" and
    its suffix (after the last "_") is "conv".
    Only links with p_nom_opt > 1e-6 are included.

    Link name format: "{src}to{dst}_{mode}"

    Returns
    -------
    pd.DataFrame with columns:
        from_market, to_market, mode, capacity_kw, capex, total_flow_kwh,
        marginal_cost_per_kwh, fuel_opex_usd
    Empty DataFrame (same columns) if no transport links were built.

    Notes
    -----
    capex is always 0 — transport is modelled as a hired service with
    capital_cost=0 and no owned-asset capex.  The actual transport cost is
    fuel_opex_usd = marginal_cost_per_kwh × total_flow_kwh.
    """
    EMPTY_COLS = [
        "from_market", "to_market", "mode",
        "capacity_kw", "capex", "total_flow_kwh",
        "marginal_cost_per_kwh", "fuel_opex_usd",
    ]
    TRANSPORT_MODES = {"conv"}

    transport_links = [
        name for name in network.links.index
        if "to" in name and name.rsplit("_", 1)[-1] in TRANSPORT_MODES
    ]

    if not transport_links:
        return pd.DataFrame(columns=EMPTY_COLS)

    weights    = network.snapshot_weightings["objective"].values
    p0_df      = network.links_t.p0    # (n_snapshots × n_links) time-series flows

    rows: list[dict] = []
    for link_name in transport_links:
        p_nom_opt = float(network.links.loc[link_name, "p_nom_opt"])
        if p_nom_opt <= 1e-6:
            continue

        # Parse "{src}to{dst}_{mode}"
        mode   = link_name.rsplit("_", 1)[1]
        src_dst = link_name.rsplit("_", 1)[0]
        to_idx  = src_dst.find("to")
        src     = src_dst[:to_idx]
        dst     = src_dst[to_idx + 2:]

        # Weighted absolute flow [kWh]
        if link_name in p0_df.columns:
            total_flow = float((p0_df[link_name].abs() * weights).sum())
        else:
            total_flow = 0.0

        mc = float(network.links.loc[link_name, "marginal_cost"])
        rows.append({
            "from_market"        : src,
            "to_market"          : dst,
            "mode"               : mode,
            "capacity_kw"        : p_nom_opt,
            "capex"              : float(network.links.loc[link_name, "capital_cost"]) * p_nom_opt,
            "total_flow_kwh"     : total_flow,
            "marginal_cost_per_kwh": mc,
            "fuel_opex_usd"      : mc * total_flow,
        })

    if not rows:
        return pd.DataFrame(columns=EMPTY_COLS)

    return pd.DataFrame(rows)


def _identify_hubs(
    capacities_df: pd.DataFrame,
    config: Config,
) -> List[str]:
    """
    Return the single market with the highest refrigeration capacity per
    state, selected as the interstate hub candidate for Config 3.

    If every market in the state has fridge_kw == 0 (no refrigeration was
    built at all), an empty list is returned so that the state is safely
    skipped in run_interstate().
    """
    if capacities_df["fridge_kw"].max() == 0:
        return []
    top_hub = capacities_df["fridge_kw"].idxmax()
    return [top_hub]


# ---------------------------------------------------------------------------
# Dispatch extraction helper
# ---------------------------------------------------------------------------

def _extract_dispatch(
    network,
    capacities_df: pd.DataFrame,
    hub_ids: List[str],
    state_name: str,
) -> Optional[pd.DataFrame]:
    """
    Extract hourly dispatch time series for one representative hub market
    and one representative spoke market after an intra-state solve.

    Hub representative  : market with the highest optimised fridge_kw.
    Spoke representative: market with fridge_kw closest to 0 (if any).

    Returns a tidy DataFrame with one row per snapshot per market,
    or None if extraction fails.
    """
    try:
        # ── Select representative markets ──────────────────────────────────
        rep_hub_mid: Optional[str] = None
        rep_spoke_mid: Optional[str] = None

        if not capacities_df.empty and "fridge_kw" in capacities_df.columns:
            sorted_cap = capacities_df["fridge_kw"].sort_values(ascending=False)
            if not sorted_cap.empty:
                rep_hub_mid = str(sorted_cap.index[0])
            spoke_candidates = sorted_cap[~sorted_cap.index.isin(hub_ids)]
            if not spoke_candidates.empty:
                rep_spoke_mid = str(
                    spoke_candidates.index[
                        spoke_candidates.values.argsort()[0]
                    ]
                )

        candidates = [m for m in [rep_hub_mid, rep_spoke_mid] if m is not None]
        if not candidates:
            return None

        snaps = network.snapshots
        records: List[pd.DataFrame] = []

        for mid in candidates:
            def _get_series(df_t, col, default=0.0):
                if col in df_t.columns:
                    return df_t[col].values.astype(float)
                return np.full(len(snaps), default)

            solar_kw        = _get_series(network.generators_t.p,    f"{mid}_solar")
            cold_demand_kw  = _get_series(network.loads_t.p_set,     f"{mid}_cold_load")
            fridge_kw       = _get_series(network.links_t.p0,        f"{mid}_fridge")
            pcm_soc_kwh     = _get_series(network.stores_t.e,        f"{mid}_pcm")
            pcm_dispatch_kw = _get_series(network.stores_t.p,        f"{mid}_pcm")
            pcm_charge_kw   = np.maximum(-pcm_dispatch_kw, 0.0)
            pcm_discharge_kw= np.maximum(pcm_dispatch_kw, 0.0)

            # Export: links where this market is the source (bus0)
            export_cols  = [
                c for c in network.links_t.p0.columns
                if c.startswith(f"{mid}to") and c.endswith("_conv")
            ]
            export_kw = (
                network.links_t.p0[export_cols].sum(axis=1).values.astype(float)
                if export_cols else np.zeros(len(snaps))
            )

            # Import: links where this market is the destination (bus1)
            import_cols = [
                c for c in network.links_t.p1.columns
                if c.endswith(f"to{mid}_conv")
            ]
            import_kw = (
                network.links_t.p1[import_cols].abs().sum(axis=1).values.astype(float)
                if import_cols else np.zeros(len(snaps))
            )

            records.append(pd.DataFrame({
                "timestamp"       : snaps,
                "state"           : state_name,
                "market_id"       : mid,
                "role"            : "hub" if mid == rep_hub_mid else "spoke",
                "solar_kw"        : solar_kw,
                "cold_demand_kw"  : cold_demand_kw,
                "fridge_kw"       : fridge_kw,
                "pcm_soc_kwh"     : pcm_soc_kwh,
                "pcm_charge_kw"   : pcm_charge_kw,
                "pcm_discharge_kw": pcm_discharge_kw,
                "export_kw"       : export_kw,
                "import_kw"       : import_kw,
            }))

        return pd.concat(records, ignore_index=True) if records else None

    except Exception as _dispatch_exc:
        import traceback as _tb
        print(f"  [DISPATCH ERROR] {state_name}: {_dispatch_exc}")
        _tb.print_exc()
        return None


# ---------------------------------------------------------------------------
# Public run functions
# ---------------------------------------------------------------------------

def run_standalone(
    state_name: str,
    market_ids: List[str],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    snapshots: pd.DatetimeIndex,
    weights: Optional[np.ndarray],
    config: Config,
) -> dict:
    """
    Config 1 — Standalone optimisation: each market solved independently.

    Every market receives solar PV, battery storage, and refrigeration.
    Passive PCM is added only when ``config.include_pcm`` is true.

    Parameters
    ----------
    state_name  : label for logging and results
    market_ids  : list of str market IDs to optimise
    solar_cfs   : market_id → hourly solar capacity-factor array
    cold_loads  : market_id → hourly cold demand array (kW)
    snapshots   : DatetimeIndex of length n_snapshots
    weights     : snapshot weights; uniform ones if None
    config      : Config instance (solver must be 'gurobi')

    Returns
    -------
    dict with keys:
        state        : str
        config       : "standalone"
        total_cost   : float  — sum of individual market objectives
        market_costs : pd.Series  — market_id → objective value
        capacities   : pd.DataFrame  — from _extract_capacities
    """
    if weights is None:
        weights = np.ones(len(snapshots))

    market_costs: dict[str, float] = {}
    cap_parts: list[pd.DataFrame] = []

    for mid in market_ids:
        network = build_standalone_network(
            market_id = mid,
            solar_cf  = solar_cfs[mid],
            cold_load = cold_loads[mid],
            snapshots = snapshots,
            weights   = weights,
            config    = config,
        )
        _solve(network, config)

        market_costs[mid] = float(network.objective)
        cap_parts.append(_extract_capacities(network, [mid]))

    capacities  = pd.concat(cap_parts)
    total_cost  = sum(market_costs.values())
    market_series = pd.Series(market_costs, name="objective")

    return {
        "state"       : state_name,
        "config"      : "standalone",
        "total_cost"  : total_cost,
        "market_ids"  : market_ids,
        "market_costs": market_series,
        "capacities"  : capacities,
        "include_pcm" : bool(config.include_pcm),
        "checkpoint_compatibility_marker": config.checkpoint_compatibility_marker,
    }


def run_intrastate(
    state_name: str,
    market_ids: List[str],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: Optional[np.ndarray],
    config: Config,
) -> dict:
    """
    Config 2 — Intra-state connected optimisation.

    All markets in the state are solved jointly.  The optimiser decides
    endogenously which markets install refrigeration (hubs) and which
    receive cold via transport (spokes): fridge capacity is extendable
    everywhere, so markets where a fridge is uneconomic will have
    fridge p_nom_opt ≈ 0.

    Parameters
    ----------
    state_name  : state label
    market_ids  : list of market IDs in this state
    solar_cfs   : market_id → solar CF array
    cold_loads  : market_id → cold demand array (kW)
    locations   : DataFrame with columns market_id, x, y
    snapshots   : DatetimeIndex
    weights     : snapshot weights; uniform ones if None
    config      : Config instance

    Returns
    -------
    dict with keys:
        state          : str
        config         : "intrastate"
        total_cost     : float  — network.objective
        capacities     : pd.DataFrame
        transport      : pd.DataFrame  — from _extract_transport_summary
        hub_market_ids : list[str]  — markets where fridge was installed
    """
    if weights is None:
        weights = np.ones(len(snapshots))

    network = build_intrastate_network(
        state_name = state_name,
        market_ids = market_ids,
        solar_cfs  = solar_cfs,
        cold_loads = cold_loads,
        locations  = locations,
        snapshots  = snapshots,
        weights    = weights,
        config     = config,
    )
    _solve(network, config)

    capacities  = _extract_capacities(network, market_ids)
    transport   = _extract_transport_summary(network)
    hub_ids     = _identify_hubs(capacities, config)
    dispatch_df = _extract_dispatch(network, capacities, hub_ids, state_name)

    return {
        "state"         : state_name,
        "config"        : "intrastate",
        "total_cost"    : float(network.objective),
        "market_ids"    : market_ids,        # full list of all markets in this state
        "capacities"    : capacities,
        "transport"     : transport,
        "hub_market_ids": hub_ids,
        "dispatch"      : dispatch_df,
        "include_pcm"   : bool(config.include_pcm),
        "checkpoint_compatibility_marker": config.checkpoint_compatibility_marker,
    }


def run_interstate(
    intrastate_results: Dict[str, dict],
    standalone_results: Dict[str, dict],
    locations: pd.DataFrame,
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    snapshots: pd.DatetimeIndex,
    weights: Optional[np.ndarray],
    config: Config,
) -> dict:
    """
    Config 3 — Hub-only national LP with intra-state and inter-state
    connections between hub markets.

    One hub per state (the market with the highest fridge_kw from Config 2)
    represents the entire state: its cold bus carries the aggregated demand
    of all markets in that state.  Hub technologies are re-optimised from
    zero so states can specialise via interstate cold trade.

    The network uses build_interstate_network(), which includes:
      • Intra-state hub-to-hub connections (no distance threshold).
      • Inter-state hub-to-hub connections (within distance threshold).
      • Full state cold demand on each hub (no spoke markets outside the LP).

    Parameters
    ----------
    intrastate_results : state_name → result dict from run_intrastate()
    standalone_results : state_name → result dict from run_standalone()
                         (retained for API compatibility; not used for cost)
    locations          : DataFrame (market_id, x, y, state) for ALL markets
    solar_cfs          : market_id → solar CF array for ALL markets
    cold_loads         : market_id → cold demand array (kW) for ALL markets
    snapshots          : DatetimeIndex
    weights            : snapshot weights; uniform ones if None
    config             : Config instance

    Returns
    -------
    dict with keys:
        config                 : "interstate"
        total_cost             : float  — network.objective (full national LP)
        total_cost_incl_spokes : float  — same as total_cost (no external spokes)
        spoke_cost             : float  — always 0.0
        spoke_cost_basis       : str
        hub_count              : int
        hub_market_ids         : list[str]
        interstate_links_built : int
        capacities             : pd.DataFrame
        transport              : pd.DataFrame
    """
    if weights is None:
        weights = np.ones(len(snapshots))

    # ------------------------------------------------------------------
    # 1. Collect hub market IDs from every state
    # ------------------------------------------------------------------
    hub_market_ids: set = set()
    for result in intrastate_results.values():
        hub_market_ids.update(result["hub_market_ids"])

    if not hub_market_ids:
        raise RuntimeError(
            "No hub markets found across any state. "
            "Check that intrastate results contain non-zero fridge capacities."
        )

    # ------------------------------------------------------------------
    # 2. Build hub_records DataFrame (type-safe market_id lookups)
    # ------------------------------------------------------------------
    loc_df = locations.copy()
    loc_df["market_id"] = loc_df["market_id"].map(normalise_id)
    loc_idx = loc_df.set_index("market_id")
    solar_lookup = _build_id_lookup(solar_cfs)
    cold_lookup  = _build_id_lookup(cold_loads)

    hub_rows: list[dict] = []
    skipped_hubs: list[str] = []
    for mid in hub_market_ids:
        mid_key = normalise_id(mid)
        if mid_key not in loc_idx.index:
            skipped_hubs.append(mid_key)
            continue
        solar = _lookup_id(solar_lookup, mid)
        if solar is None:
            skipped_hubs.append(mid_key)
            continue
        hub_rows.append({
            "market_id": mid_key,
            "x"        : float(loc_idx.loc[mid_key, "x"]),
            "y"        : float(loc_idx.loc[mid_key, "y"]),
            "state"    : str(loc_idx.loc[mid_key, "state"]),
            "solar_cf" : solar,
            "cold_load": _lookup_id(cold_lookup, mid),
        })

    if skipped_hubs:
        print(
            f"  Warning: skipping {len(skipped_hubs)} hub(s) not in the "
            f"current market set (stale checkpoint?): "
            f"{skipped_hubs[:5]}{'…' if len(skipped_hubs) > 5 else ''}"
        )

    if not hub_rows:
        raise RuntimeError(
            "No hub markets remain after filtering to the current market set. "
            "Delete results/checkpoints/ and re-run, or ensure debug market "
            "selection includes the Config 2 hub markets."
        )

    hub_records = pd.DataFrame(hub_rows)
    hub_ids     = hub_records["market_id"].tolist()

    # ------------------------------------------------------------------
    # 3. Build and solve the hub-only national network
    # ------------------------------------------------------------------
    network = build_interstate_network(
        hub_records        = hub_records,
        intrastate_results = intrastate_results,
        cold_load_arrays   = cold_loads,
        snapshots          = snapshots,
        weights            = weights,
        config             = config,
    )
    _solve(network, config)

    capacities = _extract_capacities(network, hub_ids)
    transport  = _extract_transport_summary(network)

    interstate_links = transport if not transport.empty else pd.DataFrame()
    interstate_links_built = len(interstate_links)

    lp_cost = float(network.objective)

    return {
        "config"                : "interstate",
        "total_cost"            : lp_cost,
        "total_cost_incl_spokes": lp_cost,
        "spoke_cost"            : 0.0,
        "spoke_cost_basis"      : "all demand aggregated into hub LP",
        "hub_count"             : len(hub_ids),
        "hub_market_ids"        : hub_ids,
        "interstate_links_built": interstate_links_built,
        "capacities"            : capacities,
        "transport"             : transport,
        "include_pcm"           : bool(config.include_pcm),
        "checkpoint_compatibility_marker": config.checkpoint_compatibility_marker,
    }


def run_national_connected(
    intrastate_results: Dict[str, dict],
    state_groups: Dict[str, list],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: Optional[np.ndarray],
    config: Config,
) -> dict:
    """
    Config 3 — Active implementation.  True full national mesh across all
    markets, with no hub selection, no distance threshold, and no pruning
    floor.  Every market from every state is a node; intra-state links are
    a full mesh (identical to Config 2); interstate links connect every pair
    of markets from different states with no exclusions.
    transit_efficiency(d) and diesel marginal cost alone determine how much
    flow, if any, the optimizer sends over each link.

    Intended to run on HPC given the scale (~1,984 markets nationally).

    Parameters
    ----------
    intrastate_results : state_name → result dict from run_intrastate()
    state_groups       : state_name → list[market_id]
    solar_cfs          : market_id → solar CF array
    cold_loads         : market_id → cold demand array kW
    locations          : DataFrame (market_id, x, y, state)
    snapshots          : DatetimeIndex
    weights            : snapshot weights; uniform ones if None
    config             : Config instance

    Returns
    -------
    dict with keys:
        config                 : "national_connected"
        total_cost             : float  — network.objective (ALL markets)
        interstate_links_built : int
        capacities             : pd.DataFrame
        transport              : pd.DataFrame
    """
    if weights is None:
        weights = np.ones(len(snapshots))

    # ── Build and solve the national network ──────────────────────────────
    network = build_national_network(
        state_groups = state_groups,
        solar_cfs    = solar_cfs,
        cold_loads   = cold_loads,
        locations    = locations,
        snapshots    = snapshots,
        weights      = weights,
        config       = config,
    )
    _solve(network, config, extra_solver_options=config.c3_barrier_options)

    # ── Extract results ───────────────────────────────────────────────────
    all_mids   = [mid for mids in state_groups.values() for mid in mids]
    capacities = _extract_capacities(network, all_mids)
    transport  = _extract_transport_summary(network)

    # Identify inter-state links by checking state membership
    loc_state = locations.set_index("market_id")["state"].to_dict()
    interstate_links = pd.DataFrame()
    if not transport.empty:
        def _is_interstate(row):
            s1 = loc_state.get(str(row["from_market"]), "")
            s2 = loc_state.get(str(row["to_market"]),   "")
            return s1 != s2 and s1 != "" and s2 != ""
        mask = transport.apply(_is_interstate, axis=1)
        interstate_links = transport[mask]

    return {
        "config"                : "national_connected",
        "total_cost"            : float(network.objective),
        "interstate_links_built": len(interstate_links),
        "capacities"            : capacities,
        "transport"             : transport,
        "include_pcm"           : bool(config.include_pcm),
        "checkpoint_compatibility_marker": config.checkpoint_compatibility_marker,
    }


# ---------------------------------------------------------------------------
# Smoke-test  (uses CBC since Gurobi is unavailable in this environment)
# ---------------------------------------------------------------------------
