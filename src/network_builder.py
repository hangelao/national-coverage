"""
network_builder.py
------------------
Constructs unsolved PyPSA networks for the three configurations of the
solar cold-chain optimisation pipeline.

Public API
----------
    build_standalone_network(market_id, solar_cf, cold_load,
                             snapshots, weights, config)
        -> pypsa.Network   [Config 1: one market, no transport]

    build_intrastate_network(state_name, market_ids, solar_cfs,
                             cold_loads, locations, snapshots, weights, config)
        -> pypsa.Network   [Config 2: all markets in one state, full transport mesh]

    build_interstate_network(hub_records, intrastate_results, cold_load_arrays,
                             snapshots, weights, config)
        -> pypsa.Network   [Config 3: one hub per state, full state demand, inter-state links]

Component naming convention
---------------------------
    Electricity bus  :  "{mid}_elec"
    Cold bus         :  "{mid}_cold"
    Battery aux bus  :  "{mid}_bat_bus"    (internal; not in user-facing results)
    Solar generator  :  "{mid}_solar"
    Fridge link      :  "{mid}_fridge"
    Battery store    :  "{mid}_battery"
    Batt charge link :  "{mid}_batt_charge"
    Batt disch link  :  "{mid}_batt_discharge"
    PCM store        :  "{mid}_pcm"
    Transport link   :  "{src}to{dst}_{mode}"  (mode: conv)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pypsa

from config import Config


# ---------------------------------------------------------------------------
# Internal distance helper
# ---------------------------------------------------------------------------

def _get_distance(road_distances: dict, id_a, id_b) -> float:
    """
    Return road distance (km) between two markets.
    Raises KeyError if pair not found — no silent fallback.

    The road_distance_matrix.pkl was built from integer uniq_id values
    (compute_road_distances.py reads the CSV without dtype=, so ids are
    Python ints).  market_id values in the pipeline are normalised strings
    (e.g. "121069").  Cast to int here so the lookup always matches the
    pkl key type regardless of how callers pass the id.
    """
    a = int(id_a)
    b = int(id_b)
    key = (a, b)
    if key in road_distances:
        return road_distances[key]
    key_rev = (b, a)
    if key_rev in road_distances:
        return road_distances[key_rev]
    raise KeyError(
        f"Road distance not found for market pair ({id_a}, {id_b}). "
        f"Ensure road_distance_matrix.pkl covers all market pairs."
    )


# ---------------------------------------------------------------------------
# Internal network construction helpers
# ---------------------------------------------------------------------------

def _set_snapshot_weights(
    network: pypsa.Network, weights: np.ndarray, config: Config
) -> None:
    """
    Apply annual objective weights without stretching storage dynamics.

    PyPSA uses:
      • objective  — multiplier on variable (marginal) costs in the objective
      • stores     — physical hours elapsed in Store energy balances
      • generators — multiplier on generator energy constraints
    Cluster counts belong in annual objective/generator aggregation only.
    Applying them to ``stores`` would treat each representative hour as many
    consecutive physical hours and distort state-of-charge dynamics.
    """
    network.snapshot_weightings["objective"] = weights
    network.snapshot_weightings["generators"] = weights
    network.snapshot_weightings["stores"] = config.time_resolution_hours


def _transport_availability_mask(
    snapshots: pd.DatetimeIndex, config: Config
) -> pd.Series:
    """Return the binary transport availability for each snapshot clock hour."""
    config.validate_transport_operating_window()
    if not isinstance(snapshots, pd.DatetimeIndex):
        raise TypeError("transport availability requires a DatetimeIndex")

    available = (
        (snapshots.hour >= config.transport_operating_start_hour)
        & (snapshots.hour < config.transport_operating_end_hour)
    )
    return pd.Series(available.astype(float), index=snapshots, name="p_max_pu")


def _add_market_components(
    network: pypsa.Network,
    mid: str,
    solar_cf_array: np.ndarray,
    cold_load_array: np.ndarray,
    snapshots: pd.DatetimeIndex,
    config: Config,
    n_snapshots: int,
    c2_caps: Optional[Union[pd.Series, dict]] = None,
) -> None:
    """
    Add every component required for a single off-grid cold-chain market.

    Topology
    --------
    Solar ──► elec_bus ──► fridge_link ──────────────────────► cold_bus ──► cold_load
                │                                                  │
                ▼                                                  ▼
          bat_charge_link ──► bat_bus ──► bat_discharge_link       pcm Store
                                   ▲                                  │
                                   └────────────────────── (bat_bus)  └── cold_bus

    Parameters
    ----------
    mid            : market ID string (e.g. "110081")
    solar_cf_array : hourly solar capacity factors, length = n_snapshots
    cold_load_array: hourly cold demand in kW, length = n_snapshots
    snapshots      : DatetimeIndex of length n_snapshots
    config         : Config instance
    n_snapshots    : int — number of retained operational snapshots
    c2_caps        : optional Config 2 optimised capacities for this market;
                     when provided, solar/fridge/battery/PCM are sized with
                     C2 values as floors (p_nom_min / e_nom_min) and remain
                     extendable above that level
    """
    c = config   # shorthand

    if c2_caps is not None:
        solar_floor  = float(c2_caps["solar_kw"])
        fridge_floor = float(c2_caps["fridge_kw"])
        bat_floor    = float(c2_caps["battery_kwh"])
    else:
        solar_floor = fridge_floor = bat_floor = 0.0

    # ------------------------------------------------------------------
    # Buses
    # ------------------------------------------------------------------
    network.add("Bus", f"{mid}_elec",    carrier="AC")
    network.add("Bus", f"{mid}_cold",    carrier="cold")
    network.add("Bus", f"{mid}_bat_bus", carrier="AC")    # auxiliary for battery

    # ------------------------------------------------------------------
    # Solar PV generator
    # p_max_pu is the time-varying capacity factor (upper bound per kWp).
    # capital_cost is annualised $/kWp and independent of snapshot count.
    # ------------------------------------------------------------------
    solar_kwargs: dict = dict(
        bus=f"{mid}_elec",
        p_nom_extendable=True,
        p_nom_min=solar_floor,
        p_max_pu=pd.Series(solar_cf_array, index=snapshots),
        capital_cost=c.pv_annualised_capex_per_kw,
        marginal_cost=0.0,
    )
    if c2_caps is not None:
        solar_kwargs["p_nom"] = solar_floor
    network.add("Generator", f"{mid}_solar", **solar_kwargs)

    # ------------------------------------------------------------------
    # Battery storage  (two-link model for asymmetric efficiencies)
    # ------------------------------------------------------------------
    # Store sits on an auxiliary bus; charge/discharge links bridge to elec_bus.
    battery_kwargs: dict = dict(
        bus=f"{mid}_bat_bus",
        e_nom_extendable=True,
        e_nom_min=bat_floor,
        e_cyclic=True,
        standing_loss=c.battery_standing_loss,
        capital_cost=c.battery_annualised_energy_capex_per_kwh,
    )
    if c2_caps is not None:
        battery_kwargs["e_nom"] = bat_floor
    network.add("Store", f"{mid}_battery", **battery_kwargs)
    network.add(
        "Link",
        f"{mid}_batt_charge",
        bus0=f"{mid}_elec",
        bus1=f"{mid}_bat_bus",
        p_nom_extendable=True,
        p_nom_min=0.0,
        efficiency=c.battery_charge_efficiency,
        marginal_cost=0.0,
        # One shared bidirectional converter: charge Link carries its cost.
        # optimizer._solve() equates this external AC-side rating to the
        # efficiency-adjusted discharge Link output rating.
        capital_cost=c.battery_annualised_power_capex_per_kw,
    )
    network.add(
        "Link",
        f"{mid}_batt_discharge",
        bus0=f"{mid}_bat_bus",
        bus1=f"{mid}_elec",
        p_nom_extendable=True,
        p_nom_min=0.0,
        efficiency=c.battery_discharge_efficiency,
        marginal_cost=0.0,
        capital_cost=0.0,
    )

    # ------------------------------------------------------------------
    # Conventional electrically driven vapour-compression refrigerator.
    # CAPEX is derived from USD 2,580 / 2.05361 kW-cold and is therefore on
    # a cooling-output-capacity basis. PyPSA Link p_nom is electrical input;
    # multiplying CAPEX by COP converts it to the Link input-capacity basis.
    # ------------------------------------------------------------------
    fridge_kwargs: dict = dict(
        bus0=f"{mid}_elec",
        bus1=f"{mid}_cold",
        p_nom_extendable=True,
        p_nom_min=fridge_floor,
        efficiency=c.fridge_cop,
        capital_cost=c.fridge_annualised_capex_per_kw_electric,
        marginal_cost=0.0,
    )
    if c2_caps is not None:
        fridge_kwargs["p_nom"] = fridge_floor
    network.add("Link", f"{mid}_fridge", **fridge_kwargs)

    # ------------------------------------------------------------------
    # PCM passive thermal storage, integrated directly on the cold bus.
    # It is supplementary and absent from normal Config 1–3 runs.
    # ------------------------------------------------------------------
    if c.include_pcm:
        pcm_kwargs: dict = dict(
            bus=f"{mid}_cold",
            carrier="pcm",
            e_nom_extendable=True,
            e_cyclic=True,
            standing_loss=c.pcm_standing_loss,
            capital_cost=c.pcm_annualised_capex_per_kwh,
        )
        network.add("Store", f"{mid}_pcm", **pcm_kwargs)

    # ------------------------------------------------------------------
    # Cold demand  (fixed load on cold bus)
    # ------------------------------------------------------------------
    network.add(
        "Load",
        f"{mid}_cold_load",
        bus=f"{mid}_cold",
        p_set=pd.Series(cold_load_array, index=snapshots),
    )


def _add_transport_links(
    network: pypsa.Network,
    m1: str,
    m2: str,
    distance: float,
    config: Config,
    n_snapshots: int,
) -> None:
    """
    Add bidirectional conventional diesel-truck transport links between two markets.

    Two links are created:
        {m1}to{m2}_conv,  {m2}to{m1}_conv

    Each link transfers cold energy from the source cold bus to the
    destination cold bus.  Transport is modelled as a hired service:
      • efficiency  — passive transit heat loss (Newton's law of heating);
                      see Config.transit_efficiency. Fully independent of
                      the diesel marginal cost.
      • capital_cost = 0  — no owned-asset capex; pay-per-use only.

    Parameters
    ----------
    distance   : road distance in km (from _get_distance / road_distance_matrix.pkl)
    n_snapshots: unused; retained for API compatibility
    """
    c = config
    if c.max_transport_link_hours is not None:
        max_distance_km = c.van_speed_kmh * c.max_transport_link_hours
        if distance > max_distance_km:
            return
    availability = _transport_availability_mask(network.snapshots, c)

    for src, dst in [(m1, m2), (m2, m1)]:
        network.add(
            "Link",
            f"{src}to{dst}_conv",
            bus0=f"{src}_cold",
            bus1=f"{dst}_cold",
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_max_pu=availability,
            efficiency=c.transit_efficiency(distance),
            marginal_cost=c.conv_marginal_cost(distance),
            capital_cost=0,
        )


# ---------------------------------------------------------------------------
# Public network builders
# ---------------------------------------------------------------------------

def build_standalone_network(
    market_id: str,
    solar_cf: np.ndarray,
    cold_load: np.ndarray,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
) -> pypsa.Network:
    """
    Config 1 — Standalone network for a single market.

    Each market operates entirely independently: its own solar PV, battery,
    PCM thermal storage, and refrigeration unit.  No transport links.

    Parameters
    ----------
    market_id : str market identifier
    solar_cf  : np.ndarray shape (n_snapshots,) — hourly solar capacity factors
    cold_load : np.ndarray shape (n_snapshots,) — hourly cold demand in kW
    snapshots : DatetimeIndex of length n_snapshots
    weights   : np.ndarray snapshot weights from representative period selection
    config    : Config instance

    Returns
    -------
    pypsa.Network  (unsolved; call network.optimize() externally)
    """
    n_snapshots = len(snapshots)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    _set_snapshot_weights(network, weights, config)

    _add_market_components(
        network, market_id, solar_cf, cold_load, snapshots, config, n_snapshots
    )

    return network


def build_intrastate_network(
    state_name: str,
    market_ids: list,
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
) -> pypsa.Network:
    """
    Config 2 — Intra-state connected network.

    All markets within one state share a fully-connected transport mesh.
    The optimiser decides endogenously which markets install refrigeration
    capacity (hubs) and which receive cold energy via transport (spokes),
    by setting fridge p_nom_extendable=True everywhere: markets where
    installing a fridge is uneconomic will have fridge p_nom_opt ≈ 0.

    Parameters
    ----------
    state_name  : name of the state (for logging / metadata only)
    market_ids  : list of str market IDs belonging to this state
    solar_cfs   : dict market_id → np.ndarray solar CF (length n_snapshots)
    cold_loads  : dict market_id → np.ndarray cold demand kW (length n_snapshots)
    locations   : DataFrame with columns market_id, x (longitude), y (latitude);
                  x/y are used only to index markets — distances come from
                  config.road_distances
    snapshots   : DatetimeIndex of length n_snapshots
    weights     : np.ndarray snapshot weights
    config      : Config instance

    Returns
    -------
    pypsa.Network  (unsolved)
    """
    n_snapshots = len(snapshots)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    _set_snapshot_weights(network, weights, config)
    network.name = state_name

    # Add components for every market in the state
    for mid in market_ids:
        _add_market_components(
            network, mid, solar_cfs[mid], cold_loads[mid],
            snapshots, config, n_snapshots,
        )

    # Add transport links for every ordered pair (i, j) with i < j
    for i, m1 in enumerate(market_ids):
        for m2 in market_ids[i + 1:]:
            dist = _get_distance(config.road_distances, m1, m2)
            _add_transport_links(network, m1, m2, dist, config, n_snapshots)

    return network


def build_interstate_network(
    hub_records: pd.DataFrame,
    intrastate_results: Dict[str, dict],
    cold_load_arrays: Dict[str, np.ndarray],
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
) -> pypsa.Network:
    """
    Config 3 — Inter-state connected network (one hub per state).

    Each hub carries the aggregated cold demand of all markets in its state.
    Hub technologies are re-optimised from zero (no C2 capacity floors), so
    states can specialise via interstate cold trade.

    Hub markets retain intra-state transport links between same-state hubs,
    AND gain additional links to hub markets in neighbouring states within
    a 100 km road-distance threshold (superseded by build_national_network;
    retained for run_interstate() backwards compatibility).

    Parameters
    ----------
    hub_records : DataFrame with one row per hub market, columns:
                    market_id  (str)
                    x          (float, longitude)
                    y          (float, latitude)
                    state      (str)
                    solar_cf   (np.ndarray, length n_snapshots)
                    cold_load  (np.ndarray, length n_snapshots) — hub-only;
                                 superseded by state aggregation below
    intrastate_results : state_name → result dict from run_intrastate();
                         market_ids per state used to aggregate cold demand
    cold_load_arrays   : market_id → hourly cold demand array (all markets)
    snapshots   : DatetimeIndex of length n_snapshots
    weights     : np.ndarray snapshot weights
    config      : Config instance

    Returns
    -------
    pypsa.Network  (unsolved)
    """
    n_snapshots = len(snapshots)
    threshold   = 100.0   # superseded by build_national_network full mesh

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    _set_snapshot_weights(network, weights, config)
    network.name = "interstate"

    # cold_load_arrays may use str or int keys — normalise once for lookup
    cl_lookup: dict = {}
    for k, v in cold_load_arrays.items():
        cl_lookup[str(k)] = v
        try:
            cl_lookup[int(k)] = v
        except (ValueError, TypeError):
            pass

    # Add components for every hub market (fresh optimisation from zero)
    for _, row in hub_records.iterrows():
        mid   = str(row["market_id"])
        state = str(row["state"])
        state_market_ids = intrastate_results[state].get("market_ids", [])
        cold_load_parts = []
        for m in state_market_ids:
            arr = None
            for key in (str(m), m):
                if key in cl_lookup:
                    arr = cl_lookup[key]
                    break
            if arr is not None:
                cold_load_parts.append(arr)
        if cold_load_parts:
            cold_load = sum(cold_load_parts)
        else:
            cold_load = None
            for key in (str(mid), mid):
                if key in cl_lookup:
                    cold_load = cl_lookup[key]
                    break
            if cold_load is None:
                cold_load = np.zeros(n_snapshots)
        _add_market_components(
            network, mid, row["solar_cf"], cold_load,
            snapshots, config, n_snapshots,
        )

    # Transport links:
    #   - Same-state hub pairs: always connected (preserves Config 2 topology)
    #   - Cross-state hub pairs: connected only within distance threshold
    # This ensures Config 3 cost ≤ Config 2 cost by construction, and any
    # C2→C3 saving reflects inter-state connectivity specifically.
    hub_df = hub_records.reset_index(drop=True)

    for i in range(len(hub_df)):
        row_i = hub_df.iloc[i]
        for j in range(i + 1, len(hub_df)):
            row_j      = hub_df.iloc[j]
            same_state = row_i["state"] == row_j["state"]
            dist       = _get_distance(
                config.road_distances,
                str(row_i["market_id"]),
                str(row_j["market_id"]),
            )
            if same_state or dist <= threshold:
                _add_transport_links(
                    network,
                    str(row_i["market_id"]),
                    str(row_j["market_id"]),
                    dist, config, n_snapshots,
                )

    return network


def build_national_network(
    state_groups: Dict[str, list],
    solar_cfs: Dict[str, np.ndarray],
    cold_loads: Dict[str, np.ndarray],
    locations: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    weights: np.ndarray,
    config: Config,
) -> pypsa.Network:
    """
    Config 3 — True full national mesh (ALL markets, ALL states).

    Every market from every state is a node in the LP.  Intra-state links
    are a full mesh, identical to build_intrastate_network.  Interstate links
    are built for every cross-state pair whose transit_efficiency(d) meets
    or exceeds config.transit_efficiency_floor.

    The floor is a *computational pruning* convenience — not a modelling
    assumption about where trade is allowed.  Links pruned by the floor have
    such high thermal loss and diesel cost that the optimiser would never use
    them anyway; skipping them keeps the LP tractable at national scale.
    Choose the floor value using diagnostics/count_pruned_links.py before
    an HPC submission; the diagnostic reports surviving link counts and
    estimated LP variable counts at several candidate values.

    Parameters
    ----------
    state_groups : state_name → list of market IDs in that state
    solar_cfs    : market_id → solar CF array (length n_snapshots)
    cold_loads   : market_id → cold demand array kW (length n_snapshots)
    locations    : DataFrame with columns market_id, x, y, state
    snapshots    : DatetimeIndex of length n_snapshots
    weights      : np.ndarray snapshot weights
    config       : Config instance

    Returns
    -------
    pypsa.Network  (unsolved)
    """
    n_snapshots = len(snapshots)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    _set_snapshot_weights(network, weights, config)
    network.name = "national_connected"

    # ── Step 1: Add components for every market in every state ────────────
    t_comp = time.time()
    for mids in state_groups.values():
        for mid in mids:
            _add_market_components(
                network, mid, solar_cfs[mid], cold_loads[mid],
                snapshots, config, n_snapshots,
            )
    print(f"  [build_national] market components added "
          f"[{time.time() - t_comp:.1f}s]")

    # ── Steps 2 & 3: Transport links — vectorised batch build ────────────
    # Intra-state : every within-state pair (no floor).
    # Inter-state : every cross-state pair whose transit_efficiency(d) meets
    #               or exceeds config.transit_efficiency_floor (purely a
    #               computational-pruning convenience — see docstring).
    #
    # The original code added the ~380k bidirectional links one at a time via
    # network.add(), which is O(n^2) in PyPSA (every add re-validates the
    # growing component frame) and took hours.  Here the enumeration/pruning
    # is done with numpy and the links are issued in a single vectorised
    # network.add() call — seconds instead of hours.  The resulting link set
    # (names, buses, efficiency, marginal_cost) is identical to the old
    # per-pair _add_transport_links() path.
    t_links = time.time()

    # Market order MUST match Step 1 (bus names are f"{mid}_cold").
    markets = [mid for mids in state_groups.values() for mid in mids]
    market_state = np.array(
        [st for st, mids in state_groups.items() for _ in mids]
    )
    N = len(markets)
    pos = {int(m): k for k, m in enumerate(markets)}

    # Dense symmetric road-distance matrix (NaN => pair absent from the pkl).
    D = np.full((N, N), np.nan, dtype=np.float64)
    for (a, b), d in config.road_distances.items():
        ia = pos.get(a)
        ib = pos.get(b)
        if ia is not None and ib is not None:
            D[ia, ib] = d
            D[ib, ia] = d

    # All i<j candidate pairs, vectorised.
    iu, ju   = np.triu_indices(N, k=1)
    dist_u   = D[iu, ju]
    same_st  = market_state[iu] == market_state[ju]
    # transit_efficiency(d) = exp(-(d / van_speed) / tau)  — see Config.
    eff_u    = np.exp(-(dist_u / config.van_speed_kmh)
                      / config.transit_thermal_tau_hr)

    if config.max_transport_link_hours is None:
        distance_allowed = np.ones_like(dist_u, dtype=bool)
    else:
        max_distance_km = config.van_speed_kmh * config.max_transport_link_hours
        distance_allowed = dist_u <= max_distance_km

    # Enforce the time-consistency limit in candidate construction itself.
    intra_mask = same_st & distance_allowed
    inter_keep = ((~same_st)
                  & (eff_u >= config.transit_efficiency_floor)
                  & distance_allowed)
    keep_mask  = intra_mask | inter_keep

    # Fail fast on any *used* pair missing from the road-distance matrix
    # (mirrors the original _get_distance KeyError — no silent fallback).
    if np.isnan(dist_u[keep_mask]).any():
        n_missing = int(np.isnan(dist_u[keep_mask]).sum())
        raise KeyError(
            f"Road distance not found for {n_missing} required market pair(s). "
            f"Ensure road_distance_matrix.pkl covers all market pairs."
        )

    intrastate_pairs        = int(intra_mask.sum())
    interstate_pairs        = int(inter_keep.sum())
    interstate_pairs_pruned = int(
        ((~same_st) & (eff_u < config.transit_efficiency_floor)).sum()
    )

    print(f"  [build_national] Building interstate mesh: {N} markets, "
          f"~{N * (N - 1) // 2:,} candidate pairs total (intra + inter-state), "
          f"floor={config.transit_efficiency_floor} …")

    ai       = iu[keep_mask]
    bj       = ju[keep_mask]
    dist_sel = dist_u[keep_mask]
    eff_sel  = eff_u[keep_mask]
    # conv_marginal_cost is linear through the origin: c(d) = c(1) · d.
    mc_sel   = config.conv_marginal_cost(1.0) * dist_sel

    src = [markets[i] for i in ai]
    dst = [markets[j] for j in bj]

    # Bidirectional: both directions share the distance-derived eff / cost.
    names = ([f"{s}to{d_}_conv" for s, d_ in zip(src, dst)] +
             [f"{d_}to{s}_conv" for s, d_ in zip(src, dst)])
    bus0  = [f"{s}_cold" for s in src] + [f"{d_}_cold" for d_ in dst]
    bus1  = [f"{d_}_cold" for d_ in dst] + [f"{s}_cold" for s in src]
    eff2  = np.concatenate([eff_sel, eff_sel])
    mc2   = np.concatenate([mc_sel,  mc_sel])
    availability = _transport_availability_mask(network.snapshots, config)
    transport_p_max_pu = pd.DataFrame(
        np.repeat(
            availability.to_numpy()[:, np.newaxis],
            len(names),
            axis=1,
        ),
        index=network.snapshots,
        columns=names,
    )

    network.add(
        "Link", names,
        bus0=bus0, bus1=bus1,
        p_nom_extendable=True, p_nom_min=0.0,
        p_max_pu=transport_p_max_pu,
        efficiency=eff2, marginal_cost=mc2, capital_cost=0.0,
    )

    total_markets = N
    print(f"  [build_national] {total_markets} markets, "
          f"{len(state_groups)} states, "
          f"{intrastate_pairs} intra-state pairs, "
          f"{interstate_pairs} inter-state pairs built, "
          f"{interstate_pairs_pruned} inter-state pairs pruned "
          f"(below transit_efficiency_floor={config.transit_efficiency_floor})"
          f" — approx {(interstate_pairs + intrastate_pairs) * n_snapshots:,} "
          f"transport-link time-indexed variables at {n_snapshots} snapshots  "
          f"[{time.time() - t_links:.1f}s]")

    return network


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
