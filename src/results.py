"""
results.py
----------
Compiles, compares, saves, and visualises results from all three pipeline
configurations.

Public API
----------
    compile_state_summary(standalone_results, intrastate_results,
                          interstate_total_cost, n_markets_per_state)
        -> pd.DataFrame

    compile_technology_mix(standalone_results, intrastate_results)
        -> pd.DataFrame

    compile_transport_breakdown(intrastate_results, interstate_result)
        -> pd.DataFrame

    generate_figures(state_summary, technology_mix, transport_breakdown,
                     locations, output_dir, *, interstate_result=None,
                     standalone_results=None, intrastate_results=None,
                     hub_capacity_threshold_kw=0.1)
        -> None

    save_all_results(state_summary, technology_mix, transport_breakdown,
                     interstate_result, locations, config, output_dir, *,
                     standalone_results=None, intrastate_results=None)
        -> None

    config3_fair_total(interstate_result, standalone_results)
        -> float   [DEPRECATED — kept for reference only]
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import geopandas as gpd
try:
    from adjustText import adjust_text  # type: ignore[import-untyped]
    _ADJUST_TEXT_AVAILABLE = True
except ImportError:
    _ADJUST_TEXT_AVAILABLE = False
    def adjust_text(texts, **kwargs):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# State choropleth reference data (fig19/fig20)
# ---------------------------------------------------------------------------

NIGERIA_STATES_GEOJSON = os.path.join(
    os.path.dirname(__file__), "data", "nigeria_states.geojson"
)

# Same cutoff used by optimizer._extract_transport_summary when saved link
# rows are created.  This constant changes reporting labels only.
TRANSPORT_CAPACITY_REPORTING_TOLERANCE_KW = 1e-6

# Shapefile spelling -> model spelling (reconcile mismatches)
SHAPEFILE_TO_MODEL_STATE = {
    "Nassarawa": "Nasarawa",
    "Federal Capital Territory": "FCT",
}

# Standard state capital coordinates (lon, lat)
STATE_CAPITALS = {
    "Abia": (7.4860, 5.5320), "Adamawa": (12.4780, 9.3280),
    "Akwa Ibom": (7.9340, 5.0380), "Anambra": (7.0710, 6.2100),
    "Bauchi": (9.8440, 10.3110), "Bayelsa": (6.2650, 4.9270),
    "Benue": (8.5390, 7.7320), "Borno": (13.1500, 11.8460),
    "Cross River": (8.3170, 4.9500), "Delta": (5.7000, 6.2000),
    "Ebonyi": (8.1010, 6.2650), "Edo": (5.6040, 6.3390),
    "Ekiti": (5.2110, 7.6210), "Enugu": (7.5100, 6.4500),
    "FCT": (7.4951, 9.0579), "Gombe": (11.1670, 10.2890),
    "Imo": (7.0330, 5.4840), "Jigawa": (9.5240, 12.2280),
    "Kaduna": (7.4390, 10.5230), "Kano": (8.5160, 12.0000),
    "Katsina": (7.6000, 13.0000), "Kebbi": (4.1990, 12.4530),
    "Kogi": (6.7400, 7.8000), "Kwara": (4.5590, 8.4970),
    "Lagos": (3.3792, 6.5244), "Nasarawa": (8.5210, 8.5390),
    "Niger": (6.5560, 9.6140), "Ogun": (3.3470, 7.1600),
    "Ondo": (5.1470, 7.2500), "Osun": (4.5590, 7.7630),
    "Oyo": (3.9000, 8.0000), "Plateau": (8.8940, 9.9280),
    "Rivers": (7.0130, 4.8240), "Sokoto": (5.2470, 13.0620),
    "Taraba": (11.3330, 8.8940), "Yobe": (11.7470, 11.9660),
    "Zamfara": (6.2440, 12.1660),
}


# ---------------------------------------------------------------------------
# Matplotlib style helper
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass   # fall back to matplotlib default


# ---------------------------------------------------------------------------
# Choropleth base helpers (fig19/fig20)
# ---------------------------------------------------------------------------

def _plot_nigeria_choropleth_base(ax):
    """Load and draw the state boundary background. Returns the loaded gdf."""
    states_gdf = gpd.read_file(NIGERIA_STATES_GEOJSON)
    states_gdf["state_model"] = states_gdf["NAME_1"].replace(SHAPEFILE_TO_MODEL_STATE)
    states_gdf.plot(ax=ax, color="#dbe7f2", edgecolor="#8fa9c2", linewidth=0.6)
    return states_gdf


def _plot_state_bubbles(ax, values: Dict[str, float], vmin: float, vmax: float,
                        cmap_colors=("#f5ecd7", "#4f8fc0", "#0b3d63"),
                        label_suffix: str = "%"):
    """
    values: {state_name: metric_value}, using MODEL state names (e.g.
    "Nasarawa" not "Nassarawa", "FCT" not "Federal Capital Territory").
    States not in `values` get a small pale "no data" marker instead.
    Returns a ScalarMappable for colorbar construction.
    """
    cmap = LinearSegmentedColormap.from_list("state_metric", list(cmap_colors), N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)
    texts = []
    for state, (lon, lat) in STATE_CAPITALS.items():
        val = values.get(state)
        if val is not None:
            color = cmap(norm(val))
            ax.scatter(lon, lat, s=380, color=color, edgecolor="#333333",
                      linewidth=1.6, zorder=5)
            t = ax.annotate(f"{state} {val:.1f}{label_suffix}", xy=(lon, lat),
                           fontsize=8, fontweight="bold", color="#222222",
                           zorder=6)
            texts.append(t)
        else:
            ax.scatter(lon, lat, s=90, color="#ece4d0", edgecolor="#999999",
                      linewidth=1, zorder=4, alpha=0.85)
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                                                   lw=0.5, alpha=0.6))
    return cm.ScalarMappable(norm=norm, cmap=cmap)


# ---------------------------------------------------------------------------
# compile_state_summary
# ---------------------------------------------------------------------------

def compile_state_summary(
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
    interstate_total_cost: float,
    n_markets_per_state: Dict[str, int],
    hub_capacity_threshold_kw: float = 0.1,
) -> pd.DataFrame:
    """
    One row per state summarising costs and coordination value.

    Columns: state, n_markets, cost_standalone, cost_intrastate,
             saving_intrastate_abs, saving_intrastate_pct,
             dominant_transport_mode,
             n_refrigeration_sites_above_threshold,
             refrigeration_site_pct
    Sorted by saving_intrastate_pct descending.
    """
    rows: list[dict] = []

    for state in sorted(set(standalone_results) | set(intrastate_results)):
        s_res = standalone_results.get(state, {})
        i_res = intrastate_results.get(state, {})

        cost_sa   = float(s_res.get("total_cost", np.nan))
        cost_ia   = float(i_res.get("total_cost", np.nan))
        n_markets = n_markets_per_state.get(state, 0)

        if not (np.isnan(cost_sa) or np.isnan(cost_ia) or cost_sa == 0):
            saving_abs = cost_sa - cost_ia
            saving_pct = saving_abs / cost_sa * 100.0
        else:
            saving_abs = saving_pct = np.nan

        cap_df  = i_res.get("capacities", pd.DataFrame())
        if cap_df is not None and not cap_df.empty and "fridge_kw" in cap_df.columns:
            n_sites = int((cap_df["fridge_kw"] > hub_capacity_threshold_kw).sum())
        else:
            n_sites = 0
        site_pct = (n_sites / n_markets * 100.0) if n_markets > 0 else np.nan

        transport_df  = i_res.get("transport", pd.DataFrame())
        dominant_mode = _dominant_transport_mode(transport_df)

        rows.append({
            "state"                  : state,
            "n_markets"              : n_markets,
            "cost_standalone"        : cost_sa,
            "cost_intrastate"        : cost_ia,
            "saving_intrastate_abs"  : saving_abs,
            "saving_intrastate_pct"  : saving_pct,
            "dominant_transport_mode": dominant_mode,
            "n_refrigeration_sites_above_threshold": n_sites,
            "refrigeration_site_pct" : site_pct,
        })

    _COLS = [
        "state", "n_markets", "cost_standalone", "cost_intrastate",
        "saving_intrastate_abs", "saving_intrastate_pct",
        "dominant_transport_mode",
        "n_refrigeration_sites_above_threshold", "refrigeration_site_pct",
    ]
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows)
    return df.sort_values("saving_intrastate_pct", ascending=False).reset_index(drop=True)


def _dominant_transport_mode(transport_df: pd.DataFrame) -> str:
    if transport_df is None or transport_df.empty:
        return "none"
    if "mode" not in transport_df.columns or "capacity_kw" not in transport_df.columns:
        return "none"
    mode_cap = transport_df.groupby("mode")["capacity_kw"].sum()
    if mode_cap.empty or mode_cap.max() == 0:
        return "none"
    return str(mode_cap.idxmax())


# ---------------------------------------------------------------------------
# compile_technology_mix
# ---------------------------------------------------------------------------

def compile_technology_mix(
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
) -> pd.DataFrame:
    """
    Long-format: (state, config, component, total_capacity).
    Components: solar_kw, fridge_kw, battery_kwh, battery_power_kw, pcm_kwh.
    ``pcm_kwh`` is retained for a stable schema and is zero in battery-only
    main runs.
    """
    COMPONENTS = [
        "solar_kw", "fridge_kw", "battery_kwh", "battery_power_kw", "pcm_kwh"
    ]
    rows: list[dict] = []

    for state in sorted(set(standalone_results) | set(intrastate_results)):
        for config_label, result_dict in [
            ("standalone", standalone_results.get(state, {})),
            ("intrastate", intrastate_results.get(state, {})),
        ]:
            cap_df = result_dict.get("capacities", None)
            for comp in COMPONENTS:
                if cap_df is None or cap_df.empty or comp not in cap_df.columns:
                    total = np.nan
                else:
                    total = float(cap_df[comp].sum())
                rows.append({"state": state, "config": config_label,
                              "component": comp, "total_capacity": total})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compile_transport_breakdown
# ---------------------------------------------------------------------------

def compile_transport_breakdown(
    intrastate_results: Dict[str, dict],
    interstate_result: dict,
    locations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    All saved positive transport links from C2 and C3, with configuration,
    scope, and reporting-status columns.

    When `locations` is provided, Config 3 links are split by actual state
    membership: links whose endpoints share a state are labelled scope='intrastate',
    all others scope='interstate'.  Pass `locations` whenever `interstate_result`
    comes from run_national_connected() (full-mesh Config 3), which contains both
    intra- and inter-state links in the same solve.

    Without `locations` (default), all Config 3 links are blanket-labelled
    scope='interstate' / state='inter' — correct only for the legacy hub-only
    run_interstate() path.
    """
    EMPTY_COLS = [
        "config", "from_market", "to_market", "mode", "capacity_kw", "capex",
        "total_flow_kwh", "scope", "state", "raw_mathematically_positive",
        "above_existing_capacity_tolerance", "zero_cost_degenerate",
        "meaningfully_active", "reporting_status",
    ]
    parts: list[pd.DataFrame] = []

    for state, result in intrastate_results.items():
        df = result.get("transport", pd.DataFrame())
        if df is not None and not df.empty:
            df = df.copy()
            df["config"] = "C2"
            df["scope"] = "intrastate"
            df["state"] = state
            parts.append(df)

    inter_df = interstate_result.get("transport", pd.DataFrame())
    if inter_df is not None and not inter_df.empty:
        inter_df = inter_df.copy()
        inter_df["config"] = "C3"
        if locations is not None and not locations.empty:
            loc_state = locations.set_index("market_id")["state"].to_dict()
            def _scope_and_state(row):
                s1 = loc_state.get(str(row["from_market"]), "")
                s2 = loc_state.get(str(row["to_market"]),   "")
                if s1 and s2 and s1 == s2:
                    return pd.Series({"scope": "intrastate", "state": s1})
                return pd.Series({"scope": "interstate", "state": "inter"})
            inter_df[["scope", "state"]] = inter_df.apply(_scope_and_state, axis=1)
        else:
            inter_df["scope"] = "interstate"
            inter_df["state"] = "inter"
        parts.append(inter_df)

    if not parts:
        return pd.DataFrame(columns=EMPTY_COLS)

    combined = pd.concat(parts, ignore_index=True)
    if "fuel_opex_usd" in combined.columns:
        fuel_opex = combined["fuel_opex_usd"]
    else:
        fuel_opex = (
            combined["marginal_cost_per_kwh"] * combined["total_flow_kwh"]
        )
    combined["raw_mathematically_positive"] = combined["capacity_kw"] > 0.0
    combined["above_existing_capacity_tolerance"] = (
        combined["capacity_kw"] > TRANSPORT_CAPACITY_REPORTING_TOLERANCE_KW
    )
    combined["zero_cost_degenerate"] = (
        combined["above_existing_capacity_tolerance"]
        & (combined["total_flow_kwh"] > 0.0)
        & (fuel_opex == 0.0)
    )
    combined["meaningfully_active"] = (
        combined["above_existing_capacity_tolerance"]
        & (combined["total_flow_kwh"] > 0.0)
        & (fuel_opex > 0.0)
    )
    combined["reporting_status"] = np.select(
        [
            combined["meaningfully_active"],
            combined["zero_cost_degenerate"],
            combined["raw_mathematically_positive"],
        ],
        [
            "meaningfully_active",
            "zero_cost_degenerate_not_economic_network_use",
            "positive_but_below_existing_capacity_tolerance",
        ],
        default="not_active",
    )
    return combined


# ---------------------------------------------------------------------------
# config3_fair_total
# ---------------------------------------------------------------------------

def compile_all_dispatch(intrastate_results: Dict[str, dict]) -> Optional[pd.DataFrame]:
    """
    Concatenate hourly dispatch DataFrames from all intrastate results.

    Returns a single long-format DataFrame, or None if no dispatch data exists.
    """
    parts = [
        res["dispatch"]
        for res in intrastate_results.values()
        if res.get("dispatch") is not None
    ]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def config3_fair_total(
    interstate_result: dict,
    standalone_results: Dict[str, dict],
) -> float:
    """
    DEPRECATED — no longer needed for the redesigned Config 3.

    With run_national_connected(), Config 3 total cost is a single national
    LP objective (network.objective) that already covers ALL markets, so no
    spoke-cost adjustment is required.  The function is retained for backwards
    compatibility only.

    Original purpose: interstate hub cost + standalone costs for spoke markets.
    """
    hub_cost  = float(interstate_result["total_cost"])
    spoke_ids = interstate_result.get("spoke_market_ids", [])

    flat_costs: dict[str, float] = {}
    for s_res in standalone_results.values():
        mc = s_res.get("market_costs")
        if mc is not None:
            for mid, cost in mc.items():
                flat_costs[mid] = float(cost)

    spoke_cost = sum(flat_costs.get(mid, 0.0) for mid in spoke_ids)
    return hub_cost + spoke_cost


# ---------------------------------------------------------------------------
# generate_figures
# ---------------------------------------------------------------------------

def generate_figures(
    state_summary: pd.DataFrame,
    technology_mix: pd.DataFrame,
    transport_breakdown: pd.DataFrame,
    locations: pd.DataFrame,
    output_dir: str,
    cold_loads_df: Optional[pd.DataFrame] = None,
    cold_load_arrays: Optional[Dict[str, np.ndarray]] = None,
    standalone_results: Optional[Dict[str, dict]] = None,
    intrastate_results: Optional[Dict[str, dict]] = None,
    interstate_result: Optional[dict] = None,
    hub_capacity_threshold_kw: float = 0.1,
    dispatch_df: Optional[pd.DataFrame] = None,
    config=None,
) -> None:
    """
    Produce and save up to fifteen PNG figures to {output_dir}/figures/.

    Fig 1  — National market spatial distribution
    Fig 2  — Cost comparison across configurations (per state)
    Fig 3  — Network benefit map (C1→C2 savings per state centroid)
    Fig 4  — Transport mode selection (intra-state links by state)
    Fig 5  — Technology mix: standalone vs intrastate (national totals)
    Fig 6  — Hub vs spoke market distribution
    Fig 7  — Cooling affordability (cost per kg of meat)         [needs cold_loads_df]
    Fig 8  — Hub selection vs market demand                      [needs cold_load_arrays]
    Fig 9  — Network benefit vs market spatial structure
    Fig 10 — Cold energy flow network (small multiples per state)
    Fig 11 — Inter-state cold flow map                           [needs interstate_result]
    Fig 12 — Within-market energy dispatch                       [needs dispatch_df]
    Fig 18 — Network connectivity's effect on local installed capacity
             (size = connected capacity, color = % change vs standalone)
    Fig 19 — C1→C2 cost savings % choropleth (real Nigeria state boundaries)
    Fig 20 — C1→C3 capacity reduction % choropleth (real Nigeria state boundaries)
    Fig 21 — C1→C3 saving rate % choropleth, capex proxy (real Nigeria state boundaries)
             [needs data/nigeria_states.geojson]
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _apply_style()

    # ── Always-available figures (Figs 1–6) ──────────────────────────────
    _fig1_market_distribution(locations, fig_dir)
    _fig2_cost_comparison(state_summary, standalone_results,
                          interstate_result, intrastate_results, fig_dir)
    _fig3_benefit_map(state_summary, locations, fig_dir)
    _fig4_transport_modes(transport_breakdown, state_summary, fig_dir)
    _fig5_technology_mix(technology_mix, fig_dir)
    _fig6_hub_distribution(locations, intrastate_results,
                           hub_capacity_threshold_kw, fig_dir)

    # ── Figures requiring extra data (Figs 7–10) ─────────────────────────
    if (cold_loads_df is not None and standalone_results and intrastate_results
            and "hourly_meat_kg" in cold_loads_df.columns):
        _fig7_cost_per_kg(state_summary, cold_loads_df,
                          standalone_results, intrastate_results, fig_dir)
    else:
        print("  Fig 7 skipped — cold_loads_df (with hourly_meat_kg), "
              "standalone_results, or intrastate_results not available")

    if cold_load_arrays is not None and intrastate_results:
        _fig8_hub_selection(locations, intrastate_results,
                            cold_load_arrays, hub_capacity_threshold_kw, fig_dir)
    else:
        print("  Fig 8 skipped — cold_load_arrays or intrastate_results not available")

    if not state_summary.empty:
        _fig9_network_drivers(state_summary, locations, fig_dir)
    else:
        print("  Fig 9 skipped — state_summary is empty")

    if not transport_breakdown.empty and intrastate_results:
        _fig10_cold_flow_network(locations, intrastate_results,
                                 transport_breakdown, hub_capacity_threshold_kw,
                                 fig_dir)
    else:
        print("  Fig 10 skipped — transport_breakdown or intrastate_results not available")

    if not transport_breakdown.empty:
        _fig11_interstate_flow(locations, transport_breakdown, fig_dir)
    else:
        print("  Fig 11 skipped — transport_breakdown not available")

    if standalone_results and intrastate_results and config is not None:
        _fig18_capacity_change_map(locations, standalone_results,
                                   intrastate_results, config, fig_dir)
    else:
        print("  Fig 18 skipped — standalone_results or intrastate_results not available")

    if state_summary is not None and not state_summary.empty:
        _fig19_state_savings_choropleth(state_summary, fig_dir)
    else:
        print("  Fig 19 skipped — state_summary not available")

    if standalone_results and interstate_result and config is not None:
        _fig20_state_capacity_choropleth(standalone_results, interstate_result,
                                         config, fig_dir)
    else:
        print("  Fig 20 skipped — standalone_results, interstate_result, "
              "or config not available")

    if standalone_results and interstate_result and config is not None:
        _fig21_state_c1c3_savings_choropleth(standalone_results, interstate_result,
                                              config, fig_dir)
    else:
        print("  Fig 21 skipped — standalone_results, interstate_result, "
              "or config not available")

    _fig12_market_dispatch(dispatch_df, state_summary, fig_dir)

    if cold_loads_df is not None and "hourly_meat_kg" in cold_loads_df.columns:
        _fig13_breakeven_loss(state_summary, cold_loads_df, fig_dir)
    else:
        print("  Fig 13 skipped — cold_loads_df (with hourly_meat_kg) not available")


# ── Figure helpers ──────────────────────────────────────────────────────────

def _fig1_market_distribution(locations: pd.DataFrame, fig_dir: str) -> None:
    """Scatter of all markets coloured by state."""
    fig, ax = plt.subplots(figsize=(12, 8))

    states      = locations["state"].unique()
    cmap        = plt.get_cmap("tab20", len(states))
    state_color = {s: cmap(i) for i, s in enumerate(states)}
    colors      = locations["state"].map(state_color)

    ax.scatter(locations["x"], locations["y"],
               c=colors, s=20, alpha=0.7, linewidths=0)

    patches = [mpatches.Patch(color=state_color[s], label=s)
               for s in sorted(states)]
    ax.legend(handles=patches, title="State",
              loc="lower right", fontsize=7,
              ncol=max(1, len(states) // 10))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("National Market Spatial Distribution — Nigeria 2019")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig1_market_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 1 → {path}")


def _fig2_cost_comparison(
    state_summary: pd.DataFrame,
    standalone_results: Optional[Dict[str, dict]],
    interstate_result: Optional[dict],
    intrastate_results: Optional[Dict[str, dict]],
    fig_dir: str,
) -> None:
    """Horizontal grouped bar chart of costs per state."""
    df = state_summary.sort_values("cost_standalone", ascending=True).copy()
    states  = df["state"].tolist()
    n_states = len(states)
    y = np.arange(n_states)

    configs = [("Config 1 Standalone", df["cost_standalone"].values, "tomato")]
    configs.append(("Config 2 Intrastate", df["cost_intrastate"].values, "darkorange"))

    # Config 3 per state: proportional to C2 cost share (C3_total × C2_state / C2_total).
    if interstate_result is not None:
        c3_per_state = _compute_c3_per_state(
            df, standalone_results, interstate_result, intrastate_results,
        )
        if c3_per_state is not None:
            configs.append(("Config 3 National", c3_per_state, "mediumseagreen"))

    n_configs = len(configs)
    bar_h     = 0.8 / n_configs
    fig_h     = max(8, n_states * 0.4)
    fig, ax   = plt.subplots(figsize=(10, fig_h))

    for i, (label, values, color) in enumerate(configs):
        offset = (i - n_configs / 2 + 0.5) * bar_h
        ax.barh(y + offset, values, height=bar_h, color=color,
                alpha=0.85, label=label)

    ax.set_yticks(y)
    ax.set_yticklabels(states, fontsize=8)
    ax.set_xlabel("Annualised system cost (USD/year)")
    ax.set_title("System Cost by Configuration and State")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig2_cost_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 2 → {path}")


def _compute_c3_per_state(
    df: pd.DataFrame,
    standalone_results: Optional[Dict[str, dict]],
    interstate_result: dict,
    intrastate_results: Optional[Dict[str, dict]],
) -> Optional[np.ndarray]:
    if interstate_result is None:
        return None
    c3_total = float(interstate_result.get(
        "total_cost_incl_spokes",
        interstate_result.get("total_cost", float("nan")),
    ))
    if c3_total != c3_total:
        return None
    if "cost_intrastate" not in df.columns:
        return None
    c2_costs = df.set_index("state")["cost_intrastate"]
    c2_total  = float(c2_costs.sum())
    if c2_total <= 0:
        return None

    # C3 per-state = C3_total × (C2_state / C2_total)
    # Valid because C3_total ≤ C2_total by model design (C3 has more options).
    return np.array([
        c3_total * (float(c2_costs.get(state, 0.0)) / c2_total)
        for state in df["state"]
    ])


def _fig3_benefit_map(
    state_summary: pd.DataFrame,
    locations: pd.DataFrame,
    fig_dir: str,
) -> None:
    """State centroids coloured by C1→C2 saving %."""
    # Compute state centroids
    centroids = (
        locations.groupby("state")[["x", "y"]]
        .mean()
        .reset_index()
        .rename(columns={"x": "lon", "y": "lat"})
    )
    merged = centroids.merge(
        state_summary[["state", "saving_intrastate_pct", "n_markets"]],
        on="state", how="left",
    ).dropna(subset=["saving_intrastate_pct"])

    vmax = merged["saving_intrastate_pct"].max()
    norm = Normalize(vmin=0, vmax=max(vmax, 1e-6))
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        merged["lon"], merged["lat"],
        c=merged["saving_intrastate_pct"],
        s=merged["n_markets"] * 20,
        cmap=cmap, norm=norm,
        alpha=0.85, edgecolors="gray", linewidths=0.4,
    )
    for _, row in merged.iterrows():
        ax.annotate(row["state"], (row["lon"], row["lat"]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Cost saving C1→C2 (%)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Network Benefit Map — Intra-state Coordination Savings")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig3_network_benefit_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 3 → {path}")


def _fig4_transport_modes(
    transport_breakdown: pd.DataFrame,
    state_summary: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Stacked horizontal bar: link counts by mode per state (intrastate only)."""
    MODE_COLORS = {"conv": "darkorange"}

    intra = (
        transport_breakdown[transport_breakdown["scope"] == "intrastate"]
        if not transport_breakdown.empty and "scope" in transport_breakdown.columns
        else pd.DataFrame()
    )

    all_states = state_summary["state"].tolist()
    n_states   = len(all_states)
    fig_h      = max(6, n_states * 0.35)
    fig, ax    = plt.subplots(figsize=(10, fig_h))

    if intra.empty:
        ax.text(0.5, 0.5, "No intrastate transport links built",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        mode_counts = (
            intra.groupby(["state", "mode"])
            .size()
            .unstack(fill_value=0)
            .reindex(all_states, fill_value=0)
        )
        # Ensure all mode columns present
        for mode in MODE_COLORS:
            if mode not in mode_counts.columns:
                mode_counts[mode] = 0

        lefts = np.zeros(n_states)
        y     = np.arange(n_states)
        for mode, color in MODE_COLORS.items():
            vals = mode_counts[mode].values.astype(float)
            ax.barh(y, vals, left=lefts, color=color,
                    alpha=0.85, label=mode.upper())
            lefts += vals

        # Annotate zero-transport states
        zero_states = mode_counts[mode_counts.sum(axis=1) == 0].index.tolist()
        for i, state in enumerate(all_states):
            if state in zero_states:
                ax.text(0.3, i, "No links built", va="center",
                        fontsize=7, color="gray")

        ax.set_yticks(y)
        ax.set_yticklabels(all_states, fontsize=8)

    ax.set_xlabel("Number of built transport links")
    ax.set_title("Transport Mode Selection by State (Config 2 — Intra-state)")
    ax.legend(title="Mode", loc="lower right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig4_transport_modes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 4 → {path}")


def _fig5_technology_mix(technology_mix: pd.DataFrame, fig_dir: str) -> None:
    """Grouped bar: national totals per component × config."""
    COMP_LABELS = {
        "solar_kw"    : "Solar (kWp)",
        "fridge_kw"   : "Fridge (kW)",
        "battery_kwh" : "Battery (kWh)",
        "pcm_kwh"     : "PCM (kWh)",
    }
    CONFIG_COLORS = {"standalone": "tomato", "intrastate": "steelblue"}

    # National totals
    national = (
        technology_mix.groupby(["config", "component"])["total_capacity"]
        .sum()
        .unstack("config")
        .reindex(list(COMP_LABELS.keys()))
    )

    comps    = list(COMP_LABELS.keys())
    n_comps  = len(comps)
    x        = np.arange(n_comps)
    configs  = [c for c in ["standalone", "intrastate"] if c in national.columns]
    bar_w    = 0.35
    offsets  = np.linspace(-(len(configs)-1)*bar_w/2,
                            (len(configs)-1)*bar_w/2, len(configs))

    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg_name, offset in zip(configs, offsets):
        vals = national[cfg_name].values if cfg_name in national.columns \
               else np.zeros(n_comps)
        ax.bar(x + offset, vals, width=bar_w,
               color=CONFIG_COLORS.get(cfg_name, "gray"),
               alpha=0.85, label=cfg_name.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABELS[c] for c in comps])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Total installed capacity (national, symlog scale)")
    ax.set_title("Technology Mix: Standalone vs Intra-state Connected (National Totals)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig5_technology_mix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 5 → {path}")


def _fig6_hub_distribution(
    locations: pd.DataFrame,
    intrastate_results: Optional[Dict[str, dict]],
    hub_capacity_threshold_kw: float,
    fig_dir: str,
) -> None:
    """Scatter of all markets with state convex hulls and labels; hubs in red."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # ── Identify hub market IDs from intrastate capacities ────────────────
    hub_ids: set[str] = set()
    if intrastate_results:
        for state_result in intrastate_results.values():
            cap_df = state_result.get("capacities", pd.DataFrame())
            if cap_df is not None and not cap_df.empty and "fridge_kw" in cap_df.columns:
                hubs = cap_df.index[cap_df["fridge_kw"] > hub_capacity_threshold_kw]
                hub_ids.update(hubs.tolist())

    # ── 1. Convex hulls per state (drawn first, behind markers) ───────────
    for state, grp in locations.groupby("state"):
        pts = grp[["x", "y"]].values
        if len(pts) >= 3:
            try:
                hull     = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly     = MplPolygon(
                    hull_pts, closed=True,
                    fill=False, edgecolor="gray",
                    linewidth=0.8, linestyle="--", alpha=0.5,
                )
                ax.add_patch(poly)
            except Exception:
                pass    # degenerate hull (collinear points) — skip silently
        elif len(pts) == 2:
            ax.plot(pts[:, 0], pts[:, 1],
                    color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # ── 2. Scatter: spokes then hubs ─────────────────────────────────────
    spoke_locs = locations[~locations["market_id"].isin(hub_ids)]
    hub_locs   = locations[locations["market_id"].isin(hub_ids)]

    ax.scatter(spoke_locs["x"], spoke_locs["y"],
               c="lightgray", s=15, alpha=0.6,
               linewidths=0, label=f"Spoke ({len(spoke_locs)})")
    if not hub_locs.empty:
        ax.scatter(hub_locs["x"], hub_locs["y"],
                   c="tomato", s=60, alpha=0.9,
                   linewidths=0.5, edgecolors="darkred",
                   label=f"Hub ({len(hub_locs)})")

    # ── 3. State labels at centroids (drawn last, on top) ─────────────────
    centroids = locations.groupby("state")[["x", "y"]].mean()
    for state, row in centroids.iterrows():
        ax.annotate(
            state, xy=(row["x"], row["y"]),
            fontsize=8, fontweight="bold", ha="center",
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Hub vs Spoke Market Distribution by State (Config 2)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig6_hub_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 6 → {path}")


# ---------------------------------------------------------------------------
# Fig 7–10 helpers
# ---------------------------------------------------------------------------

def _fig7_cost_per_kg(
    state_summary: pd.DataFrame,
    cold_loads_df: pd.DataFrame,
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
    fig_dir: str,
) -> None:
    """Horizontal grouped bar: cooling cost per kg of meat by state."""
    MEAT_PRICE = 3.5   # approximate USD/kg Nigerian beef

    rows = []
    for _, row in state_summary.iterrows():
        state = row["state"]
        annual_meat_kg = float(
            cold_loads_df[cold_loads_df["state"] == state]["hourly_meat_kg"].sum()
        )
        if annual_meat_kg == 0:
            continue
        c1 = standalone_results.get(state, {}).get("total_cost", np.nan)
        c2 = intrastate_results.get(state, {}).get("total_cost", np.nan)
        rows.append({
            "state"          : state,
            "cost_per_kg_c1" : c1 / annual_meat_kg if c1 == c1 else np.nan,
            "cost_per_kg_c2" : c2 / annual_meat_kg if c2 == c2 else np.nan,
        })

    if not rows:
        print("  Fig 7 skipped — no meat data found")
        return

    df      = pd.DataFrame(rows).sort_values("cost_per_kg_c1", ascending=True)
    states  = df["state"].tolist()
    n_states = len(states)
    y       = np.arange(n_states)
    bar_h   = 0.35
    fig_h   = max(8, n_states * 0.4)

    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(y - bar_h / 2, df["cost_per_kg_c1"].values, height=bar_h,
            color="tomato",     alpha=0.85, label="Config 1 Standalone")
    ax.barh(y + bar_h / 2, df["cost_per_kg_c2"].values, height=bar_h,
            color="darkorange", alpha=0.85, label="Config 2 Intrastate")

    ax.axvline(MEAT_PRICE * 0.01, color="green",  linestyle=":", linewidth=1.2,
               label="1% of meat value")
    ax.axvline(MEAT_PRICE * 0.05, color="orange", linestyle=":", linewidth=1.2,
               label="5% of meat value")
    ax.axvline(MEAT_PRICE * 0.10, color="red",    linestyle=":", linewidth=1.2,
               label="10% of meat value")

    max_val = max(df["cost_per_kg_c1"].max(), df["cost_per_kg_c2"].max())
    ax.set_xlim(0, min(max_val * 1.3, 0.06))

    ax.set_yticks(y)
    ax.set_yticklabels(states, fontsize=8)
    ax.set_xlabel("Cooling cost per kg of meat (USD/kg/year)")
    ax.set_title("Cold Chain Affordability by State")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig7_cost_per_kg.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 7 → {path}")


def _fig8_hub_selection(
    locations: pd.DataFrame,
    intrastate_results: Dict[str, dict],
    cold_load_arrays: Dict[str, np.ndarray],
    hub_capacity_threshold_kw: float,
    fig_dir: str,
) -> None:
    """Scatter: mean cold demand vs optimised fridge capacity, hubs highlighted."""
    rows = []
    for state_result in intrastate_results.values():
        cap_df = state_result.get("capacities", pd.DataFrame())
        if cap_df is None or cap_df.empty or "fridge_kw" not in cap_df.columns:
            continue
        for mid, fridge_kw in cap_df["fridge_kw"].items():
            arr = cold_load_arrays.get(str(mid))
            if arr is None:
                continue
            rows.append({
                "market_id"    : str(mid),
                "mean_cold_kw" : float(arr.mean()),
                "fridge_kw"    : float(fridge_kw),
                "is_hub"       : float(fridge_kw) > hub_capacity_threshold_kw,
            })

    if not rows:
        print("  Fig 8 skipped — no capacity data found")
        return

    df      = pd.DataFrame(rows)
    hub_df  = df[df["is_hub"]]
    spoke_df = df[~df["is_hub"]]
    n_hubs  = len(hub_df)
    n_total = len(df)
    hub_pct = n_hubs / n_total * 100 if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(spoke_df["mean_cold_kw"], spoke_df["fridge_kw"],
               c="lightgray", s=40, alpha=0.6, linewidths=0, label="Spoke")
    ax.scatter(hub_df["mean_cold_kw"],   hub_df["fridge_kw"],
               c="tomato",    s=40, alpha=0.85, linewidths=0.5,
               edgecolors="darkred", label="Hub")

    ax.text(0.98, 0.98,
            f"Hubs: {n_hubs}/{n_total} markets ({hub_pct:.0f}%)\n"
            f"(hub = fridge capacity > {hub_capacity_threshold_kw:.1f} kW)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("Mean cold demand (kW)")
    ax.set_ylabel("Optimised fridge capacity (kW)")
    ax.set_title("Hub Selection vs Market Demand (Config 2)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig8_hub_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 8 → {path}")


def _fig9_network_drivers(
    state_summary: pd.DataFrame,
    locations: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Scatter: mean inter-market distance vs C1→C2 saving %, with trend line."""
    rows = []
    for _, row in state_summary.dropna(subset=["saving_intrastate_pct"]).iterrows():
        state    = row["state"]
        state_locs = locations[locations["state"] == state][["x", "y"]].values
        n = len(state_locs)
        if n < 2:
            avg_dist = 0.0
        else:
            dists = [
                math.sqrt((state_locs[i, 0] - state_locs[j, 0]) ** 2 +
                          (state_locs[i, 1] - state_locs[j, 1]) ** 2)
                for i in range(n) for j in range(i + 1, n)
            ]
            avg_dist = float(np.mean(dists))
        rows.append({
            "state"      : state,
            "avg_dist"   : avg_dist,
            "saving_pct" : float(row["saving_intrastate_pct"]),
            "n_markets"  : int(row["n_markets"]),
        })

    if not rows:
        print("  Fig 9 skipped — insufficient data")
        return

    df   = pd.DataFrame(rows)
    x    = df["avg_dist"].values
    y    = df["saving_pct"].values
    sz   = df["n_markets"].values * 3

    norm = Normalize(vmin=y.min(), vmax=y.max())
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, y, s=sz, c=y, cmap=cmap, norm=norm,
                    alpha=0.85, edgecolors="gray", linewidths=0.4)

    for _, r in df.iterrows():
        ax.annotate(r["state"], (r["avg_dist"], r["saving_pct"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    # Trend line
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7)
        # Pearson r
        r_val = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.97, f"Pearson r = {r_val:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                color="steelblue")

    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04,
                 label="Cost saving C1→C2 (%)")
    ax.set_xlabel("Mean inter-market distance (decimal degrees)")
    ax.set_ylabel("Cost saving C1→C2 (%)")
    ax.set_title("Network Benefit vs Market Spatial Structure")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig9_network_drivers.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 9 → {path}")


def _fig10_cold_flow_network(
    locations: pd.DataFrame,
    intrastate_results: Dict[str, dict],
    transport_breakdown: pd.DataFrame,
    hub_capacity_threshold_kw: float,
    fig_dir: str,
) -> None:
    """Small-multiples grid: one subplot per state showing cold-energy flow arrows."""
    intra = transport_breakdown[
        (transport_breakdown["scope"] == "intrastate") &
        (transport_breakdown["capacity_kw"] > 0)
    ] if not transport_breakdown.empty else pd.DataFrame()

    # Identify hub market IDs
    hub_ids: set[str] = set()
    for state_result in intrastate_results.values():
        cap_df = state_result.get("capacities", pd.DataFrame())
        if cap_df is not None and not cap_df.empty and "fridge_kw" in cap_df.columns:
            hub_ids.update(
                cap_df.index[cap_df["fridge_kw"] > hub_capacity_threshold_kw]
                      .astype(str).tolist()
            )

    # Build coordinate lookup
    coord = locations.set_index("market_id")[["x", "y"]].to_dict("index")

    # Global max capacity for normalising arrow widths
    max_cap = float(intra["capacity_kw"].max()) if not intra.empty else 1.0
    if max_cap == 0:
        max_cap = 1.0

    states   = sorted(locations["state"].unique())
    n_states = len(states)
    n_cols   = max(1, math.ceil(math.sqrt(n_states)))
    n_rows   = math.ceil(n_states / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes_flat = np.array(axes).flatten() if n_states > 1 else [axes]

    for idx, state in enumerate(states):
        ax = axes_flat[idx]
        state_locs = locations[locations["state"] == state]

        # Gray circles for all markets
        ax.scatter(state_locs["x"], state_locs["y"],
                   c="lightgray", s=20, zorder=2, linewidths=0)

        # Red circles for hubs (larger, more prominent)
        hub_locs  = state_locs[state_locs["market_id"].isin(hub_ids)]
        n_hubs_st = len(hub_locs)
        n_total_st = len(state_locs)
        if not hub_locs.empty:
            ax.scatter(hub_locs["x"], hub_locs["y"],
                       c="tomato", s=120, zorder=3,
                       linewidths=0.5, edgecolors="darkred")

        # Transport arrows
        state_links = intra[intra["state"] == state] if not intra.empty \
                      else pd.DataFrame()
        for _, link in state_links.iterrows():
            src = coord.get(str(link["from_market"]))
            dst = coord.get(str(link["to_market"]))
            if src is None or dst is None:
                continue
            lw = max(0.4, link["capacity_kw"] / max_cap * 2.5)
            ax.annotate(
                "", xy=(dst["x"], dst["y"]),
                xytext=(src["x"], src["y"]),
                arrowprops=dict(arrowstyle="->", color="steelblue",
                                lw=lw, alpha=0.7),
                zorder=4,
            )

        ax.text(0.02, 0.98, f"Hubs: {n_hubs_st}/{n_total_st}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_title(state, fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_states, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Cold Energy Flow Networks by State (Config 2)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig10_cold_flow_network.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 10 → {path}")


def _fig11_interstate_flow(
    locations: pd.DataFrame,
    transport_breakdown: pd.DataFrame,
    fig_dir: str,
) -> None:
    """National map: markets with interstate flow (sized by total flow volume) + inter-state cold flow lines."""
    # Coordinate lookup
    coord = locations.set_index("market_id")[["x", "y"]].to_dict("index")

    # Interstate links
    inter = (
        transport_breakdown[
            (transport_breakdown["scope"] == "interstate") &
            (transport_breakdown["capacity_kw"] > 0)
        ]
        if not transport_breakdown.empty and "scope" in transport_breakdown.columns
        else pd.DataFrame()
    )
    max_cap = float(inter["capacity_kw"].max()) if not inter.empty else 1.0

    fig, ax = plt.subplots(figsize=(12, 8))

    # All markets — gray background
    ax.scatter(locations["x"], locations["y"],
               s=8, color="lightgray", zorder=1, linewidths=0)

    # Markets with nonzero interstate flow — sized by total flow volume
    if not inter.empty:
        flow_by_market: dict[str, float] = {}
        for _, link in inter.iterrows():
            flow_by_market[str(link["from_market"])] = (
                flow_by_market.get(str(link["from_market"]), 0.0)
                + float(link["capacity_kw"])
            )
            flow_by_market[str(link["to_market"])] = (
                flow_by_market.get(str(link["to_market"]), 0.0)
                + float(link["capacity_kw"])
            )
        active_locs = locations[
            locations["market_id"].astype(str).isin(flow_by_market.keys())
        ].copy()
        sizes = active_locs["market_id"].astype(str).map(flow_by_market)
        ax.scatter(active_locs["x"], active_locs["y"],
                   s=sizes * 2 + 15, color="tomato",
                   zorder=3, alpha=0.8, linewidths=0.5, edgecolors="darkred")

    # Inter-state flow lines
    if not inter.empty:
        for _, link in inter.iterrows():
            src = coord.get(str(link["from_market"]))
            dst = coord.get(str(link["to_market"]))
            if src is None or dst is None:
                continue
            lw = link["capacity_kw"] / max_cap * 3 + 0.5
            ax.plot([src["x"], dst["x"]], [src["y"], dst["y"]],
                    color="steelblue", alpha=0.6, lw=lw, zorder=2)
    else:
        ax.text(0.5, 0.02,
                "No inter-state links built",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="dimgray",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # State centroid labels
    centroids = locations.groupby("state")[["x", "y"]].mean()
    for state, row in centroids.iterrows():
        ax.annotate(state, xy=(row["x"], row["y"]),
                    fontsize=7, ha="center", color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              alpha=0.5, ec="none"))

    # Legend
    legend_handles = [
        mpatches.Patch(color="lightgray", label="All markets"),
        mpatches.Patch(color="tomato",
                       label="Markets with interstate flow (size ∝ total flow kW)"),
        plt.Line2D([0], [0], color="steelblue", lw=2, label="Inter-state cold flow"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Inter-state Cold Energy Exchange Network — Full National Mesh (Config 3)")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig11_interstate_flow.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 11 → {path}")


def _market_capacity_composite(cap_row: pd.Series, config) -> float:
    """
    Single comparable size metric per market, combining solar/fridge/
    battery/PCM capacity using the same annualised capex rates used
    elsewhere in the model — purely a visualisation convenience for
    comparing "how big is this market's system" on one scale, NOT an
    allocation of shared/joint cost (this is a real, direct per-market
    output for both Config 1 and Config 2 — no attribution ambiguity).
    """
    return (
        float(cap_row.get("solar_kw", 0.0))      * config.pv_annualised_capex_per_kw
        + float(cap_row.get("fridge_kw", 0.0))   * config.fridge_annualised_capex_per_kw_electric
        + float(cap_row.get("battery_kwh", 0.0)) * config.battery_annualised_capex_per_kwh
        + float(cap_row.get("pcm_kwh", 0.0))     * config.pcm_annualised_capex_per_kwh
        + float(cap_row.get("battery_power_kw", 0.0)) * config.battery_annualised_power_capex_per_kw
    )


def _fig18_capacity_change_map(
    locations: pd.DataFrame,
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
    config,
    fig_dir: str,
) -> None:
    """
    Map: circle size = actual connected (Config 2) capacity per market,
    circle color = % change vs. standalone (Config 1) — shows that
    connectivity causes specialisation (some markets grow into net
    exporters, others shrink and rely on imports), not just uniform
    capacity reduction.
    """
    rows: list[dict] = []
    for state, s_res in standalone_results.items():
        i_res = intrastate_results.get(state, {})
        c1_cap = s_res.get("capacities", pd.DataFrame())
        c2_cap = i_res.get("capacities", pd.DataFrame())
        if c1_cap is None or c1_cap.empty or c2_cap is None or c2_cap.empty:
            continue
        for mid in c1_cap.index:
            if mid not in c2_cap.index:
                continue
            size1 = _market_capacity_composite(c1_cap.loc[mid], config)
            size2 = _market_capacity_composite(c2_cap.loc[mid], config)
            if size1 <= 0:
                continue
            rows.append({
                "market_id": str(mid),
                "state": state,
                "size_standalone": size1,
                "size_connected": size2,
                "pct_change": (size2 - size1) / size1 * 100,
            })

    if not rows:
        print("  Fig 18 skipped — no matching capacity data")
        return

    df = pd.DataFrame(rows).merge(
        locations[["market_id", "x", "y", "state"]].astype({"market_id": str}),
        on=["market_id", "state"], how="left",
    )
    df = df.dropna(subset=["x", "y"])

    fig, ax = plt.subplots(figsize=(12, 9))

    # State convex hulls + labels (same style as fig6/fig11)
    for state, grp in df.groupby("state"):
        pts = grp[["x", "y"]].dropna().values
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                poly = MplPolygon(pts[hull.vertices], closed=True, fill=False,
                                  edgecolor="gray", linewidth=0.8,
                                  linestyle="--", alpha=0.5)
                ax.add_patch(poly)
            except Exception:
                pass
        if len(pts) > 0:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.annotate(state, xy=(cx, cy), fontsize=8, fontweight="bold",
                       ha="center", color="dimgray",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                 alpha=0.6, ec="none"))

    # Percentile-based symmetric color clipping so a single outlier
    # market doesn't wash out contrast for everyone else at national scale
    p_lo, p_hi = np.percentile(df["pct_change"], [2, 98])
    vmax = max(abs(p_lo), abs(p_hi))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = matplotlib.colormaps["RdBu_r"]

    size_scale = np.sqrt(df["size_connected"] / df["size_connected"].max())
    sizes = 30 + size_scale * 300

    sc = ax.scatter(df["x"], df["y"], s=sizes, c=df["pct_change"].clip(-vmax, vmax),
                    cmap=cmap, norm=norm, alpha=0.85,
                    edgecolors="black", linewidths=0.5, zorder=3)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("\u0394 capacity vs. standalone (%)")
    ax.text(0.01, 0.01,
            "red = grew (net exporter)   blue = shrank (relies on imports)",
            transform=ax.transAxes, fontsize=8, color="dimgray", va="bottom")

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Effect of Network Connectivity on Local Installed Capacity\n"
                 "(circle size = actual connected capacity)")
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig18_capacity_change_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 18 → {path}")


def _fig19_state_savings_choropleth(state_summary: pd.DataFrame, fig_dir: str) -> None:
    """C1→C2 cost savings % by state, real Nigeria shapefile choropleth."""
    if not os.path.exists(NIGERIA_STATES_GEOJSON):
        print(f"  Fig 19 skipped — {NIGERIA_STATES_GEOJSON} not found")
        return

    values = dict(zip(state_summary["state"], state_summary["saving_intrastate_pct"]))
    vmax = max(70.0, float(state_summary["saving_intrastate_pct"].max()))

    fig, ax = plt.subplots(figsize=(9, 10))
    _plot_nigeria_choropleth_base(ax)
    sm = _plot_state_bubbles(ax, values, vmin=0, vmax=vmax)

    ax.set_xlim(2.5, 15); ax.set_ylim(4, 14); ax.set_axis_off()
    ax.set_title("C1\u2192C2 Savings Rate (%) by State", fontsize=17,
                fontweight="bold", loc="left", pad=14)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.4, 0.02])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").set_label(
        "C1\u2192C2 savings (%)", fontsize=9)
    ax.scatter([], [], s=250, color="#4f8fc0", edgecolor="#333333",
              linewidth=1.5, label="Solved")
    ax.scatter([], [], s=90, color="#ece4d0", edgecolor="#999999",
              label="No data yet")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig19_state_savings_choropleth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 19 \u2192 {path}")


def _fig20_state_capacity_choropleth(
    standalone_results: Dict[str, dict],
    interstate_result: dict,
    config,
    fig_dir: str,
) -> None:
    """
    C1→C3 capacity reduction % by state. Uses capacity (a real, direct
    per-market output under both Config 1 and Config 3), NOT cost —
    Config 3's cost is a single national LP total with no unambiguous
    per-state split, unlike capacity.
    """
    if not os.path.exists(NIGERIA_STATES_GEOJSON):
        print(f"  Fig 20 skipped — {NIGERIA_STATES_GEOJSON} not found")
        return

    c3_cap = interstate_result.get("capacities", pd.DataFrame())
    values: Dict[str, float] = {}
    for state, s_res in standalone_results.items():
        c1_cap = s_res.get("capacities", pd.DataFrame())
        if c1_cap is None or c1_cap.empty or c3_cap is None or c3_cap.empty:
            continue
        total1 = total3 = 0.0
        for mid in c1_cap.index:
            if mid not in c3_cap.index:
                continue
            total1 += _market_capacity_composite(c1_cap.loc[mid], config)
            total3 += _market_capacity_composite(c3_cap.loc[mid], config)
        if total1 > 0:
            values[state] = (total1 - total3) / total1 * 100

    if not values:
        print("  Fig 20 skipped — no matching capacity data")
        return

    vmax = max(70.0, max(values.values()))
    fig, ax = plt.subplots(figsize=(9, 10))
    _plot_nigeria_choropleth_base(ax)
    sm = _plot_state_bubbles(ax, values, vmin=0, vmax=vmax,
                             cmap_colors=("#f5ecd7", "#7fb08a", "#1a5c2e"))

    ax.set_xlim(2.5, 15); ax.set_ylim(4, 14); ax.set_axis_off()
    ax.set_title("C1\u2192C3 Capacity Reduction (%) by State", fontsize=17,
                fontweight="bold", loc="left", pad=14)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.4, 0.02])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").set_label(
        "C1\u2192C3 capacity reduction (%)", fontsize=9)
    ax.scatter([], [], s=250, color="#7fb08a", edgecolor="#333333",
              linewidth=1.5, label="Solved")
    ax.scatter([], [], s=90, color="#ece4d0", edgecolor="#999999",
              label="No data yet")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig20_state_capacity_choropleth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 20 \u2192 {path}")


def _fig21_state_c1c3_savings_choropleth(
    standalone_results: Dict[str, dict],
    interstate_result: dict,
    config,
    fig_dir: str,
) -> None:
    """
    C1→C3 capex saving rate (%) by state.

    Uses _market_capacity_composite (annualised capex proxy) to estimate a
    per-state C1 and C3 cost share, because Config 3's LP objective is a
    single national total with no unambiguous per-state cost split.  The
    composite weights solar/fridge/battery/PCM kW by their annualised capex
    rates, giving a financially-grounded proxy for investment cost.
    """
    if not os.path.exists(NIGERIA_STATES_GEOJSON):
        print(f"  Fig 21 skipped — {NIGERIA_STATES_GEOJSON} not found")
        return

    c3_cap = interstate_result.get("capacities", pd.DataFrame())
    values: Dict[str, float] = {}
    for state, s_res in standalone_results.items():
        c1_cap = s_res.get("capacities", pd.DataFrame())
        if c1_cap is None or c1_cap.empty or c3_cap is None or c3_cap.empty:
            continue
        total1 = total3 = 0.0
        for mid in c1_cap.index:
            if mid not in c3_cap.index:
                continue
            total1 += _market_capacity_composite(c1_cap.loc[mid], config)
            total3 += _market_capacity_composite(c3_cap.loc[mid], config)
        if total1 > 0:
            values[state] = (total1 - total3) / total1 * 100

    if not values:
        print("  Fig 21 skipped — no matching capacity data")
        return

    vmax = max(70.0, max(values.values()))
    fig, ax = plt.subplots(figsize=(9, 10))
    _plot_nigeria_choropleth_base(ax)
    sm = _plot_state_bubbles(ax, values, vmin=0, vmax=vmax)

    ax.set_xlim(2.5, 15); ax.set_ylim(4, 14); ax.set_axis_off()
    ax.set_title("C1\u2192C3 Saving Rate (%) by State", fontsize=17,
                fontweight="bold", loc="left", pad=14)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.4, 0.02])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").set_label(
        "C1\u2192C3 saving rate (%, capex proxy)", fontsize=9)
    ax.scatter([], [], s=250, color="#4f8fc0", edgecolor="#333333",
              linewidth=1.5, label="Solved")
    ax.scatter([], [], s=90, color="#ece4d0", edgecolor="#999999",
              label="No data yet")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig21_state_c1c3_savings_choropleth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 21 \u2192 {path}")


def _fig12_market_dispatch(
    dispatch_df: pd.DataFrame,
    state_summary: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Stacked-area within-market dispatch for hub and spoke over one week."""
    if dispatch_df is None or dispatch_df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "No dispatch data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("Within-Market Energy Dispatch: Highest- vs Lowest-Capacity Market (Config 2)")
        plt.tight_layout()
        path = os.path.join(fig_dir, "fig12_market_dispatch.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Fig 12 → {path}")
        return

    # ── Pick state with highest C1→C2 saving ─────────────────────────────
    ranked = state_summary.dropna(subset=["saving_intrastate_pct"])
    best_state = ranked.iloc[0]["state"] if not ranked.empty else None
    state_dispatch = (
        dispatch_df[dispatch_df["state"] == best_state]
        if best_state and best_state in dispatch_df["state"].values
        else dispatch_df
    )

    hub_data   = state_dispatch[state_dispatch["role"] == "hub"]
    spoke_data = state_dispatch[state_dispatch["role"] == "spoke"]

    panels = []
    if not hub_data.empty:
        panels.append(("hub",   hub_data.iloc[0]["market_id"],
                       hub_data.sort_values("timestamp").head(168)))
    if not spoke_data.empty:
        panels.append(("spoke", spoke_data.iloc[0]["market_id"],
                       spoke_data.sort_values("timestamp").head(168)))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), squeeze=False)

    for ax_row, (role, mid, week_df) in zip(axes, panels):
        ax  = ax_row[0]
        hrs = np.arange(len(week_df))

        fridge    = week_df["fridge_kw"].values
        pcm_dis   = week_df["pcm_discharge_kw"].values
        imp       = week_df["import_kw"].values
        exp       = week_df["export_kw"].values
        demand    = week_df["cold_demand_kw"].values
        solar     = week_df["solar_kw"].values
        pcm_soc   = week_df["pcm_soc_kwh"].values

        # Stacked cold supply areas
        ax.fill_between(hrs, 0, fridge,
                        label="Fridge output",  color="steelblue",    alpha=0.7)
        ax.fill_between(hrs, fridge, fridge + pcm_dis,
                        label="PCM discharge",  color="lightblue",    alpha=0.7)
        ax.fill_between(hrs, fridge + pcm_dis, fridge + pcm_dis + imp,
                        label="Cold import",    color="mediumseagreen", alpha=0.7)

        # Cold demand and solar
        ax.plot(hrs, demand, color="black",  lw=1.5, linestyle="--", label="Cold demand")
        ax.plot(hrs, solar,  color="goldenrod", lw=1.2, linestyle="-.", label="Solar gen", alpha=0.8)

        # Export as negative area
        ax.fill_between(hrs, -exp, 0,
                        label="Cold export", color="orange", alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)

        # PCM state of charge on secondary axis
        ax2 = ax.twinx()
        ax2.plot(hrs, pcm_soc, color="purple", lw=1, linestyle=":", alpha=0.6)
        ax2.set_ylabel("PCM SoC (kWh)", color="purple", fontsize=8)
        ax2.tick_params(axis="y", colors="purple", labelsize=7)

        state_lbl = week_df["state"].iloc[0]
        role_lbl  = (f"Highest-capacity market {mid}" if role == "hub"
                     else f"Lowest-capacity market {mid}")
        ax.set_title(f"{role_lbl} — {state_lbl}", fontsize=10)
        ax.set_xlabel("Hour of representative week")
        ax.set_ylabel("Cold power (kW)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Within-Market Energy Dispatch: Highest- vs Lowest-Capacity Market (Config 2)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig12_market_dispatch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 12 → {path}")


def _fig13_breakeven_loss(
    state_summary: pd.DataFrame,
    cold_loads_df: pd.DataFrame,
    fig_dir: str,
    meat_price_usd: float = 3.5,
    national_loss_rate_pct: float = 40.0,
) -> None:
    """
    Break-even meat loss reduction needed to justify cold chain investment.

    For each state, computes the minimum % of meat spoilage that must be
    prevented to recover the annual system cost.  Compares against the
    documented national post-harvest loss rate of 40 %.
    """
    # ── Step 1 & 2: Annual meat throughput per state, merged ─────────────
    annual_meat = (
        cold_loads_df.groupby("state")["hourly_meat_kg"]
        .sum()
        .reset_index()
        .rename(columns={"hourly_meat_kg": "annual_meat_kg"})
    )
    df = state_summary.merge(annual_meat, on="state", how="left")

    # ── Step 3 & 4: Break-even percentages ───────────────────────────────
    df["meat_value_usd"]  = df["annual_meat_kg"] * meat_price_usd
    df["breakeven_pct"]   = df["cost_intrastate"]  / df["meat_value_usd"] * 100
    df["breakeven_pct_c1"] = df["cost_standalone"] / df["meat_value_usd"] * 100

    # Drop rows where meat value is zero or missing to avoid div/0 artefacts
    df = df[df["meat_value_usd"] > 0].copy()

    # ── Step 5: Sort ─────────────────────────────────────────────────────
    df = df.sort_values("breakeven_pct", ascending=True).reset_index(drop=True)

    # ── Step 6: Plot ─────────────────────────────────────────────────────
    n_states = len(df)
    fig_h    = max(8, n_states * 0.45)
    fig, ax  = plt.subplots(figsize=(12, fig_h))

    bar_h  = 0.35
    y      = np.arange(n_states)

    ax.barh(y + bar_h / 2, df["breakeven_pct_c1"], height=bar_h,
            color="tomato", alpha=0.65, label="Config 1 Standalone")
    ax.barh(y - bar_h / 2, df["breakeven_pct"],   height=bar_h,
            color="steelblue", alpha=0.85, label="Config 2 Intrastate")

    ax.set_yticks(y)
    ax.set_yticklabels(df["state"], fontsize=9)

    # ── Step 7: Reference line + shaded zone ─────────────────────────────
    ax.axvline(national_loss_rate_pct, color="darkred", linewidth=2,
               linestyle="--",
               label=f"Documented loss rate ({national_loss_rate_pct:.0f}%)")
    ax.axvspan(0, national_loss_rate_pct, alpha=0.06, color="green",
               label="Economically justified zone")

    # ── Step 8: Annotation ───────────────────────────────────────────────
    ylim = ax.get_ylim()
    ax.text(national_loss_rate_pct * 0.5, ylim[1] * 0.98,
            "All states below this line\nare economically justified",
            ha="center", va="top", fontsize=8, color="darkgreen")

    # ── Step 9: Labels ────────────────────────────────────────────────────
    ax.set_xlabel(
        "Minimum meat loss reduction required to break even (%)",
        fontsize=11)
    ax.set_ylabel("State", fontsize=11)
    ax.set_title(
        "Cold Chain Economic Justification:\n"
        "Break-even Meat Loss Reduction vs Documented Loss Rate",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=0)

    # ── Step 10: Stats text box ───────────────────────────────────────────
    national_avg_c2 = float(df["breakeven_pct"].mean())
    safety_margin   = national_loss_rate_pct - national_avg_c2
    ax.text(0.98, 0.02,
            f"Average break-even (C2): {national_avg_c2:.1f}%\n"
            f"Documented loss rate: {national_loss_rate_pct:.0f}%\n"
            f"Safety margin: {safety_margin:.1f} pp",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(fig_dir, "fig13_breakeven_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Fig 13 → {path}")


# ---------------------------------------------------------------------------
# compile_cost_breakdown
# ---------------------------------------------------------------------------

def compile_cost_breakdown(
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
    interstate_result: dict,
    state_summary: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Wide-format table explaining what drives cost changes across C1, C2, C3.

    Each row is one metric; columns are C1, C2, C3, C1→C2 absolute change,
    C1→C3 absolute change, C1→C2 % change, C1→C3 % change.

    Metrics
    -------
    Capacity (physical units):
        solar_kw_total, fridge_kw_total, battery_kwh_total,
        battery_power_kw_total, pcm_kwh_total,
        raw_positive_transport_links, meaningfully_active_transport_links,
        transport_capacity_kw_total, transport_flow_kwh_total

    Annualised costs (USD/yr):
        solar_capex_usd, fridge_capex_usd, battery_energy_capex_usd,
        battery_inverter_power_capex_usd, pcm_capex_usd,
        equipment_capex_total_usd,
        transport_capex_usd, transport_fuel_opex_usd,
        total_cost_computed_usd, total_cost_lp_usd

    transport_fuel_opex_usd is computed as the residual
    (total_cost_lp - equipment_capex - transport_capex) so it captures
    everything inside the LP objective that is not accounted for by direct
    capacity costs — primarily distance-weighted diesel fuel marginal costs.
    """
    cfg = config

    def _cap_totals(results_by_state: Dict[str, dict]) -> dict:
        solar = fridge = battery = pcm = battery_power = 0.0
        for r in results_by_state.values():
            cap = r.get("capacities", pd.DataFrame())
            if cap is None or cap.empty:
                continue
            solar   += float(cap.get("solar_kw",   pd.Series(0)).sum())
            fridge  += float(cap.get("fridge_kw",  pd.Series(0)).sum())
            battery += float(cap.get("battery_kwh", pd.Series(0)).sum())
            pcm     += float(cap.get("pcm_kwh",     pd.Series(0)).sum())
            battery_power += float(cap.get("battery_power_kw", pd.Series(0)).sum())
        return dict(solar=solar, fridge=fridge, battery=battery, pcm=pcm,
                    battery_power=battery_power)

    def _cap_totals_from_df(cap_df: pd.DataFrame) -> dict:
        if cap_df is None or cap_df.empty:
            return dict(solar=0.0, fridge=0.0, battery=0.0, pcm=0.0,
                        battery_power=0.0)
        return dict(
            solar   = float(cap_df.get("solar_kw",   pd.Series(0)).sum()),
            fridge  = float(cap_df.get("fridge_kw",  pd.Series(0)).sum()),
            battery = float(cap_df.get("battery_kwh", pd.Series(0)).sum()),
            pcm     = float(cap_df.get("pcm_kwh",     pd.Series(0)).sum()),
            battery_power = float(cap_df.get("battery_power_kw", pd.Series(0)).sum()),
        )

    def _transport_totals(transport_df: Optional[pd.DataFrame]) -> dict:
        if transport_df is None or transport_df.empty:
            return dict(raw_links=0, meaningful_links=0, capacity_kw=0.0,
                        fuel_opex=0.0, flow_kwh=0.0)
        if "fuel_opex_usd" in transport_df.columns:
            fuel_opex = float(transport_df["fuel_opex_usd"].sum())
        elif "marginal_cost_per_kwh" in transport_df.columns:
            fuel_opex = float((transport_df["marginal_cost_per_kwh"]
                               * transport_df["total_flow_kwh"]).sum())
        else:
            print("  WARNING: transport_df has no fuel_opex_usd or marginal_cost_per_kwh "
                  "— delete checkpoints and re-run to get accurate transport fuel costs")
            fuel_opex = 0.0
        capacity = transport_df["capacity_kw"]
        flow = transport_df["total_flow_kwh"]
        if "fuel_opex_usd" in transport_df.columns:
            fuel_by_link = transport_df["fuel_opex_usd"]
        else:
            fuel_by_link = transport_df["marginal_cost_per_kwh"] * flow
        raw_positive = capacity > 0.0
        meaningful = (
            (capacity > TRANSPORT_CAPACITY_REPORTING_TOLERANCE_KW)
            & (flow > 0.0)
            & (fuel_by_link > 0.0)
        )
        return dict(
            raw_links       = int(raw_positive.sum()),
            meaningful_links= int(meaningful.sum()),
            capacity_kw     = float(capacity.sum()),
            fuel_opex       = fuel_opex,
            flow_kwh        = float(flow.sum()),
        )

    def _equipment_capex(caps: dict) -> float:
        return (
            caps["solar"]   * cfg.pv_annualised_capex_per_kw
            + caps["fridge"]  * cfg.fridge_annualised_capex_per_kw_electric
            + caps["battery"] * cfg.battery_annualised_capex_per_kwh
            + caps["pcm"]     * cfg.pcm_annualised_capex_per_kwh
            + caps["battery_power"] * cfg.battery_annualised_power_capex_per_kw
        )

    # ── C1 ────────────────────────────────────────────────────────────────
    c1_caps  = _cap_totals(standalone_results)
    c1_trans = dict(raw_links=0, meaningful_links=0, capacity_kw=0.0,
                    fuel_opex=0.0, flow_kwh=0.0)
    c1_lp    = float(state_summary["cost_standalone"].sum())
    c1_eq_capex = _equipment_capex(c1_caps)

    # ── C2 ────────────────────────────────────────────────────────────────
    c2_caps  = _cap_totals(intrastate_results)
    c2_trans_parts = [
        r["transport"]
        for r in intrastate_results.values()
        if r.get("transport") is not None
        and not r["transport"].empty
    ]
    c2_trans_df = pd.concat(c2_trans_parts, ignore_index=True) if c2_trans_parts else None
    c2_trans = _transport_totals(c2_trans_df)
    c2_lp    = float(state_summary["cost_intrastate"].sum())
    c2_eq_capex = _equipment_capex(c2_caps)

    # ── C3 ────────────────────────────────────────────────────────────────
    c3_caps  = _cap_totals_from_df(interstate_result.get("capacities", pd.DataFrame()))
    c3_trans = _transport_totals(interstate_result.get("transport", None))
    c3_lp    = float(interstate_result.get(
        "total_cost_incl_spokes",
        interstate_result.get("total_cost", float("nan")),
    ))
    c3_eq_capex = _equipment_capex(c3_caps)

    # Fuel opex: use direct sum from transport summary where available;
    # fall back to LP-residual (total_cost - equipment_capex) to capture
    # any rounding or untracked marginal costs.
    c1_fuel = c1_trans["fuel_opex"]
    c2_fuel = c2_trans["fuel_opex"] if c2_trans["fuel_opex"] > 0 else c2_lp - c2_eq_capex
    c3_fuel = c3_trans["fuel_opex"] if c3_trans["fuel_opex"] > 0 else c3_lp - c3_eq_capex

    # ── Assemble rows ─────────────────────────────────────────────────────
    def _row(metric: str, c1: float, c2: float, c3: float) -> dict:
        def _pct(base, new):
            return (new - base) / abs(base) * 100 if base and not np.isnan(base) else np.nan
        return {
            "metric"              : metric,
            "C1"                  : c1,
            "C2"                  : c2,
            "C3"                  : c3,
            "C1_to_C2_change"     : c2 - c1,
            "C1_to_C3_change"     : c3 - c1,
            "C1_to_C2_pct_change" : _pct(c1, c2),
            "C1_to_C3_pct_change" : _pct(c1, c3),
        }

    rows = [
        # -- Physical capacities -------------------------------------------
        _row("solar_kw_total",
             c1_caps["solar"],    c2_caps["solar"],    c3_caps["solar"]),
        _row("fridge_kw_total",
             c1_caps["fridge"],   c2_caps["fridge"],   c3_caps["fridge"]),
        _row("battery_kwh_total",
             c1_caps["battery"],  c2_caps["battery"],  c3_caps["battery"]),
        _row("battery_power_kw_total",
             c1_caps["battery_power"], c2_caps["battery_power"],
             c3_caps["battery_power"]),
        _row("pcm_kwh_total",
             c1_caps["pcm"],      c2_caps["pcm"],      c3_caps["pcm"]),
        _row("raw_positive_transport_links",
             c1_trans["raw_links"], c2_trans["raw_links"],
             c3_trans["raw_links"]),
        _row("meaningfully_active_transport_links",
             c1_trans["meaningful_links"], c2_trans["meaningful_links"],
             c3_trans["meaningful_links"]),
        _row("transport_capacity_kw_total",
             c1_trans["capacity_kw"], c2_trans["capacity_kw"], c3_trans["capacity_kw"]),
        _row("transport_flow_kwh_total",
             c1_trans["flow_kwh"],    c2_trans["flow_kwh"],    c3_trans["flow_kwh"]),
        # -- Annualised costs (USD/yr) -------------------------------------
        _row("solar_capex_usd",
             c1_caps["solar"]    * cfg.pv_annualised_capex_per_kw,
             c2_caps["solar"]    * cfg.pv_annualised_capex_per_kw,
             c3_caps["solar"]    * cfg.pv_annualised_capex_per_kw),
        _row("fridge_capex_usd",
             c1_caps["fridge"]   * cfg.fridge_annualised_capex_per_kw_electric,
             c2_caps["fridge"]   * cfg.fridge_annualised_capex_per_kw_electric,
             c3_caps["fridge"]   * cfg.fridge_annualised_capex_per_kw_electric),
        _row("battery_energy_capex_usd",
             c1_caps["battery"]  * cfg.battery_annualised_capex_per_kwh,
             c2_caps["battery"]  * cfg.battery_annualised_capex_per_kwh,
             c3_caps["battery"]  * cfg.battery_annualised_capex_per_kwh),
        _row("battery_inverter_power_capex_usd",
             c1_caps["battery_power"] * cfg.battery_annualised_power_capex_per_kw,
             c2_caps["battery_power"] * cfg.battery_annualised_power_capex_per_kw,
             c3_caps["battery_power"] * cfg.battery_annualised_power_capex_per_kw),
        _row("pcm_capex_usd",
             c1_caps["pcm"]      * cfg.pcm_annualised_capex_per_kwh,
             c2_caps["pcm"]      * cfg.pcm_annualised_capex_per_kwh,
             c3_caps["pcm"]      * cfg.pcm_annualised_capex_per_kwh),
        _row("equipment_capex_total_usd",
             c1_eq_capex,  c2_eq_capex,  c3_eq_capex),
        _row("transport_fuel_opex_usd",
             c1_fuel,  c2_fuel,  c3_fuel),
        _row("total_cost_computed_usd",
             c1_eq_capex + c1_fuel,
             c2_eq_capex + c2_fuel,
             c3_eq_capex + c3_fuel),
        _row("total_cost_lp_usd",
             c1_lp,  c2_lp,  c3_lp),
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compile_market_cost_breakdown
# ---------------------------------------------------------------------------

def compile_market_cost_breakdown(
    standalone_results: Dict[str, dict],
    intrastate_results: Dict[str, dict],
    interstate_result: dict,
    locations: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Per-market cost breakdown across C1, C2, and C3.

    One row per market.  Columns (prefixed c1_ / c2_ / c3_) cover:
      - Physical capacity  : solar_kw, fridge_kw, battery_kwh,
                             battery_power_kw, pcm_kwh
        (PCM is optional and safely reported as zero when disabled.)
      - Annualised capex   : solar_capex_usd, fridge_capex_usd,
                             battery_energy_capex_usd,
                             battery_inverter_power_capex_usd, pcm_capex_usd,
                             equipment_capex_usd
      - Combined total     : equipment_and_transport_capex_usd

    Change columns (C1→C2 and C1→C3) are appended for every cost metric.

    Note: LP total cost is only available at state level (C1/C2) or national
    level (C3) so it is not included here.  transport_capex reflects
    infrastructure investment only; fuel opex is not attributable per-market
    without per-link marginal cost data.
    """
    cfg = config
    loc_state: Dict[str, str] = (
        locations.set_index("market_id")["state"].astype(str).to_dict()
        if locations is not None and not locations.empty
        else {}
    )

    # ── Build per-market capacity dicts for C1 and C2 ─────────────────────
    # c1_caps[mid] = {solar_kw, fridge_kw, battery_kwh, pcm_kwh}
    # pcm_power_kw is retained as NaN for CSV compatibility; passive PCM has
    # no independently optimised power capacity.
    ZERO_CAP = dict(solar_kw=0.0, fridge_kw=0.0, battery_kwh=0.0, pcm_kwh=0.0,
                    battery_power_kw=0.0, pcm_power_kw=float("nan"))

    c1_caps: Dict[str, dict] = {}
    for r in standalone_results.values():
        cap = r.get("capacities", pd.DataFrame())
        if cap is None or cap.empty:
            continue
        for mid, row in cap.iterrows():
            c1_caps[str(mid)] = {
                "solar_kw"   : float(row.get("solar_kw",    0.0)),
                "fridge_kw"  : float(row.get("fridge_kw",   0.0)),
                "battery_kwh": float(row.get("battery_kwh", 0.0)),
                "pcm_kwh"    : float(row.get("pcm_kwh",     0.0)),
                "battery_power_kw": float(row.get("battery_power_kw", 0.0)),
                "pcm_power_kw"    : float(row.get("pcm_power_kw", float("nan"))),
            }

    c2_caps: Dict[str, dict] = {}
    for r in intrastate_results.values():
        cap = r.get("capacities", pd.DataFrame())
        if cap is None or cap.empty:
            continue
        for mid, row in cap.iterrows():
            c2_caps[str(mid)] = {
                "solar_kw"   : float(row.get("solar_kw",    0.0)),
                "fridge_kw"  : float(row.get("fridge_kw",   0.0)),
                "battery_kwh": float(row.get("battery_kwh", 0.0)),
                "pcm_kwh"    : float(row.get("pcm_kwh",     0.0)),
                "battery_power_kw": float(row.get("battery_power_kw", 0.0)),
                "pcm_power_kw"    : float(row.get("pcm_power_kw", float("nan"))),
            }

    c3_cap_df = interstate_result.get("capacities", pd.DataFrame())
    c3_caps: Dict[str, dict] = {}
    if c3_cap_df is not None and not c3_cap_df.empty:
        for mid, row in c3_cap_df.iterrows():
            c3_caps[str(mid)] = {
                "solar_kw"   : float(row.get("solar_kw",    0.0)),
                "fridge_kw"  : float(row.get("fridge_kw",   0.0)),
                "battery_kwh": float(row.get("battery_kwh", 0.0)),
                "pcm_kwh"    : float(row.get("pcm_kwh",     0.0)),
                "battery_power_kw": float(row.get("battery_power_kw", 0.0)),
                "pcm_power_kw"    : float(row.get("pcm_power_kw", float("nan"))),
            }

    # ── Build per-market transport fuel opex (50/50 endpoint split) ───────
    # Transport has capital_cost=0 (hired service); the real cost is
    # marginal_cost_per_kwh × total_flow_kwh stored as fuel_opex_usd.
    def _build_transport_fuel_opex(transport_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if transport_df is None or transport_df.empty:
            return out
        if "fuel_opex_usd" in transport_df.columns:
            opex_col = transport_df["fuel_opex_usd"]
        elif "marginal_cost_per_kwh" in transport_df.columns:
            opex_col = transport_df["marginal_cost_per_kwh"] * transport_df["total_flow_kwh"]
        else:
            print("  WARNING: transport_df has no fuel_opex_usd or marginal_cost_per_kwh "
                  "— delete checkpoints and re-run to get accurate transport fuel costs")
            return out
        for (_, lnk), opex in zip(transport_df.iterrows(), opex_col):
            half = float(opex) / 2.0
            for endpoint in (str(lnk["from_market"]), str(lnk["to_market"])):
                out[endpoint] = out.get(endpoint, 0.0) + half
        return out

    c2_trans_parts = [
        r["transport"]
        for r in intrastate_results.values()
        if r.get("transport") is not None and not r["transport"].empty
    ]
    c2_trans_df        = pd.concat(c2_trans_parts, ignore_index=True) if c2_trans_parts else None
    c2_trans_fuel_opex = _build_transport_fuel_opex(c2_trans_df)
    c3_trans_fuel_opex = _build_transport_fuel_opex(interstate_result.get("transport", None))

    # ── Helper: capex cost for one market's capacities ────────────────────
    def _eq_capex(caps: dict) -> float:
        return (
            caps["solar_kw"]    * cfg.pv_annualised_capex_per_kw
            + caps["fridge_kw"]   * cfg.fridge_annualised_capex_per_kw_electric
            + caps["battery_kwh"] * cfg.battery_annualised_capex_per_kwh
            + caps["pcm_kwh"]     * cfg.pcm_annualised_capex_per_kwh
            + caps["battery_power_kw"] * cfg.battery_annualised_power_capex_per_kw
        )

    # ── All market IDs across all three configs ────────────────────────────
    all_markets = sorted(set(c1_caps) | set(c2_caps) | set(c3_caps))

    rows = []
    for mid in all_markets:
        c1 = c1_caps.get(mid, ZERO_CAP)
        c2 = c2_caps.get(mid, ZERO_CAP)
        c3 = c3_caps.get(mid, ZERO_CAP)

        c1_eq = _eq_capex(c1)
        c2_eq = _eq_capex(c2)
        c3_eq = _eq_capex(c3)

        c1_tf = 0.0
        c2_tf = c2_trans_fuel_opex.get(mid, 0.0)
        c3_tf = c3_trans_fuel_opex.get(mid, 0.0)

        c1_tot = c1_eq + c1_tf
        c2_tot = c2_eq + c2_tf
        c3_tot = c3_eq + c3_tf

        def _pct(base, new):
            return (new - base) / abs(base) * 100 if base != 0.0 else np.nan

        rows.append({
            "market_id"                            : mid,
            "state"                                : loc_state.get(mid, ""),
            # ── C1 ────────────────────────────────────────────────────────
            "c1_solar_kw"                          : c1["solar_kw"],
            "c1_fridge_kw"                         : c1["fridge_kw"],
            "c1_battery_kwh"                       : c1["battery_kwh"],
            "c1_battery_power_kw"                  : c1["battery_power_kw"],
            "c1_pcm_kwh"                           : c1["pcm_kwh"],
            "c1_solar_capex_usd"                   : c1["solar_kw"]    * cfg.pv_annualised_capex_per_kw,
            "c1_fridge_capex_usd"                  : c1["fridge_kw"]   * cfg.fridge_annualised_capex_per_kw_electric,
            "c1_battery_energy_capex_usd"          : c1["battery_kwh"] * cfg.battery_annualised_capex_per_kwh,
            "c1_battery_inverter_power_capex_usd"  : c1["battery_power_kw"] * cfg.battery_annualised_power_capex_per_kw,
            "c1_pcm_capex_usd"                     : c1["pcm_kwh"]     * cfg.pcm_annualised_capex_per_kwh,
            "c1_equipment_capex_usd"               : c1_eq,
            "c1_transport_fuel_opex_usd"           : c1_tf,
            "c1_total_cost_usd"                    : c1_tot,
            # ── C2 ────────────────────────────────────────────────────────
            "c2_solar_kw"                          : c2["solar_kw"],
            "c2_fridge_kw"                         : c2["fridge_kw"],
            "c2_battery_kwh"                       : c2["battery_kwh"],
            "c2_battery_power_kw"                  : c2["battery_power_kw"],
            "c2_pcm_kwh"                           : c2["pcm_kwh"],
            "c2_solar_capex_usd"                   : c2["solar_kw"]    * cfg.pv_annualised_capex_per_kw,
            "c2_fridge_capex_usd"                  : c2["fridge_kw"]   * cfg.fridge_annualised_capex_per_kw_electric,
            "c2_battery_energy_capex_usd"          : c2["battery_kwh"] * cfg.battery_annualised_capex_per_kwh,
            "c2_battery_inverter_power_capex_usd"  : c2["battery_power_kw"] * cfg.battery_annualised_power_capex_per_kw,
            "c2_pcm_capex_usd"                     : c2["pcm_kwh"]     * cfg.pcm_annualised_capex_per_kwh,
            "c2_equipment_capex_usd"               : c2_eq,
            "c2_transport_fuel_opex_usd"           : c2_tf,
            "c2_total_cost_usd"                    : c2_tot,
            # ── C3 ────────────────────────────────────────────────────────
            "c3_solar_kw"                          : c3["solar_kw"],
            "c3_fridge_kw"                         : c3["fridge_kw"],
            "c3_battery_kwh"                       : c3["battery_kwh"],
            "c3_battery_power_kw"                  : c3["battery_power_kw"],
            "c3_pcm_kwh"                           : c3["pcm_kwh"],
            "c3_solar_capex_usd"                   : c3["solar_kw"]    * cfg.pv_annualised_capex_per_kw,
            "c3_fridge_capex_usd"                  : c3["fridge_kw"]   * cfg.fridge_annualised_capex_per_kw_electric,
            "c3_battery_energy_capex_usd"          : c3["battery_kwh"] * cfg.battery_annualised_capex_per_kwh,
            "c3_battery_inverter_power_capex_usd"  : c3["battery_power_kw"] * cfg.battery_annualised_power_capex_per_kw,
            "c3_pcm_capex_usd"                     : c3["pcm_kwh"]     * cfg.pcm_annualised_capex_per_kwh,
            "c3_equipment_capex_usd"               : c3_eq,
            "c3_transport_fuel_opex_usd"           : c3_tf,
            "c3_total_cost_usd"                    : c3_tot,
            # ── C1→C2 changes ─────────────────────────────────────────────
            "c1_to_c2_solar_kw_change"             : c2["solar_kw"]    - c1["solar_kw"],
            "c1_to_c2_fridge_kw_change"            : c2["fridge_kw"]   - c1["fridge_kw"],
            "c1_to_c2_battery_kwh_change"          : c2["battery_kwh"] - c1["battery_kwh"],
            "c1_to_c2_battery_power_kw_change"     : c2["battery_power_kw"] - c1["battery_power_kw"],
            "c1_to_c2_pcm_kwh_change"              : c2["pcm_kwh"]     - c1["pcm_kwh"],
            "c1_to_c2_equipment_capex_change"      : c2_eq  - c1_eq,
            "c1_to_c2_transport_fuel_opex_added"   : c2_tf  - c1_tf,
            "c1_to_c2_total_cost_change"           : c2_tot - c1_tot,
            "c1_to_c2_total_cost_pct_change"       : _pct(c1_tot, c2_tot),
            # ── C1→C3 changes ─────────────────────────────────────────────
            "c1_to_c3_solar_kw_change"             : c3["solar_kw"]    - c1["solar_kw"],
            "c1_to_c3_fridge_kw_change"            : c3["fridge_kw"]   - c1["fridge_kw"],
            "c1_to_c3_battery_kwh_change"          : c3["battery_kwh"] - c1["battery_kwh"],
            "c1_to_c3_battery_power_kw_change"     : c3["battery_power_kw"] - c1["battery_power_kw"],
            "c1_to_c3_pcm_kwh_change"              : c3["pcm_kwh"]     - c1["pcm_kwh"],
            "c1_to_c3_equipment_capex_change"      : c3_eq  - c1_eq,
            "c1_to_c3_transport_fuel_opex_added"   : c3_tf  - c1_tf,
            "c1_to_c3_total_cost_change"           : c3_tot - c1_tot,
            "c1_to_c3_total_cost_pct_change"       : _pct(c1_tot, c3_tot),
        })

    return pd.DataFrame(rows).sort_values(["state", "market_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# save_all_results
# ---------------------------------------------------------------------------

def save_all_results(
    state_summary: pd.DataFrame,
    technology_mix: pd.DataFrame,
    transport_breakdown: pd.DataFrame,
    interstate_result: dict,
    locations: pd.DataFrame,
    config,
    output_dir: str,
    *,
    standalone_results: Optional[Dict[str, dict]] = None,
    intrastate_results: Optional[Dict[str, dict]] = None,
    cold_loads_df: Optional[pd.DataFrame] = None,
    cold_load_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Save all result artefacts to output_dir.

    Writes
    ------
    CSV tables : state_summary.csv, technology_mix.csv, transport_breakdown.csv,
                 national_summary.csv, cost_breakdown.csv,
                 market_cost_breakdown.csv
    Figures    : figures/fig1_*.png … fig10_*.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── CSVs ──────────────────────────────────────────────────────────────
    for filename, df in [
        ("state_summary.csv",       state_summary),
        ("technology_mix.csv",      technology_mix),
        ("transport_breakdown.csv", transport_breakdown),
    ]:
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        print(f"  Saved {path}  {df.shape}")

    # ── National 3-way cost comparison ────────────────────────────────────
    c1_total = float(state_summary["cost_standalone"].sum())
    c2_total = float(state_summary["cost_intrastate"].sum())
    # Config 3: one continuous national-mesh LP retaining eligible
    # intrastate and interstate network connections.
    c3_total = float(interstate_result.get(
        "total_cost_incl_spokes",
        interstate_result.get("total_cost", float("nan")),
    ))

    national_summary = pd.DataFrame([{
        "config1_standalone_total"  : c1_total,
        "config2_intrastate_total"  : c2_total,
        "config3_national_total"    : c3_total,
        "saving_c1_to_c2_abs"      : c1_total - c2_total,
        "saving_c1_to_c2_pct"      : (c1_total - c2_total) / c1_total * 100
                                      if c1_total > 0 else np.nan,
        "saving_c2_to_c3_abs"      : c2_total - c3_total,
        "saving_c2_to_c3_pct"      : (c2_total - c3_total) / c2_total * 100
                                      if c2_total > 0 else np.nan,
        "saving_c1_to_c3_abs"      : c1_total - c3_total,
        "saving_c1_to_c3_pct"      : (c1_total - c3_total) / c1_total * 100
                                      if c1_total > 0 else np.nan,
        "config3_network_formulation": "continuous_national_mesh",
        "active_interstate_network_links": interstate_result.get(
            "interstate_links_built", 0
        ),
    }])
    path = os.path.join(output_dir, "national_summary.csv")
    national_summary.to_csv(path, index=False)
    print(f"  Saved {path}")

    # ── Component cost breakdown ──────────────────────────────────────────
    if standalone_results and intrastate_results and interstate_result and config is not None:
        cost_breakdown = compile_cost_breakdown(
            standalone_results, intrastate_results, interstate_result,
            state_summary, config,
        )
        cb_path = os.path.join(output_dir, "cost_breakdown.csv")
        cost_breakdown.to_csv(cb_path, index=False)
        print(f"  Saved {cb_path}  {cost_breakdown.shape}")

        market_breakdown = compile_market_cost_breakdown(
            standalone_results, intrastate_results, interstate_result,
            locations, config,
        )
        mb_path = os.path.join(output_dir, "market_cost_breakdown.csv")
        market_breakdown.to_csv(mb_path, index=False)
        print(f"  Saved {mb_path}  {market_breakdown.shape}")
    else:
        print("  cost_breakdown.csv / market_cost_breakdown.csv skipped — "
              "missing standalone/intrastate/interstate results or config")

    # ── Dispatch samples ──────────────────────────────────────────────────
    # Always write (or overwrite) dispatch_samples.csv so stale data from a
    # previous run is never left behind.
    _DISPATCH_COLS = [
        "timestamp", "state", "market_id", "role",
        "solar_kw", "cold_demand_kw", "fridge_kw",
        "pcm_soc_kwh", "pcm_charge_kw", "pcm_discharge_kw",
        "export_kw", "import_kw",
    ]
    dispatch_df: Optional[pd.DataFrame] = None
    if intrastate_results:
        dispatch_df = compile_all_dispatch(intrastate_results)

    disp_path = os.path.join(output_dir, "dispatch_samples.csv")
    if dispatch_df is not None and not dispatch_df.empty:
        dispatch_df.to_csv(disp_path, index=False)
        print(f"  Saved {disp_path}  {dispatch_df.shape}")
    else:
        # Write empty file with correct headers so old data is never kept
        pd.DataFrame(columns=_DISPATCH_COLS).to_csv(disp_path, index=False)
        print(f"  Saved {disp_path}  (empty — no dispatch data extracted)")

    # ── Figures ───────────────────────────────────────────────────────────
    print("Generating figures …")
    generate_figures(
        state_summary, technology_mix, transport_breakdown,
        locations, output_dir,
        cold_loads_df=cold_loads_df,
        cold_load_arrays=cold_load_arrays,
        standalone_results=standalone_results,
        intrastate_results=intrastate_results,
        interstate_result=interstate_result,
        hub_capacity_threshold_kw=getattr(config, "hub_capacity_threshold_kw", 0.1),
        dispatch_df=dispatch_df,
        config=config,
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
