"""
config.py
---------
Single dataclass holding every parameter for the solar cold-chain
optimisation pipeline.  All other modules import from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Config:
    # Main Config 1–3 runs are battery-only. Passive PCM is enabled only by
    # the dedicated supplementary PCM-cost sensitivity analysis.
    include_pcm: bool = False
    # Bump whenever a formulation change makes solved checkpoints incompatible.
    checkpoint_compatibility_marker: str = (
        "battery_external_ac_power_v1_transport_window_v1"
    )
    discount_rate: float = 0.08
    pv_lifetime_years: int = 25
    battery_lifetime_years: int = 15
    pcm_lifetime_years: int = 20
    # Economic/project lifetime adopted from Rami and Allouhi (2024).
    fridge_lifetime_years: int = 20
    # ------------------------------------------------------------------
    # Technology capital costs  (USD per unit of capacity)
    # ------------------------------------------------------------------
    pv_capex_per_kw: float = 580.0
    battery_energy_capex_per_kwh: float = 241.0
    battery_power_capex_per_kw: float = 372.0
    pcm_capex_per_kwh: float = 45.0

    # --- Diesel genset (counterfactual supply, R7) ----------------------
    # Used only by the diesel base case. Nigerian market cold storage that is
    # not grid-connected today runs on small diesel gensets, so this is the
    # incumbent technology the solar proposal must beat — without it, "solar
    # coordination is cheaper" has no referent.
    #
    # Genset capex is low and fuel cost is high, the mirror image of solar.
    # That asymmetry is the whole point: under a *capital* budget (Fig. 2b)
    # diesel looks attractive, and only lifetime cost reveals the difference.
    diesel_genset_capex_per_kw: float = 400.0      # USD/kW installed, small unit
    diesel_genset_lifetime_years: int = 10         # shorter than PV's 25
    diesel_genset_kwh_per_litre: float = 3.2       # ~32% efficient; diesel LHV ~10 kWh/L
    diesel_co2_kg_per_litre: float = 2.68          # IPCC default, diesel oil
    # Single source of truth for the fuel price. Previously hardcoded inside
    # conv_marginal_cost; the genset base case needs the same number, and two
    # copies would silently drift apart under a fuel-price sensitivity.
    diesel_price_usd_per_litre: float = 0.982      # NBS 2024

    # Smallest genset that can actually be bought. Continuous sizing lets the
    # LP build 2-watt gensets as micro-peakers to shave battery capacity — an
    # artefact, not a technology choice. Enforcing "build zero, or at least
    # this much" turns the decision into a MILP but makes it physical.
    min_genset_kw: float = 5.0                     # ~5 kVA, smallest common unit

    # ------------------------------------------------------------------
    # Embodied emission factors  (kg CO2e per unit of installed capacity)
    # ------------------------------------------------------------------
    # PLACEHOLDERS pending a sourced LCA review — order-of-magnitude values
    # from the general literature, recorded here so the accounting framework
    # is complete and the sensitivity of the conclusion to them can be tested.
    # Do NOT quote these in the manuscript without replacing them with cited
    # values. Combustion factors below are, by contrast, standard.
    pv_embodied_kgco2_per_kw: float = 1500.0       # placeholder
    battery_embodied_kgco2_per_kwh: float = 100.0  # placeholder
    fridge_embodied_kgco2_per_kw: float = 150.0    # placeholder
    genset_embodied_kgco2_per_kw: float = 60.0     # placeholder
    # USD 2,580 refrigeration unit / 2.05361 kW-cold peak cooling load;
    # source CAPEX is therefore on a cooling-output-capacity basis.
    fridge_capex_per_kw_cold: float = 1256.324229

    # ------------------------------------------------------------------
    # Technology operational parameters
    # ------------------------------------------------------------------
    # Conventional electrically driven vapour-compression refrigeration.
    fridge_cop: float = 2.7
    battery_charge_efficiency: float = 0.92195
    battery_discharge_efficiency: float = 0.92195
    battery_standing_loss: float = 2.81e-5
    pcm_standing_loss: float = 0.0

    # ------------------------------------------------------------------
    # Transit thermal loss parameters (passive heat gain, uninsulated van)
    # ------------------------------------------------------------------
    van_speed_kmh: float = 40.0
    # Transport-time consistency: links deliver within the snapshot they draw
    # (PyPSA semantics), so a journey must be completable inside one dispatch
    # interval to be physical. None = legacy behaviour (no cap; admits links
    # to 240 km whose 3-6 h travel exceeds the 3 h step). Set to
    # time_resolution_hours to enforce d <= van_speed_kmh * max hours.
    max_transport_link_hours: float | None = None
    transit_thermal_tau_hr: float = 2.0
    transport_operating_start_hour: int = 4
    transport_operating_end_hour: int = 19

    # Computational pruning only — NOT a modelling assumption. Skips
    # creating transport links whose transit_efficiency(d) is negligible,
    # to keep the full national mesh solvable. Value should be chosen
    # based on diagnostics/count_pruned_links.py output, not guessed.
    transit_efficiency_floor: float = 0.05   # placeholder — replace with
                                             # the value chosen from the
                                             # diagnostic output

    # ------------------------------------------------------------------
    # Network / hub detection
    # ------------------------------------------------------------------
    # Superseded by top-1-per-state hub selection in run_intrastate().
    # Retained for backwards compatibility only.
    hub_capacity_threshold_kw: float = 0.1

    # ------------------------------------------------------------------
    # Simulation settings
    # ------------------------------------------------------------------
    solver: str = "gurobi"
    solver_threads: int = 4
    use_representative_weeks: bool = True
    n_representative_weeks: int = 1
    representative_period_seed: int = 42
    time_resolution_hours: int = 3
    hours_per_year: int = 8760

    # ------------------------------------------------------------------
    # Config 3 barrier-solve tuning (applied ONLY to the national LP)
    # ------------------------------------------------------------------
    # The full national mesh is a huge LP that is solved by the interior-point
    # (barrier) method.  Two safe accelerations:
    #   Method=2     — barrier only; skip the concurrent primal/dual simplex
    #                  that can never win on a model this size and just steals
    #                  threads + memory bandwidth from the factorisation.
    #   BarConvTol   — planning-grade tolerance; the default 1e-8 spends many
    #                  slow tail iterations closing a gap that doesn't matter.
    #                  Safe to loosen because crossover finishes to an EXACT
    #                  optimal basis regardless.
    # Crossover is deliberately LEFT ON (Gurobi default). On these short
    # representative runs it costs ~2 min (<2% of runtime) but converts the
    # "smeared" barrier interior point into a sparse vertex — essential for
    # interpretable interstate-link counts and flow maps. Only disable it
    # (add "Crossover": 0) if crossover itself ever becomes a real bottleneck.
    c3_barrier_options: dict = field(default_factory=lambda: {
        "Method": 2,
        "BarConvTol": 1e-6,
    })

    # ------------------------------------------------------------------
    # File paths  (no defaults — set at runtime in run.py)
    # ------------------------------------------------------------------
    hourly_loads_csv: str = ""
    solar_csv: str = ""
    output_dir: str = ""
    road_distances_path: str = "road_distance_matrix_full.pkl"

    # ------------------------------------------------------------------
    # Pre-loaded road distance matrix (populated by run.py at startup)
    # Dict {(uniq_id_a, uniq_id_b): road_km} computed via
    # OpenRouteService API (OpenStreetMap data).
    # ------------------------------------------------------------------
    road_distances: dict = field(default=None)

    def __post_init__(self) -> None:
        self.validate_transport_operating_window()

    def validate_transport_operating_window(self) -> None:
        """Validate a non-wrapping, half-open transport operating window."""
        start = self.transport_operating_start_hour
        end = self.transport_operating_end_hour

        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError("transport_operating_start_hour must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise TypeError("transport_operating_end_hour must be an integer")
        if not 0 <= start <= 23:
            raise ValueError(
                "transport_operating_start_hour must be between 0 and 23"
            )
        if not 1 <= end <= 24:
            raise ValueError(
                "transport_operating_end_hour must be between 1 and 24"
            )
        if start >= end:
            raise ValueError(
                "transport operating window must be non-wrapping with "
                "transport_operating_start_hour < transport_operating_end_hour"
            )

    # ==================================================================
    # Financial helper methods
    # ==================================================================

    def capital_recovery_factor(self, lifetime_years: int) -> float:
        """Return the CRF using this configuration's financial assumptions."""
        r = self.discount_rate
        if lifetime_years <= 0:
            raise ValueError("lifetime_years must be positive")
        if r == 0:
            return 1.0 / lifetime_years
        growth = (1.0 + r) ** lifetime_years
        return r * growth / (growth - 1.0)

    @property
    def pv_annualised_capex_per_kw(self) -> float:
        return self.pv_capex_per_kw * self.capital_recovery_factor(self.pv_lifetime_years)

    @property
    def diesel_genset_annualised_capex_per_kw(self) -> float:
        return self.diesel_genset_capex_per_kw * self.capital_recovery_factor(
            self.diesel_genset_lifetime_years
        )

    @property
    def diesel_marginal_cost_per_kwh(self) -> float:
        """
        Fuel cost of one kWh of genset electricity [USD/kWh_e].

        Uses the same diesel price as road transport (0.982 USD/L, NBS 2024) so
        the base case and the transport model cannot drift apart.
        """
        return self.diesel_price_usd_per_litre / self.diesel_genset_kwh_per_litre

    @property
    def diesel_co2_kg_per_kwh(self) -> float:
        """Combustion CO2 per kWh of genset electricity [kg/kWh_e]."""
        return self.diesel_co2_kg_per_litre / self.diesel_genset_kwh_per_litre

    @property
    def battery_annualised_energy_capex_per_kwh(self) -> float:
        return self.battery_energy_capex_per_kwh * self.capital_recovery_factor(
            self.battery_lifetime_years
        )

    @property
    def battery_annualised_power_capex_per_kw(self) -> float:
        return self.battery_power_capex_per_kw * self.capital_recovery_factor(
            self.battery_lifetime_years
        )

    # Backwards-compatible battery name used by existing result tables.
    @property
    def battery_annualised_capex_per_kwh(self) -> float:
        return self.battery_annualised_energy_capex_per_kwh

    @property
    def pcm_annualised_capex_per_kwh(self) -> float:
        return self.pcm_capex_per_kwh * self.capital_recovery_factor(
            self.pcm_lifetime_years
        )

    @property
    def fridge_annualised_capex_per_kw_electric(self) -> float:
        # CAPEX is per kW of cooling output, while Link p_nom is electrical
        # input capacity; multiplying by COP converts cost to the input basis.
        return (
            self.fridge_capex_per_kw_cold
            * self.fridge_cop
            * self.capital_recovery_factor(self.fridge_lifetime_years)
        )

    # ==================================================================
    # Transport cost & efficiency methods
    # (d = road distance in km between two markets, from road_distances dict)
    #
    # Two independent mechanisms are modelled separately:
    #   1. Fuel cost (marginal_cost) — diesel for vehicle propulsion only.
    #      The vehicle has no active refrigeration unit.
    #   2. Thermal loss (efficiency) — passive warming toward ambient
    #      temperature during transit, since the van is uninsulated and
    #      unrefrigerated. Modelled via Newton's law of heating.
    # ==================================================================

    # --- Conventional diesel truck ------------------------------------

    def conv_marginal_cost(self, d: float) -> float:
        """
        Variable fuel cost [USD / kWh_cold] for conventional refrigerated truck.
        d: road distance in km (from pre-computed OpenRouteService matrix).

        Derivation:
          - Diesel price:      0.982 USD/L
                               (NBS AGO Price Watch 2024 annual average;
                                ₦1,438.74/L ÷ 1,465 NGN/USD annual average)
          - Fuel efficiency:   8.635 km/L  (light refrigerated truck)
          - Vehicle payload:   2,000 kg meat
          - Cold capacity:     62.8 kWh_cold
                               (2,000 kg × 0.0314 kWh/kg;
                                c_p = 3.14 kJ/kgK, ΔT = 36 K —
                                consistent with config thermal parameters)

        Formula:
          c(d) = (p_diesel / η_fuel) × d / Q_cold
        """
        diesel_price    = self.diesel_price_usd_per_litre  # USD/litre (NBS 2024)
        fuel_efficiency = 8.635   # km/litre
        cold_per_trip   = 62.8    # kWh_cold per truck load
        return (diesel_price / fuel_efficiency) * d / cold_per_trip

    def transit_efficiency(self, d: float) -> float:
        """
        Fraction of cold energy/margin retained after road transit in an
        uninsulated, unrefrigerated van (goods warm passively toward
        ambient temperature en route).

        Newton's law of heating:
          T(t) = T_amb - (T_amb - T0) * exp(-t / tau)
        Expressed as a retained-cold-fraction efficiency:
          eta(d) = exp(-(d / van_speed_kmh) / transit_thermal_tau_hr)

        d: road distance in km.
        Placeholder parameters (van_speed_kmh, transit_thermal_tau_hr) are
        assumptions pending calibration — see sensitivity_thermal_loss.py.
        """
        transit_time_hr = d / self.van_speed_kmh
        return math.exp(-transit_time_hr / self.transit_thermal_tau_hr)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()

    print("Annualised capex (USD / unit / year, derived from Config CRFs)")
    print(f"  Passive PCM enabled: {cfg.include_pcm}")
    print(f"  PV       : {cfg.pv_annualised_capex_per_kw:.4f}  USD/kW/yr")
    print(f"  Battery energy : {cfg.battery_annualised_energy_capex_per_kwh:.4f}  USD/kWh/yr")
    print(f"  Battery power  : {cfg.battery_annualised_power_capex_per_kw:.4f}  USD/kW/yr")
    print(f"  PCM            : {cfg.pcm_annualised_capex_per_kwh:.4f}  USD/kWh/yr")
    print(f"  Fridge   : {cfg.fridge_annualised_capex_per_kw_electric:.4f}  USD/kW_electric/yr")

    print("\nTemporal configuration")
    print(f"  representative weeks : {cfg.n_representative_weeks}")
    print(f"  time resolution      : {cfg.time_resolution_hours} h")

    print("\nTransport marginal costs at d = 100.0 km (road distance)")
    d = 100.0
    for mode in ("conv",):
        mc = getattr(cfg, f"{mode}_marginal_cost")(d)
        eff = cfg.transit_efficiency(d)
        print(f"  {mode:4s}  marginal=${mc:.4f}/kWh  efficiency={eff:.4f} (d={d:.0f}km)  capex=$0")

    assert abs(cfg.transit_efficiency(100.0) - 0.2865) < 1e-3, \
        "transit_efficiency(100) sanity check failed"

    print("Hub threshold (deprecated):", cfg.hub_capacity_threshold_kw, "kW")
