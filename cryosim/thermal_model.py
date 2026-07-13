"""Lumped-parameter cryogenic tank thermal / self-pressurization model.

Node structure (all SI):

* ullage vapor (mass m_u, internal energy U_u) — single-species real vapor,
  pressure from a CoolProp (rho, u) flash;
* saturated interface at T_i = T_sat(P) — massless, closes phase change;
* warm stratified liquid *surface layer* (thin slab, temperature T_s);
* bulk liquid (temperature T_b) — the tank outlet draws from here;
* two wall nodes (dry/wet) with temperature-dependent heat capacity.

Governing equations (see README for derivation and sources):

  dm_u/dt = mdot_evap                      (evap > 0, condensation < 0)
  dU_u/dt = Q_wd->u - Q_u->i + mdot_evap*h_g(T_i) - P*dV_u/dt
  mdot_evap = (Q_u->i + Q_s->i) / h_fg(P)  (interface energy balance)
  m_s*cp*dT_s/dt = chi*Q_ww->l + Q_b<->s - Q_s->i - mdot_evap*cp*(T_i-T_s)
  m_b*cp*dT_b/dt = (1-chi)*Q_ww->l - Q_b<->s
  wall nodes: m_w*c_w(T)*dT_w/dt = Q_env - Q_w->fluid

Free convection closures: Churchill & Chu (vertical plate) for wall-fluid,
McAdams horizontal-plate correlations for the interface (Incropera & DeWitt,
"Fundamentals of Heat and Mass Transfer", ch. 9).

This formulation follows the standard lumped treatment of the NASA LeRC
self-pressurization experiments (Hasan, Lin & Van Dresar 1991; Van Dresar &
Lin, NASA TM-105411 1992) in which ullage superheating drives the early
pressure rise and interfacial evaporation from the warm surface layer drives
the quasi-steady stage.

Assumptions / limitations
-------------------------
* Single-species tank: ullage is pure propellant vapor. No helium pressurant
  (an optional *autogenous* pressurant injection at a pressure setpoint is
  provided instead). No dissolved-gas effects.
* The stratified layer is a single thin slab: no continuous temperature
  profile. Its share of the wetted-wall heat input is the *stratification
  factor* ``chi`` — a calibrated parameter (see validation), not a prediction.
* Slosh coupling enters through a mixing factor Phi >= 1 (from the slosh
  model) that enhances interface heat transfer and surface<->bulk exchange and
  correspondingly reduces the effective chi. This link is NOT quantitatively
  validated against public data (none was found); it is a parametric knob
  whose qualitative behavior (slosh -> destratification -> pressure change)
  follows Ludwig & Dreyer, Cryogenics 63 (2014).
* Liquid treated as incompressible for node volumes within a step; boundary
  work between ullage and liquid retained via P*dV_u/dt.
* The optional autogenous pressurant is an EXTERNAL mass/energy source: the
  injected vapor is not deducted from the engine/injector flow, so total
  system mass is not closed while it operates (adequate for tank-state
  studies; account for the return-line flow separately at vehicle level).
* If the ullage pressure falls below the bulk liquid's vapor pressure
  (deep blowdown), real propellant would flash-boil; no flashing model is
  included — results in that regime are invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .fluids import Fluid
from .tank_geometry import TankGeometry

G0 = 9.80665

# Approximate cp(T) tables [J/(kg K)] from the NIST cryogenic material
# properties database (curve-fit values, adequate for wall thermal inertia).
_WALL_CP_TABLES = {
    "aluminum": (
        np.array([20.0, 40.0, 60.0, 77.0, 100.0, 150.0, 200.0, 250.0, 300.0]),
        np.array([9.0, 78.0, 214.0, 336.0, 481.0, 683.0, 791.0, 855.0, 897.0]),
    ),
    "stainless": (
        np.array([20.0, 40.0, 60.0, 77.0, 100.0, 150.0, 200.0, 250.0, 300.0]),
        np.array([17.0, 74.0, 141.0, 190.0, 262.0, 350.0, 402.0, 440.0, 477.0]),
    ),
}


_WALL_K_TABLES = {
    "aluminum": (  # Al 6061-T6
        np.array([20.0, 40.0, 77.0, 150.0, 300.0]),
        np.array([28.0, 52.0, 84.0, 120.0, 155.0]),
    ),
    "stainless": (  # 304 SS
        np.array([20.0, 40.0, 77.0, 150.0, 300.0]),
        np.array([2.2, 4.7, 7.9, 11.5, 15.0]),
    ),
}


def wall_cp(material: str, T: float) -> float:
    Ttab, cptab = _WALL_CP_TABLES[material]
    return float(np.interp(T, Ttab, cptab))


def wall_k(material: str, T: float) -> float:
    Ttab, ktab = _WALL_K_TABLES[material]
    return float(np.interp(T, Ttab, ktab))


# --------------------------------------------------------------------------
# Free-convection closures (Incropera & DeWitt ch. 9)
# --------------------------------------------------------------------------
def nu_vertical_plate(Ra: float, Pr: float) -> float:
    """Churchill & Chu correlation for a vertical plate (all Ra)."""
    Ra = max(Ra, 1.0)
    return (
        0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)
    ) ** 2


def nu_horizontal_plate(Ra: float, unstable: bool) -> float:
    """McAdams horizontal-plate correlations.

    ``unstable=True``: hot surface facing up / cold facing down (buoyant).
    ``unstable=False``: stably stratified orientation.
    """
    Ra = max(Ra, 1.0)
    if unstable:
        if Ra < 1e7:
            return 0.54 * Ra ** 0.25
        return 0.15 * Ra ** (1 / 3)
    return 0.27 * Ra ** 0.25


@dataclass
class TankThermalConfig:
    """Model parameters with quiescent defaults.

    ``stratification_factor`` (chi): share of wetted-wall heat routed to the
    surface layer in quiescent conditions. Calibrated so that the predicted
    K-site self-pressurization rates land in the experimentally reported
    band relative to the homogeneous rate (see validation tests).
    """

    stratification_factor: float = 0.25
    surface_layer_thickness: float = 0.05   # m
    wall_material: str = "aluminum"
    wall_thickness: float = 2.0e-3          # m, conduction path cross-section
    wall_conduction_length: float | None = None  # m; default height/4
    #: Multiplier on the (stable flat-plate) gas-side interface coefficient.
    #: Wall-plume-driven vapor circulation makes real ullage->interface
    #: exchange much larger than the quiescent correlation; calibrated
    #: against the K-site measured/homogeneous rate ratios (see validation).
    ullage_interface_enhancement: float = 10.0
    pressurant_setpoint: float | None = None  # Pa; autogenous hold if set
    pressurant_temperature: float = 250.0     # K, injected vapor temp
    pressurant_max_flow: float = 0.05         # kg/s
    min_ullage_fraction: float = 0.005


class TankThermalModel:
    """Lumped-node tank self-pressurization / boil-off model."""

    def __init__(
        self,
        fluid: Fluid,
        geometry: TankGeometry,
        wall_mass: float,
        heat_flux: float,
        config: TankThermalConfig | None = None,
    ):
        """
        Parameters
        ----------
        fluid : working fluid (tank propellant).
        geometry : tank geometry.
        wall_mass : total tank structural mass [kg] (split by wetted area).
        heat_flux : uniform environmental heat flux into the wall [W/m^2].
        """
        self.fluid = fluid
        self.geom = geometry
        self.wall_mass = wall_mass
        self.q_env = heat_flux
        self.cfg = config or TankThermalConfig()

    # ----------------------------------------------------------- initial state
    def initial_state(
        self,
        P0: float,
        fill_fraction: float,
        T_liquid: float | None = None,
        T_wall: float | None = None,
    ) -> np.ndarray:
        """State vector [m_u, U_u, m_l, T_b, T_s, T_wd, T_ww]."""
        f = self.fluid
        T_sat = f.T_sat(P0)
        T_l = T_liquid if T_liquid is not None else T_sat
        V_l = fill_fraction * self.geom.V_total
        V_u = self.geom.V_total - V_l
        rho_l = f.sat_liquid(P0).rho if T_l >= T_sat else f.state_TP(T_l, P0).rho
        sv = f.sat_vapor(P0)
        m_u = sv.rho * V_u
        U_u = sv.u * m_u
        m_l = rho_l * V_l
        T_w = T_wall if T_wall is not None else T_sat
        return np.array([m_u, U_u, m_l, T_l, T_sat, T_w, T_w])

    # ------------------------------------------------------------------- RHS
    def _ullage_PT(self, m_u: float, U_u: float, V_u: float) -> tuple[float, float]:
        rho_u = m_u / V_u
        u_u = U_u / m_u
        try:
            return self.fluid.flash_rho_u(rho_u, u_u)
        except ValueError:
            # (rho,u) fell inside the dome numerically: treat as saturated.
            # Find P such that two-phase u at this rho matches: fall back to
            # saturated vapor at this density.
            from CoolProp.CoolProp import PropsSI
            P = PropsSI("P", "Dmass", rho_u, "Q", 1, self.fluid.name)
            T = PropsSI("T", "Dmass", rho_u, "Q", 1, self.fluid.name)
            return P, T

    def rhs(
        self,
        t: float,
        y: np.ndarray,
        mdot_out: float = 0.0,
        accel: float = G0,
        mixing_factor: float = 1.0,
    ) -> np.ndarray:
        """Time derivatives.

        ``mixing_factor`` Phi >= 1 is the slosh-coupling knob: it multiplies
        interfacial and surface<->bulk exchange and divides the effective
        stratification factor.
        """
        f, geom, cfg = self.fluid, self.geom, self.cfg
        m_u, U_u, m_l, T_b, T_s, T_wd, T_ww = y

        # --- geometry of the current fill state
        # liquid density at bulk temperature (saturated-liquid approximation)
        rho_l = f.sat_liquid(max(f.P_sat(min(T_b, f.T_crit - 0.5)), 1000.0)).rho
        V_l = m_l / rho_l
        V_u = max(geom.V_total - V_l, cfg.min_ullage_fraction * geom.V_total)
        h_fill = geom.height_at_volume(V_l)
        A_int = geom.free_surface_area(h_fill)
        A_wet = geom.wetted_wall_area(h_fill)
        A_dry = geom.dry_wall_area(h_fill)
        L_liq = max(h_fill, 1e-3)
        L_gas = max(geom.height - h_fill, 1e-3)

        # --- ullage state
        P, T_u = self._ullage_PT(m_u, U_u, V_u)
        T_i = f.T_sat(P)
        h_fg = f.h_fg(P)
        sat_v = f.sat_vapor(P)
        sat_l = f.sat_liquid(P)
        cp_l = sat_l.cp

        # --- surface layer bookkeeping
        m_s = min(rho_l * A_int * cfg.surface_layer_thickness, 0.5 * m_l)
        m_s = max(m_s, 1e-6)
        m_b = max(m_l - m_s, 1e-6)

        chi = cfg.stratification_factor / mixing_factor

        # --- convective heat flows
        def h_nat(state, L, dT, nu_func, *args):
            if L <= 0 or abs(dT) < 1e-9:
                return state.k / max(L, 1e-3)  # conduction floor
            try:
                beta = f.beta(state.T, state.P)
            except ValueError:
                # state sits exactly on the saturation line: ideal-gas estimate
                beta = 1.0 / state.T
            nu_visc = state.mu / state.rho
            alpha = state.k / (state.rho * state.cp)
            Ra = accel * abs(beta) * abs(dT) * L**3 / (nu_visc * alpha)
            Nu = nu_func(Ra, *args)
            return Nu * state.k / L

        # dry wall -> ullage (vertical wall, gas side)
        try:
            gas_film = f.state_TP(max(0.5 * (T_wd + T_u), T_i + 0.01), P)
        except ValueError:
            gas_film = sat_v
        h_wu = h_nat(gas_film, L_gas, T_wd - T_u, nu_vertical_plate, gas_film.Pr)
        Q_wd_u = h_wu * A_dry * (T_wd - T_u)

        # ullage -> interface (horizontal plate; warm gas above cold surface
        # is stable stratification)
        L_int = max(np.sqrt(A_int / np.pi) / 2.0, 1e-3)
        h_ui = h_nat(sat_v, L_int, T_u - T_i, nu_horizontal_plate, T_u < T_i)
        h_ui *= cfg.ullage_interface_enhancement
        Q_u_i = h_ui * A_int * (T_u - T_i) * mixing_factor

        # wet wall -> liquid (vertical wall, liquid side); the clip keeps the
        # film state liquid — degenerate range (very low P) falls back to T_i
        liq_hi = max(T_i - 1e-3, f.T_triple + 0.6)
        liq_film_T = float(np.clip(0.5 * (T_ww + T_b), f.T_triple + 0.5, liq_hi))
        try:
            liq_film = f.state_TP(liq_film_T, P)
        except ValueError:
            liq_film = sat_l
        h_wl = h_nat(liq_film, L_liq, T_ww - T_b, nu_vertical_plate, liq_film.Pr)
        Q_ww_l = h_wl * A_wet * (T_ww - T_b)

        # surface layer -> interface (warm liquid below colder interface is
        # unstable when T_s > T_i)
        h_si = h_nat(sat_l, L_int, T_s - T_i, nu_horizontal_plate, T_s > T_i)
        Q_s_i = h_si * A_int * (T_s - T_i) * mixing_factor

        # surface layer <-> bulk (conduction across the slab + slosh mixing)
        h_bs = sat_l.k / cfg.surface_layer_thickness * mixing_factor
        Q_b_s = h_bs * A_int * (T_b - T_s)

        # --- interface phase change
        mdot_evap = (Q_u_i + Q_s_i) / h_fg

        # --- optional autogenous pressurant (vapor injection at setpoint)
        mdot_press = 0.0
        h_press = 0.0
        if cfg.pressurant_setpoint is not None and P < cfg.pressurant_setpoint:
            deficit = (cfg.pressurant_setpoint - P) / cfg.pressurant_setpoint
            mdot_press = cfg.pressurant_max_flow * min(1.0, 20.0 * deficit)
            T_pin = max(cfg.pressurant_temperature, T_i + 1.0)
            try:
                h_press = f.state_TP(T_pin, P).h
            except ValueError:
                h_press = sat_v.h

        # --- mass balances
        dm_u = mdot_evap + mdot_press
        dm_l = -mdot_out - mdot_evap

        # --- ullage volume rate (liquid incompressible within the step)
        dV_u = -dm_l / rho_l

        # --- energy balances
        dU_u = (
            Q_wd_u
            - Q_u_i
            + mdot_evap * sat_v.h
            + mdot_press * h_press
            - P * dV_u
        )
        dT_s = (
            chi * Q_ww_l + Q_b_s - Q_s_i - mdot_evap * cp_l * max(T_i - T_s, 0.0)
        ) / (m_s * cp_l)
        dT_b = ((1.0 - chi) * Q_ww_l - Q_b_s) / (m_b * cp_l)

        # --- wall nodes, incl. axial dry->wet wall conduction across the
        # liquid line (dominant moderator of ullage superheat at high fill)
        cw_d = wall_cp(cfg.wall_material, T_wd)
        cw_w = wall_cp(cfg.wall_material, T_ww)
        A_tot = geom.A_wall_total
        mw_d = self.wall_mass * A_dry / A_tot
        mw_w = self.wall_mass * A_wet / A_tot
        L_cond = cfg.wall_conduction_length or geom.height / 4.0
        k_w = wall_k(cfg.wall_material, 0.5 * (T_wd + T_ww))
        perim = 2.0 * np.pi * geom.surface_radius(h_fill)
        Q_wd_ww = k_w * cfg.wall_thickness * perim / L_cond * (T_wd - T_ww)
        dT_wd = (self.q_env * A_dry - Q_wd_u - Q_wd_ww) / max(mw_d * cw_d, 1e-3)
        dT_ww = (self.q_env * A_wet - Q_ww_l + Q_wd_ww) / max(mw_w * cw_w, 1e-3)

        return np.array([dm_u, dU_u, dm_l, dT_b, dT_s, dT_wd, dT_ww])

    # -------------------------------------------------------------- simulate
    def simulate(
        self,
        t_span: tuple[float, float],
        y0: np.ndarray,
        mdot_out=0.0,
        accel=G0,
        mixing_factor=1.0,
        n_out: int = 200,
        rtol: float = 1e-6,
    ) -> "TankHistory":
        """Integrate the tank ODEs.

        ``mdot_out``, ``accel`` and ``mixing_factor`` may be scalars or
        callables of time (the coupled simulator passes callables).
        """
        as_fun = lambda v: v if callable(v) else (lambda t, _v=v: _v)
        f_out, f_acc, f_mix = as_fun(mdot_out), as_fun(accel), as_fun(mixing_factor)

        def fun(t, y):
            return self.rhs(t, y, f_out(t), f_acc(t), f_mix(t))

        t_eval = np.linspace(*t_span, n_out)
        sol = solve_ivp(
            fun, t_span, y0, method="RK45", t_eval=t_eval,
            rtol=rtol, atol=1e-8 * np.maximum(np.abs(y0), 1.0), max_step=(t_span[1]-t_span[0])/50,
        )
        if not sol.success:
            raise RuntimeError(f"tank integration failed: {sol.message}")
        return self._postprocess(sol.t, sol.y, f_out, f_acc, f_mix)

    def _postprocess(self, t, Y, f_out, f_acc, f_mix) -> "TankHistory":
        f, geom = self.fluid, self.geom
        n = len(t)
        P = np.empty(n); T_u = np.empty(n); fill = np.empty(n)
        boiloff = np.empty(n); h_fill = np.empty(n)
        for i in range(n):
            m_u, U_u, m_l, T_b, T_s, T_wd, T_ww = Y[:, i]
            rho_l = f.sat_liquid(max(f.P_sat(min(T_b, f.T_crit - 0.5)), 1000.0)).rho
            V_l = m_l / rho_l
            V_u = max(geom.V_total - V_l,
                      self.cfg.min_ullage_fraction * geom.V_total)
            P[i], T_u[i] = self._ullage_PT(m_u, U_u, V_u)
            fill[i] = V_l / geom.V_total
            h_fill[i] = geom.height_at_volume(V_l)
            # dm_l/dt = -mdot_out - mdot_evap  =>  mdot_evap = -dm_l/dt - mdot_out
            deriv = self.rhs(t[i], Y[:, i], f_out(t[i]), f_acc(t[i]), f_mix(t[i]))
            boiloff[i] = -deriv[2] - f_out(t[i])
        return TankHistory(
            t=t, P=P, T_ullage=T_u, T_bulk=Y[3], T_surface=Y[4],
            T_wall_dry=Y[5], T_wall_wet=Y[6], m_ullage=Y[0], m_liquid=Y[2],
            fill_fraction=fill, fill_height=h_fill, boiloff_rate=boiloff,
        )


@dataclass
class TankHistory:
    """Time histories from a tank simulation (SI units)."""

    t: np.ndarray
    P: np.ndarray
    T_ullage: np.ndarray
    T_bulk: np.ndarray
    T_surface: np.ndarray
    T_wall_dry: np.ndarray
    T_wall_wet: np.ndarray
    m_ullage: np.ndarray
    m_liquid: np.ndarray
    fill_fraction: np.ndarray
    fill_height: np.ndarray
    boiloff_rate: np.ndarray

    def pressure_rise_rate(self, t_start_frac: float = 0.3) -> float:
        """Quasi-steady dP/dt [Pa/s] from a linear fit of the late history."""
        i0 = int(len(self.t) * t_start_frac)
        return float(np.polyfit(self.t[i0:], self.P[i0:], 1)[0])


# --------------------------------------------------------------------------
# Homogeneous (thermal-equilibrium) reference model
# --------------------------------------------------------------------------
def homogeneous_pressure_rise_rate(
    fluid: Fluid, V_tank: float, fill_fraction: float, P: float, Q_dot: float
) -> float:
    """Analytic pressure-rise rate [Pa/s] of the *homogeneous* model.

    All tank contents in thermodynamic equilibrium (saturated mixture) in a
    rigid vessel receiving heat Q_dot:  dP/dt = Q_dot / (m * du/dP|_rho).
    This is the classical lower-bound reference used in the NASA LeRC
    self-pressurization reports (Hasan, Lin & Van Dresar 1991).
    """
    from CoolProp.CoolProp import PropsSI

    sl, sv = fluid.sat_liquid(P), fluid.sat_vapor(P)
    V_l = fill_fraction * V_tank
    V_u = V_tank - V_l
    m = sl.rho * V_l + sv.rho * V_u
    rho = m / V_tank
    dP = 1e-3 * P
    u1 = PropsSI("Umass", "P", P - dP, "Dmass", rho, fluid.name)
    u2 = PropsSI("Umass", "P", P + dP, "Dmass", rho, fluid.name)
    du_dP = (u2 - u1) / (2 * dP)
    return Q_dot / (m * du_dP)


def simulate_homogeneous(
    fluid: Fluid, V_tank: float, fill_fraction: float, P0: float,
    Q_dot: float, t_end: float, n_out: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the homogeneous model: returns (t, P)."""
    from CoolProp.CoolProp import PropsSI

    sl, sv = fluid.sat_liquid(P0), fluid.sat_vapor(P0)
    m = sl.rho * fill_fraction * V_tank + sv.rho * (1 - fill_fraction) * V_tank
    rho = m / V_tank
    U0 = sl.u * sl.rho * fill_fraction * V_tank + sv.u * sv.rho * (
        1 - fill_fraction) * V_tank

    def rhs(t, y):
        return [Q_dot]

    t = np.linspace(0.0, t_end, n_out)
    U = U0 + Q_dot * t
    P = np.array([
        PropsSI("P", "Dmass", rho, "Umass", Ui / m, fluid.name) for Ui in U
    ])
    return t, P
