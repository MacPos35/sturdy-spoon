"""1D marched regenerative-cooling model (Bartz / Dittus-Boelter / 1D wall).

Standard reduced-order thrust-chamber cooling analysis, per:

* D.R. Bartz, "A Simple Equation for Rapid Estimation of Rocket Nozzle
  Convective Heat Transfer Coefficients", Jet Propulsion 27(1), 1957.
* NASA SP-8087, "Liquid Rocket Engine Fluid-Cooled Combustion Chambers".
* Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant Rocket
  Engines", ch. 4 (land/fin treatment, coolant-side sizing).

At every axial station the three-resistance balance is solved for the wall:

  q = h_g (T_aw - T_wg) = (k_w / t_w)(T_wg - T_wc) = eta h_c (T_wc - T_co)

with q per unit *hot-gas-side* area, Bartz h_g (with the sigma property
correction, which depends on T_wg — solved implicitly), Dittus-Boelter
coolant-side h_c (optional Sieder-Tate viscosity-ratio and helical-curvature
corrections), and the channel lands treated as straight fins (efficiency
eta_f = tanh(m b)/(m b)). Coolant enthalpy and pressure are marched from
station to station (Haaland friction factor + momentum/acceleration term);
properties are re-evaluated locally with CoolProp, which captures methane's
transcritical property variation.

Assumptions / limitations (stated per the no-CFD design choice)
---------------------------------------------------------------
* Quasi-1D: no circumferential variation, no streamline curvature effects,
  no injector-region boundary-layer development (Bartz is known to
  over-predict near the injector and, for CH4, by ~20-30% overall —
  conservative for cooling design; see ODREC, Appl. Sci. 14(1):71, 2024;
  a ``bartz_factor`` calibration knob exposes that documented correction).
* Film cooling: optionally credited via :class:`FilmCooling` — a
  gaseous-film effectiveness decay (Hatch & Papell, NASA TN D-130) with
  no liquid-film run and no latent-heat credit, i.e. deliberately the
  conservative end of film-cooling practice.
* 1D wall conduction through the hot wall only; no axial conduction, no
  closeout-side heat loss (adiabatic outer wall), radiation neglected.
* Single-phase coolant: no boiling model. Transcritical property variation
  is captured through CoolProp, but heat-transfer deterioration near the
  pseudo-critical point is NOT modeled (documented known limitation for
  near-critical methane).
* Steady state: the coupled simulator calls this quasi-steadily (regen
  thermal time constants ~ms << tank time scales ~s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .chamber_geometry import ChamberContour, CoolingChannels
from .combustion import CombustionGas, adiabatic_wall_temperature, mach_from_area_ratio
from .fluids import Fluid


def bartz_h_g(
    gas: CombustionGas,
    Pc: float,
    Dt: float,
    r_curv: float,
    area_ratio: float,
    M: float,
    T_wg: float,
) -> float:
    """Bartz (1957) hot-gas-side convective coefficient [W/(m^2 K)]."""
    g = gas.gamma
    sigma = 1.0 / (
        (0.5 * (T_wg / gas.T_c) * (1.0 + (g - 1.0) / 2.0 * M**2) + 0.5) ** 0.68
        * (1.0 + (g - 1.0) / 2.0 * M**2) ** 0.12
    )
    return (
        0.026
        / Dt**0.2
        * (gas.mu**0.2 * gas.cp / gas.Pr**0.6)
        * (Pc / gas.c_star) ** 0.8
        * (Dt / r_curv) ** 0.1
        * (1.0 / area_ratio) ** 0.9
        * sigma
    )


def dittus_boelter_nu(Re: float, Pr: float) -> float:
    """Nu = 0.023 Re^0.8 Pr^0.4 (heating; Incropera & DeWitt eq. 8.60)."""
    return 0.023 * Re**0.8 * Pr**0.4


def haaland_friction_factor(Re: float, rel_roughness: float) -> float:
    """Haaland explicit approximation to Colebrook (Darcy f)."""
    Re = max(Re, 100.0)
    if Re < 2300.0:
        return 64.0 / Re
    inv_sqrt = -1.8 * np.log10((rel_roughness / 3.7) ** 1.11 + 6.9 / Re)
    return (1.0 / inv_sqrt) ** 2


def fin_efficiency(h_c: float, k_wall: float, land: float, height: float) -> float:
    """Straight-fin efficiency of a channel land (adiabatic tip)."""
    if land <= 0 or height <= 0:
        return 0.0
    m = np.sqrt(2.0 * h_c / (k_wall * land))
    mb = m * height
    return float(np.tanh(mb) / mb) if mb > 1e-9 else 1.0


@dataclass(frozen=True)
class FilmCooling:
    """Fuel-film cooling credit for the regen solution (gaseous-film model).

    The film is treated as a gaseous coolant layer from the injection point
    on: its effectiveness decays exponentially as the hot gas convects heat
    into it, with decay rate = (gas-side heat pickup per unit length) /
    (film capacity rate m_f cp_f) — the Hatch & Papell form (NASA TN D-130,
    1959, tangential-slot gaseous film cooling). The wall then sees
    ``T_aw_eff = T_aw - eta (T_aw - T_inject)``.

    Deliberately conservative at this tier:

    * no liquid-film run: a liquid film's evaporation heat sink (which holds
      the wall near the coolant boiling point over the first chamber section)
      is NOT credited — the film is "spent" faster than a real liquid film;
    * ``cp`` is the vapor-phase heat capacity only, no latent-heat credit;
    * ``T_inject`` should be the (hot) post-regen manifold temperature, not
      the tank temperature.
    """

    mdot: float          # kg/s, film flow injected at the face
    T_inject: float      # K, film temperature at injection
    cp: float            # J/(kg K), vapor-phase film heat capacity


@dataclass
class RegenResult:
    """Axial distributions (hot-gas-side area basis) and coolant march."""

    x: np.ndarray                 # m, from injector face
    r: np.ndarray                 # local inner-wall radius
    q: np.ndarray                 # W/m^2 heat flux (gas-side area)
    T_wg: np.ndarray              # hot-gas-side wall temperature
    T_wc: np.ndarray              # coolant-side wall temperature
    T_aw: np.ndarray              # adiabatic wall (recovery) temperature
    h_g: np.ndarray               # gas-side coefficient
    h_c: np.ndarray               # coolant-side coefficient
    T_coolant: np.ndarray         # coolant bulk temperature
    P_coolant: np.ndarray         # coolant static pressure
    velocity: np.ndarray          # coolant channel velocity
    mach: np.ndarray              # hot-gas Mach number
    Q_total: float                # W, total heat picked up
    dP_total: float               # Pa, coolant pressure drop
    coolant_inlet: object         # FluidState
    coolant_outlet: object        # FluidState
    #: True if any station's coolant state fell inside the vapor dome —
    #: single-phase correlations are invalid there (no boiling model!).
    boiling_detected: bool = False
    #: True if the marched coolant pressure hit the numerical floor: the
    #: feed pressure cannot sustain the flow (dP spiral as hot coolant
    #: density drops) and velocities/states downstream of the collapse are
    #: NOT physical. Raise the inlet pressure or enlarge the channels.
    pressure_collapsed: bool = False
    #: Film-cooling effectiveness eta(x) when a film credit was applied
    #: (None otherwise). T_aw above is then the film-reduced value.
    film_effectiveness: np.ndarray | None = None

    @property
    def peak_wall_temperature(self) -> float:
        return float(self.T_wg.max())

    @property
    def coolant_temperature_rise(self) -> float:
        return float(self.coolant_outlet.T - self.coolant_inlet.T)


class RegenCoolingModel:
    """Marched regen-cooling solution for one chamber + channel layout."""

    def __init__(
        self,
        contour: ChamberContour,
        channels: CoolingChannels,
        gas: CombustionGas,
        coolant: Fluid,
        counterflow: bool = True,
        sieder_tate: bool = True,
        film: FilmCooling | None = None,
        bartz_factor: float = 1.0,
    ):
        """``film``: optional film-cooling credit (see :class:`FilmCooling`).

        ``bartz_factor``: calibration multiplier on the Bartz coefficient.
        Bartz is known to over-predict LOX/CH4 heat flux by ~20-30% (ODREC,
        Appl. Sci. 14(1):71, 2024) — 0.8 is the documented calibration for
        that propellant class; the default 1.0 keeps Bartz's conservatism.
        """
        channels.validate_against(contour)
        self.contour = contour
        self.channels = channels
        self.gas = gas
        self.coolant = coolant
        self.counterflow = counterflow
        self.sieder_tate = sieder_tate
        self.film = film
        self.bartz_factor = float(bartz_factor)

        # Precompute hot-gas station quantities (independent of coolant).
        ar = contour.area_ratio()
        sup = contour.is_supersonic()
        self._mach = np.array([
            mach_from_area_ratio(a, gas.gamma, s) for a, s in zip(ar, sup)
        ])
        self._T_aw = np.array([
            adiabatic_wall_temperature(gas, M) for M in self._mach
        ])

    def solve(
        self,
        Pc: float,
        mdot_coolant: float,
        T_inlet: float,
        P_inlet: float,
    ) -> RegenResult:
        """March the coolant through the jacket at chamber pressure Pc."""
        ct, ch, gas, f = self.contour, self.channels, self.gas, self.coolant
        n = len(ct.x)
        order = range(n - 1, -1, -1) if self.counterflow else range(n)

        # Film-cooling credit: effectiveness decays downstream of the face
        # as the hot gas convects heat into the film (Hatch-Papell form,
        # NASA TN D-130): eta(x) = exp(-int h_g 2 pi r dx / (m_f cp_f)).
        # h_g for the decay integral is evaluated at a representative wall
        # temperature (0.8 T_aw); the Bartz sigma is only weakly sensitive.
        eta = None
        T_aw_arr = self._T_aw
        if self.film is not None and self.film.mdot > 0.0:
            eta = np.empty(n)
            cap = self.film.mdot * self.film.cp          # W/K
            expo = 0.0
            for j in range(n):
                eta[j] = np.exp(-expo / cap)
                if j < n - 1:
                    hg_j = self.bartz_factor * bartz_h_g(
                        gas, Pc, ct.Dt, ct.r_curv_throat,
                        ct.area[j] / ct.At, self._mach[j],
                        0.8 * self._T_aw[j])
                    expo += hg_j * 2.0 * np.pi * ct.r[j] \
                        * (ct.x[j + 1] - ct.x[j])
            T_aw_arr = self._T_aw - eta * (self._T_aw - self.film.T_inject)

        st = f.state_TP(T_inlet, P_inlet)
        inlet_state = st
        h_bulk, P = st.h, P_inlet

        q = np.zeros(n); T_wg = np.zeros(n); T_wc = np.zeros(n)
        h_g_arr = np.zeros(n); h_c_arr = np.zeros(n)
        T_co = np.zeros(n); P_co = np.zeros(n); vel = np.zeros(n)

        mdot_ch = mdot_coolant / ch.n_channels
        A_ch = ch.flow_area
        G = mdot_ch / A_ch  # channel mass flux
        boiling = False
        collapsed = False

        x = ct.x
        for k, i in enumerate(order):
            st = f.state_PH(P, h_bulk)
            boiling = boiling or st.two_phase
            v = G / st.rho
            Re = G * ch.D_h / st.mu
            # near the critical point the EOS can return non-physical cp
            # (hence Pr) excursions; clamp so Pr^0.4 stays real and bounded
            Pr = float(np.clip(st.Pr, 0.2, 100.0))
            Nu = dittus_boelter_nu(Re, Pr)
            h_c0 = Nu * st.k / ch.D_h
            # helical curvature enhancement (Niino/ Ito-type, mild):
            if ch.helix_angle_deg > 0:
                h_c0 *= 1.0 + 3.5 * ch.D_h / (2.0 * ct.r[i])

            land = ch.land_width(ct.r[i])
            # effective coolant-side area per unit gas-side area:
            # channels + land fins vs 2*pi*r of hot wall
            per_gas = 2.0 * np.pi * ct.r[i]

            T_aw = T_aw_arr[i]
            M = self._mach[i]
            ar = ct.area[i] / ct.At

            def residual(Twg):
                hg = self.bartz_factor * bartz_h_g(
                    gas, Pc, ct.Dt, ct.r_curv_throat, ar, M, Twg)
                qq = hg * (T_aw - Twg)
                Twc = Twg - qq * ch.t_wall / ch.k_wall
                hc = h_c0
                if self.sieder_tate:
                    # viscosity-ratio correction (mu/mu_wall)^0.14, clamped
                    try:
                        mu_w = f.state_TP(
                            float(np.clip(Twc, f.T_triple + 1, f.T_crit * 3)), P
                        ).mu
                        hc = h_c0 * float(np.clip((st.mu / mu_w) ** 0.14, 0.5, 2.0))
                    except ValueError:
                        pass
                eta_f = fin_efficiency(hc, ch.k_wall, land, ch.channel_height)
                A_cool_eff = (ch.channel_width + 2.0 * eta_f * ch.channel_height) \
                    * ch.n_channels * ch.path_factor / per_gas
                # coolant-side flux balance (per gas-side area):
                return qq - hc * A_cool_eff * (Twc - st.T)

            T_hi = T_aw - 1e-3
            try:
                Twg = brentq(residual, st.T + 1e-3, T_hi, xtol=1e-3)
            except ValueError:
                # no sign change: wall essentially at coolant/adiabatic limit
                Twg = st.T + 1.0 if residual(st.T + 1.0) < 0 else T_hi

            hg = self.bartz_factor * bartz_h_g(
                gas, Pc, ct.Dt, ct.r_curv_throat, ar, M, Twg)
            qi = hg * (T_aw - Twg)
            Twc = Twg - qi * ch.t_wall / ch.k_wall

            q[i], T_wg[i], T_wc[i] = qi, Twg, Twc
            h_g_arr[i] = hg
            h_c_arr[i] = h_c0
            T_co[i], P_co[i], vel[i] = st.T, P, v

            # --- march to next station
            if k < n - 1:
                dx = abs(x[i] - x[order[k + 1]])
                dA_gas = 2.0 * np.pi * ct.r[i] * dx
                h_bulk += qi * dA_gas / mdot_coolant
                # pressure drop over the physical coolant path length
                dL = dx * ch.path_factor
                fD = haaland_friction_factor(Re, ch.roughness / ch.D_h)
                dP_fric = fD * dL / ch.D_h * 0.5 * st.rho * v**2
                st_next = f.state_PH(max(P - dP_fric, 1e4), h_bulk)
                dP_mom = G**2 * (1.0 / st_next.rho - 1.0 / st.rho)
                P = P - dP_fric - dP_mom
                if P <= 1e4:
                    P = 1e4
                    collapsed = True

        outlet_state = f.state_PH(P, h_bulk)
        dA = 2.0 * np.pi * ct.r * np.gradient(x)
        return RegenResult(
            x=x, r=ct.r.copy(), q=q, T_wg=T_wg, T_wc=T_wc,
            T_aw=T_aw_arr.copy(), h_g=h_g_arr, h_c=h_c_arr,
            T_coolant=T_co, P_coolant=P_co, velocity=vel, mach=self._mach.copy(),
            Q_total=float(np.sum(q * dA)),
            dP_total=float(P_inlet - P),
            coolant_inlet=inlet_state, coolant_outlet=outlet_state,
            boiling_detected=boiling or outlet_state.two_phase,
            pressure_collapsed=collapsed,
            film_effectiveness=None if eta is None else eta,
        )
