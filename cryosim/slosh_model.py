"""Mechanical-analog lateral slosh model for upright cylindrical tanks.

Implements the classical equivalent-pendulum / spring-mass representation of
linear (small-amplitude) lateral sloshing per NASA SP-8009 ("Propellant Slosh
Loads", Dodge 1968), NASA SP-106 (Abramson 1966, ch. 2) and F.T. Dodge, "The
New Dynamic Behavior of Liquids in Moving Containers" (SwRI, 2000, ch. 1-4).

For a rigid upright circular cylinder (radius R, equivalent flat-bottom
liquid depth h, axial acceleration a replacing g in flight), with xi_n the
n-th root of J1'(xi) = 0 (xi_1 = 1.8412):

* natural frequency      omega_n^2 = (xi_n a / R) tanh(xi_n h / R)
* sloshing modal mass    m_n = m_liq 2 tanh(xi_n h/R) / [(h/R) xi_n (xi_n^2-1)]
* pendulum length        L_n = a / omega_n^2 = R / (xi_n tanh(xi_n h/R))
* slosh-mass depth below the free surface  d_n = (R/xi_n) tanh(xi_n h/(2R))
* rigid mass             m_0 = m_liq - sum m_n, located to conserve the CG.

Viscous smooth-wall damping ratio via the Mikishev-Dorozhkin correlation as
given in Dodge (2000):

  zeta_1 = 0.79 sqrt(nu / sqrt(a R^3)) *
           [1 + 0.318/sinh(1.84 h/R) (1 + (1 - h/R)/cosh(1.84 h/R))]

(empirical; literature scatter of order +/-30%. Baffled-tank damping is NOT
modeled — pass a user-defined damping ratio instead.)

Assumptions / limitations
-------------------------
* Linear lateral slosh only: no rotary/swirl, no nonlinear or breaking-wave
  regimes (valid for wave amplitude << R), no vortex/PMD interaction.
* Rigid, axisymmetric tank; single dominant mode analysis (first 3 modes
  computed, first mode integrated in time).
* Dome ends handled through the SP-8009 equivalent flat-bottom depth
  (equal liquid volume); accuracy degrades when the free surface is inside
  a dome — flagged via ``in_dome``.
* Bond number is assumed large (gravity/thrust-dominated regime, true for
  0.2-0.5 m tanks under >= 1 g axial acceleration); no capillary effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jnp_zeros

from .fluids import Fluid
from .tank_geometry import TankGeometry

G0 = 9.80665

#: First three roots of J1'(xi) = 0 (computed once via scipy; the classical
#: tabulated values are 1.8412, 5.3314, 8.5363).
XI_N = jnp_zeros(1, 3)


@dataclass
class SloshMode:
    """Equivalent-pendulum parameters of one lateral slosh mode."""

    n: int
    omega: float        # rad/s
    frequency: float    # Hz
    mass: float         # kg (sprung/pendulum mass)
    pendulum_length: float  # m
    depth_below_surface: float  # m (slosh-mass location)
    height_above_bottom: float  # m


@dataclass
class SloshParameters:
    """Full mechanical-analog parameter set at one fill/acceleration state."""

    R: float
    h: float                    # equivalent flat-bottom depth
    accel: float
    m_liquid: float
    modes: list[SloshMode]
    m_rigid: float
    rigid_height: float         # rigid-mass height above tank bottom
    zeta_viscous: float         # first-mode smooth-wall damping ratio
    in_dome: bool = False       # free surface inside a dome: analog degraded

    @property
    def first(self) -> SloshMode:
        return self.modes[0]

    @property
    def slosh_mass_fraction(self) -> float:
        return self.first.mass / self.m_liquid


def mikishev_dorozhkin_zeta(nu: float, R: float, h: float, accel: float) -> float:
    """Smooth-wall viscous damping ratio of the first lateral mode.

    Mikishev & Dorozhkin (1961) correlation as given in Dodge (2000);
    empirical, scatter of order +/-30%.
    """
    if h <= 0 or accel <= 0:
        return 0.0
    hR = h / R
    base = 0.79 * np.sqrt(nu / np.sqrt(accel * R**3))
    depth_corr = 1.0 + 0.318 / np.sinh(1.84 * hR) * (
        1.0 + (1.0 - hR) / np.cosh(1.84 * hR)
    )
    return float(base * depth_corr)


def slosh_parameters(
    rho: float,
    nu: float,
    R: float,
    h: float,
    accel: float = G0,
    n_modes: int = 3,
    in_dome: bool = False,
) -> SloshParameters:
    """Mechanical-analog parameters for liquid of density rho [kg/m^3] and
    kinematic viscosity nu [m^2/s] at equivalent depth h in a cylinder of
    radius R under axial acceleration accel."""
    if h <= 0:
        raise ValueError("liquid depth must be positive")
    m_liq = rho * np.pi * R**2 * h
    hR = h / R
    modes = []
    for i in range(n_modes):
        xi = XI_N[i]
        tanh_term = np.tanh(xi * hR)
        omega2 = xi * accel / R * tanh_term
        omega = np.sqrt(omega2)
        m_n = m_liq * 2.0 * tanh_term / (hR * xi * (xi**2 - 1.0))
        L_n = R / (xi * tanh_term)
        d_n = R / xi * np.tanh(xi * hR / 2.0)
        modes.append(
            SloshMode(
                n=i + 1,
                omega=float(omega),
                frequency=float(omega / (2 * np.pi)),
                mass=float(m_n),
                pendulum_length=float(L_n),
                depth_below_surface=float(d_n),
                height_above_bottom=float(h - d_n),
            )
        )
    m_rigid = m_liq - sum(m.mass for m in modes)
    # rigid-mass height from CG conservation: m_liq h/2 = m0 H0 + sum m_n H_n
    H0 = (m_liq * h / 2.0 - sum(m.mass * m.height_above_bottom for m in modes)) / m_rigid
    zeta = mikishev_dorozhkin_zeta(nu, R, h, accel)
    return SloshParameters(
        R=R, h=h, accel=accel, m_liquid=m_liq, modes=modes,
        m_rigid=float(m_rigid), rigid_height=float(H0),
        zeta_viscous=zeta, in_dome=in_dome,
    )


class SloshModel:
    """Slosh analog bound to a tank geometry and fluid.

    Uses the SP-8009 equivalent flat-bottom depth for the current liquid
    volume; ``params_at`` flags (``in_dome``) states where the free surface
    is inside a dome and the cylinder analog is only approximate.
    """

    def __init__(self, fluid: Fluid, geometry: TankGeometry,
                 zeta_override: float | None = None):
        self.fluid = fluid
        self.geom = geometry
        #: user-supplied damping (e.g. baffled tank); None = smooth-wall corr.
        self.zeta_override = zeta_override

    def params_at(
        self, V_liquid: float, P: float, accel: float = G0, n_modes: int = 3
    ) -> SloshParameters:
        sl = self.fluid.sat_liquid(P)
        h_eq = self.geom.equivalent_depth(V_liquid)
        h_actual = self.geom.height_at_volume(V_liquid)
        p = slosh_parameters(
            rho=sl.rho, nu=sl.mu / sl.rho, R=self.geom.R, h=h_eq,
            accel=accel, n_modes=n_modes,
            in_dome=self.geom.surface_in_dome(h_actual),
        )
        if self.zeta_override is not None:
            p.zeta_viscous = self.zeta_override
        return p


# --------------------------------------------------------------------------
# First-mode time-domain response (linear pendulum analog)
# --------------------------------------------------------------------------
@dataclass
class SloshHistory:
    t: np.ndarray
    displacement: np.ndarray      # slosh-mass lateral displacement x1 [m]
    cg_offset: np.ndarray         # lateral CG shift of the liquid [m]
    force: np.ndarray             # lateral force on the tank [N]
    frequency: np.ndarray         # instantaneous first-mode frequency [Hz]
    slosh_mass: np.ndarray        # instantaneous slosh mass [kg]
    zeta: np.ndarray


def simulate_first_mode(
    params_of_t,
    lateral_accel,
    t_span: tuple[float, float],
    n_out: int = 500,
) -> SloshHistory:
    """Integrate the first-mode oscillator with slowly-varying parameters.

      x1'' + 2 zeta omega x1' + omega^2 x1 = -a_lat(t)

    ``params_of_t``: callable t -> SloshParameters (quasi-static fill/accel
    variation; valid while parameter time scales >> slosh period).
    ``lateral_accel``: callable t -> lateral acceleration of the tank [m/s^2].
    """

    def rhs(t, y):
        p = params_of_t(t)
        w = p.first.omega
        z = p.zeta_viscous
        x, v = y
        return [v, -2.0 * z * w * v - w**2 * x - lateral_accel(t)]

    t_eval = np.linspace(*t_span, n_out)
    sol = solve_ivp(rhs, t_span, [0.0, 0.0], t_eval=t_eval, rtol=1e-8,
                    atol=1e-12, max_step=(t_span[1] - t_span[0]) / 200)
    if not sol.success:
        raise RuntimeError(f"slosh integration failed: {sol.message}")

    n = len(sol.t)
    cg = np.empty(n); F = np.empty(n); fr = np.empty(n)
    ms = np.empty(n); zz = np.empty(n)
    for i, ti in enumerate(sol.t):
        p = params_of_t(ti)
        m1, w = p.first.mass, p.first.omega
        x = sol.y[0, i]
        cg[i] = m1 * x / p.m_liquid
        # lateral force via spring analog: F = m1 omega^2 x  (+ rigid-mass
        # inertial reaction handled by the vehicle model, not here)
        F[i] = m1 * w**2 * x
        fr[i] = p.first.frequency
        ms[i] = m1
        zz[i] = p.zeta_viscous
    return SloshHistory(
        t=sol.t, displacement=sol.y[0], cg_offset=cg, force=F,
        frequency=fr, slosh_mass=ms, zeta=zz,
    )
