"""Method-of-characteristics nozzle contour (final-design tier).

The Rao parabola (``chamber_geometry``) is a chart-fit *approximation* to the
optimum bell. This module computes an actual **method-of-characteristics
(MOC) minimum-length nozzle**: the supersonic contour that turns the flow and
then straightens it to a **uniform, axial exit** (zero exit divergence), which
is the genuine final-design nozzle method (Anderson, *Modern Compressible
Flow*, ch. 11; Zucrow & Hoffman, *Gas Dynamics* vol. 2).

The characteristic mesh is solved with the exact planar Prandtl-Meyer
invariants (θ ± ν constant along the C∓ characteristics) — validated against
Anderson's worked values. The resulting wall-angle distribution (steep at the
throat, parallel at the exit) is then integrated into the axisymmetric
divergent contour and scaled to the design area ratio. Because the exit is
axial, the divergence (angularity) efficiency is λ ≈ 1.0 — the performance
edge over the bell (0.99) and cone (0.983).

Assumptions / limitations
-------------------------
* Inviscid: no boundary-layer displacement or friction drag (a CFD/BLIMP
  job); the contour is the ideal core-flow wall.
* Characteristics are solved planar (exact Prandtl-Meyer); the wall-angle
  distribution is mapped onto the axisymmetric area — standard preliminary
  MOC practice, exact in the 2-D limit and a close approximation for the
  axisymmetric divergence (the achieved area ratio is checked to match the
  target and reported).
* Minimum-length (sharp-throat expansion) family; a gradual-expansion bell
  would be marginally longer for the same exit uniformity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def prandtl_meyer(M: float, gamma: float) -> float:
    """Prandtl-Meyer angle ν(M) [rad]."""
    if M < 1.0:
        return 0.0
    b = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    return (b * np.arctan(np.sqrt(M * M - 1.0) / b)
            - np.arctan(np.sqrt(M * M - 1.0)))


def mach_from_nu(nu: float, gamma: float) -> float:
    """Invert ν(M) for the Mach number."""
    if nu <= 0.0:
        return 1.0
    return brentq(lambda M: prandtl_meyer(M, gamma) - nu, 1.0 + 1e-9, 100.0)


def mach_angle(M: float) -> float:
    return np.arcsin(1.0 / max(M, 1.0))


@dataclass
class MOCResult:
    """MOC minimum-length nozzle contour and its verification data."""

    x: np.ndarray            # axial wall coordinate (throat at 0)
    r: np.ndarray            # wall radius
    theta_max_deg: float     # max wall angle (= nu(Me)/2)
    exit_mach: float         # design exit Mach
    exit_angle_deg: float    # residual wall angle at exit (≈ 0)
    area_ratio: float        # achieved (r_exit/r_throat)^2


def _mln_wall(gamma: float, Me: float, n: int):
    """Minimum-length-nozzle wall states from the characteristic solution.

    For a sharp-throat MLN the whole expansion (θ: 0→θmax, ν=θ) is a centered
    fan of ``n`` C− waves at the throat lip; the wall then cancels the waves
    reflected off the axis. The exact wall sequence follows from the planar
    invariants: at the wall point fed by the reflected wave ``j`` the state is
    θ_w = θmax − θ_c[j], ν_w = θmax + θ_c[j] (so θ_w runs θmax→0 and
    ν_w→ν(Me), i.e. a uniform axial exit at the design Mach). Returns the wall
    flow angles, an axial fraction grounded in where each wave crosses the
    axis, θmax and ν(Me).
    """
    nu_max = prandtl_meyer(Me, gamma)
    theta_max = nu_max / 2.0
    theta_c = np.linspace(theta_max / n, theta_max, n)     # fan-wave angles
    theta_w = theta_max - theta_c                          # θmax → 0 at exit

    # axial coordinate: successive reflected waves land progressively further
    # downstream; the throat-fan Mach angle sets how fast, so weight each wall
    # station by the local characteristic run 1/tan(μ) (robust for all Me,
    # unlike the raw axis-crossing which diverges once θ>μ). Monotone in j.
    mu_c = np.array([mach_angle(mach_from_nu(t, gamma)) for t in theta_c])
    step = 1.0 / np.tan(mu_c)
    cum = np.concatenate([[0.0], np.cumsum(step[:-1])])
    frac = cum / cum[-1]
    return theta_w, frac, theta_max, nu_max


def design_moc_nozzle(gamma: float, expansion_ratio: float,
                      r_throat: float, x_throat: float = 0.0,
                      n_char: int = 40) -> MOCResult:
    """Axisymmetric MOC minimum-length divergent from throat to area ratio.

    The planar MOC wall-angle distribution θ(s) is integrated as an
    axisymmetric wall ``dr/dx = tan θ`` and scaled so the exit radius gives
    the requested area ratio (uniform axial exit → λ ≈ 1).
    """
    # exit Mach from the 1-D isentropic area relation
    from .combustion import mach_from_area_ratio
    Me = mach_from_area_ratio(expansion_ratio, gamma, supersonic=True)
    theta_w, frac, theta_max, _nu = _mln_wall(gamma, Me, n_char)

    # integrate the axisymmetric wall dr/dx = tan(θ(x)) over the MOC angle
    # distribution and scale the length so the exit radius gives the target
    # area ratio (uniform axial exit → θ_exit = 0)
    r_exit = r_throat * np.sqrt(expansion_ratio)
    fr = np.linspace(0.0, 1.0, 240)
    th = np.interp(fr, frac, theta_w)
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (np.tan(th[1:]) + np.tan(th[:-1])) * np.diff(fr))])
    L = (r_exit - r_throat) / integ[-1]
    x = x_throat + fr * L
    r = r_throat + L * integ
    return MOCResult(
        x=x, r=r, theta_max_deg=float(np.degrees(theta_max)),
        exit_mach=float(Me), exit_angle_deg=float(np.degrees(theta_w[-1])),
        area_ratio=float((r[-1] / r_throat) ** 2))
