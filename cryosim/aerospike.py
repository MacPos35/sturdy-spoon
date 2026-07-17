"""Aerospike (plug-nozzle) engine: spike contour, annular chamber, losses.

The aerospike is the configuration LEAP 71's Noyron model is best known
for: a toroidal combustion chamber discharging through an annular throat
onto a central spike whose surface completes the expansion externally.
Because the outer plume boundary is free, the nozzle altitude-compensates
up to its design pressure ratio, and the (truncated) spike exits the flow
axially — divergence efficiency ~1 by construction.

Methods (all classical, reduced-order):

* **Spike contour** — Angelino's approximate method (G. Angelino,
  "Approximate Method for Plug Nozzle Design", AIAA Journal 2(10), 1964):
  the full expansion is a centered Prandtl-Meyer fan at the cowl lip; each
  Mach line from the lip closes the continuity balance
  ``R_lip^2 - r^2 = (A_t/pi) eps(M) M sin(theta+mu)`` on the spike surface.
  The ideal spike tip closes exactly on the axis at the design area ratio
  (``pi R_lip^2 = eps A_t``); practical spikes are truncated.
* **Truncation loss** — the removed tip's pressure thrust minus the base
  recovery, with a closed-wake base-pressure model ``P_b = P_a``
  (conservative at altitude, where a real truncated plug keeps a small
  positive base pressure; Hagemann, Immich, Nguyen & Dumnov, "Advanced
  Rocket Nozzles", J. Propulsion & Power 14(5), 1998). The loss is
  scale-invariant, so it is computed once on a unit-throat contour.
* **Annular chamber** — toroidal chamber of mean radius R_m (the throat
  annulus mean), sized by the same contraction-ratio and L* rules as the
  cylindrical chamber; both walls (outer cowl, inner spike side) are
  exposed as duck-typed "contours" the regen model can cool.
* **Stability** — thin-gap annular acoustics: azimuthal modes
  ``f_m = m a / (2 pi R_m)`` (the racetrack wavelength ``2 pi R_m / m``),
  longitudinal ``a/2L``; same n-tau band and chug guard as the
  cylindrical screen. A screening estimate only.

Assumptions / limitations
-------------------------
* Angelino's method is the standard preliminary spike design (isentropic,
  axisymmetric source-flow approximation along conical Mach surfaces) —
  a final contour would come from an MOC/CFD pass.
* Base pressure is the softest number in any reduced-order aerospike
  model; the closed-wake choice is conservative below the design
  pressure ratio and the truncation loss is reported, not hidden.
* Bartz on an annular throat uses the hydraulic scale D_t = 2 x gap and
  the shoulder arc as the curvature radius — a documented reduced-order
  choice, not a validated annular-throat correlation.
* Channel land widths on the spike use the convex-side convention of
  ``CoolingChannels`` (coolant assumed outside the wall); on the spike the
  coolant is inside, making lands ~2 t_wall/r optimistic (<2% here) —
  well inside the process-floor margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .combustion import (R_UNIV, area_ratio_from_mach, mach_from_area_ratio)
from .combustion_stability import StabilityResult, _TAU_BAND
from .moc_nozzle import mach_angle, mach_from_nu, prandtl_meyer


# ----------------------------------------------------------------------
# Spike contour (Angelino 1964)
# ----------------------------------------------------------------------

@dataclass
class SpikeContour:
    """Truncated Angelino spike + the data the loss model needs."""

    x: np.ndarray            # axial coordinate, 0 at the cowl-lip plane
    r: np.ndarray            # spike surface radius
    mach: np.ndarray         # surface Mach number
    p_ratio: np.ndarray      # surface static p / Pc
    R_lip: float             # cowl lip radius (pi R_lip^2 = eps A_t)
    r_throat: float          # spike shoulder radius
    gap: float               # geometric throat gap R_lip - r_throat
    exit_mach: float
    nu_exit: float           # Prandtl-Meyer angle of the exit Mach [rad]
    length_fraction: float   # retained fraction of the full spike length
    r_base: float            # spike radius at the truncation plane
    length_full: float       # full (ideal) spike length from the shoulder
    length: float            # truncated length from the shoulder


def design_spike_contour(gamma: float, expansion_ratio: float, A_t: float,
                         length_fraction: float = 0.30,
                         n_points: int = 160) -> SpikeContour:
    """Angelino approximate spike for an annular throat of area ``A_t``.

    ``length_fraction`` retains that fraction of the full ideal spike
    length (1.0 = full spike closing on the axis; LEAP 71-class spikes
    retain ~0.2-0.4).
    """
    if not 0.05 <= length_fraction <= 1.0:
        raise ValueError("length_fraction must be in [0.05, 1]")
    if expansion_ratio <= 1.0:
        raise ValueError("expansion_ratio must exceed 1")
    Me = mach_from_area_ratio(expansion_ratio, gamma, supersonic=True)
    nu_e = prandtl_meyer(Me, gamma)
    R_lip = float(np.sqrt(expansion_ratio * A_t / np.pi))

    Ms = np.linspace(1.0, Me, n_points)
    r = np.empty(n_points)
    x = np.empty(n_points)
    for i, M in enumerate(Ms):
        theta = nu_e - prandtl_meyer(M, gamma)      # flow angle from axis
        phi = theta + mach_angle(M)                 # Mach line angle
        eps = area_ratio_from_mach(float(M), gamma)
        r2 = R_lip**2 - (A_t / np.pi) * eps * M * np.sin(phi)
        r[i] = np.sqrt(max(r2, 0.0))
        x[i] = (R_lip - r[i]) / np.tan(phi)
    # shift so the shoulder (M=1 point) is x=0; downstream positive
    x = x - x[0]
    g = 1.0 + (gamma - 1.0) / 2.0 * Ms**2
    p_ratio = g ** (-gamma / (gamma - 1.0))

    L_full = float(x[-1])
    L = length_fraction * L_full
    keep = x <= L + 1e-12
    xk, rk, mk, pk = x[keep], r[keep], Ms[keep], p_ratio[keep]
    if xk[-1] < L - 1e-9:       # close the contour exactly at the cut plane
        r_cut = float(np.interp(L, x, r))
        m_cut = float(np.interp(L, x, Ms))
        p_cut = float(np.interp(L, x, p_ratio))
        xk = np.append(xk, L)
        rk = np.append(rk, r_cut)
        mk = np.append(mk, m_cut)
        pk = np.append(pk, p_cut)

    return SpikeContour(
        x=xk, r=rk, mach=mk, p_ratio=pk,
        R_lip=R_lip, r_throat=float(r[0]), gap=R_lip - float(r[0]),
        exit_mach=float(Me), nu_exit=float(nu_e),
        length_fraction=length_fraction, r_base=float(rk[-1]),
        length_full=L_full, length=float(xk[-1]))


def truncation_cf_loss(gamma: float, expansion_ratio: float,
                       Pa_over_Pc: float, length_fraction: float,
                       n_points: int = 400) -> float:
    """Delta-Cf lost by truncating the ideal spike (closed-wake P_b = Pa).

    The removed tip carried pressure thrust ``int (p - P_b) dA_proj``; with
    the closed-wake base pressure the loss is
    ``sum (p/Pc - Pa/Pc) pi (r_i^2 - r_i+1^2) / A_t`` over the tip —
    scale-invariant, so it is evaluated on a unit-throat contour.
    Conservative: a real truncated plug's recirculating base holds
    P_b > Pa below the design pressure ratio (Hagemann et al. 1998).
    """
    if length_fraction >= 1.0:
        return 0.0
    sp = design_spike_contour(gamma, expansion_ratio, 1.0,
                              length_fraction=1.0, n_points=n_points)
    L_cut = length_fraction * sp.length_full
    mask = sp.x >= L_cut
    xs = np.concatenate([[L_cut], sp.x[mask]])
    rs = np.concatenate([[float(np.interp(L_cut, sp.x, sp.r))],
                         sp.r[mask]])
    ps = np.concatenate([[float(np.interp(L_cut, sp.x, sp.p_ratio))],
                         sp.p_ratio[mask]])
    dCf = 0.0
    for i in range(len(xs) - 1):
        p_avg = 0.5 * (ps[i] + ps[i + 1])
        dA = np.pi * (rs[i]**2 - rs[i + 1]**2)      # projected annular ring
        dCf += max(p_avg - Pa_over_Pc, 0.0) * dA
    return float(dCf)        # already per unit A_t (Pc-normalized)


# ----------------------------------------------------------------------
# Cooled-surface duck-type for the regen model
# ----------------------------------------------------------------------

@dataclass
class AerospikeSurface:
    """One cooled wall of the aerospike engine, regen-model compatible.

    Exposes exactly the attributes ``RegenCoolingModel`` /
    ``optimize_channels`` / ``design_manifolds`` read from a
    ``ChamberContour``: ``x, r, area, At, Dt, r_curv_throat, Rt,
    x_throat, i_throat, area_ratio(), is_supersonic()``. ``area`` is the
    shared annular gas-path flow area (the same hot gas heats both
    walls); ``r`` is this wall's own radius, which is what sets channel
    packing and the per-unit-area heat pickup.
    """

    name: str
    x: np.ndarray
    r: np.ndarray
    area: np.ndarray
    At: float
    Dt: float                 # Bartz scale: 2 x throat gap (hydraulic)
    r_curv_throat: float
    Rt: float                 # min wall radius (channel-land check)
    x_throat: float
    i_throat: int
    supersonic: np.ndarray
    nozzle_type: str = "aerospike"
    theta_exit_deg: float = 0.0

    def area_ratio(self) -> np.ndarray:
        return self.area / self.At

    def is_supersonic(self) -> np.ndarray:
        return self.supersonic

    def divergence_efficiency(self) -> float:
        return 1.0            # axial exit by construction (truncation loss
        # is carried separately in Cf)


@dataclass
class AerospikeChamber:
    """Toroidal chamber + annular throat + truncated spike geometry."""

    spike: SpikeContour
    cowl: AerospikeSurface        # outer wall: injector face -> cowl lip
    inner: AerospikeSurface       # inner wall: face -> shoulder -> base
    R_mean: float                 # throat annulus mean radius
    gap_throat: float
    gap_chamber: float
    L_chamber: float              # cylindrical (annular) chamber length
    L_convergent: float
    contraction_ratio: float
    expansion_ratio: float


def design_aerospike_chamber(A_t: float, expansion_ratio: float,
                             contraction_ratio: float, L_star: float,
                             gamma: float, length_fraction: float = 0.30,
                             conv_angle_deg: float = 30.0,
                             n_points: int = 200) -> AerospikeChamber:
    """Size the annular chamber + spike and build both cooled surfaces.

    Same sizing rules as the cylindrical chamber: annulus flow area
    ``CR x A_t``, cylindrical-section length ``0.8 L*/CR`` (the convergent
    holds the remaining chamber volume), 30-deg convergent walls.
    """
    spike = design_spike_contour(gamma, expansion_ratio, A_t,
                                 length_fraction, n_points=n_points)
    R_lip, r_t = spike.R_lip, spike.r_throat
    g = spike.gap
    R_m = 0.5 * (R_lip + r_t)

    h_c = contraction_ratio * A_t / (2.0 * np.pi * R_m)
    if R_m - 0.5 * h_c < 0.2 * R_lip:
        raise ValueError(
            f"annular chamber gap {h_c*1e3:.1f} mm closes the center at "
            f"mean radius {R_m*1e3:.1f} mm - reduce the contraction ratio")
    L_c = 0.8 * L_star / contraction_ratio
    beta = np.radians(conv_angle_deg)
    L_conv = max(0.5 * (h_c - g), 0.0) / np.tan(beta)
    x_th = L_c + L_conv

    # subsonic gas path (shared by both walls): face -> throat annulus
    n_sub = max(int(0.55 * n_points), 40)
    x_sub = np.linspace(0.0, x_th, n_sub)

    def wall(radius_ch, radius_th):
        rr = np.full(n_sub, radius_ch)
        m = x_sub > L_c
        rr[m] = radius_ch + (radius_th - radius_ch) \
            * (x_sub[m] - L_c) / max(L_conv, 1e-12)
        return rr

    r_out = wall(R_m + 0.5 * h_c, R_lip)
    r_in = wall(R_m - 0.5 * h_c, r_t)
    area_sub = np.pi * (r_out**2 - r_in**2)
    area_sub[-1] = A_t                      # exact aerodynamic throat

    Dt = 2.0 * g
    r_curv = 1.5 * g

    cowl = AerospikeSurface(
        name="cowl", x=x_sub.copy(), r=r_out, area=area_sub.copy(),
        At=A_t, Dt=Dt, r_curv_throat=r_curv, Rt=float(r_out.min()),
        x_throat=x_th, i_throat=n_sub - 1,
        supersonic=np.zeros(n_sub, dtype=bool))

    # inner wall continues onto the spike (supersonic external expansion)
    x_sp = x_th + (spike.x[1:] - spike.x[0])
    a_sp = np.array([area_ratio_from_mach(float(M), gamma) * A_t
                     for M in spike.mach[1:]])
    x_all = np.concatenate([x_sub, x_sp])
    r_all = np.concatenate([r_in, spike.r[1:]])
    area_all = np.concatenate([area_sub, a_sp])
    sup = np.concatenate([np.zeros(n_sub, dtype=bool),
                          np.ones(len(x_sp), dtype=bool)])
    inner = AerospikeSurface(
        name="spike", x=x_all, r=r_all, area=area_all,
        At=A_t, Dt=Dt, r_curv_throat=r_curv, Rt=float(r_all.min()),
        x_throat=x_th, i_throat=n_sub - 1, supersonic=sup)

    return AerospikeChamber(
        spike=spike, cowl=cowl, inner=inner, R_mean=R_m, gap_throat=g,
        gap_chamber=h_c, L_chamber=L_c, L_convergent=L_conv,
        contraction_ratio=contraction_ratio,
        expansion_ratio=expansion_ratio)


# ----------------------------------------------------------------------
# Annular-chamber stability screen
# ----------------------------------------------------------------------

def annular_stability_screen(gas, R_mean: float, L_gas: float,
                             stiffness: float,
                             min_stiffness: float = 0.15
                             ) -> StabilityResult:
    """Acoustic screen for a thin annular (toroidal) chamber.

    Azimuthal modes travel the mean circumference: ``f_m = m a / (2 pi
    R_mean)`` (thin-gap limit); longitudinal as in the cylindrical screen.
    Same n-tau sensitive band and chug-stiffness guard. Screening only —
    annular chambers additionally support spinning modes the thin-gap
    formula does not resolve.
    """
    a = float(np.sqrt(gas.gamma * (R_UNIV / gas.molar_mass) * gas.T_c))
    L = max(float(L_gas), 1e-3)
    modes = {"1L": a / (2.0 * L), "2L": a / L}
    for m in (1, 2, 3):
        modes[f"{m}T-ann"] = m * a / (2.0 * np.pi * R_mean)

    f_lo, f_hi = 1.0 / _TAU_BAND[1], 1.0 / _TAU_BAND[0]
    sensitive = [k for k, f in modes.items() if f_lo <= f <= f_hi]
    stiffness_ok = stiffness >= min_stiffness
    notes = ["annular thin-gap acoustics - screening estimate only "
             "(spinning/radial-gap modes not resolved)"]
    hard = [m for m in sensitive if not m.endswith("L")]
    if hard:
        notes.append(f"annular mode(s) {hard} fall in the n-tau sensitive "
                     "band - verify with a combustion-response analysis")
    if not stiffness_ok:
        notes.append(f"injector stiffness {stiffness*100:.0f}% Pc below "
                     f"the {min_stiffness*100:.0f}% chug guard")
    return StabilityResult(
        sound_speed=a, modes=modes, stiffness=stiffness,
        stiffness_ok=stiffness_ok, sensitive_modes=sensitive,
        tau_band_hz=(f_lo, f_hi),
        verdict_ok=stiffness_ok and not hard, notes=notes)
