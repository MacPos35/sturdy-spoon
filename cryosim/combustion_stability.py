"""Combustion-stability screen: chamber acoustic modes + coupling check.

The standard final-design stability screen (before a rig test): compute the
chamber's acoustic eigenmodes and check whether the injector is stiff enough
and whether any mode sits in the combustion-response-sensitive band where it
could couple with the heat release (Crocco's n–τ sensitivity; Harrje & Reardon,
NASA SP-194; Sutton ch. 9).

Chamber modeled as a closed–open cylinder (radius R_c, length L_c) filled with
the combustion gas at sound speed ``a = √(γ R T_c)``:

* longitudinal ``f_L,n = n a / 2L`` ,
* tangential/radial ``f_mn = a s_mn / 2πR`` with ``s_mn`` the roots of J′_m
  (1T = 1.841, 2T = 3.054, 1R = 3.832 …).

The **1T (first tangential)** mode is historically the most dangerous in liquid
engines (F-1, RS-27) and is flagged specially.

Assumptions / limitations
-------------------------
* Linear acoustics with a uniform, quiescent, cold-length-corrected gas — a
  *screen*, not a nonlinear combustion-instability simulation (no wave
  steepening, no mode coupling, no baffle/cavity damping model beyond the
  qualitative note). Passing the screen means "no obvious linear coupling and
  adequate stiffness", not "proven stable".
* The n–τ sensitive band uses a representative combustion time lag; the real
  τ needs test data. Injector stiffness ≥ ~0.15·Pc is the classical chug
  guard (Huzel & Huang ch. 4), not a proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import jnp_zeros

from .combustion import R_UNIV

#: J'_m roots labelled by acoustic mode (m tangential order, radial index).
_MODE_ROOTS = {
    "1T": (1, 1), "2T": (2, 1), "3T": (3, 1),
    "1R": (0, 1), "1T1R": (1, 2), "2R": (0, 2),
}

#: representative combustion time lag band [s] → sensitive frequency band.
_TAU_BAND = (0.2e-3, 1.0e-3)      # τ ~ 0.2–1 ms → ~1–5 kHz sensitive


def _bessel_prime_root(m: int, n: int) -> float:
    """n-th positive root of J'_m (n>=1); root 0 of J'_0 is 0 (skip)."""
    if m == 0:
        # J'_0 roots are the J_1 roots; jnp_zeros(0,·) gives them
        return float(jnp_zeros(0, n)[n - 1])
    return float(jnp_zeros(m, n)[n - 1])


@dataclass
class StabilityResult:
    sound_speed: float                 # m/s, chamber acoustic speed
    modes: dict                        # label -> frequency [Hz]
    stiffness: float                   # injector dP/Pc
    stiffness_ok: bool
    sensitive_modes: list              # labels in the n-τ band
    tau_band_hz: tuple
    verdict_ok: bool
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        ms = ", ".join(f"{k} {v/1e3:.1f}kHz" for k, v in self.modes.items())
        return (f"stability screen: a={self.sound_speed:.0f} m/s; {ms}; "
                f"injector stiffness {self.stiffness*100:.0f}% Pc "
                f"({'ok' if self.stiffness_ok else 'LOW'}); "
                + ("no mode in the sensitive band"
                   if not self.sensitive_modes else
                   f"IN-BAND: {', '.join(self.sensitive_modes)}"))


def stability_screen(gas, contour, stiffness: float,
                     min_stiffness: float = 0.15) -> StabilityResult:
    """Acoustic-mode + injector-coupling screen.

    ``gas`` a ``CombustionGas`` (chamber sound speed), ``contour`` a
    ``ChamberContour`` (radius R_c and cylindrical length), ``stiffness`` the
    injector dP/Pc.
    """
    a = np.sqrt(gas.gamma * (R_UNIV / gas.molar_mass) * gas.T_c)
    Rc = float(contour.Rc_chamber)
    # gas-column length for the longitudinal mode: injector face to throat
    # (the convergent nozzle is the ~closed downstream end)
    Lc = max(float(contour.x_throat), 1e-3)

    modes = {"1L": a / (2.0 * Lc), "2L": a / Lc}
    for label, (m, n) in _MODE_ROOTS.items():
        s = _bessel_prime_root(m, n)
        modes[label] = a * s / (2.0 * np.pi * Rc)

    f_lo, f_hi = 1.0 / _TAU_BAND[1], 1.0 / _TAU_BAND[0]     # Hz band
    sensitive = [k for k, f in modes.items() if f_lo <= f <= f_hi]

    stiffness_ok = stiffness >= min_stiffness
    notes = []
    if "1T" in sensitive:
        notes.append("1T mode in the sensitive band — the historically most "
                     "dangerous liquid-engine mode; baffles/acoustic cavities "
                     "are the usual fix")
    if not stiffness_ok:
        notes.append(f"injector stiffness {stiffness*100:.0f}% Pc below the "
                     f"{min_stiffness*100:.0f}% chug guard")
    # screen passes if stiff enough and no tangential/radial mode is in-band
    hard = [m for m in sensitive if m not in ("1L", "2L")]
    verdict = stiffness_ok and not hard
    if hard and not notes:
        notes.append(f"chamber mode(s) {hard} fall in the n-τ sensitive band "
                     "— verify with a combustion-response (n-τ) analysis")
    return StabilityResult(
        sound_speed=float(a), modes=modes, stiffness=stiffness,
        stiffness_ok=stiffness_ok, sensitive_modes=sensitive,
        tau_band_hz=(f_lo, f_hi), verdict_ok=verdict, notes=notes)
