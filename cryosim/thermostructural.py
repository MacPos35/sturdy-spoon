"""Hot-wall thermo-structural analysis + low-cycle-fatigue life.

The regen model gives the wall *temperatures*; this module gives the wall
*stress and life* — the analysis that actually sizes a reusable regen
chamber and is the classic life-limiter (the "doghouse" thermal-stress
ratcheting of the hot wall between cooling channels; Quentmeyer, NASA
TM-103282; Huzel & Huang ch. 4).

Two stresses act on the thin hot wall (final-design tier, closed-form):

* **Thermal** — the through-thickness gradient the coolant sustains puts the
  hot face in compression: ``σ_th = E α (T_wg − T_wc) / (2(1−ν))``.
* **Pressure** — the coolant-to-gas Δp bends the hot wall as a plate built in
  at the channel lands (span = channel width): ``σ_p = Δp (w/t_w)² / 2``.

The combined stress is checked against the temperature-derated yield, and the
total strain range per thermal cycle is put through the **Manson–Coffin**
low-cycle-fatigue law to get cycles-to-failure.

Assumptions / limitations
-------------------------
* Closed-form thin-wall / built-in-plate stresses — not a 3-D FEA; no stress
  concentration at land fillets, no creep–fatigue interaction, no ratcheting
  accumulation beyond the Manson–Coffin estimate.
* Material elastic/fatigue constants are representative literature values
  (copper alloys, IN718), not lot-specific data — the *formulae* are exact,
  the absolute cycle life is order-of-magnitude. Yield is derated linearly to
  ~half at the liner temperature limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

#: Elastic + fatigue properties per liner material (SI; representative).
#: E [Pa], alpha [1/K], nu, sigma_y0 [Pa] @293 K, T_limit [K]; Manson-Coffin
#: sigma_f' [Pa], eps_f' [-], b, c (Δε/2 = σf'/E (2N)^b + εf' (2N)^c).
STRUCTURAL = {
    "cucrzr": dict(E=130e9, alpha=17.5e-6, nu=0.34, sigma_y0=320e6,
                   T_limit=800.0, sf=520e6, ef=0.30, b=-0.10, c=-0.60),
    "grcop-42": dict(E=127e9, alpha=17.0e-6, nu=0.33, sigma_y0=210e6,
                     T_limit=850.0, sf=460e6, ef=0.35, b=-0.10, c=-0.62),
    "inconel718": dict(E=200e9, alpha=13.0e-6, nu=0.29, sigma_y0=1030e6,
                       T_limit=1150.0, sf=1900e6, ef=0.28, b=-0.08, c=-0.70),
}


def yield_at(mat: dict, T: float) -> float:
    """Yield strength derated linearly to ~half at the temperature limit."""
    frac = np.clip((T - 293.0) / (mat["T_limit"] - 293.0), 0.0, 1.0)
    return mat["sigma_y0"] * (1.0 - 0.5 * frac)


def manson_coffin_life(delta_eps: float, mat: dict) -> float:
    """Cycles to failure N for a total strain range Δε (Manson–Coffin)."""
    if delta_eps <= 0:
        return np.inf
    E, sf, ef, b, c = mat["E"], mat["sf"], mat["ef"], mat["b"], mat["c"]

    def resid(logN):
        twoN = 2.0 * 10.0 ** logN
        return (sf / E) * twoN ** b + ef * twoN ** c - delta_eps / 2.0
    # bracket in log10(N) between 0 (1 cycle) and 8 (1e8, high-cycle)
    lo, hi = resid(0.0), resid(8.0)
    if lo < 0:            # even one cycle exceeds the strain capacity
        return 1.0
    if hi > 0:            # essentially infinite life
        return 1e8
    return float(10.0 ** brentq(resid, 0.0, 8.0))


@dataclass
class ThermoStructuralResult:
    x: np.ndarray
    sigma_thermal: np.ndarray     # Pa, through-wall thermal stress
    sigma_pressure: np.ndarray    # Pa, coolant Δp plate bending
    sigma_total: np.ndarray       # Pa, combined
    yield_margin: np.ndarray      # σ_y(T)/σ_total − 1
    peak_stress: float            # Pa, max combined (at/near throat)
    min_margin: float             # worst yield margin along the wall
    delta_eps_max: float          # peak total strain range
    cycle_life: float             # Manson–Coffin N at the worst station
    material: str

    def describe(self) -> str:
        return (f"thermo-structural ({self.material}): peak wall stress "
                f"{self.peak_stress/1e6:.0f} MPa, min yield margin "
                f"{self.min_margin*100:.0f}%, LCF life {self.cycle_life:.0f} "
                f"cycles (Δε {self.delta_eps_max*100:.2f}%)")


def analyze(regen, channels, gas, Pc: float, material: str
            ) -> ThermoStructuralResult:
    """Thermo-structural + LCF analysis along the regen solution.

    ``regen`` is a :class:`~cryosim.regen_model.RegenResult`; ``channels`` a
    ``CoolingChannels``; ``gas`` a ``CombustionGas`` (for the hot-gas static
    pressure); ``Pc`` chamber pressure; ``material`` a key of ``STRUCTURAL``.
    """
    key = material.strip().lower()
    if key not in STRUCTURAL:
        raise KeyError(f"no structural data for {material!r}; have "
                       f"{list(STRUCTURAL)}")
    mat = STRUCTURAL[key]
    dT = np.maximum(regen.T_wg - regen.T_wc, 0.0)
    sig_th = mat["E"] * mat["alpha"] * dT / (2.0 * (1.0 - mat["nu"]))

    # hot-gas static pressure from the local Mach; coolant Δp bends the wall
    g = gas.gamma
    p_gas = Pc * (1.0 + (g - 1.0) / 2.0 * regen.mach**2) ** (-g / (g - 1.0))
    dp = np.maximum(regen.P_coolant - p_gas, 0.0)
    sig_p = 0.5 * dp * (channels.channel_width / channels.t_wall) ** 2

    sig_tot = sig_th + sig_p
    sy = np.array([yield_at(mat, T) for T in regen.T_wg])
    margin = sy / np.maximum(sig_tot, 1.0) - 1.0

    # total strain range per cycle: free thermal strain + pressure elastic
    delta_eps = mat["alpha"] * dT + sig_p / mat["E"]
    i = int(np.argmax(delta_eps))
    N = manson_coffin_life(float(delta_eps[i]), mat)

    return ThermoStructuralResult(
        x=regen.x, sigma_thermal=sig_th, sigma_pressure=sig_p,
        sigma_total=sig_tot, yield_margin=margin,
        peak_stress=float(sig_tot.max()), min_margin=float(margin.min()),
        delta_eps_max=float(delta_eps[i]), cycle_life=N, material=key)
