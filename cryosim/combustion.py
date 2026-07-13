"""Hot-gas (combustion product) properties and quasi-1D isentropic flow.

The regen model needs stagnation-condition combustion-gas properties
(T_c, gamma, molar mass, viscosity, Prandtl number). The proper source is a
chemical-equilibrium run (NASA CEA / RocketCEA / RPA) for the actual
propellants, mixture ratio and chamber pressure — pass those in directly.
For convenience, nominal CEA-derived presets are included for LOX/CH4 and
LOX/ethanol at typical student-engine conditions.

Transport approximations (standard practice, per Bartz 1957 and Huzel &
Huang eq. 4-15/4-16):

* viscosity  mu = 1.184e-7 * M[g/mol]^0.5 * T[K]^0.6   [Pa s]
* Prandtl    Pr = 4 gamma / (9 gamma - 5)

Assumptions / limitations
-------------------------
* Frozen composition along the nozzle (no shifting equilibrium, no kinetics).
* Calorically perfect gas for the isentropic relations (gamma constant).
* Preset tables are nominal single-point values, NOT a substitute for a CEA
  run at your actual O/F and Pc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

R_UNIV = 8.31446  # J/(mol K)


@dataclass
class CombustionGas:
    """Stagnation-state combustion-gas description.

    Parameters: chamber (stagnation) temperature T_c [K], ratio of specific
    heats gamma [-], molar mass M [kg/mol]; viscosity/Prandtl optional
    (Bartz approximations used when omitted).
    """

    T_c: float
    gamma: float
    molar_mass: float
    mu: float | None = None
    Pr: float | None = None

    def __post_init__(self):
        if self.mu is None:
            self.mu = 1.184e-7 * (self.molar_mass * 1e3) ** 0.5 * self.T_c ** 0.6
        if self.Pr is None:
            self.Pr = 4.0 * self.gamma / (9.0 * self.gamma - 5.0)

    @property
    def R_specific(self) -> float:
        return R_UNIV / self.molar_mass

    @property
    def cp(self) -> float:
        g = self.gamma
        return g * self.R_specific / (g - 1.0)

    @property
    def c_star(self) -> float:
        """Characteristic velocity [m/s] (ideal)."""
        g = self.gamma
        return np.sqrt(g * self.R_specific * self.T_c) / (
            g * np.sqrt((2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        )

    @property
    def recovery_factor(self) -> float:
        """Turbulent boundary-layer recovery factor r = Pr^(1/3)."""
        return self.Pr ** (1.0 / 3.0)


#: Nominal CEA-derived presets (frozen, chamber conditions, typical student
#: engine O/F and Pc ~ 20-55 bar). Use a real CEA run for design work.
GAS_PRESETS = {
    "lox/ch4": CombustionGas(T_c=3430.0, gamma=1.14, molar_mass=21.3e-3),
    "lox/ethanol": CombustionGas(T_c=3200.0, gamma=1.13, molar_mass=23.8e-3),
}


def gas_preset(name: str) -> CombustionGas:
    key = name.strip().lower()
    if key not in GAS_PRESETS:
        raise KeyError(f"no gas preset {name!r}; available: {list(GAS_PRESETS)}")
    return GAS_PRESETS[key]


# --------------------------------------------------------------------------
# Quasi-1D isentropic relations
# --------------------------------------------------------------------------
def area_ratio_from_mach(M: float, gamma: float) -> float:
    g = gamma
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + (g - 1.0) / 2.0 * M**2)) ** (
        (g + 1.0) / (2.0 * (g - 1.0))
    )


def mach_from_area_ratio(ar: float, gamma: float, supersonic: bool) -> float:
    """Invert A/A* for the requested branch."""
    if ar < 1.0:
        ar = max(ar, 1.0)  # numerical guard at the throat
    if abs(ar - 1.0) < 1e-12:
        return 1.0
    if supersonic:
        return brentq(lambda M: area_ratio_from_mach(M, gamma) - ar, 1.0 + 1e-12, 100.0)
    return brentq(lambda M: area_ratio_from_mach(M, gamma) - ar, 1e-8, 1.0 - 1e-12)


def static_over_stagnation_T(M: float, gamma: float) -> float:
    return 1.0 / (1.0 + (gamma - 1.0) / 2.0 * M**2)


def adiabatic_wall_temperature(gas: CombustionGas, M: float) -> float:
    """T_aw with turbulent recovery factor (Bartz 1957)."""
    g, r = gas.gamma, gas.recovery_factor
    num = 1.0 + r * (g - 1.0) / 2.0 * M**2
    den = 1.0 + (g - 1.0) / 2.0 * M**2
    return gas.T_c * num / den
