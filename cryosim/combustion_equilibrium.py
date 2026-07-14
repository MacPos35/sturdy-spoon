"""Chemical-equilibrium combustion for CxHyOz fuels + LOX.

Computes the adiabatic flame temperature, equilibrium product composition
and the derived stagnation-gas properties (T_c, frozen gamma, mean molar
mass, c*) from the *actual* propellants, mixture ratio and chamber
pressure — replacing hand-entered preset gas tables. This is the "based on
a real physics model" step: chamber conditions come out of thermochemistry,
so any O/F or Pc gives a physically consistent gas, and the mixture ratio
can be optimized for peak performance.

Method
------
Gibbs free-energy minimization by the element-potential (Lagrange) method
used in NASA CEA (Gordon & McBride, NASA RP-1311, 1994): at a temperature T
and pressure P the equilibrium moles are

    n_i = n_tot * exp( Σ_k λ_k a_ik − g_i°(T)/RT − ln P_bar )

with element potentials λ_k solved from the element balances Σ_i a_ik n_i =
b_k and Σ_i n_i = n_tot. The chamber temperature is the T that closes the
adiabatic energy balance H_products(T_c) = H_reactants (reactants referenced
to 298.15 K). Standard-state Gibbs/enthalpy come from NASA 7-coefficient
thermodynamic polynomials for the major C/H/O combustion species.

Species: CO2, CO, H2O, H2, O2, OH, H, O (no solid carbon, no nitrogen —
LOX/hydrocarbon and LOX/H2 only).

Assumptions / limitations
-------------------------
* Reactants referenced at 298.15 K: real cryogenic injection is colder, so
  the true T_c is a little lower (a documented, conservative simplification;
  CEA's default reactant-temperature option makes the same choice).
* Ideal-gas products, no ionization, no condensed phases; the 8-species set
  omits minor species (HO2, H2O2, CH*, soot) — negligible for the mixture
  ratios of interest but not for very rich operation.
* Frozen composition is used downstream in the nozzle (matches the regen
  model's existing assumption); this routine sets the *chamber* state.
* Validated against the shipped presets and published CEA points in
  ``validation/test_combustion_equilibrium.py`` — read that before trusting
  absolute numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .combustion import R_UNIV, CombustionGas

# ----------------------------------------------------------------------
# Thermodynamic data
# ----------------------------------------------------------------------
# NASA 7-coefficient polynomials, high-temperature range (1000-3500 K),
# Gordon & McBride / GRI-Mech 3.0 values. Order: a1..a5 (cp/R poly), a6
# (enthalpy integration const, K), a7 (entropy const).
#   cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
#   H/RT = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T
#   S/R  = a1 lnT + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a7
_NASA7_HIGH = {
    "H2O": (2.67214561e0, 3.05629289e-3, -8.73026011e-7, 1.20099639e-10,
            -6.39161787e-15, -2.98992090e4, 6.86281681e0),
    "CO2": (4.45362282e0, 3.14016873e-3, -1.27841054e-6, 2.39399667e-10,
            -1.66903319e-14, -4.89669609e4, -9.55395877e-1),
    "CO":  (3.02507806e0, 1.44268852e-3, -5.63082779e-7, 1.01858133e-10,
            -6.91095156e-15, -1.42687352e4, 6.10821180e0),
    "H2":  (2.99142337e0, 7.00064411e-4, -5.63382869e-8, -9.23157818e-12,
            1.58275179e-15, -8.35033997e2, -1.35511017e0),
    "O2":  (3.69757819e0, 6.13519689e-4, -1.25884199e-7, 1.77528148e-11,
            -1.13643531e-15, -1.23393018e3, 3.18916559e0),
    "OH":  (2.86472886e0, 1.05650448e-3, -2.59082758e-7, 3.05218674e-11,
            -1.33195876e-15, 3.68362875e3, 5.70164073e0),
    "H":   (2.50000000e0, 0.0, 0.0, 0.0, 0.0, 2.54716270e4, -4.60117638e-1),
    "O":   (2.54205966e0, -2.75506191e-5, -3.10280335e-9, 4.55106742e-12,
            -4.36805150e-16, 2.92308027e4, 4.92030811e0),
}

#: element vector [C, H, O] per species.
_ELEMENTS = {
    "H2O": (0, 2, 1), "CO2": (1, 0, 2), "CO": (1, 0, 1), "H2": (0, 2, 0),
    "O2": (0, 0, 2), "OH": (0, 1, 1), "H": (0, 1, 0), "O": (0, 0, 1),
}

#: molar masses [kg/mol].
_MOLAR = {
    "H2O": 18.0153e-3, "CO2": 44.0095e-3, "CO": 28.0101e-3, "H2": 2.01588e-3,
    "O2": 31.9988e-3, "OH": 17.0073e-3, "H": 1.00794e-3, "O": 15.9994e-3,
}

_SPECIES = list(_NASA7_HIGH)
_A = np.array([_ELEMENTS[s] for s in _SPECIES], float)   # (n_species, 3)
_M = np.array([_MOLAR[s] for s in _SPECIES])


@dataclass(frozen=True)
class Fuel:
    """A CxHyOz fuel: atoms per molecule + standard enthalpy of formation."""

    name: str
    nC: int
    nH: int
    nO: int
    dHf: float            # J/mol, at 298.15 K (liquid for storables)

    @property
    def molar_mass(self) -> float:
        return (self.nC * 12.0107 + self.nH * 1.00794
                + self.nO * 15.9994) * 1e-3


#: Common fuels (ΔHf standard values; liquid phase for storables).
FUELS = {
    "ch4": Fuel("methane", 1, 4, 0, -74600.0),
    "methane": Fuel("methane", 1, 4, 0, -74600.0),
    "h2": Fuel("hydrogen", 0, 2, 0, 0.0),
    "hydrogen": Fuel("hydrogen", 0, 2, 0, 0.0),
    "ethanol": Fuel("ethanol", 2, 6, 1, -277000.0),   # liquid
    "c3h8": Fuel("propane", 3, 8, 0, -104700.0),
    "propane": Fuel("propane", 3, 8, 0, -104700.0),
}

_M_O2 = 31.9988e-3


def stoichiometric_of(fuel: str) -> float:
    """Stoichiometric mass O/F for complete combustion to CO2 + H2O."""
    key = fuel.strip().lower()
    if key not in FUELS:
        raise KeyError(f"unknown fuel {fuel!r}")
    f = FUELS[key]
    o_atoms = 2 * f.nC + f.nH / 2.0 - f.nO      # O atoms needed per fuel mol
    return o_atoms / 2.0 * _M_O2 / f.molar_mass


def _poly_H_RT(coeffs, T):
    a1, a2, a3, a4, a5, a6, _a7 = coeffs
    return (a1 + a2 * T / 2 + a3 * T**2 / 3 + a4 * T**3 / 4
            + a5 * T**4 / 5 + a6 / T)


def _poly_S_R(coeffs, T):
    a1, a2, a3, a4, a5, _a6, a7 = coeffs
    return (a1 * np.log(T) + a2 * T + a3 * T**2 / 2 + a4 * T**3 / 3
            + a5 * T**4 / 4 + a7)


def _poly_cp_R(coeffs, T):
    a1, a2, a3, a4, a5, _a6, _a7 = coeffs
    return a1 + a2 * T + a3 * T**2 + a4 * T**3 + a5 * T**4


def _g_RT(T):
    """Standard-state g°/RT for every species at T."""
    return np.array([_poly_H_RT(_NASA7_HIGH[s], T) - _poly_S_R(_NASA7_HIGH[s], T)
                     for s in _SPECIES])


def _h_RT(T):
    return np.array([_poly_H_RT(_NASA7_HIGH[s], T) for s in _SPECIES])


@dataclass
class EquilibriumResult:
    """Chamber-equilibrium solution."""

    T_c: float
    P: float
    of_ratio: float
    mole_fractions: dict
    molar_mass: float          # kg/mol
    gamma: float               # frozen, at T_c
    cp: float                  # J/(kg K), frozen
    c_star: float              # m/s, ideal
    fuel: str

    def as_gas(self) -> CombustionGas:
        """Package as the CombustionGas the regen/design code consumes."""
        return CombustionGas(T_c=self.T_c, gamma=self.gamma,
                             molar_mass=self.molar_mass)


def _initial_moles(b):
    """Complete-combustion composition as a feasible starting point."""
    nC, nH, nO = b
    nCO2 = nC
    nH2O = nH / 2.0
    o_used = 2 * nCO2 + nH2O
    nO2 = max((nO - o_used) / 2.0, 0.0)
    nH2 = 0.0
    if o_used > nO:                       # oxygen-lean: shift CO2->CO, H2O->H2
        deficit = o_used - nO
        shift = min(deficit, nCO2)
        nCO2 -= shift
        deficit -= shift
        nH2O = max(nH2O - deficit, 0.0)
        nH2 = nH / 2.0 - nH2O
        nO2 = 0.0
    g = {"CO2": nCO2, "CO": nC - nCO2, "H2O": nH2O, "H2": nH2, "O2": nO2}
    return np.array([max(g.get(s, 0.0), 1e-8) for s in _SPECIES])


def _solve_composition(T, P_bar, b):
    """Equilibrium moles at (T, P) by the element-potential method.

    Stationarity of the ideal-gas Gibbs energy gives n_i = N exp(Σ_k λ_k a_ik
    − g_i°/RT − ln P). A damped Newton iteration on the element potentials
    λ_k and ln N drives the element balances Σ_i a_ik n_i = b_k and Σ_i n_i =
    N to zero. The problem is convex, so damped Newton converges to the unique
    global minimum — deterministically, unlike a general NLP solver.
    """
    g0 = _g_RT(T) + np.log(P_bar)         # g_i°/RT + ln P_bar
    n0 = _initial_moles(b)
    N0 = n0.sum()
    # seed element potentials by least-squares fit of the major-species
    # stationarity condition  Σ_k λ_k a_ik ≈ ln(n_i/N) + g0_i
    target = np.log(np.clip(n0, 1e-12, None) / N0) + g0
    lam, *_ = np.linalg.lstsq(_A, target, rcond=None)
    y = np.concatenate([lam, [np.log(N0)]])

    for _ in range(200):
        lam, lnN = y[:3], y[3]
        n = np.exp(lnN + np.clip(_A @ lam - g0, -400.0, 60.0))
        R = np.empty(4)
        R[:3] = _A.T @ n - b
        R[3] = n.sum() - np.exp(lnN)
        scale = np.concatenate([b + 1e-12, [n.sum() + 1e-12]])
        if np.max(np.abs(R) / scale) < 1e-12:
            break
        J = np.zeros((4, 4))
        for k in range(3):
            for j in range(3):
                J[k, j] = np.sum(_A[:, k] * _A[:, j] * n)
            J[k, 3] = np.sum(_A[:, k] * n)
        J[3, :3] = [np.sum(_A[:, j] * n) for j in range(3)]
        J[3, 3] = n.sum() - np.exp(lnN)
        try:
            dy = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(J, -R, rcond=None)[0]
        # damp so no species' log-mole moves more than ~2 per step
        dexpo = _A @ dy[:3] + dy[3]
        step = min(1.0, 2.0 / max(np.max(np.abs(dexpo)), 1e-9))
        y = y + step * dy

    lam, lnN = y[:3], y[3]
    return np.exp(lnN + np.clip(_A @ lam - g0, -400.0, 60.0))


def equilibrium_combustion(fuel: str, of_ratio: float, Pc: float,
                           ) -> EquilibriumResult:
    """Solve chamber equilibrium for ``fuel`` + LOX at mass ratio ``of_ratio``.

    ``Pc`` in Pa. Returns temperature, composition and derived gas
    properties.
    """
    key = fuel.strip().lower()
    if key not in FUELS:
        raise KeyError(f"unknown fuel {fuel!r}; available: "
                       f"{sorted(set(FUELS))}")
    f = FUELS[key]
    if of_ratio <= 0 or Pc <= 0:
        raise ValueError("of_ratio and Pc must be positive")

    # per 1 mol fuel:
    n_O2 = of_ratio * f.molar_mass / _M_O2
    b = np.array([float(f.nC), float(f.nH), f.nO + 2.0 * n_O2])
    H_react = f.dHf   # O2 gas ΔHf = 0 at 298.15 K
    P_bar = Pc / 1e5

    def energy_residual(T):
        n = _solve_composition(T, P_bar, b)
        H_prod = float((n * _h_RT(T)).sum() * R_UNIV * T)
        return H_prod - H_react

    lo, hi = 500.0, 4500.0
    f_lo, f_hi = energy_residual(lo), energy_residual(hi)
    if f_lo > 0:            # flame colder than 500 K (extreme off-stoich)
        T_c = lo
    elif f_hi < 0:          # hotter than the 4500 K data range
        T_c = hi
    else:
        T_c = brentq(energy_residual, lo, hi, xtol=0.5)
    return _package(T_c, Pc, b, f, of_ratio)


def _package(T_c, Pc, b, fuel, of_ratio) -> EquilibriumResult:
    """Derive gas properties from the converged chamber equilibrium."""
    n = _solve_composition(T_c, Pc / 1e5, b)
    x = n / n.sum()
    M = float((x * _M).sum())
    cp_molar = float((x * np.array(
        [_poly_cp_R(_NASA7_HIGH[s], T_c) for s in _SPECIES])).sum() * R_UNIV)
    cp_mass = cp_molar / M
    R_spec = R_UNIV / M
    gamma = cp_mass / (cp_mass - R_spec)
    gas = CombustionGas(T_c=T_c, gamma=gamma, molar_mass=M)
    return EquilibriumResult(
        T_c=T_c, P=Pc, of_ratio=of_ratio,
        mole_fractions={s: float(xi) for s, xi in zip(_SPECIES, x)
                        if xi > 1e-4},
        molar_mass=M, gamma=gamma, cp=cp_mass, c_star=gas.c_star,
        fuel=fuel.name)


def optimize_of(fuel: str, Pc: float, objective: str = "c_star",
                expansion_ratio: float | None = None,
                bounds: tuple = (1.0, 8.0)) -> tuple[float, EquilibriumResult]:
    """Mixture ratio that maximizes performance (physics-driven O/F search).

    ``objective``: ``"c_star"`` (chamber performance, ε-independent) or
    ``"isp_vac"`` (needs ``expansion_ratio``). Returns (O/F, result).
    Peak c* sits fuel-rich of stoichiometric because dissociation and low
    product molar mass favor excess fuel — the classic result.
    """
    from .combustion import mach_from_area_ratio

    def perf(of):
        res = equilibrium_combustion(fuel, of, Pc)
        if objective == "c_star":
            return res.c_star
        if objective == "isp_vac":
            if expansion_ratio is None:
                raise ValueError("isp_vac objective needs expansion_ratio")
            g = res.gamma
            Me = mach_from_area_ratio(expansion_ratio, g, supersonic=True)
            Pe_Pc = (1 + (g - 1) / 2 * Me**2) ** (-g / (g - 1))
            CF = np.sqrt(2 * g**2 / (g - 1)
                         * (2 / (g + 1)) ** ((g + 1) / (g - 1))
                         * (1 - Pe_Pc ** ((g - 1) / g))) \
                + expansion_ratio * Pe_Pc
            return CF * res.c_star / 9.80665
        raise ValueError("objective must be 'c_star' or 'isp_vac'")

    ofs = np.linspace(bounds[0], bounds[1], 22)
    vals = np.array([perf(o) for o in ofs])
    i = int(np.argmax(vals))
    lo = ofs[max(i - 1, 0)]
    hi = ofs[min(i + 1, len(ofs) - 1)]
    # golden-ish refine
    best_of, best_v = ofs[i], vals[i]
    for o in np.linspace(lo, hi, 15):
        v = perf(o)
        if v > best_v:
            best_of, best_v = o, v
    return float(best_of), equilibrium_combustion(fuel, best_of, Pc)
