"""Validate the equilibrium combustion solver against CEA anchor points.

The absolute numbers a rocket designer trusts (T_c, c*, product mole
fractions) are checked against published NASA CEA results for LOX/CH4,
LOX/H2 and LOX/ethanol. The 8-species, frozen-gamma model is expected to
land within a few percent; wider tolerances than the analytical checks
reflect that (and are stated, not hidden).
"""

import numpy as np
import pytest

from cryosim.combustion_equilibrium import (
    equilibrium_combustion,
    optimize_of,
    shifting_c_star,
    shifting_isp,
    stoichiometric_of,
)

pytestmark = pytest.mark.validation


# (fuel, O/F, Pc[Pa], Tc range [K], c* range [m/s]) — CEA reference bands
CASES = [
    ("ch4", 3.2, 20e5, (3350, 3560), (1770, 1870)),
    ("ch4", 3.5, 100e5, (3550, 3720), (1810, 1910)),
    ("h2", 6.0, 20e5, (3350, 3560), (2250, 2470)),
    ("ethanol", 1.7, 20e5, (3230, 3450), (1650, 1810)),
    # kerolox: the classic 1000-psia CEA anchor (Sutton tab. 5-5: Tc ~3670 K,
    # c* ~1800 m/s at O/F 2.27) and the 20-bar operating point
    ("rp1", 2.27, 68.05 * 101325, (3550, 3720), (1730, 1840)),
    ("rp1", 2.3, 20e5, (3390, 3560), (1700, 1820)),
]


@pytest.mark.parametrize("fuel,of,Pc,Trange,crange", CASES)
def test_equilibrium_vs_cea_bands(fuel, of, Pc, Trange, crange):
    r = equilibrium_combustion(fuel, of, Pc)
    assert Trange[0] <= r.T_c <= Trange[1], f"T_c={r.T_c:.0f} K"
    assert crange[0] <= r.c_star <= crange[1], f"c*={r.c_star:.0f} m/s"
    # frozen gamma of hot combustion products sits ~1.14-1.25
    assert 1.13 <= r.gamma <= 1.26
    # mole fractions normalize and are all physical
    assert abs(sum(r.mole_fractions.values()) - 1.0) < 0.02
    assert all(0.0 <= v <= 1.0 for v in r.mole_fractions.values())


def test_element_conservation():
    """Products must conserve C, H, O from the reactants (per mol fuel)."""
    from cryosim.combustion_equilibrium import (_solve_composition, _A, FUELS,
                                                _M_O2)
    f = FUELS["ch4"]
    of = 3.2
    nO2 = of * f.molar_mass / _M_O2
    b = np.array([f.nC, f.nH, f.nO + 2 * nO2], float)
    n = _solve_composition(3400.0, 20.0, b)
    assert np.allclose(_A.T @ n, b, rtol=1e-6, atol=1e-6)


def test_richer_at_low_of_gives_more_CO_and_H2():
    """Fuel-rich operation shifts CO2->CO and makes H2 (incomplete combustion)."""
    lean = equilibrium_combustion("ch4", 3.6, 20e5).mole_fractions
    rich = equilibrium_combustion("ch4", 2.6, 20e5).mole_fractions
    assert rich.get("CO", 0) > lean.get("CO", 0)
    assert rich.get("H2", 0) > lean.get("H2", 0)


def test_peak_cstar_is_fuel_rich_of_stoichiometric():
    """Peak c* sits rich of stoichiometric (low product molar mass wins)."""
    of_opt, _ = optimize_of("ch4", 20e5, objective="c_star")
    stoich = stoichiometric_of("ch4")
    assert of_opt < stoich          # rich of stoichiometric
    assert 2.0 < of_opt < stoich    # but not absurdly so


def test_stoichiometric_values():
    # CH4 + 2 O2 -> CO2 + 2 H2O : O/F = 2*32/16.04 = 3.99
    assert stoichiometric_of("ch4") == pytest.approx(3.99, rel=0.02)
    # 2 H2 + O2 -> 2 H2O : O/F = 32/(2*2.016) = 7.94
    assert stoichiometric_of("h2") == pytest.approx(7.94, rel=0.02)
    # CH1.9423 + 1.4856 O2 -> CO2 + 0.9712 H2O : O/F = 3.40
    assert stoichiometric_of("rp1") == pytest.approx(3.40, rel=0.02)


def test_pressure_raises_temperature():
    """Higher chamber pressure suppresses dissociation -> hotter flame."""
    lo = equilibrium_combustion("ch4", 3.4, 10e5).T_c
    hi = equilibrium_combustion("ch4", 3.4, 200e5).T_c
    assert hi > lo + 50.0


# --- shifting-equilibrium expansion (the real O/F-optimum fix) ----------

def test_shifting_cstar_beats_frozen_and_matches_cea():
    """Shifting c* > frozen (recombination credit) and within ~2% of CEA."""
    froz = equilibrium_combustion("ch4", 3.2, 20e5).c_star
    shift = shifting_c_star("ch4", 3.2, 20e5)
    assert shift > froz
    assert 1810 <= shift <= 1900          # CEA ~1837


def test_shifting_isp_magnitude():
    """Vacuum Isp (eps=40) for LOX/CH4 lands in the CEA band."""
    isp = shifting_isp("ch4", 3.4, 20e5, expansion_ratio=40.0)
    assert 355 <= isp <= 380              # CEA ~365-372


@pytest.mark.parametrize("fuel,of_lo,of_hi", [
    ("ch4", 3.0, 3.6),        # CEA peak-Isp ~3.3-3.5
    ("ethanol", 1.7, 2.0),    # CEA ~1.8-1.9
])
def test_shifting_of_optimum_near_cea(fuel, of_lo, of_hi):
    """The key fix: peak-Isp O/F on the shifting model sits near the true
    (mildly rich) CEA optimum, not the frozen model's fuel-rich bias."""
    of_opt, _ = optimize_of(fuel, 20e5, objective="isp_vac",
                            expansion_ratio=40.0)
    assert of_lo <= of_opt <= of_hi
    # and clearly leaner than the frozen-c* optimum
    of_frozen_cstar, _ = optimize_of(fuel, 20e5, objective="c_star")
    assert of_opt > of_frozen_cstar
