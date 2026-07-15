"""Validation of the final-design-tier methods against exact references."""

import numpy as np
import pytest

from cryosim.chamber_geometry import (ChamberContour,
                                      nozzle_divergence_efficiency)
from cryosim.combustion_stability import _bessel_prime_root, stability_screen
from cryosim.combustion import CombustionGas
from cryosim.moc_nozzle import (design_moc_nozzle, mach_from_nu,
                                prandtl_meyer)
from cryosim.thermostructural import STRUCTURAL, manson_coffin_life

pytestmark = pytest.mark.validation


# --- MOC nozzle vs Anderson (Modern Compressible Flow, ch. 11 / App. C) --

def test_prandtl_meyer_table():
    for M, deg in [(1.5, 11.905), (2.0, 26.380), (3.0, 49.757),
                   (4.0, 65.785)]:
        assert np.degrees(prandtl_meyer(M, 1.4)) == pytest.approx(deg,
                                                                  abs=0.02)


def test_moc_nozzle_uniform_parallel_exit():
    for eps in (4.0, 10.0, 30.0):
        r = design_moc_nozzle(1.2, eps, r_throat=0.02, n_char=80)
        assert abs(r.exit_angle_deg) < 1e-3            # axial exit
        assert r.area_ratio == pytest.approx(eps, rel=1e-3)
        assert mach_from_nu(2 * np.radians(r.theta_max_deg), 1.2) == \
            pytest.approx(r.exit_mach, rel=1e-6)       # θmax = ν(Me)/2
    # ideal MOC beats the bell beats the cone
    assert (nozzle_divergence_efficiency(10.0, "moc")
            > nozzle_divergence_efficiency(10.0, "bell")
            > nozzle_divergence_efficiency(10.0, "conical"))


# --- thermo-structural closed forms + Manson-Coffin ---------------------

def test_thermal_stress_and_life_reference():
    m = STRUCTURAL["cucrzr"]
    # E α ΔT / (2(1−ν)) is exact by construction
    dT = 250.0
    assert (m["E"] * m["alpha"] * dT / (2 * (1 - m["nu"]))) == \
        pytest.approx(130e9 * 17.5e-6 * 250 / (2 * 0.66))
    # Manson-Coffin: 0.5% strain range gives a long-but-finite copper life
    assert 3000 < manson_coffin_life(0.005, m) < 30000


# --- acoustic stability vs closed-form cylinder modes -------------------

def test_cylinder_acoustic_modes_closed_form():
    for m, ref in [(1, 1.8412), (2, 3.0542), (3, 4.2012)]:
        assert _bessel_prime_root(m, 1) == pytest.approx(ref, abs=1e-3)


def test_1T_frequency_order_of_magnitude():
    # a ~1 m radius chamber at a ~1000 m/s sound speed → 1T ~ hundreds of Hz
    gas = CombustionGas(T_c=3400.0, gamma=1.15, molar_mass=22e-3)
    contour = ChamberContour(throat_radius=0.1, contraction_ratio=4.0,
                             expansion_ratio=4.0, chamber_length=0.3)
    r = stability_screen(gas, contour, stiffness=0.2)
    assert 100.0 < r.modes["1T"] < 20000.0
