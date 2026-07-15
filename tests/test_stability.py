"""Combustion-stability acoustic screen."""

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour
from cryosim.combustion import CombustionGas
from cryosim.combustion_stability import _bessel_prime_root, stability_screen


def test_bessel_roots_exact():
    # standard acoustic-mode eigenvalues (roots of J'_m)
    assert _bessel_prime_root(1, 1) == pytest.approx(1.8412, abs=1e-3)   # 1T
    assert _bessel_prime_root(2, 1) == pytest.approx(3.0542, abs=1e-3)   # 2T
    assert _bessel_prime_root(0, 1) == pytest.approx(3.8317, abs=1e-3)   # 1R


@pytest.fixture
def screen():
    gas = CombustionGas(T_c=3400.0, gamma=1.14, molar_mass=21e-3)
    contour = ChamberContour(throat_radius=0.03, contraction_ratio=8.0,
                             expansion_ratio=4.0, chamber_length=0.15)
    return gas, contour


def test_mode_frequencies_scale_with_sound_speed(screen):
    gas, contour = screen
    r = stability_screen(gas, contour, stiffness=0.2)
    a = np.sqrt(gas.gamma * (8.31446 / gas.molar_mass) * gas.T_c)
    assert r.sound_speed == pytest.approx(a, rel=1e-9)
    # 1T = a·1.8412/(2πRc)
    assert r.modes["1T"] == pytest.approx(
        a * 1.8412 / (2 * np.pi * contour.Rc_chamber), rel=1e-3)
    # higher tangential modes are higher frequency
    assert r.modes["2T"] > r.modes["1T"]


def test_stiffness_guard(screen):
    gas, contour = screen
    assert stability_screen(gas, contour, stiffness=0.20).stiffness_ok
    assert not stability_screen(gas, contour, stiffness=0.05).stiffness_ok


def test_describe_runs(screen):
    gas, contour = screen
    txt = stability_screen(gas, contour, stiffness=0.2).describe()
    assert "1T" in txt and "stiffness" in txt
