"""Unit tests for tank geometry against closed-form solutions."""

import numpy as np
import pytest

from cryosim.tank_geometry import TankGeometry

R, L = 0.2, 1.0  # student-scale tank


def test_total_volume_hemispherical():
    tank = TankGeometry(R, L, "hemispherical", "hemispherical")
    V_exact = np.pi * R**2 * L + 4.0 / 3.0 * np.pi * R**3
    assert tank.V_total == pytest.approx(V_exact, rel=1e-4)


def test_total_volume_elliptical_2to1():
    tank = TankGeometry(R, L, "elliptical", "elliptical", dome_aspect=2.0)
    # each 2:1 semi-ellipsoidal dome: (2/3) pi R^2 (R/2)
    V_exact = np.pi * R**2 * L + 2.0 * (2.0 / 3.0) * np.pi * R**2 * (R / 2.0)
    assert tank.V_total == pytest.approx(V_exact, rel=1e-4)


def test_flat_cylinder_relations():
    tank = TankGeometry(R, L, "flat", "flat")
    h = 0.4
    assert tank.volume_at_height(h) == pytest.approx(np.pi * R**2 * h, rel=1e-4)
    # wetted wall = bottom plate + cylinder side
    assert tank.wetted_wall_area(h) == pytest.approx(
        np.pi * R**2 + 2 * np.pi * R * h, rel=1e-4
    )
    assert tank.liquid_cg_height(h) == pytest.approx(h / 2, rel=1e-4)
    assert tank.surface_radius(h) == pytest.approx(R)


def test_height_volume_inverse():
    tank = TankGeometry(R, L, "elliptical", "elliptical")
    for fill in (0.1, 0.5, 0.9):
        V = fill * tank.V_total
        h = tank.height_at_volume(V)
        assert tank.volume_at_height(h) == pytest.approx(V, rel=1e-6)


def test_monotonic_volume():
    tank = TankGeometry(R, L, "hemispherical", "elliptical")
    z = np.linspace(0, tank.height, 100)
    V = [tank.volume_at_height(zi) for zi in z]
    assert np.all(np.diff(V) >= 0)


def test_dry_plus_wet_is_total():
    tank = TankGeometry(R, L, "elliptical", "elliptical")
    h = 0.7
    assert tank.wetted_wall_area(h) + tank.dry_wall_area(h) == pytest.approx(
        tank.A_wall_total, rel=1e-9
    )


def test_surface_in_dome_flag():
    tank = TankGeometry(R, L, "elliptical", "elliptical")
    assert tank.surface_in_dome(0.05)          # in bottom dome (b = 0.1)
    assert not tank.surface_in_dome(0.5)       # in cylinder
    assert tank.surface_in_dome(tank.height - 0.01)


def test_hemisphere_wetted_area():
    # liquid exactly filling the bottom hemisphere: wetted area = 2 pi R^2
    tank = TankGeometry(R, 1.0, "hemispherical", "flat")
    assert tank.wetted_wall_area(R) == pytest.approx(2 * np.pi * R**2, rel=1e-3)
