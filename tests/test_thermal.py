"""Fast unit/sanity tests for the tank thermal model (student-scale LOX tank)."""

import numpy as np
import pytest

from cryosim.fluids import Fluid
from cryosim.tank_geometry import TankGeometry
from cryosim.thermal_model import (
    TankThermalModel,
    TankThermalConfig,
    homogeneous_pressure_rise_rate,
    wall_cp,
    wall_k,
    nu_vertical_plate,
    nu_horizontal_plate,
)


@pytest.fixture(scope="module")
def lox_tank():
    lox = Fluid("LOX")
    tank = TankGeometry(0.15, 0.8, "elliptical", "elliptical")  # ~0.3 m dia
    model = TankThermalModel(lox, tank, wall_mass=12.0, heat_flux=50.0)
    return lox, tank, model


def test_self_pressurization_rises(lox_tank):
    lox, tank, model = lox_tank
    y0 = model.initial_state(2e5, 0.7)
    hist = model.simulate((0, 300.0), y0, n_out=20, rtol=1e-5)
    assert hist.P[-1] > hist.P[0]
    assert np.all(np.diff(hist.P) > -1.0)  # monotone rise (numerical noise ok)


def test_boiloff_positive_and_mass_conserved(lox_tank):
    lox, tank, model = lox_tank
    y0 = model.initial_state(2e5, 0.7)
    hist = model.simulate((0, 300.0), y0, n_out=20, rtol=1e-5)
    # zero at t=0 (saturated-equilibrium start), positive once heating acts
    assert hist.boiloff_rate[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(hist.boiloff_rate[1:] > 0)
    # no outflow: total fluid mass constant
    total = hist.m_liquid + hist.m_ullage
    assert total[-1] == pytest.approx(total[0], rel=1e-6)


def test_outflow_drains_tank(lox_tank):
    lox, tank, model = lox_tank
    y0 = model.initial_state(2.5e5, 0.8)
    mdot = 0.5  # kg/s
    hist = model.simulate((0, 60.0), y0, mdot_out=mdot, n_out=15, rtol=1e-5)
    dm = hist.m_liquid[0] - hist.m_liquid[-1]
    boiled = np.trapezoid(hist.boiloff_rate, hist.t)
    assert dm == pytest.approx(mdot * 60.0 + boiled, rel=0.02)
    # blowdown: ullage expands with no pressurant -> pressure falls
    assert hist.P[-1] < hist.P[0]


def test_pressurant_holds_setpoint(lox_tank):
    lox, tank, _ = lox_tank
    cfg = TankThermalConfig(pressurant_setpoint=2.5e5, pressurant_max_flow=0.2)
    model = TankThermalModel(lox, tank, 12.0, 50.0, cfg)
    y0 = model.initial_state(2.5e5, 0.8)
    hist = model.simulate((0, 60.0), y0, mdot_out=0.5, n_out=15, rtol=1e-5)
    assert hist.P[-1] > 0.9 * 2.5e5


def test_mixing_reduces_stratification(lox_tank):
    """Slosh-coupling knob: strong mixing must pull the surface layer toward
    the bulk temperature (destratification)."""
    lox, tank, model = lox_tank
    y0 = model.initial_state(2e5, 0.7)
    quiet = model.simulate((0, 300.0), y0, mixing_factor=1.0, n_out=15, rtol=1e-5)
    mixed = model.simulate((0, 300.0), y0, mixing_factor=20.0, n_out=15, rtol=1e-5)
    dT_quiet = quiet.T_surface[-1] - quiet.T_bulk[-1]
    dT_mixed = mixed.T_surface[-1] - mixed.T_bulk[-1]
    assert abs(dT_mixed) < abs(dT_quiet)


def test_homogeneous_rate_scales_linearly():
    lox = Fluid("LOX")
    r1 = homogeneous_pressure_rise_rate(lox, 0.1, 0.5, 2e5, 10.0)
    r2 = homogeneous_pressure_rise_rate(lox, 0.1, 0.5, 2e5, 20.0)
    assert r2 == pytest.approx(2 * r1, rel=1e-9)
    assert r1 > 0


def test_wall_property_tables():
    assert wall_cp("aluminum", 300.0) == pytest.approx(897.0)
    assert wall_cp("aluminum", 20.0) == pytest.approx(9.0)
    assert wall_k("stainless", 77.0) == pytest.approx(7.9)


def test_nu_correlations_positive():
    assert nu_vertical_plate(1e9, 0.7) > nu_vertical_plate(1e6, 0.7) > 1
    assert nu_horizontal_plate(1e8, True) > nu_horizontal_plate(1e8, False)
