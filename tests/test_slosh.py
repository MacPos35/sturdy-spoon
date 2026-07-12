"""Unit tests for the slosh mechanical analog."""

import numpy as np
import pytest

from cryosim.fluids import Fluid
from cryosim.slosh_model import (
    XI_N,
    SloshModel,
    mikishev_dorozhkin_zeta,
    simulate_first_mode,
    slosh_parameters,
)
from cryosim.tank_geometry import TankGeometry

RHO, NU = 1000.0, 1.0e-6  # water
G = 9.80665


def test_bessel_roots_match_tabulated():
    # Abramson NASA SP-106, ch. 2: roots of J1'(xi)=0
    assert XI_N[0] == pytest.approx(1.8412, abs=1e-4)
    assert XI_N[1] == pytest.approx(5.3314, abs=1e-4)
    assert XI_N[2] == pytest.approx(8.5363, abs=1e-4)


def test_deep_tank_limits():
    p = slosh_parameters(RHO, NU, R=0.2, h=2.0, accel=G)  # h/R = 10
    # omega^2 R / a -> xi_1
    assert p.first.omega**2 * 0.2 / G == pytest.approx(1.8412, rel=1e-4)
    # pendulum length -> R / xi_1
    assert p.first.pendulum_length == pytest.approx(0.2 / 1.8412, rel=1e-4)
    # slosh-mass depth below surface -> R / xi_1
    assert p.first.depth_below_surface == pytest.approx(0.2 / 1.8412, rel=1e-3)


def test_mass_bookkeeping():
    p = slosh_parameters(RHO, NU, R=0.2, h=0.3, accel=G)
    total_modal = sum(m.mass for m in p.modes)
    assert p.m_rigid + total_modal == pytest.approx(p.m_liquid, rel=1e-12)
    assert 0 < p.m_rigid < p.m_liquid
    # CG conservation
    cg = (p.m_rigid * p.rigid_height +
          sum(m.mass * m.height_above_bottom for m in p.modes)) / p.m_liquid
    assert cg == pytest.approx(p.h / 2, rel=1e-12)


def test_slosh_mass_fraction_h_over_R_1():
    # SP-8009 curve: m1/m ~ 0.43 at h/R = 1 (flat-bottom cylinder)
    p = slosh_parameters(RHO, NU, R=0.2, h=0.2, accel=G)
    assert p.slosh_mass_fraction == pytest.approx(0.432, abs=0.01)


def test_frequency_increases_with_accel():
    p1 = slosh_parameters(RHO, NU, 0.2, 0.3, accel=G)
    p3 = slosh_parameters(RHO, NU, 0.2, 0.3, accel=3 * G)
    assert p3.first.omega == pytest.approx(np.sqrt(3) * p1.first.omega, rel=1e-9)


def test_damping_viscosity_scaling():
    z1 = mikishev_dorozhkin_zeta(NU, 0.2, 0.6, G)
    z2 = mikishev_dorozhkin_zeta(4 * NU, 0.2, 0.6, G)
    assert z2 == pytest.approx(2 * z1, rel=1e-9)
    assert 0 < z1 < 0.05


def test_free_decay_matches_zeta():
    """Ring-down of the pendulum analog must decay at the imposed zeta."""
    p = slosh_parameters(RHO, NU, R=0.2, h=0.4, accel=G)
    p.zeta_viscous = 0.01
    w = p.first.omega
    # short lateral impulse to excite the mode
    a_lat = lambda t: 1.0 if t < 0.05 else 0.0
    T = 2 * np.pi / w
    hist = simulate_first_mode(lambda t: p, a_lat, (0.0, 30 * T), n_out=6000)
    x = hist.displacement
    i0 = np.argmax(np.abs(x))
    # amplitude envelope from peaks after the impulse
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.abs(x[i0:]))
    amp = np.abs(x[i0:])[pk]
    n_cycles = np.arange(len(amp)) / 2.0  # two |x| peaks per cycle
    fit = np.polyfit(n_cycles, np.log(amp), 1)[0]
    zeta_meas = -fit / (2 * np.pi)
    assert zeta_meas == pytest.approx(0.01, rel=0.05)


def test_slosh_model_with_lox_tank():
    lox = Fluid("LOX")
    tank = TankGeometry(0.15, 0.8, "elliptical", "elliptical")
    model = SloshModel(lox, tank)
    p = model.params_at(V_liquid=0.6 * tank.V_total, P=2e5, accel=3 * G)
    # student-scale LOX tank under 3 g: first mode of order 1-2 Hz
    assert 0.5 < p.first.frequency < 5.0
    assert not p.in_dome
    p_low = model.params_at(V_liquid=0.02 * tank.V_total, P=2e5)
    assert p_low.in_dome


def test_zeta_override():
    lox = Fluid("LOX")
    tank = TankGeometry(0.15, 0.8)
    model = SloshModel(lox, tank, zeta_override=0.05)  # e.g. baffled tank
    p = model.params_at(0.5 * tank.V_total, 2e5)
    assert p.zeta_viscous == 0.05
