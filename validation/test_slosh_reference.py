"""Validation: slosh analog vs. the published linear-theory solution.

The mechanical analog implemented in ``slosh_model`` IS the closed-form
linear potential-flow solution of NASA SP-106 (Abramson, 1966, ch. 2) /
SP-8009, whose frequency predictions were experimentally confirmed to within
a few percent in the programs those handbooks summarize. These tests verify
the implementation against:

* the published eigenvalues xi_n of J1'(xi)=0 and dimensionless frequency
  table values Lambda = omega^2 R / g of SP-106 ch. 2;
* independently hand-evaluated values of the Mikishev-Dorozhkin damping
  correlation as printed in Dodge (2000) — note the correlation itself is
  empirical with ~+/-30% scatter in the underlying experiments;
* resonant response of the pendulum analog vs. the exact steady-state
  magnification factor 1/(2 zeta) of a linear oscillator.

Honesty note: digitized experimental frequency/damping traces (e.g. NASA
TN D-1367) were not accessible from the development environment; validation
is therefore against the published analytical solution and tabulated
constants, which for LINEAR slosh is the accepted design reference.
"""

import numpy as np
import pytest

from cryosim.slosh_model import (
    mikishev_dorozhkin_zeta,
    simulate_first_mode,
    slosh_parameters,
)

G = 9.80665


@pytest.mark.validation
def test_dimensionless_frequency_table():
    """SP-106 ch.2: Lambda(h/R) = xi_1 tanh(xi_1 h/R)."""
    cases = {  # h/R : Lambda = omega^2 R / g
        0.5: 1.8412 * np.tanh(1.8412 * 0.5),
        1.0: 1.8412 * np.tanh(1.8412),
        2.0: 1.8412 * np.tanh(3.6824),
    }
    for hR, lam in cases.items():
        p = slosh_parameters(1000.0, 1e-6, R=1.0, h=hR, accel=G)
        # expected values use the 5-digit tabulated xi_1 = 1.8412
        assert p.first.omega**2 / G == pytest.approx(lam, rel=1e-4)


@pytest.mark.validation
def test_slosh_mass_curve_endpoints():
    """SP-8009 fig. limits: m1/m -> 0.837 (shallow) and ~0 (very deep)."""
    # shallow limit: m1/m -> 2/(xi_1^2 - 1) = 0.837
    p_sh = slosh_parameters(1000.0, 1e-6, R=1.0, h=0.01, accel=G)
    assert p_sh.slosh_mass_fraction == pytest.approx(2 / (1.8412**2 - 1), rel=1e-3)
    p_dp = slosh_parameters(1000.0, 1e-6, R=1.0, h=20.0, accel=G)
    assert p_dp.slosh_mass_fraction < 0.06


@pytest.mark.validation
def test_mikishev_dorozhkin_reference_values():
    """Hand-evaluated correlation points (Dodge 2000 form)."""
    # deep tank, water, R = 0.5 m:
    # base = 0.79 sqrt(1e-6 / sqrt(9.80665 * 0.125)) = 0.79*sqrt(9.034e-7)
    z = mikishev_dorozhkin_zeta(1e-6, 0.5, 5.0, G)
    base = 0.79 * np.sqrt(1e-6 / np.sqrt(G * 0.125))
    assert z == pytest.approx(base, rel=1e-3)  # depth correction -> 1 deep
    # shallow correction increases damping: bracket factor at h/R = 0.5 is
    # 1 + 0.318/sinh(0.92) * (1 + 0.5/cosh(0.92)) = 1.405
    z_shallow = mikishev_dorozhkin_zeta(1e-6, 0.5, 0.25, G)
    assert z_shallow / z == pytest.approx(1.405, abs=0.01)


@pytest.mark.validation
def test_resonant_magnification():
    """Steady-state resonant amplitude of x1 must equal the linear-oscillator
    result a0/(2 zeta omega^2)."""
    p = slosh_parameters(1000.0, 1e-6, R=0.3, h=0.6, accel=G)
    p.zeta_viscous = 0.02
    w = p.first.omega
    a0 = 0.1
    a_lat = lambda t: a0 * np.sin(w * t)
    T = 2 * np.pi / w
    hist = simulate_first_mode(lambda t: p, a_lat, (0.0, 120 * T), n_out=12000)
    steady = np.abs(hist.displacement[-2000:]).max()
    assert steady == pytest.approx(a0 / (2 * 0.02 * w**2), rel=0.03)


@pytest.mark.validation
def test_slosh_mass_matches_published_1_over_2p2_form():
    """Literature cross-check: the widely printed first-mode slosh mass
    m1 = m_F (R/2.2h) tanh(1.84 h/R)  is algebraically our
    2 tanh(xi h/R) / ((h/R) xi (xi^2-1)) with xi = 1.8412:
    the coefficient 2/(xi(xi^2-1)) equals 1/2.2 to 4 significant figures."""
    xi = 1.8412
    assert 2.0 / (xi * (xi**2 - 1.0)) == pytest.approx(1.0 / 2.2, rel=2e-4)
    p = slosh_parameters(1000.0, 1e-6, R=0.3, h=0.45, accel=G)
    m_published = p.m_liquid * (0.3 / (2.2 * 0.45)) * np.tanh(1.84 * 0.45 / 0.3)
    assert p.first.mass == pytest.approx(m_published, rel=2e-3)
