"""Swirl-injector theory checks against the published relations."""

import numpy as np
import pytest

from cryosim.injector_design import (
    A_from_phi,
    design_injector,
    discharge_coefficient,
    phi_from_A,
    phi_from_spray_angle,
    spray_half_angle,
)


def test_maximum_flow_relation_roundtrip():
    for A in (0.3, 0.8, 1.5, 3.0, 6.0):
        phi = phi_from_A(A)
        assert A_from_phi(phi) == pytest.approx(A, rel=1e-9)


def test_relations_hand_evaluated():
    # hand evaluation of the published closed forms at phi = 0.5:
    # A  = 0.5*sqrt(2)/(0.5*sqrt(0.5)) = 2
    # mu = 0.5*sqrt(0.5)/sqrt(1.5)     = 0.288675...
    assert A_from_phi(0.5) == pytest.approx(2.0, rel=1e-12)
    assert discharge_coefficient(0.5) == pytest.approx(
        0.5 * np.sqrt(0.5) / np.sqrt(1.5), rel=1e-12)


def test_literature_anchor_points():
    """A = 1 gives mu ~ 0.44 and half-angle ~ 30-35 deg in the classical
    charts (Bazarov ch. 2 / Bayvel & Orzechowski)."""
    phi = phi_from_A(1.0)
    assert 0.40 <= discharge_coefficient(phi) <= 0.48
    assert 28.0 <= np.degrees(spray_half_angle(phi)) <= 38.0


def test_monotonic_trends():
    """More swirl (larger A): lower Cd, wider cone."""
    A = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    phis = [phi_from_A(a) for a in A]
    cds = [discharge_coefficient(p) for p in phis]
    angs = [spray_half_angle(p) for p in phis]
    assert all(np.diff(cds) < 0)
    assert all(np.diff(angs) > 0)


def test_spray_angle_inversion():
    for deg in (25.0, 40.0, 55.0):
        phi = phi_from_spray_angle(np.radians(deg))
        assert np.degrees(spray_half_angle(phi)) == pytest.approx(deg,
                                                                  abs=1e-6)


@pytest.fixture
def demo_design():
    return design_injector(
        Pc=20e5, thrust=5e3, mdot_ox=1.5, mdot_fuel=0.5,
        rho_ox=1140.0, mu_ox=2.0e-4, rho_fuel=60.0, face_radius=0.06)


def test_element_mass_flow_closure(demo_design):
    d = demo_design
    ox_total = d.n_elements * d.element.mdot
    assert ox_total == pytest.approx(d.mdot_ox, rel=1e-9)
    fuel_core = d.n_elements * d.annulus.mdot
    film = 0.0 if d.film is None else d.film.mdot
    assert fuel_core + film == pytest.approx(d.mdot_fuel, rel=1e-9)


def test_orifice_equation_consistency(demo_design):
    """The sized swirler must reproduce its flow through mdot = mu A sqrt."""
    e = demo_design.element
    mdot = e.Cd * np.pi * e.r_nozzle**2 * np.sqrt(2 * e.rho * e.dP)
    assert mdot == pytest.approx(e.mdot, rel=1e-9)


def test_geometric_characteristic_consistency(demo_design):
    e = demo_design.element
    A_geo = e.R_swirl_arm * e.r_nozzle / (e.n_tangential * e.r_tangential**2)
    assert A_geo == pytest.approx(e.A, rel=1e-9)


def test_face_packing(demo_design):
    d = demo_design
    assert sum(n for _, n in d.rings) == d.n_elements
    # outermost ring must clear the wall margin
    r_out = max(r for r, _ in d.rings)
    assert r_out + d.d_element_env / 2.0 <= d.face_radius


def test_gas_like_fuel_flagged(demo_design):
    assert any("gas-like" in w for w in demo_design.warnings)


def test_stiffness_reported(demo_design):
    assert demo_design.stiffness_ox == pytest.approx(0.20)
    assert demo_design.P_feed_ox > demo_design.Pc


def test_describe_runs(demo_design):
    text = demo_design.describe()
    assert "swirler" in text and "stiffness" in text
