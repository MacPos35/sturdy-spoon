"""Tests for the automatic torus-manifold design module."""

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.manifold_design import (
    coupled_maldistribution,
    design_manifolds,
    header_pressure_profile,
    _torus_wall,
)


@pytest.fixture(scope="module")
def geom():
    ct = ChamberContour(0.019, 6.0, 4.5, 0.09, n_points=80)
    ch = CoolingChannels(60, 1.0e-3, 1.4e-3, 0.7e-3, k_wall=330.0)
    return ct, ch


@pytest.fixture(scope="module")
def design(geom):
    ct, ch = geom
    # 5 kN methane jacket conditions (cold dense inlet, hot gas-like outlet)
    return design_manifolds(
        ct, ch, mdot_coolant=0.425, dp_channel=21e5,
        rho_in=390.0, mu_in=8.5e-5, rho_out=35.0, mu_out=3.0e-5,
        mawp=135e5, target=0.03,
    )


def test_header_profile_shapes():
    theta = 2 * np.pi * (np.arange(60) + 0.5) / 60
    kw = dict(n_feeders=1, mdot_total=0.4, rho=400.0, mu=1e-4,
              d=8e-3, r_centerline=0.05)
    dp_div = header_pressure_profile(theta, dividing=True, **kw)
    dp_comb = header_pressure_profile(theta, dividing=False, **kw)
    # zero-mean deviations
    assert abs(dp_div.mean()) < 1e-6 * np.abs(dp_div).max()
    # port at theta=0: dividing header pressure is LOWEST at the port,
    # highest opposite (momentum regain dominates at these sizes)
    i_port = 0
    i_far = np.argmin(np.abs(theta - np.pi))
    assert dp_div[i_port] < dp_div[i_far]
    assert dp_comb[i_port] < dp_comb[i_far]  # collector also low at port
    # symmetric about the port
    assert dp_div[1] == pytest.approx(dp_div[-2], rel=1e-6)


def test_bigger_duct_more_uniform():
    theta = 2 * np.pi * (np.arange(60) + 0.5) / 60
    kw = dict(n_feeders=1, mdot_total=0.4, rho=400.0, mu=1e-4,
              r_centerline=0.05, dividing=True)
    small = header_pressure_profile(theta, d=5e-3, **kw)
    big = header_pressure_profile(theta, d=12e-3, **kw)
    assert np.ptp(big) < 0.2 * np.ptp(small)


def test_stiffer_channels_tolerate_worse_manifold():
    theta = 2 * np.pi * (np.arange(40) + 0.5) / 40
    dp = header_pressure_profile(theta, 1, 0.4, 400.0, 1e-4, 6e-3, 0.05,
                                 True)
    soft, _ = coupled_maldistribution(theta, dp, lambda th: 0 * th, 0.0,
                                      2e5)
    stiff, _ = coupled_maldistribution(theta, dp, lambda th: 0 * th, 0.0,
                                       20e5)
    assert stiff < soft


def test_torus_wall_exceeds_cylinder_and_converges():
    # inner-crotch factor > 1, approaching the cylinder value for R >> r
    t_tight = _torus_wall(100e5, 20e-3, 0.03, "316L")
    t_loose = _torus_wall(100e5, 20e-3, 3.0, "316L")
    assert t_tight > t_loose
    from cryosim.line_sizing import DESIGN_FACTOR, MATERIALS
    t_cyl = DESIGN_FACTOR * 100e5 * 10e-3 / MATERIALS["316L"]["S_allow"]
    assert t_loose == pytest.approx(max(t_cyl, 0.9e-3), rel=0.02)


def test_design_meets_target(design):
    assert design.met_target
    assert design.maldistribution <= 0.03
    assert len(design.flow_relative) == 60
    assert design.flow_relative.mean() == pytest.approx(1.0, rel=1e-9)


def test_velocity_cap_respected(design):
    assert design.inlet.V_max <= 30.0 + 1e-9
    assert design.outlet.V_max <= 30.0 + 1e-9


def test_hot_collector_is_bigger(design):
    # gas-like outlet flow needs a much larger duct than the dense inlet
    assert design.outlet.duct_diameter > design.inlet.duct_diameter


def test_wall_and_mass_positive(design):
    for m in (design.inlet, design.outlet):
        assert m.wall_thickness >= 0.9e-3
        assert m.mass > 0
        assert m.r_centerline > m.r_attach


def test_report_and_drawing(design, geom, tmp_path):
    import os

    from cryosim.manifold_design import draw_manifolds

    txt = design.report()
    assert "uniformity" in txt and "NOT MET" not in txt
    assert "stress analysis" in txt
    ct, _ = geom
    out = draw_manifolds(design, ct, str(tmp_path / "manifold.png"))
    assert os.path.getsize(out) > 30000
