"""Unit tests for the thrust-chamber contour and cooling channels."""

import numpy as np
import pytest

from cryosim.chamber_geometry import (ChamberContour, CoolingChannels,
                                      nozzle_divergence_efficiency)


@pytest.fixture
def contour():
    return ChamberContour(
        throat_radius=0.02,
        contraction_ratio=8.0,
        expansion_ratio=4.0,
        chamber_length=0.12,
        n_points=400,
    )


def test_throat_is_minimum(contour):
    assert contour.r.min() == pytest.approx(contour.Rt, rel=1e-3)
    assert contour.x[np.argmin(contour.r)] == pytest.approx(
        contour.x_throat, abs=contour.length / 100
    )


def test_end_area_ratios(contour):
    ar = contour.area_ratio()
    assert ar[0] == pytest.approx(8.0, rel=1e-3)
    assert ar[-1] == pytest.approx(4.0, rel=1e-2)


def test_contour_shape(contour):
    # converging then diverging
    i_t = contour.i_throat
    assert np.all(np.diff(contour.r[:i_t]) <= 1e-12)
    assert np.all(np.diff(contour.r[i_t:]) >= -1e-12)


def test_channels_geometry():
    ch = CoolingChannels(
        n_channels=40, channel_width=1.5e-3, channel_height=2.0e-3,
        t_wall=1.0e-3,
    )
    assert ch.flow_area == pytest.approx(3.0e-6)
    assert ch.D_h == pytest.approx(2 * 1.5e-3 * 2e-3 / (1.5e-3 + 2e-3))
    # land at r_wall = 0.02: 2 pi (0.021)/40 - 1.5e-3
    assert ch.land_width(0.02) == pytest.approx(
        2 * np.pi * 0.021 / 40 - 1.5e-3, rel=1e-9
    )


def test_channels_overlap_rejected(contour):
    ch = CoolingChannels(
        n_channels=200, channel_width=1.5e-3, channel_height=2e-3, t_wall=1e-3
    )
    with pytest.raises(ValueError, match="land"):
        ch.validate_against(contour)


def test_helix_path_factor():
    ch = CoolingChannels(30, 1e-3, 2e-3, 1e-3, helix_angle_deg=30.0)
    assert ch.path_factor == pytest.approx(1.0 / np.cos(np.radians(30)))


# --- bell (thrust-optimized) nozzle ------------------------------------

def test_bell_exit_radius_and_area_ratio():
    bell = ChamberContour(throat_radius=0.02, contraction_ratio=8.0,
                          expansion_ratio=6.0, chamber_length=0.1,
                          nozzle_type="bell")
    assert bell.r[-1] == pytest.approx(0.02 * np.sqrt(6.0), rel=2e-3)
    assert bell.area_ratio()[-1] == pytest.approx(6.0, rel=5e-3)
    assert bell.r.min() == pytest.approx(bell.Rt, rel=1e-3)


def test_bell_shorter_and_more_efficient_than_cone():
    kw = dict(throat_radius=0.02, contraction_ratio=8.0, expansion_ratio=8.0,
              chamber_length=0.1)
    cone = ChamberContour(nozzle_type="conical", **kw)
    bell = ChamberContour(nozzle_type="bell", **kw)
    assert bell.length < cone.length            # 80% bell is shorter
    assert bell.theta_exit_deg < 15.0           # gentler exit than 15deg cone
    assert bell.divergence_efficiency() > cone.divergence_efficiency()


def test_divergence_efficiency_values():
    # 15-deg cone: 0.5(1+cos15) = 0.9830
    assert nozzle_divergence_efficiency(8.0, "conical") == pytest.approx(
        0.5 * (1 + np.cos(np.radians(15.0))), rel=1e-9)
    # bell beats the cone and stays below 1
    lam = nozzle_divergence_efficiency(8.0, "bell")
    assert 0.983 < lam < 1.0


def test_bell_contour_monotonic_divergent():
    bell = ChamberContour(throat_radius=0.02, contraction_ratio=8.0,
                          expansion_ratio=10.0, chamber_length=0.1,
                          nozzle_type="bell", n_points=300)
    div = bell.r[bell.x > bell.x_throat]
    assert np.all(np.diff(div) > -1e-9)         # radius grows to the exit


def test_bad_nozzle_type_rejected():
    with pytest.raises(ValueError, match="nozzle_type"):
        ChamberContour(throat_radius=0.02, nozzle_type="aerospike")
