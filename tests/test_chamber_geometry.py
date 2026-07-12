"""Unit tests for the thrust-chamber contour and cooling channels."""

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels


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
