"""Tests for the slosh-coupled gimbal (TVC) stabilization model."""

import numpy as np
import pytest

from cryosim.fluids import Fluid
from cryosim.gimbal_control import (
    GimbalController,
    VehicleModel,
    closed_loop_eigenvalues,
    simulate_gimbal,
)
from cryosim.slosh_model import SloshModel
from cryosim.tank_geometry import TankGeometry


@pytest.fixture(scope="module")
def slosh_params():
    """Slosh analog of the demo LCH4 tank at 70% fill under 4 g."""
    ch4 = Fluid("LCH4")
    tank = TankGeometry(0.2, 1.0, "elliptical", "elliptical")
    model = SloshModel(ch4, tank)
    return model.params_at(0.7 * tank.V_total, 3e5, accel=4 * 9.81)


@pytest.fixture(scope="module")
def vehicle(slosh_params):
    """Imaginary 3 m student rocket: engine at station 0, nose at ~3 m."""
    return VehicleModel.from_components(
        dry_mass=60.0, dry_cg=1.7, dry_inertia=45.0,
        tank_bottom=0.9, slosh=slosh_params,
        engine_station=0.0, thrust=5000.0,
    )


def test_vehicle_assembly(vehicle, slosh_params):
    # composite-CG consistency is enforced by the constructor
    assert vehicle.x_engine < 0            # engine below CG
    assert vehicle.m_slosh == pytest.approx(slosh_params.first.mass)
    M = vehicle.mass_matrix()
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)  # physically valid mass matrix


def test_auto_tune_gains_scale_with_inertia(vehicle):
    c1 = GimbalController.auto_tune(vehicle, bandwidth_hz=1.0)
    # doubling inertia at fixed thrust must double the gains
    import dataclasses
    heavy = dataclasses.replace(vehicle, I_rigid=vehicle.I_rigid
                                + vehicle.I_theta)
    c2 = GimbalController.auto_tune(heavy, bandwidth_hz=1.0)
    assert c2.Kp / c1.Kp == pytest.approx(heavy.I_theta / vehicle.I_theta,
                                          rel=1e-9)


def test_closed_loop_stable_and_recovers_from_gust(vehicle):
    ctl = GimbalController.auto_tune(vehicle, bandwidth_hz=1.0)
    gust = lambda t: 120.0 if 1.0 < t < 1.4 else 0.0  # N lateral pulse
    hist = simulate_gimbal(vehicle, ctl, t_end=15.0, disturbance=gust)
    assert hist.stable
    assert hist.settled
    assert hist.max_theta_deg < 3.0
    assert np.all(np.abs(hist.delta) <= ctl.delta_max + 1e-12)


def test_slosh_mode_visible_and_baffles_damp_it(vehicle):
    """Smooth-wall damping (zeta ~ 5e-4) barely decays the slosh mode over
    a burn — the classic argument for baffles. With baffle-level damping
    (zeta = 0.01) the mode dies out."""
    import dataclasses

    ctl = GimbalController.auto_tune(vehicle, bandwidth_hz=1.0)
    smooth = simulate_gimbal(vehicle, ctl, t_end=15.0,
                             theta0=np.radians(2.0))
    assert np.abs(smooth.slosh_displacement).max() > 1e-5  # slosh excited
    baffled = simulate_gimbal(
        dataclasses.replace(vehicle, zeta_slosh=0.01), ctl, t_end=15.0,
        theta0=np.radians(2.0))
    tail = np.abs(baffled.slosh_displacement[baffled.t > 12.0]).max()
    assert tail < 0.3 * np.abs(baffled.slosh_displacement).max()
    # smooth wall decays slower than baffled
    tail_smooth = np.abs(smooth.slosh_displacement[smooth.t > 12.0]).max()
    assert tail_smooth > tail


def test_interaction_warning_near_slosh_frequency(vehicle):
    ws_hz = vehicle.omega_slosh / (2 * np.pi)
    ctl = GimbalController.auto_tune(vehicle, bandwidth_hz=ws_hz)
    assert any("slosh" in w.lower() for w in ctl.warnings)


def test_eigenvalues_structure(vehicle):
    ctl = GimbalController.auto_tune(vehicle, bandwidth_hz=1.0)
    eig = closed_loop_eigenvalues(vehicle, ctl)
    assert len(eig) == 7
    # no unstable pole; the lateral-drift double integrator sits at 0
    assert np.all(eig.real < 1e-9)
    assert np.sum(np.abs(eig) < 1e-9) == 2
    # the lightly damped slosh pair appears near omega_slosh
    ws = vehicle.omega_slosh
    pair = eig[np.abs(eig.imag) > 0.5 * ws]
    assert len(pair) >= 2
    assert np.min(np.abs(np.abs(pair.imag) - ws)) < 0.1 * ws


def test_station_convention_enforced():
    with pytest.raises(ValueError, match="composite CG"):
        VehicleModel(m_rigid=100.0, I_rigid=50.0, x_rigid=0.5,
                     m_slosh=10.0, x_slosh=0.5, L_slosh=0.1,
                     zeta_slosh=0.002, x_engine=-1.0, thrust=5000.0)


def test_plot_written(vehicle, tmp_path):
    from cryosim.gimbal_control import plot_gimbal

    ctl = GimbalController.auto_tune(vehicle, bandwidth_hz=1.0)
    hist = simulate_gimbal(vehicle, ctl, t_end=5.0,
                           theta0=np.radians(1.0), n_out=200)
    out = plot_gimbal(hist, str(tmp_path / "gimbal.png"))
    import os
    assert os.path.getsize(out) > 20000
