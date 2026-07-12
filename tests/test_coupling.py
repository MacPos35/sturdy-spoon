"""Tests for the coupled tank -> regen -> injector simulation."""

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.combustion import gas_preset
from cryosim.coupling import (
    BurnProfile,
    CoupledConfig,
    CoupledSimulator,
    Engine,
)
from cryosim.fluids import Fluid
from cryosim.slosh_model import SloshModel
from cryosim.tank_geometry import TankGeometry
from cryosim.thermal_model import TankThermalConfig, TankThermalModel


@pytest.fixture(scope="module")
def sim():
    """5 kN-class LOX/CH4 engine, methane tank as coolant source, pump-fed."""
    ch4 = Fluid("LCH4")
    tank = TankGeometry(0.2, 1.0, "elliptical", "elliptical")
    tcfg = TankThermalConfig(pressurant_setpoint=3e5, pressurant_max_flow=0.1)
    tmodel = TankThermalModel(ch4, tank, wall_mass=25.0, heat_flux=150.0,
                              config=tcfg)
    engine = Engine(
        contour=ChamberContour(0.019, 6.0, 4.5, 0.09, n_points=80),
        channels=CoolingChannels(60, 1.0e-3, 1.4e-3, 0.7e-3, k_wall=330.0),
        gas=gas_preset("lox/ch4"),
        mixture_ratio=3.4,
    )
    return CoupledSimulator(
        ch4, tank, tmodel, SloshModel(ch4, tank), engine,
        CoupledConfig(dt=0.5, regen_interval=1.5, pump_dp=87e5),
    )


@pytest.fixture(scope="module")
def burn(sim):
    prof = BurnProfile(
        pc_of_t=lambda t: 30e5,
        axial_accel_of_t=lambda t: (3 + 0.2 * t) * 9.81,
        lateral_accel_of_t=lambda t: 0.8 * np.sin(2 * np.pi * 0.9 * t),
        t_end=5.0,
    )
    return sim.run(prof, P0=3e5, fill0=0.9, T_liquid0=118.0)


def test_run_completes_with_consistent_arrays(burn):
    a = burn.asarrays()
    n = len(a["t"])
    assert n >= 9
    for key, arr in a.items():
        assert len(arr) == n, key
    assert len(burn.regen_snapshots) >= 3


def test_mass_closure(burn):
    a = burn.asarrays()
    dt = np.diff(a["t"])
    drained = np.sum(a["mdot_coolant"][:-1] * dt)
    boiled = np.sum(a["boiloff_rate"][:-1] * dt)
    dm = a["m_liquid"][0] - a["m_liquid"][-1]
    assert dm == pytest.approx(drained + boiled, rel=5e-3)


def test_tank_drains_and_cg_drops(burn):
    a = burn.asarrays()
    assert a["fill_fraction"][-1] < a["fill_fraction"][0]
    assert a["cg_axial"][-1] < a["cg_axial"][0]


def test_slosh_frequency_follows_acceleration(burn):
    a = burn.asarrays()
    # accel ramps 3g -> 4g over the burn; f ~ sqrt(a) modulo depth change
    assert a["slosh_freq"][-1] > a["slosh_freq"][0]
    assert np.all(a["slosh_zeta"] > 0)


def test_injector_inlet_state_tracked(burn):
    a = burn.asarrays()
    ok = ~np.isnan(a["coolant_outlet_T"])
    assert ok.sum() >= 3
    # heated coolant enters the injector warmer than it left the tank
    assert np.all(a["coolant_outlet_T"][ok] > a["coolant_inlet_T"][ok])
    assert np.all(a["coolant_outlet_T"][ok] > a["T_bulk"][ok])
    # and the regen inlet temperature IS the tank outlet temperature
    assert np.allclose(a["coolant_inlet_T"][ok], a["T_bulk"][ok], atol=0.5)


def test_slosh_mixing_engages(burn):
    a = burn.asarrays()
    assert a["mixing_factor"].max() > 1.0
    assert np.abs(a["slosh_amplitude"]).max() > 0


def test_engine_off_no_regen(sim):
    prof = BurnProfile(pc_of_t=lambda t: 0.0, t_end=1.5)
    h = sim.run(prof, P0=3e5, fill0=0.5, T_liquid0=118.0)
    a = h.asarrays()
    assert np.all(np.isnan(a["coolant_outlet_T"]))
    assert np.all(a["mdot_coolant"] == 0.0)
    # only boil-off changes the liquid mass
    assert a["m_liquid"][0] - a["m_liquid"][-1] < 0.05


def test_fill_cutoff_stops_run(sim):
    prof = BurnProfile(pc_of_t=lambda t: 30e5, t_end=500.0)
    h = sim.run(prof, P0=3e5, fill0=0.10, T_liquid0=118.0)
    a = h.asarrays()
    assert a["t"][-1] < 500.0 - 1.0  # stopped early at the cutoff
    assert a["fill_fraction"][-1] >= sim.cfg.fill_cutoff * 0.9
