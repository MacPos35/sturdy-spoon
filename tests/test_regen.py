"""Unit tests for the regen-cooling model building blocks."""

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.combustion import (
    CombustionGas,
    adiabatic_wall_temperature,
    area_ratio_from_mach,
    gas_preset,
    mach_from_area_ratio,
)
from cryosim.fluids import Fluid
from cryosim.regen_model import (
    RegenCoolingModel,
    bartz_h_g,
    dittus_boelter_nu,
    fin_efficiency,
    haaland_friction_factor,
)


def test_dittus_boelter_hand_value():
    # 0.023 * (1e5)^0.8 * 4^0.4 ; (1e5)^0.8 = 1e4
    assert dittus_boelter_nu(1e5, 1.0) == pytest.approx(230.0, rel=1e-9)
    assert dittus_boelter_nu(1e5, 4.0) == pytest.approx(230.0 * 4**0.4, rel=1e-9)


def test_haaland_regimes():
    assert haaland_friction_factor(1000.0, 0.0) == pytest.approx(0.064, rel=1e-9)
    # Moody chart, smooth pipe, Re = 1e5: f ~ 0.018
    assert haaland_friction_factor(1e5, 0.0) == pytest.approx(0.018, rel=0.03)


def test_fin_efficiency_limits():
    assert fin_efficiency(1e3, 350.0, 1e-3, 1e-9) == pytest.approx(1.0, abs=1e-6)
    # long fin: eta -> 1/(m b)
    h, k, tl, b = 5e4, 350.0, 1e-3, 0.05
    m = np.sqrt(2 * h / (k * tl))
    assert fin_efficiency(h, k, tl, b) == pytest.approx(1 / (m * b), rel=1e-3)


def test_isentropic_mach_table_values():
    # Anderson, Modern Compressible Flow, A/A* = 2.0, gamma = 1.4
    assert mach_from_area_ratio(2.0, 1.4, supersonic=True) == pytest.approx(
        2.197, abs=2e-3
    )
    assert mach_from_area_ratio(2.0, 1.4, supersonic=False) == pytest.approx(
        0.306, abs=2e-3
    )
    # round trip
    M = 1.7
    assert mach_from_area_ratio(
        area_ratio_from_mach(M, 1.2), 1.2, True
    ) == pytest.approx(M, rel=1e-9)


def test_adiabatic_wall_limits():
    gas = gas_preset("lox/ch4")
    assert adiabatic_wall_temperature(gas, 0.0) == pytest.approx(gas.T_c)
    # supersonic: T_aw below T_c (r < 1) but well above static T
    assert adiabatic_wall_temperature(gas, 3.0) < gas.T_c


def test_bartz_scalings():
    gas = gas_preset("lox/ch4")
    kw = dict(Dt=0.06, r_curv=0.04, area_ratio=1.0, M=1.0, T_wg=800.0)
    h1 = bartz_h_g(gas, 20e5, **kw)
    h2 = bartz_h_g(gas, 40e5, **kw)
    assert h2 / h1 == pytest.approx(2**0.8, rel=1e-9)  # h ~ Pc^0.8
    ha = bartz_h_g(gas, 20e5, Dt=0.06, r_curv=0.04, area_ratio=4.0, M=2.9,
                   T_wg=800.0)
    assert ha < 0.35 * h1  # (1/AR)^0.9 decay away from throat


@pytest.fixture(scope="module")
def small_engine():
    """~2 kN student-scale LOX/CH4 engine, methane-cooled."""
    gas = gas_preset("lox/ch4")
    ct = ChamberContour(0.013, contraction_ratio=8.0, expansion_ratio=4.5,
                        chamber_length=0.10, n_points=120)
    ch = CoolingChannels(n_channels=42, channel_width=1.2e-3,
                         channel_height=1.8e-3, t_wall=0.8e-3, k_wall=330.0)
    model = RegenCoolingModel(ct, ch, gas, Fluid("Methane"))
    Pc = 20e5
    mdot_f = Pc * ct.At / gas.c_star / 4.4  # fuel-only at MR 3.4
    return model, ct, Pc, mdot_f


def test_heat_flux_peaks_at_throat(small_engine):
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 60e5)
    assert abs(res.x[np.argmax(res.q)] - ct.x_throat) < 0.01
    assert abs(res.x[np.argmax(res.T_wg)] - ct.x_throat) < 0.02


def test_energy_consistency(small_engine):
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 60e5)
    dH = mdot * (res.coolant_outlet.h - res.coolant_inlet.h)
    assert res.Q_total == pytest.approx(dH, rel=0.03)


def test_wall_temperature_ordering(small_engine):
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 60e5)
    assert np.all(res.T_wg >= res.T_wc - 1e-9)
    assert np.all(res.T_wc > res.T_coolant)
    assert np.all(res.T_aw > res.T_wg)


def test_pressure_drops_monotonically(small_engine):
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 60e5)
    assert res.dP_total > 0
    assert res.coolant_outlet.P == pytest.approx(60e5 - res.dP_total, rel=1e-6)


def test_counterflow_injector_end_hottest(small_engine):
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 60e5)
    # counterflow: coolant enters at nozzle exit, hottest at injector face
    assert res.T_coolant[0] > res.T_coolant[-1]


def test_more_flow_cools_better(small_engine):
    model, ct, Pc, mdot = small_engine
    r1 = model.solve(Pc, mdot, 110.0, 60e5)
    r2 = model.solve(Pc, 1.5 * mdot, 110.0, 60e5)
    assert r2.peak_wall_temperature < r1.peak_wall_temperature
    assert r2.coolant_outlet.T < r1.coolant_outlet.T


def test_combustion_gas_properties():
    gas = CombustionGas(T_c=3430.0, gamma=1.14, molar_mass=21.3e-3)
    # c* for these values ~ 1800 m/s (LOX/CH4 class)
    assert gas.c_star == pytest.approx(1818.0, rel=0.01)
    assert gas.Pr == pytest.approx(4 * 1.14 / (9 * 1.14 - 5), rel=1e-12)
    assert 0.8 < gas.recovery_factor < 1.0


def test_pressure_collapse_flagged(small_engine):
    """Insufficient feed pressure must be flagged, not silently reported."""
    model, ct, Pc, mdot = small_engine
    res = model.solve(Pc, mdot, 110.0, 13e5)  # far too little feed pressure
    assert res.pressure_collapsed
    ok = model.solve(Pc, mdot, 110.0, 60e5)
    assert not ok.pressure_collapsed
