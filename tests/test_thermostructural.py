"""Hot-wall thermo-structural + Manson-Coffin life."""

import numpy as np
import pytest

from cryosim.thermostructural import (STRUCTURAL, manson_coffin_life,
                                      yield_at)


def test_thermal_stress_formula_exact():
    m = STRUCTURAL["cucrzr"]
    dT = 150.0
    sig = m["E"] * m["alpha"] * dT / (2 * (1 - m["nu"]))
    assert sig == pytest.approx(130e9 * 17.5e-6 * 150 / (2 * (1 - 0.34)))


def test_manson_coffin_monotonic():
    m = STRUCTURAL["cucrzr"]
    lives = [manson_coffin_life(de, m) for de in (0.002, 0.005, 0.01, 0.03)]
    assert all(np.diff(lives) < 0)          # more strain → fewer cycles
    # 1% strain range for a copper alloy → hundreds-to-thousands of cycles
    assert 200 < manson_coffin_life(0.01, m) < 5000


def test_yield_derates_with_temperature():
    m = STRUCTURAL["cucrzr"]
    assert yield_at(m, 293.0) == pytest.approx(m["sigma_y0"])
    assert yield_at(m, m["T_limit"]) == pytest.approx(0.5 * m["sigma_y0"])
    assert yield_at(m, 550.0) < yield_at(m, 350.0)


def test_analyze_on_design():
    from cryosim.engine_design import EngineSpec, design_engine
    d = design_engine(EngineSpec(thrust=5e3), n_random=8, n_polish=8)
    ts = d.thermostructural
    assert ts is not None
    assert ts.peak_stress > 0
    assert ts.sigma_total.shape == d.channel_design.result.x.shape
    # combined stress = thermal + pressure
    assert np.allclose(ts.sigma_total,
                       ts.sigma_thermal + ts.sigma_pressure)
    assert ts.cycle_life > 0
