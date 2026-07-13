"""Tests for the cooling-channel design optimizer."""

import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.combustion import gas_preset
from cryosim.fluids import Fluid
from cryosim.optimize import optimize_channels


@pytest.fixture(scope="module")
def setup():
    gas = gas_preset("lox/ch4")
    ct = ChamberContour(0.019, 6.0, 4.5, 0.09, n_points=50)
    Pc = 30e5
    mdot = Pc * ct.At / gas.c_star / 4.4
    return ct, gas, Fluid("Methane"), Pc, mdot


@pytest.fixture(scope="module")
def optimum(setup):
    ct, gas, ch4, Pc, mdot = setup
    baseline = CoolingChannels(60, 1.0e-3, 1.4e-3, 0.7e-3, k_wall=330.0)
    return optimize_channels(
        ct, gas, ch4, Pc, mdot, 118.0, 90e5, dp_budget=25e5,
        n_random=12, n_polish=10, seed=3, baseline=baseline,
    )


def test_optimizer_beats_baseline(optimum):
    assert optimum.baseline is not None
    assert optimum.peak_T_wg < optimum.baseline["peak"]
    assert optimum.dp < 25e5 + 1e5  # respects the budget (within penalty)


def test_optimum_is_feasible(optimum):
    assert optimum.land_at_throat > 0.15e-3
    assert not optimum.result.boiling_detected
    assert optimum.n_evaluations > 10


def test_report_content(optimum):
    txt = optimum.report()
    assert "peak T_wg" in txt and "vs baseline" in txt
    # unconstrained-manufacturing optimum lands in HARCC territory:
    # narrow/deep channels flagged for non-conventional manufacturing
    if optimum.manufacturability_warnings:
        assert "manufacturability" in txt


def test_infeasible_space_raises(setup):
    ct, gas, ch4, Pc, mdot = setup
    with pytest.raises(RuntimeError, match="no feasible"):
        optimize_channels(
            ct, gas, ch4, Pc, mdot, 118.0, 90e5,
            bounds={"n_channels": (400, 500),        # lands close up
                    "channel_width": (2.5e-3, 3e-3)},
            n_random=6, n_polish=0, seed=1,
        )
