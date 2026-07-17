"""Fast unit tests for the equilibrium combustion API (not the CEA bands —
those live in validation/test_combustion_equilibrium.py)."""

import numpy as np
import pytest

from cryosim.combustion import CombustionGas
from cryosim.combustion_equilibrium import (
    equilibrium_combustion,
    optimize_of,
    stoichiometric_of,
)


def test_returns_gas_and_reasonable_numbers():
    r = equilibrium_combustion("ch4", 3.2, 20e5)
    assert 3000 < r.T_c < 3800
    assert 1.13 < r.gamma < 1.26
    assert 0.015 < r.molar_mass < 0.030
    gas = r.as_gas()
    assert isinstance(gas, CombustionGas)
    assert gas.T_c == r.T_c and gas.gamma == r.gamma


def test_deterministic():
    a = equilibrium_combustion("ch4", 3.2, 20e5)
    b = equilibrium_combustion("ch4", 3.2, 20e5)
    assert a.T_c == b.T_c
    assert a.molar_mass == b.molar_mass


def test_unknown_fuel_raises():
    with pytest.raises(KeyError):
        equilibrium_combustion("kerosene_blend_x", 2.0, 20e5)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        equilibrium_combustion("ch4", -1.0, 20e5)
    with pytest.raises(ValueError):
        equilibrium_combustion("ch4", 3.0, -20e5)


def test_mole_fractions_sum_to_one():
    r = equilibrium_combustion("h2", 6.0, 20e5)
    assert sum(r.mole_fractions.values()) == pytest.approx(1.0, abs=0.02)


def test_optimize_of_returns_result():
    of, res = optimize_of("ch4", 20e5, objective="c_star")
    assert 1.5 < of < stoichiometric_of("ch4")
    assert res.of_ratio == pytest.approx(of)


def test_shifting_functions_return_sane_values():
    from cryosim.combustion_equilibrium import shifting_c_star, shifting_isp
    cs = shifting_c_star("ch4", 3.2, 20e5)
    assert 1700 < cs < 1950
    # shifting c* exceeds the frozen c* (recombination credit)
    assert cs > equilibrium_combustion("ch4", 3.2, 20e5).c_star
    isp = shifting_isp("ch4", 3.4, 20e5, expansion_ratio=40.0)
    assert 340 < isp < 390


def test_extreme_off_stoich_does_not_crash():
    # very lean and very rich should still return (clamped) results
    assert equilibrium_combustion("ch4", 0.5, 20e5).T_c > 400
    assert equilibrium_combustion("ch4", 12.0, 20e5).T_c > 400


def test_soot_predicted_only_when_very_rich():
    # normal / lean operation: no soot; extreme rich: carbon supersaturated
    assert not equilibrium_combustion("ch4", 3.2, 20e5).soot_predicted
    assert not equilibrium_combustion("ch4", 2.0, 20e5).soot_predicted
    rich = equilibrium_combustion("ch4", 1.0, 20e5)
    assert rich.soot_predicted and rich.carbon_activity >= 1.0


# ---------------------------------------------------------------- RP-1


def test_rp1_fractional_formula():
    """RP-1 is the CEA CH1.9423 surrogate: fractional atoms must work."""
    from cryosim.combustion_equilibrium import FUELS
    f = FUELS["rp1"]
    assert f.nH == pytest.approx(1.9423)
    # M(CH1.9423) = 12.0107 + 1.9423*1.00794 = 13.97 g/mol
    assert f.molar_mass == pytest.approx(13.97e-3, rel=1e-3)
    # complete combustion needs 2 + 1.9423/4 mol O per C -> O/F 3.40
    assert stoichiometric_of("rp1") == pytest.approx(3.40, rel=0.01)


def test_rp1_aliases_identical():
    a = equilibrium_combustion("rp1", 2.3, 20e5)
    b = equilibrium_combustion("kerosene", 2.3, 20e5)
    c = equilibrium_combustion("RP-1", 2.3, 20e5)
    assert a.T_c == b.T_c == c.T_c


def test_rp1_reasonable_numbers():
    r = equilibrium_combustion("rp1", 2.3, 20e5)
    assert 3300 < r.T_c < 3600          # CEA ~3540 K at 20 bar
    assert 1.15 < r.gamma < 1.26
    assert 0.019 < r.molar_mass < 0.025
    assert not r.soot_predicted         # chamber gas at O/F 2.3 is not sooting
