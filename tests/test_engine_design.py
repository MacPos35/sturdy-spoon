"""Autonomous design pipeline: convergence, determinism, ledger, trace."""

import numpy as np
import pytest

from cryosim.engine_design import (
    DesignError,
    DesignTrace,
    EngineSpec,
    _c_star,
    _optimum_expansion,
    _thrust_coefficient,
    design_engine,
)
from cryosim.combustion import gas_preset

# small search so the whole module runs in ~a minute
FAST = dict(n_random=8, n_polish=8)


@pytest.fixture(scope="module")
def demo():
    spec = EngineSpec(thrust=5e3, propellants="lox/ch4",
                      chamber_pressure=20e5, name="pytest-5kN")
    return design_engine(spec, **FAST)


def test_c_star_hand_check():
    """c* for gamma=1.2, M=22 g/mol, Tc=3000 K vs the closed form."""
    from cryosim.combustion import CombustionGas, R_UNIV
    gas = CombustionGas(T_c=3000.0, gamma=1.2, molar_mass=22e-3)
    g = 1.2
    R = R_UNIV / 22e-3
    Gamma = np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    assert _c_star(gas) == pytest.approx(np.sqrt(R * 3000.0) / Gamma,
                                         rel=1e-12)


def test_thrust_coefficient_matched_exit():
    """At perfect expansion the pressure term vanishes; Cf for gamma=1.2,
    eps chosen so Pe=Pa, cross-checked against Sutton eq. 3-30 evaluated
    by hand."""
    gas = gas_preset("lox/ch4")
    Pc, Pa = 20e5, 101325.0
    eps, _ = _optimum_expansion(gas, Pc, Pa, cap=50.0)
    Cf, Pe = _thrust_coefficient(gas, Pc, eps, Pa)
    assert Pe == pytest.approx(Pa, rel=1e-3)
    g = gas.gamma
    term = (2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1))
            * (1 - (Pa / Pc) ** ((g - 1) / g)))
    assert Cf == pytest.approx(np.sqrt(term), rel=1e-3)


def test_vacuum_design_uses_cap():
    gas = gas_preset("lox/ch4")
    eps, why = _optimum_expansion(gas, 20e5, 0.0, cap=25.0)
    assert eps == 25.0
    assert "cap" in why


def test_pipeline_converges_all_constraints(demo):
    assert all(item.ok for item in demo.ledger)
    assert demo.iterations >= 1
    assert len(demo.trace.entries) > 10


def test_predicted_performance_plausible(demo):
    """5 kN LOX/CH4 at 20 bar SL: the numbers must land in known bands."""
    assert 220.0 <= demo.Isp_ambient <= 280.0
    assert demo.Isp_vac > demo.Isp_ambient
    assert 0.03 <= 2 * demo.throat_radius <= 0.08        # Dt 30-80 mm
    assert 1.5 <= demo.mdot <= 3.0
    assert demo.channel_design.peak_T_wg <= 800.0
    # thrust closure: F = Cf * Pc * At
    At = np.pi * demo.throat_radius**2
    assert demo.Cf * demo.spec.chamber_pressure * At == pytest.approx(
        demo.spec.thrust, rel=1e-6)
    # mass balance
    assert demo.mdot_ox + demo.mdot_fuel == pytest.approx(demo.mdot)
    assert demo.mdot_ox / demo.mdot_fuel == pytest.approx(demo.of_ratio)


def test_injector_fed_from_regen_outlet(demo):
    """The coupling that matters: fuel injection density is the hot regen
    outlet state, not the tank state."""
    outlet = demo.channel_design.result.coolant_outlet
    assert demo.injector.annulus.rho == pytest.approx(outlet.rho)


def test_determinism():
    spec = EngineSpec(thrust=3e3, chamber_pressure=15e5, name="det")
    a = design_engine(spec, **FAST)
    b = design_engine(spec, **FAST)
    assert a.throat_radius == b.throat_radius
    assert a.channel_design.channels.n_channels == \
        b.channel_design.channels.n_channels
    assert a.channel_design.channels.channel_width == \
        b.channel_design.channels.channel_width
    assert a.injector.n_elements == b.injector.n_elements
    assert str(a.trace) == str(b.trace)


def test_trace_markdown(demo):
    md = demo.trace.as_markdown()
    assert md.startswith("# Design trace")
    assert "A. performance" in md and "B. thermal" in md


def test_ledger_in_describe(demo):
    text = demo.describe()
    assert "[PASS]" in text and "Constraint ledger" in text


def test_spec_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown spec keys"):
        EngineSpec.from_dict({"thrust": 1e3, "warp_drive": True})


def test_spec_yaml_roundtrip(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("name: yam\nthrust: 2.0e3\nchamber_pressure: 1.5e6\n"
                 "propellants: lox/ch4\n")
    spec = EngineSpec.from_yaml(str(p))
    assert spec.thrust == 2e3
    assert spec.name == "yam"


def test_infeasible_design_fails_loudly():
    """A ceiling below what the jacket needs must raise, with the ledger."""
    spec = EngineSpec(thrust=5e3, chamber_pressure=60e5,
                      max_feed_pressure=70e5, name="doomed")
    with pytest.raises(DesignError):
        design_engine(spec, **FAST)


def test_example_spec_parses():
    spec = EngineSpec.from_yaml("examples/engine_5kn_methalox.yaml")
    assert spec.thrust == 5e3
    assert spec.propellants == "lox/ch4"
