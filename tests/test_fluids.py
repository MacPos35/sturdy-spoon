"""Unit tests for the CoolProp fluid wrapper.

Reference values: NIST WebBook / standard cryogenic property tables.
"""

import pytest

from cryosim.fluids import Fluid, canonical_name


def test_aliases():
    assert canonical_name("LOX") == "Oxygen"
    assert canonical_name("lch4") == "Methane"
    assert canonical_name("Ethanol") == "Ethanol"
    # RP-1 maps to its standard single-component surrogate
    assert canonical_name("RP-1") == "n-Dodecane"
    assert canonical_name("kerosene") == "n-Dodecane"


def test_rp1_surrogate_liquid_density():
    rp1 = Fluid("rp1")
    # n-dodecane at 288 K, 1 atm: ~750 kg/m3 (NIST); RP-1 itself is ~810 —
    # the surrogate is the documented, slightly conservative stand-in
    assert rp1.state_TP(288.15, 101325).rho == pytest.approx(750.0, rel=0.02)


def test_lox_saturation_at_1atm():
    lox = Fluid("LOX")
    # NIST: O2 normal boiling point 90.19 K, sat. liquid density ~1141 kg/m3,
    # latent heat ~213 kJ/kg.
    assert lox.T_sat(101325) == pytest.approx(90.19, abs=0.05)
    assert lox.sat_liquid(101325).rho == pytest.approx(1141, rel=0.01)
    assert lox.h_fg(101325) == pytest.approx(213e3, rel=0.02)


def test_methane_saturation_at_1atm():
    ch4 = Fluid("LCH4")
    # NIST: CH4 normal boiling point 111.67 K, sat. liquid density ~422.4 kg/m3
    assert ch4.T_sat(101325) == pytest.approx(111.67, abs=0.05)
    assert ch4.sat_liquid(101325).rho == pytest.approx(422.4, rel=0.01)


def test_flash_rho_u_roundtrip():
    lox = Fluid("LOX")
    st = lox.state_TP(150.0, 5e5)  # superheated O2 vapor
    P, T = lox.flash_rho_u(st.rho, st.u)
    assert P == pytest.approx(5e5, rel=1e-4)
    assert T == pytest.approx(150.0, rel=1e-4)


def test_state_ph_roundtrip():
    ch4 = Fluid("Methane")
    st = ch4.state_TP(120.0, 60e5)  # compressed liquid methane (coolant inlet)
    st2 = ch4.state_PH(60e5, st.h)
    assert st2.T == pytest.approx(120.0, rel=1e-5)
    assert st2.rho == pytest.approx(st.rho, rel=1e-5)


def test_prandtl_positive_liquid_and_vapor():
    lox = Fluid("LOX")
    assert lox.sat_liquid(2e5).Pr > 0
    assert lox.sat_vapor(2e5).Pr > 0


def test_surface_tension():
    lox = Fluid("LOX")
    # NIST: sigma(O2, 90 K) ~ 13.2 mN/m
    assert lox.surface_tension(90.0) == pytest.approx(13.2e-3, rel=0.05)
