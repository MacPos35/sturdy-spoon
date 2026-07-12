"""Validation of the regen-cooling model against published references.

Three layers (each labeled for exactly what it establishes):

1. **Closure-level, exact**: the Haaland friction factor is validated against
   the implicit Colebrook equation solved independently here (Haaland is its
   published explicit approximation, error < 2%); the Bartz coefficient is
   validated against an independent hand-assembled evaluation of Bartz's
   1957 equation.

2. **System-level band validation vs the HYPROB program** (CIRA 30 kN
   LOX/LCH4 demonstrator, regeneratively cooled with methane through 96
   axial channels, Pc = 55 bar class; see Ricci et al. and the ODREC paper,
   Kose & Celik, Appl. Sci. 14(1):71 (2024), which validated the same 1D
   methodology against the HYPROB MTP hot-fire data). The full tabulated
   HYPROB test data was NOT accessible from the development environment, so
   this test asserts the model lands in the operating bands documented
   across the HYPROB literature: throat heat flux of tens of MW/m^2, copper
   hot-wall peak at the throat below ~1000 K, methane leaving the jacket
   transcritical at ~350-550 K, jacket pressure drop of order tens of bar.
   This is a BAND validation, not a point validation — stated honestly.

3. **Known bias, documented**: Bartz over-predicts heat flux for LOX/CH4
   thrust chambers by ~20-30% (ODREC; JAXA subscale tests, EUCASS 2017-381).
   The tool is therefore conservative for cooling design. No correction
   factor is silently applied.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.combustion import gas_preset
from cryosim.fluids import Fluid
from cryosim.regen_model import (
    RegenCoolingModel,
    bartz_h_g,
    haaland_friction_factor,
)


@pytest.mark.validation
def test_haaland_vs_colebrook():
    """Haaland must track the implicit Colebrook-White equation within 2%."""

    def colebrook(Re, rr):
        def eq(f):
            return 1 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))
        return brentq(eq, 1e-4, 0.2)

    for Re in (5e3, 1e4, 1e5, 1e6, 1e7):
        for rr in (0.0, 1e-5, 1e-4, 1e-3):
            fH = haaland_friction_factor(Re, rr)
            fC = colebrook(Re, rr)
            assert fH == pytest.approx(fC, rel=0.02), (Re, rr)


@pytest.mark.validation
def test_bartz_independent_hand_evaluation():
    """Assemble Bartz's 1957 equation independently and compare."""
    gas = gas_preset("lox/ch4")
    Pc, Dt, rc, ar, M, Twg = 55e5, 0.068, 0.032, 1.0, 1.0, 800.0
    g = gas.gamma
    # sigma per Bartz eq. (independent expression)
    a = 0.5 * Twg / gas.T_c * (1 + (g - 1) / 2 * M**2) + 0.5
    sigma = a ** -0.68 * (1 + (g - 1) / 2 * M**2) ** -0.12
    h_ref = (
        (0.026 / Dt**0.2)
        * (gas.mu**0.2 * gas.cp / gas.Pr**0.6)
        * (Pc / gas.c_star) ** 0.8
        * (Dt / rc) ** 0.1
        * ar ** -0.9
        * sigma
    )
    assert bartz_h_g(gas, Pc, Dt, rc, ar, M, Twg) == pytest.approx(h_ref, rel=1e-12)
    # order of magnitude for a 55 bar LOX/CH4 throat: tens of kW/(m^2 K)
    assert 1e4 < h_ref < 1e5


@pytest.fixture(scope="module")
def hyprob_like():
    """HYPROB-class 30 kN LOX/LCH4 chamber, 96 axial channels, Pc 55 bar.

    Channel dimensions are representative (exact HYPROB jacket dimensions
    are not public in full); results are asserted against documented bands.
    """
    gas = gas_preset("lox/ch4")
    Pc = 55e5
    ct = ChamberContour(0.034, contraction_ratio=8.5, expansion_ratio=4.0,
                        chamber_length=0.20, n_points=150)
    ch = CoolingChannels(n_channels=96, channel_width=1.3e-3,
                         channel_height=2.5e-3, t_wall=0.9e-3, k_wall=330.0)
    model = RegenCoolingModel(ct, ch, gas, Fluid("Methane"))
    mdot_f = Pc * ct.At / gas.c_star / 4.4  # full fuel flow at MR 3.4
    res = model.solve(Pc, mdot_f, T_inlet=110.0, P_inlet=160e5)
    return ct, res


@pytest.mark.validation
def test_hyprob_band_throat_heat_flux(hyprob_like):
    ct, res = hyprob_like
    q_throat = res.q[ct.i_throat]
    assert 30e6 < q_throat < 80e6  # tens of MW/m^2, documented HYPROB class


@pytest.mark.validation
def test_hyprob_band_wall_temperature(hyprob_like):
    ct, res = hyprob_like
    assert 700.0 < res.peak_wall_temperature < 1000.0  # copper liner class
    assert abs(res.x[np.argmax(res.T_wg)] - ct.x_throat) < 0.02


@pytest.mark.validation
def test_hyprob_band_coolant_outlet(hyprob_like):
    ct, res = hyprob_like
    ch4_Tcrit, ch4_Pcrit = 190.56, 4.599e6
    out = res.coolant_outlet
    assert 350.0 < out.T < 550.0          # documented outlet class
    assert out.P > ch4_Pcrit and out.T > ch4_Tcrit  # leaves transcritical
    assert 5e5 < res.dP_total < 50e5      # tens of bar jacket drop


@pytest.mark.validation
def test_hyprob_energy_closure(hyprob_like):
    ct, res = hyprob_like
    # total picked-up heat ~ MW class for a 30 kN regen chamber
    assert 1e6 < res.Q_total < 6e6
