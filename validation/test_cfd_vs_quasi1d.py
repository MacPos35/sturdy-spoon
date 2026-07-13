"""Validation: axisymmetric Euler CFD vs the quasi-1D nozzle solution.

The quasi-1D isentropic solution IS the reference here — for a smooth
converging-diverging nozzle without shocks the 1D theory is the textbook
exact limit (Anderson, Modern Compressible Flow, ch. 5), and the CFD's job
in this package is to quantify 2D deviations from it on the real contour.
These tests pin the solver's behavior on the shipped 5 kN contour at a
fixed coarse grid:

* conservation: axisymmetric mass flow constant along the duct (scatter),
* the mass flow and exit Mach agree with quasi-1D within the documented
  first-order-scheme error band (bias shrinks under grid refinement:
  +3.2% -> +2.5% mdot and -8.6% -> -6.2% exit Mach going from 100x20 to
  160x32 in development runs — first-order convergence as expected),
* the flow goes transonic at the geometric throat.

An inviscid solver cannot be validated for heat transfer — none is
attempted (see module docstring).
"""

import numpy as np
import pytest

from cryosim.cfd_nozzle import NozzleEulerCFD
from cryosim.chamber_geometry import ChamberContour
from cryosim.combustion import gas_preset


@pytest.fixture(scope="module")
def result():
    ct = ChamberContour(0.019, 6.0, 4.5, 0.09, n_points=150)
    solver = NozzleEulerCFD(ct, gas_preset("lox/ch4"), 30e5,
                            n_axial=100, n_radial=20)
    return solver.run(max_iter=6000)


@pytest.mark.validation
def test_mass_flow_matches_quasi1d(result):
    x, mdot = result.mdot_profile()
    m1d = result.mdot_quasi1d()
    bias = (np.mean(mdot[5:-5]) - m1d) / m1d
    assert abs(bias) < 0.06          # first-order coarse-grid band


@pytest.mark.validation
def test_mass_flow_conserved_along_duct(result):
    x, mdot = result.mdot_profile()
    scatter = np.std(mdot[5:-5]) / result.mdot_quasi1d()
    assert scatter < 0.04


@pytest.mark.validation
def test_exit_mach_vs_quasi1d(result):
    mc = result.centerline_mach()[1]
    m1d = result.quasi1d_mach()
    assert mc[-1] == pytest.approx(m1d[-1], rel=0.12)
    assert mc[-1] > 2.0              # clearly supersonic exit


@pytest.mark.validation
def test_transonic_at_geometric_throat(result):
    xc, mc = result.centerline_mach()
    i_t = np.argmin(np.abs(xc - result.contour.x_throat))
    assert 0.75 < mc[i_t] < 1.2
    # supersonic shortly downstream
    assert mc[min(i_t + 8, len(mc) - 1)] > 1.0


@pytest.mark.validation
def test_mach_monotone_through_nozzle(result):
    xc, mc = result.centerline_mach()
    conv = xc > 0.09  # from the convergent section onward
    dm = np.diff(mc[conv])
    # allow small numerical wiggles, no real reversals
    assert np.all(dm > -0.02)
