"""Fast unit tests for the axisymmetric Euler solver internals."""

import numpy as np
import pytest

from cryosim.cfd_nozzle import NozzleEulerCFD
from cryosim.chamber_geometry import ChamberContour
from cryosim.combustion import gas_preset


@pytest.fixture(scope="module")
def solver():
    ct = ChamberContour(0.019, 6.0, 4.5, 0.09, n_points=100)
    return NozzleEulerCFD(ct, gas_preset("lox/ch4"), 30e5,
                          n_axial=40, n_radial=10)


def test_mesh_geometry(solver):
    assert np.all(solver.area > 0)
    assert np.all(solver.rc > 0)
    # wall follows the contour (coarse axial sampling of the throat arc)
    assert solver.rn[:, -1].min() == pytest.approx(solver.contour.Rt,
                                                   rel=0.02)


def test_prim_cons_roundtrip(solver):
    rho = np.array([1.5]); u = np.array([300.0])
    v = np.array([-20.0]); p = np.array([2e6])
    U = solver._prim_to_cons(rho, u, v, p)
    r2, u2, v2, p2 = solver._cons_to_prim(U)
    assert r2[0] == pytest.approx(1.5)
    assert u2[0] == pytest.approx(300.0)
    assert v2[0] == pytest.approx(-20.0)
    assert p2[0] == pytest.approx(2e6)


def test_rusanov_consistency(solver):
    """F(U, U) must equal the exact normal flux (consistency)."""
    U = solver._prim_to_cons(np.array([2.0]), np.array([400.0]),
                             np.array([10.0]), np.array([1e6]))
    n = np.array([[0.02, 0.005]])
    F = solver._normal_flux(U, U, n)
    L = np.hypot(*n[0])
    nx, nr = n[0] / L
    rho, u, v, p = 2.0, 400.0, 10.0, 1e6
    Vn = u * nx + v * nr
    E = p / (solver.g - 1) + 0.5 * rho * (u**2 + v**2)
    exact = np.array([rho * Vn, rho * u * Vn + p * nx,
                      rho * v * Vn + p * nr, (E + p) * Vn]) * L
    assert np.allclose(F[0], exact, rtol=1e-12)


def test_initialization_is_quasi1d(solver):
    rho, u, v, p = solver._cons_to_prim(solver.U)
    # chamber cells near stagnation pressure, exit strongly expanded
    assert p[0, 0] == pytest.approx(30e5, rel=0.05)
    assert p[-1, 0] < 5e5
    assert np.all(v == 0)


def test_short_run_stays_finite(solver):
    res = solver.run(max_iter=50)
    for f in (res.rho, res.u, res.p, res.mach):
        assert np.all(np.isfinite(f))
    assert np.all(res.p > 0)
