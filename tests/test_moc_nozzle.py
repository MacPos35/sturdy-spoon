"""Method-of-characteristics nozzle: exact Prandtl-Meyer + parallel exit."""

import numpy as np
import pytest

from cryosim.chamber_geometry import (ChamberContour,
                                      nozzle_divergence_efficiency)
from cryosim.moc_nozzle import (design_moc_nozzle, mach_from_nu,
                                prandtl_meyer)


def test_prandtl_meyer_anderson_values():
    # Anderson, Modern Compressible Flow, App. C: ν(2.0, γ=1.4) = 26.38 deg
    assert np.degrees(prandtl_meyer(2.0, 1.4)) == pytest.approx(26.38, abs=0.02)
    assert np.degrees(prandtl_meyer(3.0, 1.4)) == pytest.approx(49.76, abs=0.05)


def test_prandtl_meyer_inversion():
    for M in (1.5, 2.5, 4.0):
        nu = prandtl_meyer(M, 1.2)
        assert mach_from_nu(nu, 1.2) == pytest.approx(M, rel=1e-6)


@pytest.mark.parametrize("eps", [3.5, 8.0, 25.0])
def test_moc_exit_is_parallel_and_matches_area(eps):
    r = design_moc_nozzle(1.2, eps, r_throat=0.02, n_char=60)
    # uniform axial exit: the defining property of the MOC nozzle
    assert abs(r.exit_angle_deg) < 1e-3
    # exit Mach = isentropic Mach for the area ratio; θmax = ν(Me)/2
    assert r.theta_max_deg == pytest.approx(
        0.5 * np.degrees(prandtl_meyer(r.exit_mach, 1.2)), rel=1e-6)
    assert r.area_ratio == pytest.approx(eps, rel=1e-3)
    assert np.all(np.diff(r.r) >= -1e-12)          # monotone divergent


def test_moc_efficiency_beats_bell_and_cone():
    lam_moc = nozzle_divergence_efficiency(8.0, "moc")
    lam_bell = nozzle_divergence_efficiency(8.0, "bell")
    lam_cone = nozzle_divergence_efficiency(8.0, "conical")
    assert lam_moc == pytest.approx(1.0, abs=1e-9)   # axial exit → no loss
    assert lam_moc > lam_bell > lam_cone


def test_contour_moc_type_builds():
    c = ChamberContour(throat_radius=0.02, contraction_ratio=8.0,
                       expansion_ratio=6.0, chamber_length=0.08,
                       nozzle_type="moc", gamma=1.2, n_points=200)
    assert c.r[-1] == pytest.approx(0.02 * np.sqrt(6.0), rel=2e-3)
    assert c.theta_exit_deg == pytest.approx(0.0, abs=1e-3)
    assert c.divergence_efficiency() == pytest.approx(1.0, abs=1e-9)
    # throat is Rt; the longer MOC contour samples it a touch coarsely
    assert c.r.min() == pytest.approx(c.Rt, rel=3e-3)


def test_bad_nozzle_type_rejected():
    with pytest.raises(ValueError, match="nozzle_type"):
        ChamberContour(throat_radius=0.02, nozzle_type="aerospike")
