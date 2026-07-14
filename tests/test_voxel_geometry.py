"""Voxel kernel: analytic volumes, watertightness, STL round-trip."""

import struct

import numpy as np
import pytest

from cryosim.chamber_geometry import ChamberContour, CoolingChannels
from cryosim.voxel_geometry import (
    Cylinder,
    CylinderX,
    HalfSpaceX,
    HelicalChannels,
    RingHolesX,
    SmoothDifference,
    SmoothUnion,
    Sphere,
    TorusX,
    build_chamber_jacket,
    build_injector_head,
    concat_meshes,
    mesh_qa,
    mesh_solid,
    smooth_union,
)


def test_sphere_volume_and_watertight():
    r = 0.05
    m = mesh_solid(Sphere((0, 0, 0), r), (-r,) * 3, (r,) * 3, 0.002)
    assert m.is_watertight()
    assert m.volume() == pytest.approx(4 / 3 * np.pi * r**3, rel=0.02)


def test_torus_volume():
    R, r = 0.06, 0.015
    m = mesh_solid(TorusX(0.0, R, r), (-0.02, -0.08, -0.08),
                   (0.02, 0.08, 0.08), 0.0015)
    assert m.is_watertight()
    assert m.volume() == pytest.approx(2 * np.pi**2 * R * r**2, rel=0.02)


def test_boolean_difference_tube():
    solid = CylinderX(0, 0.1, 0.03) - CylinderX(-1, 1, 0.02)
    m = mesh_solid(solid, (0, -0.04, -0.04), (0.1, 0.04, 0.04), 0.0015)
    assert m.is_watertight()
    assert m.volume() == pytest.approx(np.pi * 0.1 * (0.03**2 - 0.02**2),
                                       rel=0.02)


def test_oblique_cylinder_volume():
    p0, p1, r = (0, 0, 0), (0.05, 0.05, 0.05), 0.01
    L = np.sqrt(3) * 0.05
    m = mesh_solid(Cylinder(p0, p1, r), (-0.02, -0.02, -0.02),
                   (0.07, 0.07, 0.07), 0.001)
    assert m.is_watertight()
    assert m.volume() == pytest.approx(np.pi * r**2 * L, rel=0.03)


def test_ring_holes_removes_material():
    body = CylinderX(0, 0.01, 0.05)
    holes = RingHolesX(-1, 1, 0.03, 8, 0.004)
    m0 = mesh_solid(body, (0, -0.06, -0.06), (0.01, 0.06, 0.06), 0.0008)
    m1 = mesh_solid(body - holes, (0, -0.06, -0.06), (0.01, 0.06, 0.06),
                    0.0008)
    removed = m0.volume() - m1.volume()
    expected = 8 * np.pi * 0.002**2 * 0.01
    assert removed == pytest.approx(expected, rel=0.10)
    assert m1.is_watertight()


def test_helical_channels_carve_expected_volume():
    x = np.linspace(0.0, 0.1, 50)
    r_base = np.full_like(x, 0.03)
    n, w, h = 20, 0.002, 0.003
    shell_in, shell_out = 0.03, 0.04
    from cryosim.voxel_geometry import RevolvedAnnulus
    shell = RevolvedAnnulus(x, np.full_like(x, shell_in),
                            np.full_like(x, shell_out))
    ch = HelicalChannels(x, r_base, n, w, h, helix_angle_deg=20.0)
    m0 = mesh_solid(shell, (0, -0.05, -0.05), (0.1, 0.05, 0.05), 0.0007)
    m1 = mesh_solid(shell - ch, (0, -0.05, -0.05), (0.1, 0.05, 0.05), 0.0007)
    # channel band volume: n * w * h * path length (1/cos(helix)) * L
    expected = n * w * h * 0.1 / np.cos(np.radians(20.0))
    removed = m0.volume() - m1.volume()
    assert removed == pytest.approx(expected, rel=0.10)
    assert m1.is_watertight()


def test_stl_roundtrip(tmp_path):
    m = mesh_solid(Sphere((0, 0, 0), 0.02), (-0.02,) * 3, (0.02,) * 3, 0.002)
    p = tmp_path / "s.stl"
    m.save_stl(str(p))
    raw = p.read_bytes()
    (nf,) = struct.unpack("<I", raw[80:84])
    assert nf == m.n_triangles
    assert len(raw) == 84 + 50 * nf
    # first facet's 3 vertices (9 floats after the normal) are in mm and
    # must lie inside the bounding sphere
    vals = struct.unpack("<9f", raw[84 + 12:84 + 48])
    assert max(abs(v) for v in vals) <= 25.0


def test_concat_meshes_watertight_and_additive():
    """Two disjoint watertight shells concatenate into one watertight,
    volume-additive multi-shell mesh."""
    a = mesh_solid(Sphere((0, 0, 0), 0.02), (-0.02,) * 3, (0.02,) * 3, 0.002)
    b = mesh_solid(Sphere((0.1, 0, 0), 0.015), (0.08, -0.02, -0.02),
                   (0.12, 0.02, 0.02), 0.002)
    both = concat_meshes([a, b], 0.002)
    assert both.is_watertight()
    assert both.n_triangles == a.n_triangles + b.n_triangles
    assert both.volume() == pytest.approx(a.volume() + b.volume(), rel=1e-9)


def test_smooth_union_adds_fillet_and_stays_watertight():
    a = Sphere((-0.015, 0, 0), 0.02)
    b = Sphere((0.015, 0, 0), 0.02)
    hard = mesh_solid(a | b, (-0.04, -0.025, -0.025), (0.04, 0.025, 0.025),
                      0.001)
    soft = mesh_solid(SmoothUnion(a, b, 0.01), (-0.04, -0.025, -0.025),
                      (0.04, 0.025, 0.025), 0.001)
    assert soft.is_watertight()
    # the fillet fills the neck, so the smooth union has more material
    assert soft.volume() > hard.volume()


def test_smooth_union_reduces_to_hard_union_as_k_shrinks():
    a = Sphere((-0.015, 0, 0), 0.02)
    b = Sphere((0.015, 0, 0), 0.02)
    hard = mesh_solid(a | b, (-0.04, -0.025, -0.025), (0.04, 0.025, 0.025),
                      0.001).volume()
    tiny = mesh_solid(SmoothUnion(a, b, 1e-5), (-0.04, -0.025, -0.025),
                      (0.04, 0.025, 0.025), 0.001).volume()
    assert tiny == pytest.approx(hard, rel=0.02)


def test_smooth_union_helper_folds_list():
    parts = [Sphere((0.02 * i, 0, 0), 0.015) for i in range(3)]
    m = mesh_solid(smooth_union(parts, 0.008), (-0.02, -0.02, -0.02),
                   (0.06, 0.02, 0.02), 0.001)
    assert m.is_watertight()


def test_smooth_difference_watertight():
    body = Sphere((0, 0, 0), 0.03)
    tool = CylinderX(-1, 1, 0.012)
    m = mesh_solid(SmoothDifference(body, tool, 0.006),
                   (-0.04, -0.04, -0.04), (0.04, 0.04, 0.04), 0.001)
    assert m.is_watertight()
    assert m.volume() > 0


def test_halfspace_cut_halves_volume():
    full = mesh_solid(Sphere((0, 0, 0), 0.03), (-0.035,) * 3, (0.035,) * 3,
                      0.0012)
    half = mesh_solid(Sphere((0, 0, 0), 0.03) & HalfSpaceX(0.0, below=True),
                      (-0.035,) * 3, (0.035,) * 3, 0.0012)
    assert half.is_watertight()
    assert half.volume() == pytest.approx(0.5 * full.volume(), rel=0.03)


def test_empty_solid_raises():
    with pytest.raises(ValueError, match="empty"):
        mesh_solid(Sphere((10, 10, 10), 0.001), (0, 0, 0),
                   (0.01, 0.01, 0.01), 0.002)


def test_mesh_qa_fields():
    m = mesh_solid(Sphere((0, 0, 0), 0.02), (-0.02,) * 3, (0.02,) * 3, 0.002)
    qa = mesh_qa(m, "test", rho=1000.0)
    assert qa.watertight
    assert qa.mass_kg == pytest.approx(m.volume() * 1000.0)
    assert "watertight" in qa.describe()


@pytest.fixture(scope="module")
def small_chamber():
    contour = ChamberContour(throat_radius=0.012, contraction_ratio=6.0,
                             expansion_ratio=3.0, chamber_length=0.05,
                             n_points=120)
    channels = CoolingChannels(n_channels=40, channel_width=1.5e-3,
                               channel_height=2.5e-3, t_wall=0.7e-3)
    return contour, channels


def test_chamber_jacket_watertight(small_chamber):
    contour, channels = small_chamber
    solid, lo, hi = build_chamber_jacket(contour, channels, 1.5e-3)
    m = mesh_solid(solid, lo, hi, 0.0006)
    assert m.is_watertight()
    # sanity: solid volume below the full annular envelope, above zero
    assert 0.0 < m.volume() < np.pi * (hi[1]) ** 2 * (hi[0] - lo[0])


def test_injector_head_watertight():
    from cryosim.injector_design import design_injector
    inj = design_injector(Pc=20e5, thrust=3e3, mdot_ox=0.9, mdot_fuel=0.3,
                          rho_ox=1140.0, mu_ox=2.0e-4, rho_fuel=400.0,
                          face_radius=0.04, film_fraction=0.0)
    solid, lo, hi = build_injector_head(inj)          # domed (blend default)
    m = mesh_solid(solid, lo, hi, 0.0005)
    assert m.is_watertight()
    assert m.volume() > 0
    # the dome extends the head behind the barrel (x below the face plane)
    assert lo[0] < -1e-3


def test_organic_bell_jacket_with_manifolds_watertight():
    from cryosim.manifold_design import design_manifolds
    contour = ChamberContour(throat_radius=0.014, contraction_ratio=6.0,
                             expansion_ratio=3.5, chamber_length=0.05,
                             nozzle_type="bell", n_points=140)
    channels = CoolingChannels(n_channels=50, channel_width=1.0e-3,
                               channel_height=2.0e-3, t_wall=0.6e-3)
    man = design_manifolds(contour, channels, mdot_coolant=0.4,
                           dp_channel=12e5, rho_in=420.0, mu_in=1.0e-4,
                           rho_out=25.0, mu_out=2.0e-5, mawp=60e5,
                           material="316L")
    solid, lo, hi = build_chamber_jacket(contour, channels, 1.5e-3, man,
                                         blend=3.0e-3)
    m = mesh_solid(solid, lo, hi, 0.0008)
    assert m.is_watertight()
    assert m.volume() > 0
