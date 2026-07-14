"""Implicit/voxel geometry kernel -> watertight STL (PicoGK-style).

A small signed-distance-function (SDF) engine in the spirit of LEAP 71's
open-source PicoGK voxel kernel: solids are implicit functions f(x, y, z)
(negative inside, positive outside) built from primitives and boolean
operators, sampled on a regular voxel grid, and meshed with marching cubes
into a *closed* triangle surface written as binary STL. Because the field is
sampled with a padded all-outside border, the extracted surface is always
watertight — the property that makes voxel geometry attractive for
generative design of 3D-printed hardware.

Engine-specific builders at the bottom assemble the implicit model of a
regeneratively cooled thrust chamber (revolved contour shell minus helical
cooling channels, plus torus manifolds with feeder stubs and distribution
slots) and of a coaxial-swirl injector head (body of revolution minus
swirler bores, fuel annuli, tangential ports and film-cooling holes) from
the design objects produced elsewhere in this package.

Assumptions / limitations
-------------------------
* Composite fields are *implicit functions with the correct sign*, not exact
  Euclidean distances (booleans via min/max and the revolved-profile field
  break the distance property) — irrelevant for marching cubes, which only
  needs sign + continuity, but do not use field values as clearances.
* Geometric fidelity is voxel-limited: features smaller than ~2 voxels
  (thin film-cooling holes, sharp land corners) come out rounded or may
  close up. The mesh QA report states the voxel size next to every result.
* Meshes are for visualization/printing preparation; no CAD B-rep, no
  tolerancing, no support-generation. Print-readiness of overhangs etc. is
  not checked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

try:
    from skimage.measure import marching_cubes as _marching_cubes
except ImportError as _e:  # pragma: no cover
    _marching_cubes = None
    _skimage_err = _e


# ----------------------------------------------------------------------
# Implicit solids
# ----------------------------------------------------------------------

class Solid:
    """Base implicit solid: ``sdf(X, Y, Z) -> field`` (negative inside)."""

    def sdf(self, X, Y, Z):  # pragma: no cover - abstract
        raise NotImplementedError

    def __or__(self, other: "Solid") -> "Solid":       # union
        return _Union(self, other)

    def __and__(self, other: "Solid") -> "Solid":      # intersection
        return _Intersection(self, other)

    def __sub__(self, other: "Solid") -> "Solid":      # difference
        return _Difference(self, other)


class _Union(Solid):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def sdf(self, X, Y, Z):
        return np.minimum(self.a.sdf(X, Y, Z), self.b.sdf(X, Y, Z))


class _Intersection(Solid):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def sdf(self, X, Y, Z):
        return np.maximum(self.a.sdf(X, Y, Z), self.b.sdf(X, Y, Z))


class _Difference(Solid):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def sdf(self, X, Y, Z):
        return np.maximum(self.a.sdf(X, Y, Z), -self.b.sdf(X, Y, Z))


class Empty(Solid):
    """The empty solid (union identity)."""

    def sdf(self, X, Y, Z):
        return np.full(np.broadcast(X, Y, Z).shape, 1.0, dtype=np.float32)


def union(solids) -> Solid:
    out: Solid = Empty()
    for s in solids:
        out = out | s
    return out


# ----------------------------------------------------------------------
# Smooth (filleted) booleans — the organic-blend operators
# ----------------------------------------------------------------------
#
# Iñigo Quílez's polynomial smooth-min: instead of a hard crease where two
# solids meet (plain min/max), the surfaces merge over a blend radius ``k``,
# leaving a fillet. This is what gives generatively-designed hardware its
# "grown, not assembled" look. The composite field is not an exact distance
# (the primitives here aren't all unit-gradient), but the sign is correct and
# the fillet radius is ~k where the merged surfaces are near-distance — good
# for geometry/printing, not for metrology.

def _smin(a, b, k):
    """Polynomial smooth-min of two fields with blend radius k > 0."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


class SmoothUnion(Solid):
    """Union of two solids blended over radius ``k`` (fillet at the joint)."""

    def __init__(self, a, b, k):
        self.a, self.b, self.k = a, b, float(k)

    def sdf(self, X, Y, Z):
        return _smin(self.a.sdf(X, Y, Z), self.b.sdf(X, Y, Z), self.k)


class SmoothDifference(Solid):
    """``a`` minus ``b`` with a smooth (filleted) cut of radius ``k``."""

    def __init__(self, a, b, k):
        self.a, self.b, self.k = a, b, float(k)

    def sdf(self, X, Y, Z):
        da, db = self.a.sdf(X, Y, Z), self.b.sdf(X, Y, Z)
        # -smin(-da, db, k): smooth max(da, -db)
        return -_smin(-da, db, self.k)


def smooth_union(solids, k: float) -> Solid:
    """Left-fold a list of solids with :class:`SmoothUnion` (radius ``k``)."""
    solids = list(solids)
    if not solids:
        return Empty()
    out = solids[0]
    for s in solids[1:]:
        out = SmoothUnion(out, s, k)
    return out


class Sphere(Solid):
    def __init__(self, center, r):
        self.c, self.r = np.asarray(center, float), float(r)

    def sdf(self, X, Y, Z):
        c = self.c
        return np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) - self.r


class HalfSpaceX(Solid):
    """Solid half-space along x: keeps x <= x0 (``below=True``) or x >= x0.

    Handy for capping domes and for cutaway renders (intersect a part with a
    half-space through the axis to expose its internals).
    """

    def __init__(self, x0, below: bool = True):
        self.x0 = float(x0)
        self.sign = 1.0 if below else -1.0

    def sdf(self, X, Y, Z):
        return self.sign * (X - self.x0) + np.zeros_like(Y)


class CylinderX(Solid):
    """Capped cylinder along the x axis: x in [x0, x1], radius r about
    (y0, z0)."""

    def __init__(self, x0, x1, r, y0=0.0, z0=0.0):
        self.x0, self.x1, self.r = float(x0), float(x1), float(r)
        self.y0, self.z0 = float(y0), float(z0)

    def sdf(self, X, Y, Z):
        rad = np.sqrt((Y - self.y0)**2 + (Z - self.z0)**2) - self.r
        ax = np.maximum(self.x0 - X, X - self.x1)
        return np.maximum(rad, ax)


class Cylinder(Solid):
    """Capped cylinder between two arbitrary points (Quilez formulation)."""

    def __init__(self, p0, p1, r):
        self.p0 = np.asarray(p0, float)
        self.d = np.asarray(p1, float) - self.p0
        self.L2 = float(self.d @ self.d)
        self.r = float(r)

    def sdf(self, X, Y, Z):
        px, py, pz = X - self.p0[0], Y - self.p0[1], Z - self.p0[2]
        t = (px * self.d[0] + py * self.d[1] + pz * self.d[2]) / self.L2
        tc = np.clip(t, 0.0, 1.0)
        dx = px - tc * self.d[0]
        dy = py - tc * self.d[1]
        dz = pz - tc * self.d[2]
        rad = np.sqrt(dx**2 + dy**2 + dz**2) - self.r
        # inside the infinite slab this is the radial distance; the clip
        # handles the caps (approximate near edges — sign is correct)
        cap = (np.abs(t - 0.5) - 0.5) * np.sqrt(self.L2)
        return np.maximum(rad, cap)


class TorusX(Solid):
    """Torus whose centerline circles the x axis at x = xc."""

    def __init__(self, xc, R, r):
        self.xc, self.R, self.r = float(xc), float(R), float(r)

    def sdf(self, X, Y, Z):
        q = np.sqrt(Y**2 + Z**2) - self.R
        return np.sqrt(q**2 + (X - self.xc)**2) - self.r


class RevolvedAnnulus(Solid):
    """Solid of revolution about x between profiles r_in(x) and r_out(x).

    Profiles are given as sampled arrays over ``x`` and interpolated
    linearly. The field is exact in sign, approximate in distance.
    """

    def __init__(self, x, r_in, r_out):
        self.x = np.asarray(x, float)
        self.r_in = np.asarray(r_in, float)
        self.r_out = np.asarray(r_out, float)
        if not (len(self.x) == len(self.r_in) == len(self.r_out)):
            raise ValueError("profile arrays must share a length")

    def sdf(self, X, Y, Z):
        R = np.sqrt(Y**2 + Z**2)
        ri = np.interp(X, self.x, self.r_in)
        ro = np.interp(X, self.x, self.r_out)
        f = np.maximum(ri - R, R - ro)
        ax = np.maximum(self.x[0] - X, X - self.x[-1])
        return np.maximum(f, ax)


class HelicalChannels(Solid):
    """The cooling-channel void band: N rectangular channels of width w
    (arc length) and height h, lying on r in (r_base(x), r_base(x) + h),
    following a constant helix angle beta (deg) from x_start to x_end.

    The channel centerline twist obeys d(theta)/dx = tan(beta) / r_mid(x),
    integrated over the sampled profile.
    """

    def __init__(self, x, r_base, n_channels, width, height,
                 helix_angle_deg=0.0, x_start=None, x_end=None):
        self.x = np.asarray(x, float)
        self.r_base = np.asarray(r_base, float)
        self.N = int(n_channels)
        self.w, self.h = float(width), float(height)
        self.x0 = self.x[0] if x_start is None else float(x_start)
        self.x1 = self.x[-1] if x_end is None else float(x_end)
        r_mid = self.r_base + self.h / 2.0
        dtheta = np.tan(np.radians(helix_angle_deg)) / r_mid
        self.twist = np.concatenate(
            ([0.0], np.cumsum(0.5 * (dtheta[1:] + dtheta[:-1])
                              * np.diff(self.x))))

    def sdf(self, X, Y, Z):
        R = np.sqrt(Y**2 + Z**2)
        theta = np.arctan2(Z, Y)
        rb = np.interp(X, self.x, self.r_base)
        tw = np.interp(X, self.x, self.twist)
        pitch = 2.0 * np.pi / self.N
        # angular distance to the nearest channel centerline
        d = np.mod(theta - tw + pitch / 2.0, pitch) - pitch / 2.0
        arc = np.abs(d) * np.maximum(R, 1e-9) - self.w / 2.0
        rad = np.maximum(rb - R, R - (rb + self.h))
        ax = np.maximum(self.x0 - X, X - self.x1)
        return np.maximum(np.maximum(arc, rad), ax)


class RingHolesX(Solid):
    """n axial cylindrical holes (along x) equally spaced on a ring."""

    def __init__(self, x0, x1, ring_radius, n, d_hole, phase=0.0):
        self.x0, self.x1 = float(x0), float(x1)
        self.rr, self.n = float(ring_radius), int(n)
        self.r = float(d_hole) / 2.0
        self.phase = float(phase)

    def sdf(self, X, Y, Z):
        # distance in the (y, z) plane to the nearest hole center
        theta = np.arctan2(Z, Y) - self.phase
        pitch = 2.0 * np.pi / self.n
        tn = (np.round(theta / pitch)) * pitch + self.phase
        cy, cz = self.rr * np.cos(tn), self.rr * np.sin(tn)
        rad = np.sqrt((Y - cy)**2 + (Z - cz)**2) - self.r
        ax = np.maximum(self.x0 - X, X - self.x1)
        return np.maximum(rad, ax)


# ----------------------------------------------------------------------
# Meshing + STL + QA
# ----------------------------------------------------------------------

@dataclass
class TriMesh:
    """Indexed triangle mesh (m)."""

    vertices: np.ndarray   # (nv, 3) float
    faces: np.ndarray      # (nf, 3) int
    voxel_size: float

    @property
    def n_triangles(self) -> int:
        return len(self.faces)

    def volume(self) -> float:
        """Enclosed volume [m^3] via the divergence theorem."""
        v = self.vertices
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)

    def area(self) -> float:
        v = self.vertices
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def is_watertight(self) -> bool:
        """True when every undirected edge is shared by exactly 2 faces."""
        f = self.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def save_stl(self, path: str, name: bytes = b"cryosim") -> None:
        """Write binary STL (units: mm, the de-facto printing convention)."""
        v = self.vertices * 1e3
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
        nf = len(self.faces)
        rec = np.zeros(nf, dtype=[("n", "<3f4"), ("a", "<3f4"),
                                  ("b", "<3f4"), ("c", "<3f4"),
                                  ("attr", "<u2")])
        rec["n"], rec["a"], rec["b"], rec["c"] = n, a, b, c
        with open(path, "wb") as fh:
            fh.write(name.ljust(80, b"\0")[:80])
            fh.write(struct.pack("<I", nf))
            fh.write(rec.tobytes())


def mesh_solid(solid: Solid, bounds_min, bounds_max, voxel_size: float,
               slab: int = 24) -> TriMesh:
    """Sample ``solid`` on a voxel grid and extract the closed 0-surface.

    The grid is padded by one all-outside voxel layer on every side so
    marching cubes always produces a closed (watertight) mesh. The field is
    evaluated in x-slabs of ``slab`` layers to bound peak memory.
    """
    if _marching_cubes is None:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for meshing") from _skimage_err
    lo = np.asarray(bounds_min, float) - 2.0 * voxel_size
    hi = np.asarray(bounds_max, float) + 2.0 * voxel_size
    n = np.maximum(np.ceil((hi - lo) / voxel_size).astype(int) + 1, 2)
    xs = lo[0] + voxel_size * np.arange(n[0])
    ys = lo[1] + voxel_size * np.arange(n[1])
    zs = lo[2] + voxel_size * np.arange(n[2])
    field = np.empty((n[0], n[1], n[2]), dtype=np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i0 in range(0, n[0], slab):
        i1 = min(i0 + slab, n[0])
        Xs = xs[i0:i1, None, None]
        field[i0:i1] = solid.sdf(Xs, Y[None, :, :], Z[None, :, :])
    # enforce the outside border (guarantees closure even if the solid
    # touches the requested bounds)
    big = np.float32(voxel_size)
    field[[0, -1], :, :] = big
    field[:, [0, -1], :] = big
    field[:, :, [0, -1]] = big
    if field.min() >= 0.0:
        raise ValueError("solid is empty within the given bounds")
    # keep the level set off the grid nodes: an exactly-zero node makes
    # marching cubes emit duplicate vertices / pinched topology, breaking
    # the closed-mesh invariant (a 1e-6-voxel nudge is far below fidelity)
    field[field == 0.0] = np.float32(-1e-6 * voxel_size)
    verts, faces, _, _ = _marching_cubes(
        field, level=0.0, spacing=(voxel_size,) * 3)
    verts = verts.astype(np.float64) + lo
    return TriMesh(vertices=verts,
                   faces=np.ascontiguousarray(faces), voxel_size=voxel_size)


@dataclass
class MeshQA:
    """Quality/QA summary reported next to every generated part."""

    name: str
    n_triangles: int
    watertight: bool
    volume_cm3: float
    area_cm2: float
    mass_kg: float | None
    voxel_mm: float
    bbox_mm: tuple[float, float, float]

    def describe(self) -> str:
        m = ("n/a" if self.mass_kg is None else f"{self.mass_kg:.3f} kg")
        return (f"{self.name}: {self.n_triangles} tris, "
                f"{'watertight' if self.watertight else 'NOT WATERTIGHT'}, "
                f"V {self.volume_cm3:.1f} cm^3, mass {m}, "
                f"voxel {self.voxel_mm:.2f} mm, bbox "
                f"{self.bbox_mm[0]:.0f}x{self.bbox_mm[1]:.0f}"
                f"x{self.bbox_mm[2]:.0f} mm")


def concat_meshes(meshes: list[TriMesh], voxel_size: float) -> TriMesh:
    """Combine several meshes into one multi-shell mesh.

    Vertices/faces are concatenated with index offsets, so each input shell
    keeps its own independent topology. A set of individually watertight
    shells therefore yields a watertight (edge-paired) combined mesh — the
    right way to assemble parts that abut or interpenetrate, without the
    coarse re-meshing a boolean union would require. Slicers treat the
    overlapping shells as a union at print time.
    """
    if not meshes:
        raise ValueError("no meshes to concatenate")
    verts, faces, offset = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return TriMesh(vertices=np.vstack(verts),
                   faces=np.ascontiguousarray(np.vstack(faces)),
                   voxel_size=voxel_size)


def mesh_qa(mesh: TriMesh, name: str, rho: float | None = None) -> MeshQA:
    lo, hi = mesh.bounds()
    ext = (hi - lo) * 1e3
    vol = mesh.volume()
    return MeshQA(
        name=name, n_triangles=mesh.n_triangles,
        watertight=mesh.is_watertight(),
        volume_cm3=vol * 1e6, area_cm2=mesh.area() * 1e4,
        mass_kg=None if rho is None else rho * vol,
        voxel_mm=mesh.voxel_size * 1e3,
        bbox_mm=(float(ext[0]), float(ext[1]), float(ext[2])),
    )


# ----------------------------------------------------------------------
# Engine part builders (implicit models from cryosim design objects)
# ----------------------------------------------------------------------

def build_chamber_jacket(contour, channels, t_closeout: float,
                         manifolds=None, blend: float = 3.0e-3,
                         features: bool = True, n_instrument_ports: int = 4,
                         n_feet: int = 3) -> tuple[Solid, tuple, tuple]:
    """Implicit regen chamber: wall shell - channels + torus manifolds.

    ``contour``/``channels`` are :class:`~cryosim.chamber_geometry.
    ChamberContour` / ``CoolingChannels``; ``manifolds`` an optional
    :class:`~cryosim.manifold_design.ManifoldSystemDesign`. ``blend`` is the
    smooth-union fillet radius [m] where the torus manifolds and feeder stubs
    meet the chamber wall — the organic "grown" transition instead of a hard
    crease (0 for sharp booleans). With ``features`` on, a ring of
    ``n_instrument_ports`` instrumentation/igniter bosses and ``n_feet``
    mounting feet are blended on — the practical hardware detail real printed
    engines carry. Returns (solid, bounds_min, bounds_max).
    """
    x, r = contour.x, contour.r
    r_hot = r + channels.t_wall                       # channel floor
    r_out = r_hot + channels.channel_height + t_closeout
    shell = RevolvedAnnulus(x, r, r_out)
    ch = HelicalChannels(
        x, r_hot, channels.n_channels, channels.channel_width,
        channels.channel_height, channels.helix_angle_deg)
    solid = shell - ch
    r_max = float(r_out.max())
    x_lo, x_hi = float(x[0]), float(x[-1])

    if manifolds is not None:
        for m, x_c in ((manifolds.inlet, x[-1]), (manifolds.outlet, x[0])):
            duct_r = m.duct_diameter / 2.0
            R_T = m.r_centerline           # torus centerline clear of jacket
            wall = m.wall_thickness
            body = TorusX(x_c, R_T, duct_r + wall)
            cavity = TorusX(x_c, R_T, duct_r)
            # distribution slot: annular ring connecting the channel band
            # to the torus cavity over one channel height of axial extent
            slot_h = min(channels.channel_height,
                         2.0 * duct_r)
            r_band = np.interp(x_c, x, r_hot)
            slot = RevolvedAnnulus(
                np.array([x_c - slot_h / 2.0, x_c + slot_h / 2.0]),
                np.array([r_band, r_band]),
                np.array([R_T, R_T]))
            # feeder stubs: radial bores + bosses on the torus
            stub_len = 3.0 * duct_r
            feeders_body = []
            feeders_bore = []
            for k in range(m.n_feeders):
                ang = 2.0 * np.pi * k / m.n_feeders
                p0 = (x_c, R_T * np.cos(ang), R_T * np.sin(ang))
                p1 = (x_c, (R_T + stub_len) * np.cos(ang),
                      (R_T + stub_len) * np.sin(ang))
                feeders_body.append(Cylinder(p0, p1, duct_r * 0.8 + wall))
                feeders_bore.append(Cylinder(p0, p1, duct_r * 0.8))
                # flared inlet flange: a rounded boss at the stub tip
                feeders_body.append(Sphere(p1, duct_r + 1.5 * wall))
            # organic attach: fillet the feeders onto the torus and the torus
            # onto the shell (smooth-union), instead of hard-creased booleans
            manifold = smooth_union([body] + feeders_body, blend) \
                if blend > 0 else union([body] + feeders_body)
            solid = (SmoothUnion(solid, manifold, blend) if blend > 0
                     else solid | manifold)
            solid = solid - cavity - slot - union(feeders_bore)
            r_max = max(r_max, R_T + duct_r + wall + stub_len)
            x_lo = min(x_lo, x_c - duct_r - wall)
            x_hi = max(x_hi, x_c + duct_r + wall)

    if features:
        solid, r_feat, x_lo2, x_hi2 = _add_functional_features(
            solid, contour, channels, t_closeout, blend,
            n_instrument_ports, n_feet)
        r_max = max(r_max, r_feat)
        x_lo, x_hi = min(x_lo, x_lo2), max(x_hi, x_hi2)

    b = r_max
    return solid, (x_lo, -b, -b), (x_hi, b, b)


def _add_functional_features(solid, contour, channels, t_closeout, blend,
                             n_instrument_ports, n_feet):
    """Bolt-on hardware detail: instrumentation/igniter bosses + feet.

    Adds the practical features real printed engines carry — a ring of
    filleted instrumentation ports (one enlarged as an igniter) on the
    chamber wall and rounded mounting feet near the aft end — all
    smooth-unioned so they blend into the shell like generatively-designed
    hardware. Bores are drilled through the ports (subtracted). Geometry
    only; positions/sizes are representative, not stress- or flow-designed.
    """
    x, r = contour.x, contour.r
    r_out = r + channels.t_wall + channels.channel_height + t_closeout
    k = max(blend, 1e-4)

    # --- instrumentation / igniter boss ring on the cylindrical chamber ---
    x_ring = float(x[max(1, int(0.35 * contour.i_throat))])
    r_wall = float(np.interp(x_ring, x, r_out))
    boss_r = max(0.6 * channels.channel_height + t_closeout, 3.0e-3)
    bores = []
    r_feat = r_out.max()
    for i in range(n_instrument_ports):
        ang = 2.0 * np.pi * i / max(n_instrument_ports, 1)
        big = (i == 0)                       # element 0 = igniter (larger)
        rb = boss_r * (1.7 if big else 1.0)
        reach = r_wall + rb * 1.6
        cy, cz = np.cos(ang), np.sin(ang)
        p0 = (x_ring, (r_wall - boss_r) * cy, (r_wall - boss_r) * cz)
        p1 = (x_ring, reach * cy, reach * cz)
        boss = SmoothUnion(Cylinder(p0, p1, rb), Sphere(p1, rb), k)
        solid = SmoothUnion(solid, boss, k)
        bore_r = 0.5 * rb if big else 0.35 * rb
        bores.append(Cylinder((x_ring, (r_wall - 2 * boss_r) * cy,
                               (r_wall - 2 * boss_r) * cz),
                              (x_ring, (reach + boss_r) * cy,
                               (reach + boss_r) * cz), bore_r))
        r_feat = max(r_feat, reach + rb)
    solid = solid - union(bores)

    # --- mounting feet near the aft (nozzle) end --------------------------
    x_foot = float(x[min(len(x) - 1, contour.i_throat
                         + int(0.35 * (len(x) - contour.i_throat)))])
    r_wall_f = float(np.interp(x_foot, x, r_out))
    leg_r = 1.4 * boss_r
    span = 2.5 * leg_r
    feet = []
    for i in range(n_feet):
        ang = 2.0 * np.pi * (i + 0.5) / max(n_feet, 1)
        cy, cz = np.cos(ang), np.sin(ang)
        base = (x_foot, (r_wall_f - leg_r) * cy, (r_wall_f - leg_r) * cz)
        pad = (x_foot + span, (r_wall_f + span) * cy, (r_wall_f + span) * cz)
        leg = SmoothUnion(Cylinder(base, pad, leg_r), Sphere(pad, leg_r), k)
        feet.append(leg)
        r_feat = max(r_feat, r_wall_f + span + leg_r)
    if feet:
        solid = SmoothUnion(solid, smooth_union(feet, k), k)
    x_hi2 = x_foot + span + leg_r
    return solid, float(r_feat), float(x[0]), float(x_hi2)


def build_injector_head(inj, plate_thickness: float | None = None,
                        blend: float = 2.5e-3
                        ) -> tuple[Solid, tuple, tuple]:
    """Implicit coaxial-swirl injector head from an ``InjectorDesign``.

    A body of revolution sitting at x in [-H, 0] (chamber side at x = 0),
    with per element: the vortex-chamber bore, the exit-nozzle bore, the
    fuel annulus, and the tangential LOX ports; plus the film-cooling ring.
    The propellant side is closed by a smooth spherical dome (the classic
    injector/manifold dome) blended onto the barrel with fillet radius
    ``blend`` — the organic form, not a flat plate. Internal plena are
    simple cavities: a geometric model for visualization/printing studies,
    not a flow-balanced dome design.
    """
    e = inj.element
    a = inj.annulus
    t_face = plate_thickness or max(3.0e-3, 2.0 * a.gap)
    L_noz = e.L_nozzle + t_face
    H = e.L_vortex + L_noz + 2.0 * t_face      # barrel height (face to back)
    R_body = inj.face_radius + 4.0e-3

    # domed back: spherical cap of rise ~0.5 R_body, smooth-blended to the
    # barrel so the propellant side is a rounded dome rather than a flat disc
    dome_rise = 0.5 * R_body
    R_dome = (R_body**2 + dome_rise**2) / (2.0 * dome_rise)
    cx = -H + (R_dome - dome_rise)
    dome_cap = Sphere((cx, 0.0, 0.0), R_dome) & HalfSpaceX(-H, below=True)
    barrel = CylinderX(-H, 0.0, R_body)
    body = SmoothUnion(barrel, dome_cap, blend) if blend > 0 else \
        (barrel | dome_cap)
    x_back = -H - dome_rise
    cuts: list[Solid] = []
    x_vc0 = -L_noz - e.L_vortex                # vortex-chamber span
    for r_ring, n_on_ring in inj.rings:
        for k in range(n_on_ring):
            ang = 2.0 * np.pi * k / max(n_on_ring, 1)
            cy, cz = r_ring * np.cos(ang), r_ring * np.sin(ang)
            # nozzle bore to the face, vortex chamber above it
            cuts.append(CylinderX(-L_noz, 0.0, e.r_nozzle, cy, cz))
            cuts.append(CylinderX(x_vc0, -L_noz, e.r_vortex, cy, cz))
            # fuel annulus: annular gap ending at the face, fed from a
            # shallow circumferential groove
            ann_depth = a.recess + t_face
            cuts.append(CylinderX(-ann_depth, 0.0, a.r_outer, cy, cz)
                        - CylinderX(-ann_depth - 1.0, 1.0, a.r_inner, cy, cz))
            # tangential ports: horizontal drills tangent to the vortex wall
            for j in range(e.n_tangential):
                pang = ang + 2.0 * np.pi * j / e.n_tangential
                # entry point on the swirl arm, drilled tangentially
                tx = x_vc0 + e.r_vortex
                ey = cy + e.R_swirl_arm * np.cos(pang)
                ez = cz + e.R_swirl_arm * np.sin(pang)
                dy, dz = -np.sin(pang), np.cos(pang)
                L_port = 2.5 * e.r_vortex
                cuts.append(Cylinder(
                    (tx, ey - dy * L_port / 2, ez - dz * L_port / 2),
                    (tx, ey + dy * L_port / 2, ez + dz * L_port / 2),
                    e.r_tangential))
    # ox plenum hollowing the dome (leaves ~t_face dome wall)
    r_plenum = min(R_body - 3.0e-3,
                   max(rr for rr, _ in inj.rings) + e.r_vortex)
    if r_plenum > 0:
        cuts.append(CylinderX(x_back + t_face, x_vc0, r_plenum))
    if inj.film is not None:
        f = inj.film
        cuts.append(RingHolesX(-H, 0.0, f.ring_radius, f.n_holes, f.d_hole))
    solid = body - union(cuts)
    return solid, (x_back, -R_body, -R_body), (0.0, R_body, R_body)
