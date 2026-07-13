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


class Sphere(Solid):
    def __init__(self, center, r):
        self.c, self.r = np.asarray(center, float), float(r)

    def sdf(self, X, Y, Z):
        c = self.c
        return np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) - self.r


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
                         manifolds=None) -> tuple[Solid, tuple, tuple]:
    """Implicit regen chamber: wall shell - channels + torus manifolds.

    ``contour``/``channels`` are :class:`~cryosim.chamber_geometry.
    ChamberContour` / ``CoolingChannels``; ``manifolds`` an optional
    :class:`~cryosim.manifold_design.ManifoldSystemDesign`. Returns
    (solid, bounds_min, bounds_max).
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
            solid = (solid | body | union(feeders_body)) \
                - cavity - slot - union(feeders_bore)
            r_max = max(r_max, R_T + duct_r + wall + stub_len)
            x_lo = min(x_lo, x_c - duct_r - wall)
            x_hi = max(x_hi, x_c + duct_r + wall)

    b = r_max
    return solid, (x_lo, -b, -b), (x_hi, b, b)


def build_injector_head(inj, plate_thickness: float | None = None
                        ) -> tuple[Solid, tuple, tuple]:
    """Implicit coaxial-swirl injector head from an ``InjectorDesign``.

    A body of revolution sitting at x in [-H, 0] (chamber side at x = 0),
    with per element: the vortex-chamber bore, the exit-nozzle bore, the
    fuel annulus, and the tangential LOX ports; plus the film-cooling ring.
    Internal ox/fuel distribution domes are represented as simple plenum
    cavities — this is a geometric model for visualization/printing studies,
    not a flow-balanced dome design.
    """
    e = inj.element
    a = inj.annulus
    t_face = plate_thickness or max(3.0e-3, 2.0 * a.gap)
    L_noz = e.L_nozzle + t_face
    H = e.L_vortex + L_noz + 2.0 * t_face      # total head height
    R_body = inj.face_radius + 4.0e-3

    body = CylinderX(-H, 0.0, R_body)
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
    # simple ox plenum above the vortex chambers
    r_plenum = min(R_body - 3.0e-3,
                   max(rr for rr, _ in inj.rings) + e.r_vortex)
    if r_plenum > 0:
        cuts.append(CylinderX(-H + t_face, x_vc0, r_plenum))
    if inj.film is not None:
        f = inj.film
        cuts.append(RingHolesX(-H, 0.0, f.ring_radius, f.n_holes, f.d_hole))
    solid = body - union(cuts)
    return solid, (-H, -R_body, -R_body), (0.0, R_body, R_body)
