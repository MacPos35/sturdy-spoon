"""Axisymmetric propellant tank geometry: cylinder with dome ends.

Supports flat, hemispherical, and semi-elliptical (oblate spheroid, default
2:1) domes on either end. All fill-level relations (volume, wetted area,
liquid CG, free-surface radius) are evaluated on a cached fine axial grid from
the piecewise-analytic radius profile r(z), which keeps every dome shape and
partial-fill case uniform and robust.

Conventions: z measured upward from the lowest point of the tank (bottom dome
apex), SI units. "Fill height" h is the liquid free-surface height above z=0.

Assumptions / limitations
-------------------------
* Rigid, axisymmetric tank; no internal hardware (PMDs, baffles) volumes.
* Slosh calculations use the *equivalent flat-bottom cylinder* depth
  h_eq = V_liquid / (pi R^2) per NASA SP-8009 practice; this is a good
  approximation when the free surface lies in the cylindrical section and
  degrades as the surface enters the domes (flagged by `surface_in_dome`).
"""

from __future__ import annotations

import numpy as np

DOME_TYPES = ("flat", "hemispherical", "elliptical")


def _dome_height(radius: float, dome: str, aspect: float) -> float:
    if dome == "flat":
        return 0.0
    if dome == "hemispherical":
        return radius
    if dome == "elliptical":
        return radius / aspect
    raise ValueError(f"unknown dome type {dome!r}; expected one of {DOME_TYPES}")


class TankGeometry:
    """Cylindrical tank of radius R with dome ends.

    Parameters
    ----------
    radius : tank (cylinder) radius [m]
    cyl_length : length of the cylindrical section [m]
    bottom_dome, top_dome : 'flat' | 'hemispherical' | 'elliptical'
    dome_aspect : semi-major/semi-minor axis ratio a/b for elliptical domes
        (2.0 = the common 2:1 elliptical head). Ignored for other types.
    n_grid : number of axial grid points used for the cached integrals.
    """

    def __init__(
        self,
        radius: float,
        cyl_length: float,
        bottom_dome: str = "elliptical",
        top_dome: str = "elliptical",
        dome_aspect: float = 2.0,
        n_grid: int = 4001,
    ):
        if radius <= 0 or cyl_length < 0:
            raise ValueError("radius must be > 0 and cyl_length >= 0")
        self.R = radius
        self.L_cyl = cyl_length
        self.bottom_dome = bottom_dome
        self.top_dome = top_dome
        self.dome_aspect = dome_aspect

        self.b_bottom = _dome_height(radius, bottom_dome, dome_aspect)
        self.b_top = _dome_height(radius, top_dome, dome_aspect)
        self.height = self.b_bottom + cyl_length + self.b_top
        if self.height <= 0:
            raise ValueError("tank has zero height")

        self._build_grid(n_grid)

    # ------------------------------------------------------------------ grid
    def _radius_profile(self, z: np.ndarray) -> np.ndarray:
        """Cross-section radius r(z) of the tank wall."""
        r = np.full_like(z, self.R, dtype=float)
        b_b, b_t = self.b_bottom, self.b_top
        if b_b > 0:
            zz = z[z < b_b]
            r[z < b_b] = self.R * np.sqrt(
                np.clip(1.0 - ((b_b - zz) / b_b) ** 2, 0.0, 1.0)
            )
        if b_t > 0:
            z0 = b_b + self.L_cyl
            zz = z[z > z0]
            r[z > z0] = self.R * np.sqrt(
                np.clip(1.0 - ((zz - z0) / b_t) ** 2, 0.0, 1.0)
            )
        return r

    def _build_grid(self, n: int) -> None:
        z = np.linspace(0.0, self.height, n)
        r = self._radius_profile(z)
        area = np.pi * r**2

        # Cumulative volume and first moment (for liquid CG).
        from scipy.integrate import cumulative_trapezoid

        V = cumulative_trapezoid(area, z, initial=0.0)
        Mz = cumulative_trapezoid(area * z, z, initial=0.0)

        # Lateral (wall) surface area of revolution: dA = 2 pi r ds.
        # ds from finite differences; the dome apex (r'->inf) is handled by
        # the piecewise segment lengths of the sampled profile itself.
        dz = np.diff(z)
        dr = np.diff(r)
        ds = np.sqrt(dz**2 + dr**2)
        r_mid = 0.5 * (r[1:] + r[:-1])
        A_wall = np.concatenate([[0.0], np.cumsum(2.0 * np.pi * r_mid * ds)])
        if self.bottom_dome == "flat":
            A_wall = A_wall + np.pi * self.R**2  # flat bottom plate is wetted

        self._z = z
        self._r = r
        self._V = V
        self._Mz = Mz
        self._Awall = A_wall
        self.V_total = float(V[-1])
        self.A_wall_total = float(
            A_wall[-1] + (np.pi * self.R**2 if self.top_dome == "flat" else 0.0)
        )

    # ------------------------------------------------------------- relations
    def volume_at_height(self, h: float) -> float:
        """Liquid volume [m^3] below free-surface height h [m]."""
        h = float(np.clip(h, 0.0, self.height))
        return float(np.interp(h, self._z, self._V))

    def height_at_volume(self, V: float) -> float:
        """Free-surface height h [m] for liquid volume V [m^3]."""
        V = float(np.clip(V, 0.0, self.V_total))
        return float(np.interp(V, self._V, self._z))

    def fill_fraction_to_volume(self, fill: float) -> float:
        return fill * self.V_total

    def surface_radius(self, h: float) -> float:
        """Free-surface radius [m] at fill height h."""
        h = float(np.clip(h, 0.0, self.height))
        return float(np.interp(h, self._z, self._r))

    def free_surface_area(self, h: float) -> float:
        return np.pi * self.surface_radius(h) ** 2

    def wetted_wall_area(self, h: float) -> float:
        """Tank-wall area in contact with liquid at fill height h [m^2]."""
        h = float(np.clip(h, 0.0, self.height))
        return float(np.interp(h, self._z, self._Awall))

    def dry_wall_area(self, h: float) -> float:
        """Tank-wall area in contact with ullage gas [m^2]."""
        return self.A_wall_total - self.wetted_wall_area(h)

    def liquid_cg_height(self, h: float) -> float:
        """Axial CG of the liquid volume below height h, measured from z=0."""
        h = float(np.clip(h, 0.0, self.height))
        V = np.interp(h, self._z, self._V)
        if V <= 0.0:
            return 0.0
        Mz = np.interp(h, self._z, self._Mz)
        return float(Mz / V)

    def equivalent_depth(self, V: float) -> float:
        """Equal-volume flat-bottom-cylinder depth used by the slosh analog
        (NASA SP-8009 practice)."""
        return V / (np.pi * self.R**2)

    def surface_in_dome(self, h: float) -> bool:
        """True when the free surface lies in a dome (slosh analog degraded)."""
        return h < self.b_bottom or h > self.b_bottom + self.L_cyl
