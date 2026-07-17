"""Thrust-chamber contour and cooling-channel geometry.

Builds a conventional axisymmetric chamber/throat/nozzle wall contour from a
few design parameters (cylindrical chamber, conical convergent section blended
into circular throat arcs, conical divergent section), and describes the
regenerative cooling channels milled into the wall (rectangular cross-section,
constant width and height, straight or helical).

The contour is the classic layout used in student/early-design tools
(cf. Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant Rocket
Engines", ch. 4): upstream throat arc radius 1.5*Rt and downstream 0.382*Rt by
default, 15 deg conical divergent section.

Assumptions / limitations
-------------------------
* Conical (not Rao bell) divergent section: for regen heat-load purposes the
  difference is small; the tool targets heat transfer, not nozzle performance.
* Channels: constant width/height along the contour, so the land width varies
  with local radius; the constructor rejects geometries whose land closes up
  at the throat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _bell_angles(expansion_ratio: float, bell_percent: float
                 ) -> tuple[float, float]:
    """Parabola start/exit wall angles (theta_n, theta_e) in radians.

    Rao's thrust-optimized-parabola method reads these two angles off a chart
    keyed to expansion ratio and percent-length. The exact chart is not
    reproducible offline, so we use a smooth documented fit anchored to its
    well-known 80%-bell values: theta_n rises and theta_e falls with area
    ratio (a longer, gentler exit as the nozzle grows), and a shorter bell
    (smaller percent) trades a steeper exit angle for length. Good enough to
    generate a representative bell contour and its divergence loss; not a
    substitute for a method-of-characteristics design.
    """
    ln_eps = np.log(max(expansion_ratio, 1.5))
    theta_n = np.radians(np.clip(22.0 + 4.0 * ln_eps, 20.0, 40.0))
    theta_e = np.radians(np.clip(14.0 - 2.6 * ln_eps, 3.0, 18.0))
    # shorter-than-80% bells exit steeper (more divergence loss)
    theta_e = theta_e + np.radians(14.0 * (0.8 - bell_percent))
    return float(theta_n), float(max(theta_e, np.radians(2.0)))


def nozzle_divergence_efficiency(expansion_ratio: float, nozzle_type: str,
                                 bell_percent: float = 0.8,
                                 div_angle_deg: float = 15.0) -> float:
    """Divergence (angularity) efficiency lambda without building a contour.

    Lets the sizing loop credit the bell's reduced exit angle before the
    throat radius is known (lambda depends only on the exit geometry, not on
    scale). Same ``0.5(1 + cos theta_exit)`` model as
    :meth:`ChamberContour.divergence_efficiency`.
    """
    if nozzle_type in ("moc", "aerospike"):
        theta_e = 0.0       # uniform axial exit → λ = 1 (the aerospike's
        # truncation loss is carried separately in Cf, not in λ)
    elif nozzle_type == "bell":
        _, theta_e = _bell_angles(expansion_ratio, bell_percent)
    else:
        theta_e = np.radians(div_angle_deg)
    return 0.5 * (1.0 + np.cos(theta_e))


def _bell_divergent(Rt: float, Re: float, rho_d: float,
                    expansion_ratio: float, bell_percent: float,
                    x_throat: float):
    """Build the (x, r) polyline of a Rao-style bell divergent section.

    Downstream throat arc of radius ``rho_d`` swept to the parabola-start
    angle ``theta_n``, then a quadratic Bezier (the classic parabola
    approximation) to the exit lip at radius ``Re`` and wall angle
    ``theta_e``. Returns (xb, rb, x_exit, theta_exit_rad).
    """
    theta_n, theta_e = _bell_angles(expansion_ratio, bell_percent)
    # reference 15-deg conical length from throat to exit (Rao normalization)
    a15 = np.radians(15.0)
    r_td15 = Rt + rho_d * (1.0 - np.cos(a15))
    Ln_ref = (rho_d * np.sin(a15)
              + max(Re - r_td15, 0.0) / np.tan(a15))
    Ln = bell_percent * Ln_ref

    # downstream throat arc, throat -> N (center at (x_throat, Rt + rho_d))
    phi = np.linspace(0.0, theta_n, 30)
    xa = x_throat + rho_d * np.sin(phi)
    ra = Rt + rho_d * (1.0 - np.cos(phi))
    Nx, Nr = xa[-1], ra[-1]
    Ex, Er = x_throat + Ln, Re

    # control point Q = intersection of the two tangent lines
    #   N + t (cos tn, sin tn) = E - s (cos te, sin te)
    A = np.array([[np.cos(theta_n), np.cos(theta_e)],
                  [np.sin(theta_n), np.sin(theta_e)]])
    rhs = np.array([Ex - Nx, Er - Nr])
    try:
        t, _s = np.linalg.solve(A, rhs)
        Qx, Qr = Nx + t * np.cos(theta_n), Nr + t * np.sin(theta_n)
    except np.linalg.LinAlgError:  # parallel tangents (degenerate) -> straight
        Qx, Qr = 0.5 * (Nx + Ex), 0.5 * (Nr + Er)

    u = np.linspace(0.0, 1.0, 140)
    xbz = (1 - u) ** 2 * Nx + 2 * (1 - u) * u * Qx + u ** 2 * Ex
    rbz = (1 - u) ** 2 * Nr + 2 * (1 - u) * u * Qr + u ** 2 * Er
    xb = np.concatenate([xa, xbz[1:]])
    rb = np.concatenate([ra, rbz[1:]])
    # guarantee monotonic x for interpolation (a sane bell already is)
    keep = np.concatenate([[True], np.diff(xb) > 0])
    return xb[keep], rb[keep], float(Ex), theta_e


class ChamberContour:
    """Axisymmetric inner-wall contour r(x); x measured from injector face.

    Parameters
    ----------
    throat_radius : Rt [m]
    contraction_ratio : Ac/At (chamber-to-throat area ratio)
    expansion_ratio : Ae/At
    chamber_length : length of the cylindrical chamber section [m]
    conv_angle_deg : convergent cone half-angle [deg]
    div_angle_deg : divergent cone half-angle [deg] (conical nozzle only)
    r_conv_factor, r_div_factor : throat arc radii / Rt (upstream, downstream)
    n_points : total sampling points along the contour
    nozzle_type : "conical" (straight divergent) or "bell" (thrust-optimized
        parabolic-approximation / Rao TOP contour)
    bell_percent : bell length as a fraction of the reference 15-deg cone
        (0.8 = the near-universal "80% bell"; bell nozzles only)
    """

    def __init__(
        self,
        throat_radius: float,
        contraction_ratio: float = 8.0,
        expansion_ratio: float = 4.0,
        chamber_length: float = 0.15,
        conv_angle_deg: float = 30.0,
        div_angle_deg: float = 15.0,
        r_conv_factor: float = 1.5,
        r_div_factor: float = 0.382,
        n_points: int = 200,
        nozzle_type: str = "conical",
        bell_percent: float = 0.8,
        gamma: float = 1.2,
    ):
        if contraction_ratio <= 1.0 or expansion_ratio < 1.0:
            raise ValueError("contraction_ratio must be > 1 and expansion_ratio >= 1")
        if nozzle_type not in ("conical", "bell", "moc"):
            raise ValueError("nozzle_type must be 'conical', 'bell' or 'moc'")
        Rt = throat_radius
        Rc = Rt * np.sqrt(contraction_ratio)
        Re = Rt * np.sqrt(expansion_ratio)
        beta = np.radians(conv_angle_deg)
        alpha = np.radians(div_angle_deg)
        rho_u = r_conv_factor * Rt   # upstream throat arc radius
        rho_d = r_div_factor * Rt    # downstream throat arc radius

        # Upstream arc: from tangency with the convergent cone (angle beta)
        # down to the throat. r at cone tangency:
        r_tu = Rt + rho_u * (1.0 - np.cos(beta))
        if r_tu > Rc:
            raise ValueError("convergent arc taller than chamber; reduce conv angle "
                             "or increase contraction ratio")
        # Convergent cone axial run from chamber wall to arc tangency:
        L_cone_c = (Rc - r_tu) / np.tan(beta)
        L_arc_u = rho_u * np.sin(beta)

        x_throat = chamber_length + L_cone_c + L_arc_u

        # Divergent section: conical (straight) or a thrust-optimized bell.
        if nozzle_type == "conical":
            r_td = Rt + rho_d * (1.0 - np.cos(alpha))
            L_arc_d = rho_d * np.sin(alpha)
            L_cone_d = max(Re - r_td, 0.0) / np.tan(alpha)
            x_exit = x_throat + L_arc_d + L_cone_d
            theta_exit = alpha

            def div_r(xi: float) -> float:
                if xi <= x_throat + L_arc_d:
                    dx = xi - x_throat
                    return Rt + rho_d - np.sqrt(max(rho_d**2 - dx**2, 0.0))
                return r_td + (xi - (x_throat + L_arc_d)) * np.tan(alpha)
        elif nozzle_type == "bell":
            xb, rb, x_exit, theta_exit = _bell_divergent(
                Rt, Re, rho_d, expansion_ratio, bell_percent, x_throat)

            def div_r(xi: float) -> float:
                return float(np.interp(xi, xb, rb))
        else:  # method-of-characteristics (uniform axial exit)
            from .moc_nozzle import design_moc_nozzle
            # short throat arc to ~half the MOC start angle, then the MOC wall
            moc = design_moc_nozzle(gamma, expansion_ratio, Rt,
                                    x_throat=x_throat)
            xb = np.concatenate([[x_throat], moc.x])
            rb = np.concatenate([[Rt], moc.r])
            x_exit = float(moc.x[-1])
            theta_exit = np.radians(moc.exit_angle_deg)

            def div_r(xi: float) -> float:
                return float(np.interp(xi, xb, rb))

        def r_of_x(x: np.ndarray) -> np.ndarray:
            r = np.empty_like(x)
            for i, xi in enumerate(x):
                if xi <= chamber_length:
                    r[i] = Rc
                elif xi <= chamber_length + L_cone_c:
                    r[i] = Rc - (xi - chamber_length) * np.tan(beta)
                elif xi <= x_throat:
                    # upstream arc, center at (x_throat, Rt + rho_u)
                    dx = x_throat - xi
                    r[i] = Rt + rho_u - np.sqrt(max(rho_u**2 - dx**2, 0.0))
                else:
                    r[i] = div_r(xi)
            return r

        self.x = np.linspace(0.0, x_exit, n_points)
        self.r = r_of_x(self.x)
        self.area = np.pi * self.r**2

        self.Rt = Rt
        self.Dt = 2.0 * Rt
        self.At = np.pi * Rt**2
        self.Rc_chamber = Rc
        self.x_throat = x_throat
        self.i_throat = int(np.argmin(np.abs(self.x - x_throat)))
        self.contraction_ratio = contraction_ratio
        self.expansion_ratio = expansion_ratio
        # Bartz's correlation uses the throat radius of curvature; with two
        # different arcs the mean is common practice (e.g. Huzel & Huang).
        self.r_curv_throat = 0.5 * (rho_u + rho_d)
        self.length = x_exit
        self.nozzle_type = nozzle_type
        self.bell_percent = bell_percent if nozzle_type == "bell" else 1.0
        #: wall angle at the nozzle exit lip [deg] — the divergence-loss driver
        self.theta_exit_deg = float(np.degrees(theta_exit))

    def divergence_efficiency(self) -> float:
        """Nozzle divergence (angularity) efficiency lambda in (0, 1].

        The axial-thrust fraction of the exit momentum for a radially
        diverging supersonic exhaust, ``lambda = 0.5 (1 + cos theta)`` with
        ``theta`` the exit-lip wall half-angle (Sutton & Biblarz eq. 3-34;
        Huzel & Huang eq. 4-7). A 15-deg cone gives 0.983; a bell's small
        exit angle recovers most of that ~1.7% loss — which is exactly why
        bells are used. This is the standard preliminary-design correction;
        it is NOT a method-of-characteristics contour optimization.
        """
        return 0.5 * (1.0 + np.cos(np.radians(self.theta_exit_deg)))

    def area_ratio(self) -> np.ndarray:
        """Local A/At along the contour."""
        return self.area / self.At

    def is_supersonic(self) -> np.ndarray:
        """Station-wise flag: False upstream of the throat, True downstream."""
        return self.x > self.x_throat


@dataclass
class CoolingChannels:
    """Rectangular regen cooling channels milled into the chamber wall.

    Parameters (SI): number of channels, channel width w (circumferential),
    channel height b (radial), inner-wall thickness t_wall (hot wall between
    gas and coolant), land width is *derived* from the local radius, wall
    conductivity k_wall, optional helix angle (deg, 0 = axial channels) and
    absolute roughness for the friction factor.
    """

    n_channels: int
    channel_width: float
    channel_height: float
    t_wall: float
    k_wall: float = 350.0            # W/(m K), default: copper alloy
    helix_angle_deg: float = 0.0
    roughness: float = 3.0e-6        # m, machined copper
    flow_area: float = field(init=False)
    D_h: float = field(init=False)

    def __post_init__(self):
        w, b = self.channel_width, self.channel_height
        if min(w, b, self.t_wall) <= 0 or self.n_channels < 1:
            raise ValueError("channel dimensions and count must be positive")
        self.flow_area = w * b
        self.D_h = 2.0 * w * b / (w + b)

    def land_width(self, r_wall: float) -> float:
        """Land (rib) width at coolant-side radius r_wall + t_wall."""
        r_cool = r_wall + self.t_wall
        land = 2.0 * np.pi * r_cool / self.n_channels - self.channel_width
        return land

    def validate_against(self, contour: ChamberContour) -> None:
        """Reject channel layouts whose lands vanish at the throat."""
        land_min = self.land_width(contour.Rt)
        if land_min <= 0:
            raise ValueError(
                f"channels overlap at the throat (land = {land_min:.2e} m); "
                "reduce n_channels or channel_width"
            )

    @property
    def path_factor(self) -> float:
        """Coolant path length per unit axial length (1/cos helix)."""
        return 1.0 / np.cos(np.radians(self.helix_angle_deg))
