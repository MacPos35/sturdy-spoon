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


class ChamberContour:
    """Axisymmetric inner-wall contour r(x); x measured from injector face.

    Parameters
    ----------
    throat_radius : Rt [m]
    contraction_ratio : Ac/At (chamber-to-throat area ratio)
    expansion_ratio : Ae/At
    chamber_length : length of the cylindrical chamber section [m]
    conv_angle_deg : convergent cone half-angle [deg]
    div_angle_deg : divergent cone half-angle [deg]
    r_conv_factor, r_div_factor : throat arc radii / Rt (upstream, downstream)
    n_points : total sampling points along the contour
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
    ):
        if contraction_ratio <= 1.0 or expansion_ratio < 1.0:
            raise ValueError("contraction_ratio must be > 1 and expansion_ratio >= 1")
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

        # Downstream arc to divergence angle alpha, then cone to exit radius.
        r_td = Rt + rho_d * (1.0 - np.cos(alpha))
        L_arc_d = rho_d * np.sin(alpha)
        L_cone_d = max(Re - r_td, 0.0) / np.tan(alpha)

        x_throat = chamber_length + L_cone_c + L_arc_u
        x_exit = x_throat + L_arc_d + L_cone_d

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
                elif xi <= x_throat + L_arc_d:
                    dx = xi - x_throat
                    r[i] = Rt + rho_d - np.sqrt(max(rho_d**2 - dx**2, 0.0))
                else:
                    r[i] = r_td + (xi - (x_throat + L_arc_d)) * np.tan(alpha)
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
