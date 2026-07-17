"""Coaxial-swirl injector head sizing (LOX-centered swirler + fuel annulus).

Sizes a complete injector head for a bi-propellant engine from the chamber
pressure, propellant flow rates and injection states: number of elements,
per-element liquid-oxidizer swirl injector (Bazarov ideal-injector theory),
the surrounding fuel annulus (shear-coax, plain-orifice equation), the face
layout on concentric rings, and an optional outer film-cooling ring.

Theory — ideal (inviscid) open swirl injector, principle of maximum flow
(Abramovich/Kliachko as presented by Bazarov, Yang & Puri, "Design and
Dynamics of Jet and Swirl Injectors", in *Liquid Rocket Thrust Chambers*,
AIAA Progress in Astronautics and Aeronautics vol. 200, 2004, ch. 2; same
relations in Bayvel & Orzechowski, *Liquid Atomization*, 1993):

geometric characteristic  A  = R_in * r_n / (n_t * r_t^2)
maximum-flow condition    A  = (1 - phi) * sqrt(2) / (phi * sqrt(phi))
discharge coefficient     mu = phi * sqrt(phi) / sqrt(2 - phi)
spray half-angle          tan(alpha) = 2 mu A / sqrt((1 + sqrt(1-phi))^2
                                                     - 4 mu^2 A^2)
mass flow                 mdot = mu * pi * r_n^2 * sqrt(2 rho dP)

where phi is the "coefficient of passage fullness" (1 - relative air-core
area) at the nozzle exit, R_in the swirl arm (axis to tangential-port
centerline), r_n the exit-nozzle radius, n_t / r_t the tangential-port count
and radius.

Design procedure per element (Bazarov's classical algorithm): pick the spray
half-angle -> invert the angle relation for A (and hence phi, mu); size r_n
from the element flow and the injector pressure drop; choose the tangential
port count and the swirl-arm ratio R_in/r_n; back out r_t from A. The vortex
chamber radius/length and nozzle length follow standard proportions
(R_s = R_in + r_t, L_s ~ 2 R_s, l_n ~ r_n).

Assumptions / limitations
-------------------------
* Ideal-liquid (inviscid) swirl theory: no viscous losses in the tangential
  ports or the vortex chamber; real discharge coefficients are typically a
  few percent lower, spray angles a few degrees narrower (Bazarov ch. 2
  reports ~5-10% for well-made injectors). No hydraulic-loss correction is
  applied — stated, not hidden.
* The fuel annulus uses the incompressible orifice equation with the local
  density. Regen-heated methane arrives *supercritical and gas-like*; the
  incompressible form with the real (low) density is the common first-pass
  practice, but compressibility is not modeled — the design flags fuel
  densities below ``GAS_LIKE_RHO`` kg/m^3.
* No atomization/combustion modeling: drop sizes, mixing efficiency, C*
  efficiency and combustion stability are NOT predicted. Stiffness
  (dP/Pc >= ~0.15-0.25) is the classical chug-margin heuristic (Huzel &
  Huang ch. 4; NASA SP-8089), not a stability proof.
* Element count from a thrust-per-element band plus manufacturability and
  face-packing rules — a heuristic, adjustable via ``thrust_per_element``.
* Purely hydraulic sizing at the design point; no throttling analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

G0 = 9.80665

#: Below this density [kg/m^3] the "liquid" orifice equation is suspect.
GAS_LIKE_RHO = 200.0


# ----------------------------------------------------------------------
# Ideal swirl-injector relations (Bazarov / maximum-flow principle)
# ----------------------------------------------------------------------

def A_from_phi(phi: float) -> float:
    """Geometric characteristic A for passage fullness phi (max-flow)."""
    if not 0.0 < phi < 1.0:
        raise ValueError("phi must be in (0, 1)")
    return (1.0 - phi) * np.sqrt(2.0) / (phi * np.sqrt(phi))


def phi_from_A(A: float) -> float:
    """Invert the maximum-flow condition for phi (A > 0)."""
    if A <= 0:
        raise ValueError("A must be positive")
    return brentq(lambda p: A_from_phi(p) - A, 1e-6, 1.0 - 1e-9, xtol=1e-12)


def discharge_coefficient(phi: float) -> float:
    """Ideal discharge coefficient mu(phi)."""
    return phi * np.sqrt(phi) / np.sqrt(2.0 - phi)


def spray_half_angle(phi: float) -> float:
    """Ideal spray half-angle [rad] at the nozzle exit."""
    A = A_from_phi(phi)
    mu = discharge_coefficient(phi)
    s = (1.0 + np.sqrt(1.0 - phi)) ** 2 - 4.0 * mu**2 * A**2
    if s <= 0.0:  # numerically degenerate only as phi -> 0 (alpha -> 90 deg)
        return np.pi / 2.0
    return float(np.arctan(2.0 * mu * A / np.sqrt(s)))


def phi_from_spray_angle(alpha_rad: float) -> float:
    """Invert alpha(phi) for the passage fullness (alpha in ~(5, 80) deg)."""
    lo, hi = 1e-4, 1.0 - 1e-6  # alpha(phi) is monotonically decreasing
    a_lo, a_hi = spray_half_angle(lo), spray_half_angle(hi)
    if not a_hi <= alpha_rad <= a_lo:
        raise ValueError(
            f"target spray half-angle {np.degrees(alpha_rad):.1f} deg outside "
            f"the ideal-injector range ({np.degrees(a_hi):.1f}, "
            f"{np.degrees(a_lo):.1f}) deg"
        )
    return brentq(lambda p: spray_half_angle(p) - alpha_rad, lo, hi,
                  xtol=1e-12)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class SwirlElement:
    """One LOX swirl element (ideal open swirler), all dimensions in m."""

    mdot: float                  # element liquid flow [kg/s]
    dP: float                    # design pressure drop [Pa]
    rho: float                   # liquid density at injection [kg/m^3]
    A: float                     # geometric characteristic [-]
    phi: float                   # passage fullness [-]
    Cd: float                    # ideal discharge coefficient [-]
    alpha_deg: float             # spray half-angle [deg]
    r_nozzle: float              # exit-nozzle radius r_n
    n_tangential: int            # tangential-port count
    r_tangential: float          # tangential-port radius r_t
    R_swirl_arm: float           # axis -> port centerline R_in
    r_vortex: float              # vortex-chamber radius R_s
    L_vortex: float              # vortex-chamber length
    L_nozzle: float              # nozzle length
    v_tangential: float          # port injection velocity [m/s]
    Re_tangential: float         # port Reynolds number [-]


@dataclass
class FuelAnnulus:
    """Concentric fuel gap around the swirler nozzle (per element)."""

    mdot: float
    dP: float
    rho: float
    Cd: float
    r_inner: float               # gap inner radius (over swirler wall)
    r_outer: float               # gap outer radius
    gap: float                   # radial gap width
    velocity: float              # injection velocity [m/s]
    recess: float                # swirler tip recess below fuel exit


@dataclass
class FilmCoolingRing:
    """Outer wall film-cooling orifice ring (fuel)."""

    mdot: float
    n_holes: int
    d_hole: float
    ring_radius: float
    velocity: float
    fraction_of_fuel: float


@dataclass
class InjectorDesign:
    """Complete injector head design + hydraulic summary."""

    n_elements: int
    element: SwirlElement
    annulus: FuelAnnulus
    film: FilmCoolingRing | None
    rings: list[tuple[float, int]]      # (ring radius, elements on ring)
    pitch: float                        # element center spacing
    d_element_env: float                # element envelope outer diameter
    face_radius: float                  # chamber radius at the face
    Pc: float
    stiffness_ox: float                 # dP_ox / Pc
    stiffness_fuel: float
    P_feed_ox: float                    # required manifold feed pressures
    P_feed_fuel: float
    mdot_ox: float
    mdot_fuel: float
    face_r_inner: float = 0.0           # >0: annular face (aerospike)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        e, a = self.element, self.annulus
        lines = [
            f"Coaxial-swirl injector head: {self.n_elements} elements on "
            f"{len(self.rings)} ring(s) (pitch {self.pitch*1e3:.1f} mm, "
            f"face R {self.face_radius*1e3:.1f} mm)",
            f"  LOX swirler: r_n {e.r_nozzle*1e3:.2f} mm, A {e.A:.2f}, "
            f"Cd {e.Cd:.3f}, spray half-angle {e.alpha_deg:.0f} deg, "
            f"{e.n_tangential}x d{2*e.r_tangential*1e3:.2f} mm ports "
            f"(Re {e.Re_tangential:.2e})",
            f"  fuel annulus: gap {a.gap*1e3:.2f} mm at "
            f"r {a.r_inner*1e3:.2f}-{a.r_outer*1e3:.2f} mm, "
            f"v {a.velocity:.0f} m/s, recess {a.recess*1e3:.1f} mm",
            f"  stiffness: ox {self.stiffness_ox*100:.0f}% Pc, "
            f"fuel {self.stiffness_fuel*100:.0f}% Pc; feed "
            f"{self.P_feed_ox/1e5:.1f}/{self.P_feed_fuel/1e5:.1f} bar",
        ]
        if self.film is not None:
            f = self.film
            lines.append(
                f"  film cooling: {f.n_holes}x d{f.d_hole*1e3:.2f} mm at "
                f"r {f.ring_radius*1e3:.1f} mm "
                f"({f.fraction_of_fuel*100:.0f}% of fuel)"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Face packing
# ----------------------------------------------------------------------

def _pack_face(n_elements: int, d_env: float, face_radius: float,
               wall_margin: float, pitch_factor: float,
               face_r_inner: float = 0.0
               ) -> tuple[list[tuple[float, int]], float] | None:
    """Place elements on concentric rings; None if they don't fit.

    Ring radii step by one pitch; each ring holds floor(2 pi r / pitch)
    elements; a center element is used when the count calls for it.
    ``face_r_inner > 0`` packs an ANNULAR face (aerospike/toroidal
    chamber): no center element, rings start clear of the inner wall.
    """
    pitch = pitch_factor * d_env
    r_max = face_radius - wall_margin - d_env / 2.0
    if r_max < 0:
        return None
    rings: list[tuple[float, int]] = []
    remaining = n_elements
    if face_r_inner > 0.0:
        r = face_r_inner + wall_margin + d_env / 2.0
        if r > r_max + 1e-12:
            return None
    else:
        if remaining % 2 == 1 or remaining == 1:   # center element (odd)
            rings.append((0.0, 1))
            remaining -= 1
        r = pitch
    while remaining > 0 and r <= r_max + 1e-12:
        cap = int(np.floor(2.0 * np.pi * r / pitch))
        take = min(cap, remaining)
        rings.append((r, take))
        remaining -= take
        r += pitch
    if remaining > 0:
        return None
    return rings, pitch


# ----------------------------------------------------------------------
# Main design routine
# ----------------------------------------------------------------------

def design_injector(
    Pc: float,
    thrust: float,
    mdot_ox: float,
    mdot_fuel: float,
    rho_ox: float,
    mu_ox: float,
    rho_fuel: float,
    face_radius: float,
    face_r_inner: float = 0.0,
    stiffness: float = 0.20,
    spray_half_angle_deg: float = 45.0,
    thrust_per_element: float = 1.5e3,
    n_tangential: int = 3,
    swirl_arm_ratio: float = 2.5,
    Cd_annulus: float = 0.70,
    Cd_film: float = 0.65,
    film_fraction: float = 0.10,
    tip_wall: float = 0.8e-3,
    recess_factor: float = 1.0,
    min_orifice_d: float = 0.4e-3,
    wall_margin: float = 3.0e-3,
    pitch_factor: float = 1.35,
    dp_manifold_frac: float = 0.15,
) -> InjectorDesign:
    """Size the full coaxial-swirl injector head.

    Parameters (SI unless noted): chamber pressure, sea-level thrust (element
    count heuristic only), circuit flows, injection-state densities (fuel =
    regen-jacket outlet state), ox viscosity (port Re), chamber radius at the
    injector face. ``face_r_inner`` > 0 packs an annular face between the
    inner and outer chamber walls (aerospike/toroidal chambers).
    ``stiffness`` sets dP = stiffness * Pc for both circuits.
    ``film_fraction`` of the fuel goes to the wall film ring (0 disables).
    ``dp_manifold_frac`` adds a manifold/dome loss allowance on top of the
    element drop when reporting required feed pressures.
    """
    if min(Pc, thrust, mdot_ox, mdot_fuel, rho_ox, rho_fuel) <= 0:
        raise ValueError("Pc, thrust, flows and densities must be positive")
    if not 0.0 <= film_fraction < 0.5:
        raise ValueError("film_fraction must be in [0, 0.5)")

    notes: list[str] = []
    warnings: list[str] = []
    dP = stiffness * Pc

    # --- swirler hydraulic character from the spray-angle target ---------
    phi = phi_from_spray_angle(np.radians(spray_half_angle_deg))
    A = A_from_phi(phi)
    Cd = discharge_coefficient(phi)
    notes.append(
        f"spray half-angle target {spray_half_angle_deg:.0f} deg -> "
        f"phi={phi:.3f}, A={A:.2f}, Cd={Cd:.3f} (ideal max-flow theory)"
    )

    # --- element count: thrust band, then adjust to fit the face ---------
    n_elem = max(1, int(round(thrust / thrust_per_element)))
    mdot_film = film_fraction * mdot_fuel
    mdot_fuel_core = mdot_fuel - mdot_film

    def build_element(n: int) -> tuple[SwirlElement, FuelAnnulus, float]:
        me_ox = mdot_ox / n
        r_n = np.sqrt(me_ox / (Cd * np.pi * np.sqrt(2.0 * rho_ox * dP)))
        R_in = swirl_arm_ratio * r_n
        r_t = np.sqrt(R_in * r_n / (n_tangential * A))
        R_s = R_in + r_t
        v_t = me_ox / (n_tangential * np.pi * r_t**2 * rho_ox)
        Re_t = rho_ox * v_t * 2.0 * r_t / mu_ox
        elem = SwirlElement(
            mdot=me_ox, dP=dP, rho=rho_ox, A=A, phi=phi, Cd=Cd,
            alpha_deg=spray_half_angle_deg, r_nozzle=r_n,
            n_tangential=n_tangential, r_tangential=r_t, R_swirl_arm=R_in,
            r_vortex=R_s, L_vortex=2.0 * R_s, L_nozzle=r_n,
            v_tangential=v_t, Re_tangential=Re_t,
        )
        me_f = mdot_fuel_core / n
        area_f = me_f / (Cd_annulus * np.sqrt(2.0 * rho_fuel * dP))
        r_i = r_n + tip_wall
        r_o = np.sqrt(r_i**2 + area_f / np.pi)
        ann = FuelAnnulus(
            mdot=me_f, dP=dP, rho=rho_fuel, Cd=Cd_annulus, r_inner=r_i,
            r_outer=r_o, gap=r_o - r_i,
            velocity=me_f / (rho_fuel * area_f),
            recess=recess_factor * r_n,
        )
        d_env = 2.0 * max(R_s, r_o + tip_wall)
        return elem, ann, d_env

    # shrink the element count until the face packs (deterministic loop)
    packing = None
    while n_elem >= 1:
        elem, ann, d_env = build_element(n_elem)
        packing = _pack_face(n_elem, d_env, face_radius, wall_margin,
                             pitch_factor, face_r_inner)
        if packing is not None:
            break
        n_elem -= 1
    if packing is None:  # pragma: no cover - n_elem=1 always packs or raises
        raise ValueError("injector face too small for even one element")
    if n_elem != max(1, int(round(thrust / thrust_per_element))):
        notes.append(
            f"element count reduced to {n_elem} to fit the face "
            f"(pitch rule {pitch_factor:.2f} x envelope)"
        )
    else:
        notes.append(
            f"{n_elem} elements from ~{thrust_per_element/1e3:.1f} kN per "
            "element"
        )
    rings, pitch = packing

    # --- film-cooling ring ------------------------------------------------
    film = None
    if film_fraction > 0.0:
        ring_r = face_radius - wall_margin / 2.0
        v_film = Cd_film * np.sqrt(2.0 * dP / rho_fuel)
        area_film = mdot_film / (rho_fuel * v_film)
        # choose hole count so the drill is comfortably manufacturable
        d_h = min_orifice_d
        n_h = max(4, int(round(area_film / (np.pi * d_h**2 / 4.0))))
        d_h = np.sqrt(4.0 * area_film / (np.pi * n_h))
        film = FilmCoolingRing(
            mdot=mdot_film, n_holes=n_h, d_hole=d_h, ring_radius=ring_r,
            velocity=v_film, fraction_of_fuel=film_fraction,
        )
        notes.append(
            f"film cooling: {film_fraction*100:.0f}% of fuel through "
            f"{n_h} holes on the wall ring"
        )

    # --- checks -----------------------------------------------------------
    if 2.0 * elem.r_tangential < min_orifice_d:
        warnings.append(
            f"tangential ports d={2*elem.r_tangential*1e3:.2f} mm below the "
            f"{min_orifice_d*1e3:.2f} mm manufacturability floor — fewer "
            "elements or fewer ports needed"
        )
    if ann.gap < 0.3e-3:
        warnings.append(
            f"fuel annulus gap {ann.gap*1e3:.2f} mm is very tight for "
            "printing/machining"
        )
    if rho_fuel < GAS_LIKE_RHO:
        warnings.append(
            f"fuel density {rho_fuel:.0f} kg/m^3 is gas-like (supercritical "
            "regen outlet): incompressible orifice equation is approximate"
        )
    if elem.Re_tangential < 1e4:
        warnings.append(
            f"tangential-port Re {elem.Re_tangential:.1e} < 1e4: viscous "
            "losses will erode the ideal Cd/spray angle"
        )

    P_feed = Pc + dP * (1.0 + dp_manifold_frac)
    return InjectorDesign(
        n_elements=n_elem, element=elem, annulus=ann, film=film,
        rings=rings, pitch=pitch, d_element_env=d_env,
        face_radius=face_radius, face_r_inner=face_r_inner, Pc=Pc,
        stiffness_ox=dP / Pc, stiffness_fuel=dP / Pc,
        P_feed_ox=P_feed, P_feed_fuel=P_feed,
        mdot_ox=mdot_ox, mdot_fuel=mdot_fuel,
        notes=notes, warnings=warnings,
    )
