"""Automatic design of the regen-jacket torus manifolds.

Sizes the inlet (distribution) torus at the nozzle end and the outlet
(collection) torus at the injector end of a regeneratively cooled thrust
chamber for a target channel-to-channel flow uniformity, and picks the
feeder count and the angular clocking between the two manifolds' ports.

Physics — classical header/manifold flow-distribution theory:

* Along a **dividing** header the static pressure RISES away from the feed
  port (momentum recovery as the stream sheds mass) and falls with
  friction; along a **combining** header both acceleration and friction
  DEPRESS the pressure toward the take-off port. Both arcs are marched
  numerically with  dp = -2*k_m * rho * V dV - f/d * rho V^2/2 ds
  (k_m = momentum/regain coefficient, default 0.7, the mid-range of
  measured values; Bajura & Jones, "Flow Distribution Manifolds", ASME
  J. Fluids Eng. 98(4), 1976; also Shah & Sekulic, "Fundamentals of Heat
  Exchanger Design", ch. 12).
* Each channel sees driving pressure  dp_ch + dp_in(theta) - dp_out(theta);
  since mdot ~ sqrt(dp), stiff channels (large dp_ch) tolerate sloppier
  manifolds — the head ratio (manifold dynamic head / channel drop) is the
  controlling group, exactly the "area ratio" trend of the manifold
  literature.
* The duct diameter is the smallest (0.5 mm steps) meeting the requested
  maldistribution target; one feeder is tried first, then more (each
  feeder halves the arc length and the arc flow). Inlet/outlet port
  clocking is grid-searched for the minimum coupled maldistribution —
  offsetting the ports lets the dividing and combining profiles partially
  cancel (the U- vs Z-configuration effect of the header literature).
* Wall thickness by the toroidal-shell membrane formula with its inner-
  crotch stress concentration,  sigma_max = (P r / t) (2R - r)/(2(R - r))
  (Roark / pressure-vessel practice; reduces to the cylinder formula as
  R -> infinity), with the same design factor and material allowables as
  :mod:`cryosim.line_sizing`.

Context: manifold-induced maldistribution measurably shifts wall
temperatures in LOX/CH4 regen chambers (Effect of Inlet and Outlet
Manifolds on Regenerative Cooling in LOX/Methane Thrust Chambers,
J. Thermal Science 29, 2020) — a few percent of flow deficit on the
worst channel is the difference between margin and none at the throat.

Assumptions / limitations
-------------------------
* One-dimensional header theory: no port entrance/exit loss modeling
  (lumped into k_m), no swirl, no thermal gradients around the torus.
* Uniform-bleed velocity profile along each arc (first-order; adequate at
  the low head ratios the sizing itself enforces).
* The torus is treated as a smooth shell — bosses, welds, and the
  channel-slot cutouts need their own stress analysis before manufacture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chamber_geometry import ChamberContour, CoolingChannels
from .line_sizing import DESIGN_FACTOR, MATERIALS, size_line
from .regen_model import haaland_friction_factor

RHO_MATERIAL = {"316L": 7980.0, "AL6061T6": 2700.0}  # kg/m^3


@dataclass
class ManifoldSpec:
    """One torus manifold (SI units)."""

    name: str
    duct_diameter: float        # m, internal
    n_feeders: int
    r_attach: float             # m, radius where the torus meets the jacket
    r_centerline: float         # m, torus centerline radius R
    wall_thickness: float       # m
    material: str
    V_max: float                # m/s, peak header velocity
    head_ratio: float           # (rho V^2/2) / dp_channel
    mass: float                 # kg, shell estimate
    warnings: list = field(default_factory=list)


@dataclass
class ManifoldSystemDesign:
    """Coupled inlet+outlet manifold design for one jacket."""

    inlet: ManifoldSpec
    outlet: ManifoldSpec
    clocking_deg: float          # outlet ports rotated vs inlet ports
    maldistribution: float       # (max-min)/mean channel flow
    target: float
    theta_deg: np.ndarray        # channel angular positions
    flow_relative: np.ndarray    # per-channel flow / mean
    dp_channel: float
    feeder_line: object          # LineSpec for each inlet feeder branch
    outlet_line: object          # LineSpec for the collected outlet line
    met_target: bool

    def report(self) -> str:
        lines = [
            "Automatic torus-manifold design (1D header theory — see module "
            "docstring for sources/limits)",
            "=" * 72,
            f"  channel bank      : dp_channel = {self.dp_channel/1e5:.1f} bar",
            f"  flow uniformity   : (max-min)/mean = "
            f"{self.maldistribution*100:.2f}%  (target "
            f"{self.target*100:.1f}%){'' if self.met_target else '  NOT MET'}",
            f"  port clocking     : outlet ports rotated "
            f"{self.clocking_deg:.0f} deg from inlet feeders",
        ]
        for m in (self.inlet, self.outlet):
            lines += [
                f"  {m.name}:",
                f"    duct ID x wall  : {m.duct_diameter*1e3:.1f} x "
                f"{m.wall_thickness*1e3:.2f} mm {m.material}",
                f"    feeders         : {m.n_feeders}",
                f"    torus R (attach): {m.r_centerline*1e3:.1f} "
                f"({m.r_attach*1e3:.1f}) mm",
                f"    peak velocity   : {m.V_max:.1f} m/s "
                f"(head ratio {m.head_ratio*100:.1f}% of dp_ch)",
                f"    shell mass      : {m.mass*1e3:.0f} g",
            ]
            lines += [f"    ! {w}" for w in m.warnings]
        lines.append(
            f"  inlet feeder line : {self.feeder_line.od_in:.3g}\" x "
            f"{self.feeder_line.wall_in:.3f}\" per feeder"
        )
        if self.outlet_line is not None:
            lines.append(
                f"  outlet line       : {self.outlet_line.od_in:.3g}\" x "
                f"{self.outlet_line.wall_in:.3f}\""
            )
        else:
            lines.append("  outlet line       : internal discharge "
                         "(no external tube)")
        lines.append("  NOTE: smooth-shell sizing only; bosses, welds and "
                     "channel cutouts need their own stress analysis.")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Header pressure profiles
# --------------------------------------------------------------------------
def header_pressure_profile(
    theta: np.ndarray,
    n_feeders: int,
    mdot_total: float,
    rho: float,
    mu: float,
    d: float,
    r_centerline: float,
    dividing: bool,
    k_m: float = 0.7,
    n_march: int = 200,
) -> np.ndarray:
    """Static-pressure deviation [Pa] at angles ``theta`` (rad, feeders at
    multiples of 2*pi/n_feeders), relative to the arc mean.

    Marches  dp = -2 k_m rho V dV - f/d (rho V^2/2) ds  along each
    feeder-to-stagnation arc with a linear uniform-bleed velocity profile
    (V from V0 at the port to 0 at the arc end for dividing flow; the
    combining header is the mirrored problem with the sign of dV flipped
    by the marching itself).
    """
    A = np.pi / 4 * d**2
    L_arc = np.pi * r_centerline / n_feeders
    V0 = (mdot_total / (2 * n_feeders)) / (rho * A)

    # march from the port (xi=0) to the stagnation point (xi=1)
    xi = np.linspace(0.0, 1.0, n_march + 1)
    V = V0 * (1.0 - xi)
    ds = L_arc / n_march
    Vm = 0.5 * (V[1:] + V[:-1])
    dV = np.diff(V)
    Re = rho * np.maximum(Vm, 1e-6) * d / mu
    f = np.array([haaland_friction_factor(r, 1.5e-6 / d) for r in Re])
    # walking port -> stagnation: dV < 0, so the momentum term
    # (-2 k_m rho V dV) is a pressure RISE for both header types (the port
    # is where the header runs fastest). Friction drops pressure along the
    # LOCAL flow direction: away from the port in a dividing header
    # (subtract), toward the port in a combining one (add when walking
    # against the flow).
    regain = -2.0 * k_m * rho * Vm * dV
    fric = f / d * 0.5 * rho * Vm**2 * ds
    dp_seg = regain - fric if dividing else regain + fric
    p_xi = np.concatenate([[0.0], np.cumsum(dp_seg)])

    # map channel angles onto arc fraction (distance to nearest port)
    pitch = 2 * np.pi / n_feeders
    dtheta = np.abs((theta + pitch / 2) % pitch - pitch / 2)
    frac = np.clip(dtheta / (pitch / 2), 0.0, 1.0)
    dp = np.interp(frac, xi, p_xi)
    return dp - dp.mean()


def coupled_maldistribution(
    theta: np.ndarray,
    dp_in: np.ndarray,
    dp_out_fn,
    clocking: float,
    dp_channel: float,
) -> tuple[float, np.ndarray]:
    """Per-channel relative flow for a given outlet-port clocking."""
    dp_out = dp_out_fn(theta - clocking)
    drive = np.maximum(dp_channel + dp_in - dp_out, 1e-3)
    flow = np.sqrt(drive)
    flow = flow / flow.mean()
    return float((flow.max() - flow.min()) / flow.mean()), flow


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------
def _torus_wall(mawp: float, d: float, R: float, material: str) -> float:
    """Torus shell thickness with the inner-crotch factor (min 0.9 mm)."""
    S = MATERIALS[material]["S_allow"]
    r = d / 2
    factor = (2 * R - r) / (2 * (R - r))
    return max(DESIGN_FACTOR * mawp * r * factor / S, 0.9e-3)


def _shell_mass(d: float, R: float, t: float, material: str) -> float:
    return 4 * np.pi**2 * R * (d / 2) * t * RHO_MATERIAL.get(material, 8000.0)


def _size_torus(
    name: str,
    mdot: float,
    rho: float,
    mu: float,
    dp_channel: float,
    r_attach: float,
    mawp: float,
    theta: np.ndarray,
    dividing: bool,
    target_solo: float,
    material: str,
    max_feeders: int,
) -> tuple[ManifoldSpec, object]:
    """Smallest duct (0.5 mm steps) whose SOLO maldistribution contribution
    meets ``target_solo``. Returns (spec, dp_profile_function)."""
    v_cap = 30.0  # m/s header cap (erosion/acoustics; H&H-class practice)
    best = None
    for k in range(1, max_feeders + 1):
        for d in np.arange(4e-3, 60e-3, 0.5e-3):
            R = r_attach + d / 2
            V0_try = (mdot / (2 * k)) / (rho * np.pi / 4 * d**2)
            if V0_try > v_cap:
                continue
            dp = header_pressure_profile(theta, k, mdot, rho, mu, d, R,
                                         dividing)
            flow = np.sqrt(np.maximum(dp_channel + dp, 1e-3))
            flow /= flow.mean()
            solo = (flow.max() - flow.min()) / flow.mean()
            if solo <= target_solo:
                best = (k, d, R)
                break
        if best:
            break
    if best is None:
        # fall back to the largest sensible duct with max feeders
        k, d = max_feeders, 60e-3
        R = r_attach + d / 2
    else:
        k, d, R = best

    A = np.pi / 4 * d**2
    V0 = (mdot / (2 * k)) / (rho * A)
    t = _torus_wall(mawp, d, R, material)
    warnings = []
    if best is None:
        warnings.append("could not meet the solo uniformity target within "
                        "a 60 mm duct — increase feeders or channel dp")
    if d > 0.6 * r_attach:
        warnings.append(f"duct ({d*1e3:.0f} mm) is bulky vs the attach "
                        f"radius ({r_attach*1e3:.0f} mm) — check envelope")
    if V0 > v_cap:
        warnings.append(f"header velocity {V0:.0f} m/s exceeds the "
                        f"{v_cap:.0f} m/s cap (fallback design) — recheck "
                        "erosion/acoustics")

    spec = ManifoldSpec(
        name=name, duct_diameter=d, n_feeders=k, r_attach=r_attach,
        r_centerline=R, wall_thickness=t, material=material,
        V_max=V0, head_ratio=0.5 * rho * V0**2 / dp_channel,
        mass=_shell_mass(d, R, t, material), warnings=warnings,
    )

    def dp_fn(th):
        return header_pressure_profile(th, k, mdot, rho, mu, d, R, dividing)

    return spec, dp_fn


def design_manifolds(
    contour: ChamberContour,
    channels: CoolingChannels,
    mdot_coolant: float,
    dp_channel: float,
    rho_in: float,
    mu_in: float,
    rho_out: float,
    mu_out: float,
    mawp: float,
    coolant_name: str = "LCH4",
    cryogenic: bool = True,
    target: float = 0.03,
    material: str = "316L",
    max_feeders: int = 2,
    external_outlet: bool = True,
) -> ManifoldSystemDesign:
    """Design both jacket manifolds for a channel-flow uniformity target.

    ``dp_channel``: pressure drop of the channel bank alone (use the regen
    solve's dP_total); ``rho/mu_in``: coolant at the jacket inlet (cold),
    ``rho/mu_out``: at the outlet (hot) — the hot, low-density collector is
    usually the harder problem. ``external_outlet=False`` skips sizing the
    collected outlet tube (circuits that discharge internally, e.g. an
    aerospike spike feeding the injector through face ports).
    """
    N = channels.n_channels
    theta = 2 * np.pi * (np.arange(N) + 0.5) / N
    # attach radii: inlet torus at the nozzle-exit end, outlet at injector
    r_in = contour.r[-1] + channels.t_wall + channels.channel_height
    r_out = contour.r[0] + channels.t_wall + channels.channel_height

    inlet, dp_in_fn = _size_torus(
        "inlet manifold (nozzle end, dividing)", mdot_coolant, rho_in,
        mu_in, dp_channel, r_in, mawp, theta, True, target / 2,
        material, max_feeders,
    )
    outlet, dp_out_fn = _size_torus(
        "outlet manifold (injector end, combining)", mdot_coolant, rho_out,
        mu_out, dp_channel, r_out, mawp, theta, False, target / 2,
        material, max_feeders,
    )

    # clocking search: rotate the outlet ports for best cancellation
    dp_in = dp_in_fn(theta)
    best = (np.inf, 0.0, None)
    for clk in np.linspace(0, 2 * np.pi / outlet.n_feeders, 25):
        mal, flow = coupled_maldistribution(theta, dp_in, dp_out_fn, clk,
                                            dp_channel)
        if mal < best[0]:
            best = (mal, clk, flow)
    mal, clk, flow = best

    feeder = size_line(
        "manifold feeder", "pump_discharge", coolant_name,
        mdot_coolant / inlet.n_feeders, rho_in, mawp, mu=mu_in,
        material=material, cryogenic=cryogenic,
    )
    # the heated outlet stream is usually gas-like (supercritical, low
    # density): size it with a gas velocity target, not a liquid one
    outlet_service = "pressurant_gas" if rho_out < 200.0 else "pump_discharge"
    outlet_line = None
    if external_outlet:
        outlet_line = size_line(
            "jacket outlet line", outlet_service, coolant_name,
            mdot_coolant, rho_out, mawp, mu=mu_out, material=material,
            cryogenic=cryogenic,
        )

    return ManifoldSystemDesign(
        inlet=inlet, outlet=outlet, clocking_deg=float(np.degrees(clk)),
        maldistribution=mal, target=target,
        theta_deg=np.degrees(theta), flow_relative=flow,
        dp_channel=dp_channel, feeder_line=feeder, outlet_line=outlet_line,
        met_target=bool(mal <= target),
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_manifolds(design: ManifoldSystemDesign, contour: ChamberContour,
                   path: str) -> str:
    """Meridional section with both torus ducts + the flow-distribution
    plot around the circumference."""
    from . import plots

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

    # meridional cross-section
    x_cm = contour.x * 100
    for sgn in (1, -1):
        ax0.plot(x_cm, sgn * contour.r * 100, color="#222", lw=1.4)
    r_env = 0.0
    for spec, x_pos, dx in ((design.inlet, contour.x[-1], 4.0),
                            (design.outlet, contour.x[0], -1.5)):
        r_c = spec.r_centerline * 100
        r_d = spec.duct_diameter / 2 * 100
        r_env = max(r_env, r_c + r_d)
        for sgn in (1, -1):
            ax0.add_patch(Circle((x_pos * 100, sgn * r_c), r_d,
                                 facecolor="none",
                                 edgecolor=plots.C_ORANGE, lw=1.8))
        ax0.annotate(
            f"{spec.name.split('(')[0].strip()}\n"
            f"ID {spec.duct_diameter*1e3:.0f} mm, "
            f"{spec.n_feeders} feeder(s)",
            (x_pos * 100, -(r_c + r_d)),
            (x_pos * 100 + dx, -(r_c + r_d) - 2.6),
            fontsize=7, ha="center", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#777"),
        )
    ax0.set(xlabel="x [cm]", ylabel="r [cm]",
            title="torus manifolds on the jacket (meridional section)")
    ax0.set_ylim(-(r_env + 5.5), r_env + 1.5)
    ax0.set_aspect("equal")

    # per-channel flow distribution
    ax1.axhline(1.0, color="#999", lw=0.8)
    ax1.plot(design.theta_deg, design.flow_relative, color=plots.C_BLUE)
    band = design.target / 2
    ax1.axhspan(1 - band, 1 + band, color=plots.C_GREEN, alpha=0.12)
    ax1.set(xlabel="channel angular position [deg]",
            ylabel="channel flow / mean")
    ax1.set_title(f"distribution: (max−min)/mean = "
                  f"{design.maldistribution*100:.2f}% "
                  f"(target {design.target*100:.0f}%)", fontsize=9)
    ax1.set_xlim(0, 360)
    fig.suptitle("Automatic manifold design — 1D header theory "
                 "(Bajura & Jones class)", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
