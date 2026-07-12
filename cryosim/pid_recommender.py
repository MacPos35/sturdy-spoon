"""Rule-based feed-system P&ID recommender.

Given the propulsion-system configuration selected in the simulator
(pressurization scheme, propellant(s) and their cryogenic/storable nature,
which side feeds the regen jacket), this module recommends a feed-system
component list — each with a one-line engineering rationale — and draws a
simplified schematic (SVG/PNG via matplotlib).

The rules encode common student-team bi-propellant practice (DARE/EuRoC
class vehicles): redundant overpressure protection on every tank (relief
valve + burst disc), remote vent for cryogens, inert-gas purge, filters
upstream of fast valves, instrumentation at every state-changing interface.

IMPORTANT: this is a design *checklist aid* to start a P&ID discussion, not
a safety-reviewed P&ID. Component sizing, redundancy philosophy, and range
selection must go through your team's own safety/review process (and, for
EuRoC, the DEWG/ESRA design guidelines).
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRESSURIZATION_SCHEMES = ("regulated", "blowdown", "autogenous", "pump")


@dataclass
class TankSpec:
    """One propellant tank in the feed system."""

    name: str                 # e.g. "LCH4", "LOX"
    cryogenic: bool = True
    is_coolant: bool = False  # feeds the regen jacket
    oxidizer: bool = False


@dataclass
class FeedSystemConfig:
    pressurization: str = "regulated"
    tanks: list[TankSpec] = field(default_factory=lambda: [
        TankSpec("LOX", cryogenic=True, oxidizer=True),
        TankSpec("LCH4", cryogenic=True, is_coolant=True),
    ])
    regen_cooled: bool = True

    def __post_init__(self):
        if self.pressurization not in PRESSURIZATION_SCHEMES:
            raise ValueError(
                f"pressurization must be one of {PRESSURIZATION_SCHEMES}"
            )


@dataclass
class Component:
    tag: str
    name: str
    subsystem: str      # "pressurant" | tank name | "engine"
    rationale: str


@dataclass
class FeedSystemRecommendation:
    config: FeedSystemConfig
    components: list[Component]
    warnings: list[str]

    def by_subsystem(self) -> dict[str, list[Component]]:
        out: dict[str, list[Component]] = {}
        for c in self.components:
            out.setdefault(c.subsystem, []).append(c)
        return out

    def as_text(self) -> str:
        lines = [
            f"Feed system recommendation — pressurization: "
            f"{self.config.pressurization}",
            "=" * 66,
        ]
        for sub, comps in self.by_subsystem().items():
            lines.append(f"\n[{sub}]")
            for c in comps:
                lines.append(f"  {c.tag:<8} {c.name}")
                lines.append(f"           - {c.rationale}")
        if self.warnings:
            lines.append("\nWarnings:")
            lines += [f"  ! {w}" for w in self.warnings]
        lines.append(
            "\nNote: checklist aid only — NOT a safety-reviewed P&ID."
        )
        return "\n".join(lines)


def recommend_feed_system(cfg: FeedSystemConfig) -> FeedSystemRecommendation:
    """Apply the rule set to the configuration."""
    comps: list[Component] = []
    warnings: list[str] = []
    n = [0]

    def add(prefix, name, subsystem, rationale):
        n[0] += 1
        comps.append(Component(f"{prefix}-{n[0]:03d}", name, subsystem, rationale))

    # ---------------------------------------------------------- pressurant
    if cfg.pressurization == "regulated":
        add("BT", "High-pressure pressurant bottle (N2/He)", "pressurant",
            "stores pressurant at high density; He for cryogens (N2 dissolves/"
            "condenses in LOX/LCH4 at low temperature)")
        add("HV", "Pressurant isolation valve (remote)", "pressurant",
            "isolates the HP bottle until tank pressurization is commanded")
        add("PR", "Dome/spring pressure regulator", "pressurant",
            "holds tank pressure setpoint through the burn (constant feed "
            "pressure -> constant mdot)")
        add("RV", "Regulator-outlet relief valve", "pressurant",
            "protects the LP side against regulator failure (creep/burst)")
        add("CV", "Check valve per tank branch", "pressurant",
            "prevents propellant vapor migration/mixing back into the "
            "common pressurant manifold")
    elif cfg.pressurization == "blowdown":
        add("QD", "Pressurant charge quick-disconnect", "pressurant",
            "one-time ullage charge before flight; no in-flight regulation")
        warnings.append(
            "blowdown: feed pressure and thrust decay through the burn — "
            "verify the regen coolant pressure stays above the required "
            "injector/chamber margin at end of burn"
        )
    elif cfg.pressurization == "autogenous":
        add("HX", "Autogenous pressurization tap/heat exchanger", "pressurant",
            "taps heated coolant vapor (e.g. from the regen outlet) to "
            "self-pressurize its own tank — single species preserved")
        add("FCV", "Pressurization flow-control valve", "pressurant",
            "throttles vapor flow to hold the tank setpoint")
        add("CV", "Check valve on pressurization line", "pressurant",
            "prevents liquid backflow into the vapor line at engine shutdown")
        warnings.append(
            "autogenous pressurization couples engine state to tank pressure "
            "— verify start-up transient with a separate ground pressurant"
        )
    elif cfg.pressurization == "pump":
        add("PU", "Electric feed pump (coolant side)", "engine",
            "raises channel pressure above the propellant critical pressure "
            "(supercritical cooling, avoids two-phase flow in the jacket)")
        add("QD", "Low-pressure ullage pad charge port", "pressurant",
            "pump-fed tanks still need a small ullage pad (1-4 bar) for pump "
            "NPSH and structural stability")
        warnings.append(
            "pump-fed: verify pump NPSH against tank pressure minus vapor "
            "pressure at the warmest expected bulk temperature"
        )

    # -------------------------------------------------------------- per tank
    for tk in cfg.tanks:
        sub = tk.name
        add("FDV", f"{tk.name} fill/drain valve"
            + (" (cryo-rated)" if tk.cryogenic else ""), sub,
            "ground fill and safe de-tanking path; cryo service needs "
            "cold-rated seats/seals" if tk.cryogenic else
            "ground fill and safe de-tanking path")
        if tk.cryogenic:
            add("VV", f"{tk.name} remote vent valve", sub,
                "boil-off management during hold; venting is the primary "
                "abort/safe action for a cryogenic tank")
        add("RV", f"{tk.name} relief valve", sub,
            "primary overpressure protection sized for the worst-case "
            "heating (self-pressurization) rate")
        add("BD", f"{tk.name} burst disc", sub,
            "redundant, non-reclosing overpressure protection in parallel "
            "with the relief valve (standard practice for flight tanks)")
        add("PT", f"{tk.name} tank pressure transducer", sub,
            "ullage pressure: feed margin monitoring + abort criteria")
        if tk.cryogenic:
            add("TC", f"{tk.name} tank temperature sensors (2x)", sub,
                "bulk liquid + ullage temperature: fill level inference and "
                "stratification/boil-off monitoring")
        add("FLT", f"{tk.name} feed-line filter", sub,
            "protects injector orifices and valve seats from debris")
        add("MV", f"{tk.name} main valve (remote, fast-acting)", sub,
            "engine start/stop; abort isolation of the propellant supply")
        if tk.oxidizer:
            add("PGV", f"{tk.name} line purge check valve (inert gas)", sub,
                "pre-ignition inerting and post-fire clearing of the "
                "oxidizer line (fire safety)")

    # --------------------------------------------------------------- engine
    coolant_tanks = [t for t in cfg.tanks if t.is_coolant]
    if cfg.regen_cooled:
        if not coolant_tanks:
            warnings.append(
                "regen_cooled is set but no tank is marked is_coolant — "
                "the jacket has no propellant source in this configuration"
            )
        add("PT", "Coolant manifold inlet pressure transducer", "engine",
            "confirms jacket inlet pressure (supercritical margin / "
            "injector stiffness)")
        add("TC", "Coolant outlet temperature sensor", "engine",
            "the regen outlet state IS the injector inlet state — monitors "
            "heat pickup and coking/decomposition margin")
        add("RV", "Coolant-jacket relief path", "engine",
            "protects trapped coolant volume between valves against "
            "heat-soak overpressure after shutdown")
    add("PT", "Chamber pressure transducer", "engine",
        "primary performance and abort parameter")
    add("IGN", "Igniter (with its own isolation)", "engine",
        "bi-propellant engines are not hypergolic — positive ignition "
        "source with independent control")

    return FeedSystemRecommendation(cfg, comps, warnings)


# --------------------------------------------------------------------------
# Optimal architecture selection
# --------------------------------------------------------------------------
def select_optimal_architecture(
    oxidizer: str = "LOX",
    fuel: str = "LCH4",
    coolant_is_fuel: bool = True,
    chamber_pressure: float = 30e5,
    burn_time: float = 20.0,
    injector_dp_fraction: float = 0.2,
    jacket_dp_estimate: float = 25e5,
) -> tuple[FeedSystemConfig, list[str]]:
    """Pick the most sensible feed architecture for the given system.

    Rule set (each choice is returned with its rationale):

    1. Required regen channel feed pressure ~ Pc*(1+injector stiffness) +
       jacket drop. If that exceeds ~85% of the coolant's critical pressure,
       single-phase (supercritical) cooling is impossible pressure-fed with
       a saturated-ullage cryogenic tank -> **pump-fed** (low-pressure tank,
       pump raises only the coolant line).
    2. Otherwise, short burns at modest Pc tolerate **blowdown** (simplest,
       but decaying thrust); longer burns get **regulated** pressure-fed.
    3. Autogenous pressurization is suggested (as a note) for a cryogenic
       coolant tank whenever the regen outlet provides warm vapor anyway.

    This is deliberately transparent design logic, not an optimizer over a
    cost function — every branch is inspectable and cited in the returned
    rationale.
    """
    from .fluids import Fluid

    rationale: list[str] = []
    coolant_name = fuel if coolant_is_fuel else oxidizer
    p_req = chamber_pressure * (1.0 + injector_dp_fraction) + jacket_dp_estimate
    try:
        p_crit = Fluid(coolant_name).P_crit
    except Exception:
        p_crit = float("inf")

    if p_req > 0.85 * p_crit:
        scheme = "pump"
        rationale.append(
            f"required coolant feed pressure ~{p_req/1e5:.0f} bar exceeds "
            f"85% of {coolant_name}'s critical pressure "
            f"({p_crit/1e5:.0f} bar): a saturated-ullage cryo tank cannot be "
            "pressure-fed that high, and subcritical heated coolant would "
            "boil in the jacket -> electric pump on the coolant line, tank "
            "held at a low autogenous pad (2-4 bar)"
        )
    elif burn_time <= 8.0 and chamber_pressure <= 15e5:
        scheme = "blowdown"
        rationale.append(
            f"short burn ({burn_time:.0f} s) at modest Pc "
            f"({chamber_pressure/1e5:.0f} bar): blowdown is the simplest "
            "reliable option; verify end-of-burn injector margin"
        )
    else:
        scheme = "regulated"
        rationale.append(
            f"required feed pressure {p_req/1e5:.0f} bar is well below the "
            f"coolant critical pressure ({p_crit/1e5:.0f} bar): regulated "
            "pressure-fed keeps mdot and Pc constant over the "
            f"{burn_time:.0f} s burn with minimum moving parts"
        )

    def _cryo(name: str) -> bool:
        try:
            f = Fluid(name)
            return f.T_sat(101325.0) < 200.0
        except Exception:
            return False

    tanks = [
        TankSpec(oxidizer, cryogenic=_cryo(oxidizer), oxidizer=True,
                 is_coolant=not coolant_is_fuel),
        TankSpec(fuel, cryogenic=_cryo(fuel), is_coolant=coolant_is_fuel),
    ]
    coolant_tank = tanks[1] if coolant_is_fuel else tanks[0]
    if coolant_tank.cryogenic and scheme == "pump":
        rationale.append(
            "autogenous tank pad is a natural upgrade here: the regen outlet "
            "already provides warm vapor of the same species (see the "
            "'autogenous' scheme for that variant)"
        )
    return FeedSystemConfig(pressurization=scheme, tanks=tanks,
                            regen_cooled=True), rationale


def draw_pid(rec: FeedSystemRecommendation, path: str) -> str:
    """Render the recommendation as an ISA-5.1-flavored schematic with drawn
    valve/instrument symbols, routed lines, legend, and title block.

    Implementation lives in :mod:`cryosim.pid_drawing`.
    """
    from .pid_drawing import draw_pid as _draw

    return _draw(rec, path)
