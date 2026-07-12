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
# Schematic drawing
# --------------------------------------------------------------------------
_SUBSYSTEM_COLORS = {
    "pressurant": "#7aa6c2",
    "engine": "#c27a7a",
}


def draw_pid(rec: FeedSystemRecommendation, path: str) -> str:
    """Draw a simplified feed-system schematic and save it (SVG/PNG by
    file extension). One column per subsystem, flow top -> bottom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    groups = rec.by_subsystem()
    order = ["pressurant"] + [t.name for t in rec.config.tanks] + ["engine"]
    cols = [g for g in order if g in groups]

    n_rows = max(len(groups[c]) for c in cols)
    fig_h = 1.2 + 0.62 * n_rows
    fig, ax = plt.subplots(figsize=(3.4 * len(cols), fig_h))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(-0.5, n_rows + 1.2)
    ax.axis("off")

    for j, colname in enumerate(cols):
        color = _SUBSYSTEM_COLORS.get(colname, "#8fbc8f")
        ax.text(j + 0.5, n_rows + 0.8, colname, ha="center", va="center",
                fontsize=12, fontweight="bold")
        comps = groups[colname]
        for i, c in enumerate(comps):
            y = n_rows - i
            ax.add_patch(FancyBboxPatch(
                (j + 0.08, y - 0.22), 0.84, 0.44,
                boxstyle="round,pad=0.02", linewidth=1.2,
                edgecolor=color, facecolor="white",
            ))
            ax.text(j + 0.5, y, f"{c.tag}  {c.name}", ha="center",
                    va="center", fontsize=7.2, wrap=True)
            if i < len(comps) - 1:  # flow line to next component
                ax.plot([j + 0.5, j + 0.5], [y - 0.24, y - 0.78],
                        color=color, lw=1.2)
        # connect column to the engine column baseline
        if colname != "engine":
            y_last = n_rows - len(comps) + 1
            ax.plot([j + 0.5, j + 0.5], [y_last - 0.24, -0.3], color=color,
                    lw=1.2, linestyle=":")
    ax.plot([0.5, len(cols) - 0.5], [-0.3, -0.3], color="#555", lw=1.5,
            linestyle=":")
    ax.text(len(cols) / 2, -0.45,
            "simplified schematic — checklist aid, not a safety-reviewed P&ID",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.suptitle(
        f"Recommended feed system ({rec.config.pressurization}-fed)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
