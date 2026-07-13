"""Autonomous requirements-to-engine design pipeline (Noyron-style).

Inspired by LEAP 71's Noyron computational model: from *top-level
requirements only* — thrust, propellants, chamber pressure, ambient — this
module autonomously produces a complete regeneratively cooled engine design
(contour, cooling channels, coax-swirl injector head, torus manifolds,
structural closeout) plus a predicted-performance summary. Like Noyron, the
"AI" here is **encoded engineering knowledge**: deterministic physics models
and design rules iterated against an explicit constraint ledger, with every
decision appended to a :class:`DesignTrace` so the result is traceable,
explainable and exactly reproducible — no machine learning is involved.

Pipeline stages
---------------
A. Performance sizing: c*/Cf from the isentropic relations
   (``combustion.py``) with fixed efficiency factors, optimum expansion for
   the stated ambient (Summerfield separation guard at sea level), throat
   size, contraction-ratio and L* rules -> :class:`ChamberContour`.
B. Thermal design: cooling-channel search (``optimize.optimize_channels``)
   under the pressure-drop budget; constraint: peak hot-wall temperature
   below the liner limit; feed pressure raised above the coolant critical
   pressure when the jacket goes two-phase (the pump-fed finding from the
   coupled demo, automated as a rule).
C. Injector: coaxial-swirl head (``injector_design``) with the fuel
   injection state taken from the *regen jacket outlet* of stage B.
D. Manifolds: torus inlet/outlet headers (``manifold_design``) for the
   stage-B channels at the stage-B/C pressures.
E. Structure: closeout shell thickness from hoop stress against the
   ``line_sizing.MATERIALS`` allowables (Barlow, x1.25 design factor).

On a constraint violation, ordered repair rules fire (raise the dp budget,
go supercritical on the feed, coarsen the element count, ...) and the loop
re-runs; if the rule budget is exhausted the design **fails loudly** with
the full ledger — no silent acceptance.

Assumptions / limitations
-------------------------
* Performance model is ideal + fixed efficiency factors (eta_c* = 0.95,
  eta_Cf = 0.97 — Huzel & Huang ch. 1 typical mid-range values); no
  shifting-equilibrium CEA coupling, no boundary-layer (use a real CEA run
  and pass ``combustion_gas`` for design work).
* All sub-model limitations (Bartz-class cooling, ideal swirl theory,
  reduced-order manifolds) carry through unchanged — see each module.
* Film cooling is sized hydraulically (injector ring) but its thermal
  benefit is NOT modeled in the regen solution — conservative.
* The rules encode small-engine (kN-class) student/early-design practice;
  numbers must be re-examined for other classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import yaml

from .chamber_geometry import ChamberContour
from .combustion import (CombustionGas, area_ratio_from_mach, gas_preset,
                         R_UNIV)
from .fluids import Fluid
from .injector_design import InjectorDesign, design_injector
from .line_sizing import MATERIALS
from .manifold_design import ManifoldSystemDesign, design_manifolds
from .optimize import ChannelDesignResult, optimize_channels

G0 = 9.80665

#: Default mixture ratios for the shipped gas presets (nominal CEA-ish).
DEFAULT_OF = {"lox/ch4": 3.2, "lox/ethanol": 1.7}

#: Liner materials: conductivity + hot-wall temperature limit.
LINER_MATERIALS = {
    "cucrzr": {"k_wall": 330.0, "T_limit": 800.0, "rho": 8900.0,
               "note": "CuCrZr, printed (typ. limit ~800 K)"},
    "grcop-42": {"k_wall": 290.0, "T_limit": 850.0, "rho": 8800.0,
                 "note": "GRCop-42, LPBF"},
    "inconel718": {"k_wall": 12.0, "T_limit": 1150.0, "rho": 8190.0,
                   "note": "IN718 — poor conductor, thin walls only"},
}

#: Manufacturing-process feature floors [m].
PROCESSES = {
    "lpbf": {"min_channel_width": 0.5e-3, "min_wall": 0.4e-3,
             "min_orifice": 0.4e-3, "min_land": 0.4e-3,
             "note": "laser powder-bed fusion"},
    "machined": {"min_channel_width": 0.8e-3, "min_wall": 0.5e-3,
                 "min_orifice": 0.5e-3, "min_land": 0.8e-3,
                 "note": "milled slots + brazed closeout"},
}


# ----------------------------------------------------------------------
# Spec, trace, ledger
# ----------------------------------------------------------------------

@dataclass
class EngineSpec:
    """Top-level requirements. Everything else is derived by rules."""

    thrust: float                       # N, at the stated ambient
    propellants: str = "lox/ch4"        # preset key "ox/fuel"
    chamber_pressure: float = 20e5      # Pa
    ambient_pressure: float = 101325.0  # Pa (0 -> vacuum design)
    of_ratio: float | None = None       # None -> preset default
    combustion_gas: dict | None = None  # override: {T_c, gamma, molar_mass}
    liner: str = "cucrzr"
    closeout_material: str = "316L"
    process: str = "lpbf"
    dp_budget: float = 30e5             # Pa, jacket pressure-drop budget
    coolant_inlet_T: float | None = None  # None -> near-saturation rule
    max_feed_pressure: float = 200e5    # Pa, pump/tank capability ceiling
    stiffness: float = 0.20             # injector dP / Pc
    spray_half_angle_deg: float = 45.0
    film_fraction: float = 0.10
    thrust_per_element: float = 1.5e3   # N
    expansion_ratio_cap: float = 25.0   # vacuum-design cap
    helix_angle_deg: float = 0.0
    name: str = "engine"

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec keys: {sorted(unknown)}")
        d = dict(d)
        for k, v in d.items():
            # YAML 1.1 reads '2.0e3' (no sign) as a string — coerce all
            # non-string spec fields
            if k not in ("propellants", "liner", "closeout_material",
                         "process", "name", "combustion_gas") \
                    and isinstance(v, str):
                d[k] = float(v)
        if isinstance(d.get("combustion_gas"), dict):
            d["combustion_gas"] = {k: float(v) for k, v
                                   in d["combustion_gas"].items()}
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "EngineSpec":
        with open(path) as fh:
            d = yaml.safe_load(fh)
        if "engine" in d and isinstance(d["engine"], dict):  # allow nesting
            flat = {}
            for v in d.values():
                flat.update(v if isinstance(v, dict) else {})
            d = flat
        return cls.from_dict(d)


@dataclass
class TraceEntry:
    stage: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.stage}] {self.rule}: {self.detail}"


class DesignTrace:
    """The explainable record: every rule that fired, in order."""

    def __init__(self):
        self.entries: list[TraceEntry] = []
        self.t0 = time.perf_counter()

    def log(self, stage: str, rule: str, detail: str) -> None:
        self.entries.append(TraceEntry(stage, rule, detail))

    def as_markdown(self) -> str:
        lines = ["# Design trace", "",
                 "Every design decision, in the order it was made. "
                 "Deterministic: the same spec reproduces this file "
                 "exactly.", ""]
        stage = None
        for e in self.entries:
            if e.stage != stage:
                stage = e.stage
                lines += [f"## {stage}", ""]
            lines.append(f"- **{e.rule}** — {e.detail}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return "\n".join(str(e) for e in self.entries)


@dataclass
class LedgerItem:
    name: str
    ok: bool
    value: str
    requirement: str

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name}: {self.value} (req: {self.requirement})"


class DesignError(RuntimeError):
    """Raised when the rule budget is exhausted; carries the ledger."""

    def __init__(self, message: str, ledger: list[LedgerItem], trace):
        lines = [message, "Constraint ledger at failure:"]
        lines += [str(i) for i in ledger]
        super().__init__("\n".join(lines))
        self.ledger = ledger
        self.trace = trace


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

@dataclass
class EngineDesign:
    """Converged engine design + predicted performance."""

    spec: EngineSpec
    gas: CombustionGas
    # performance
    mdot: float
    mdot_ox: float
    mdot_fuel: float
    of_ratio: float
    expansion_ratio: float
    c_star: float
    Cf: float
    Isp_ambient: float
    Isp_vac: float
    throat_radius: float
    contraction_ratio: float
    chamber_length: float
    L_star: float
    contour: ChamberContour
    # subsystems
    channel_design: ChannelDesignResult
    injector: InjectorDesign
    manifolds: ManifoldSystemDesign
    t_closeout: float
    P_coolant_inlet: float
    # bookkeeping
    ledger: list[LedgerItem] = field(default_factory=list)
    trace: DesignTrace | None = None
    iterations: int = 1

    def performance_table(self) -> list[tuple[str, str]]:
        s, cd = self.spec, self.channel_design
        e = self.contour
        return [
            ("thrust (design ambient)", f"{s.thrust/1e3:.2f} kN"),
            ("chamber pressure", f"{s.chamber_pressure/1e5:.1f} bar"),
            ("propellants / O/F", f"{s.propellants} / {self.of_ratio:.2f}"),
            ("mass flow (ox + fuel)",
             f"{self.mdot:.3f} kg/s ({self.mdot_ox:.3f} + "
             f"{self.mdot_fuel:.3f})"),
            ("Isp (design ambient / vacuum)",
             f"{self.Isp_ambient:.0f} / {self.Isp_vac:.0f} s"),
            ("c* (delivered) / Cf", f"{self.c_star:.0f} m/s / {self.Cf:.3f}"),
            ("throat / exit diameter",
             f"{2*self.throat_radius*1e3:.1f} / "
             f"{2*e.r[-1]*1e3:.1f} mm"),
            ("expansion / contraction ratio",
             f"{self.expansion_ratio:.1f} / {self.contraction_ratio:.1f}"),
            ("chamber length / L*",
             f"{self.chamber_length*1e3:.0f} mm / {self.L_star:.2f} m"),
            ("cooling channels",
             f"{cd.channels.n_channels} x "
             f"{cd.channels.channel_width*1e3:.2f} x "
             f"{cd.channels.channel_height*1e3:.2f} mm, hot wall "
             f"{cd.channels.t_wall*1e3:.2f} mm"),
            ("peak hot-wall temperature", f"{cd.peak_T_wg:.0f} K"),
            ("jacket pressure drop", f"{cd.dp/1e5:.1f} bar"),
            ("coolant outlet (injector inlet)",
             f"{cd.coolant_outlet_T:.0f} K"),
            ("coolant feed pressure", f"{self.P_coolant_inlet/1e5:.0f} bar"),
            ("injector", f"{self.injector.n_elements} coax-swirl elements, "
             f"stiffness {self.injector.stiffness_ox*100:.0f}% Pc"),
            ("manifold maldistribution",
             f"{self.manifolds.maldistribution*100:.1f}% "
             f"(target {self.manifolds.target*100:.0f}%)"),
            ("closeout shell", f"{self.t_closeout*1e3:.2f} mm "
             f"{self.spec.closeout_material}"),
        ]

    def describe(self) -> str:
        lines = [f"Engine design '{self.spec.name}' "
                 f"(converged in {self.iterations} iteration(s))",
                 "=" * 72]
        lines += [f"  {k:34s} {v}" for k, v in self.performance_table()]
        lines.append("Constraint ledger:")
        lines += [str(i) for i in self.ledger]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Stage A helpers
# ----------------------------------------------------------------------

def _c_star(gas: CombustionGas) -> float:
    g = gas.gamma
    R = R_UNIV / gas.molar_mass
    Gamma = np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    return float(np.sqrt(R * gas.T_c) / Gamma)


def _thrust_coefficient(gas: CombustionGas, Pc: float, eps: float,
                        Pa: float) -> tuple[float, float]:
    """(Cf, Pe): ideal thrust coefficient at expansion ratio eps."""
    from .combustion import mach_from_area_ratio
    g = gas.gamma
    Me = mach_from_area_ratio(eps, g, supersonic=True)
    Pe = Pc * (1.0 + (g - 1.0) / 2.0 * Me**2) ** (-g / (g - 1.0))
    term = (2.0 * g**2 / (g - 1.0)
            * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
            * (1.0 - (Pe / Pc) ** ((g - 1.0) / g)))
    return float(np.sqrt(term) + eps * (Pe - Pa) / Pc), float(Pe)


def _optimum_expansion(gas: CombustionGas, Pc: float, Pa: float,
                       cap: float) -> tuple[float, str]:
    """Expansion ratio rule: Pe = Pa, capped (vacuum -> cap)."""
    g = gas.gamma
    if Pa < 5e3:  # effectively vacuum: optimum is unbounded
        return cap, (f"ambient < 0.05 bar: expansion ratio set to the "
                     f"spec cap {cap:.0f} (vacuum optimum is unbounded)")
    Me = np.sqrt(max(((Pc / Pa) ** ((g - 1.0) / g) - 1.0)
                     * 2.0 / (g - 1.0), 1.001))
    eps = area_ratio_from_mach(float(Me), g)
    if eps > cap:
        return cap, (f"optimum expansion {eps:.1f} exceeds the spec cap; "
                     f"capped at {cap:.0f}")
    return float(eps), (f"perfect expansion at Pa = {Pa/1e3:.1f} kPa "
                        f"-> area ratio {eps:.2f}")


def _contraction_ratio_rule(Rt: float) -> tuple[float, str]:
    """Small engines need larger contraction (Huzel & Huang fig. 4-9 trend)."""
    Dt_mm = 2e3 * Rt
    cr = float(np.clip(16.0 * (Dt_mm / 30.0) ** -0.6, 4.0, 15.0))
    return cr, (f"throat D {Dt_mm:.1f} mm -> contraction ratio {cr:.1f} "
                "(small-engine sizing trend, clipped to [4, 15])")


def _l_star_rule(propellants: str) -> tuple[float, str]:
    table = {"lox/ch4": 1.1, "lox/ethanol": 1.5}
    L = table.get(propellants, 1.2)
    return L, (f"characteristic length L* = {L:.1f} m "
               f"({propellants} class value, Huzel & Huang table 4-1 range)")


# ----------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------

def design_engine(spec: EngineSpec, n_random: int = 40, n_polish: int = 40,
                  verbose: bool = False) -> EngineDesign:
    """Run the full autonomous design; raises DesignError on failure."""
    trace = DesignTrace()

    def log(stage, rule, detail):
        trace.log(stage, rule, detail)
        if verbose:
            print(f"  {TraceEntry(stage, rule, detail)}")

    # ---------------- propellants & gas ----------------------------------
    if "/" not in spec.propellants:
        raise ValueError("propellants must be 'ox/fuel', e.g. 'lox/ch4'")
    ox_name, fuel_name = spec.propellants.split("/", 1)
    if spec.combustion_gas is not None:
        gas = CombustionGas(**spec.combustion_gas)
        log("0. setup", "combustion gas",
            f"user-supplied CEA values: T_c={gas.T_c:.0f} K, "
            f"gamma={gas.gamma:.3f}, M={gas.molar_mass*1e3:.1f} g/mol")
    else:
        gas = gas_preset(spec.propellants)
        log("0. setup", "combustion gas",
            f"preset '{spec.propellants}': T_c={gas.T_c:.0f} K, "
            f"gamma={gas.gamma:.3f}, M={gas.molar_mass*1e3:.1f} g/mol "
            "(nominal — supply combustion_gas from a CEA run for design)")
    of = spec.of_ratio or DEFAULT_OF.get(spec.propellants, 2.5)
    log("0. setup", "mixture ratio",
        f"O/F = {of:.2f}" + ("" if spec.of_ratio else
                             " (preset default)"))

    if spec.liner not in LINER_MATERIALS:
        raise ValueError(f"unknown liner {spec.liner!r}; "
                         f"available: {list(LINER_MATERIALS)}")
    liner = LINER_MATERIALS[spec.liner]
    proc = PROCESSES[spec.process]
    log("0. setup", "materials/process",
        f"liner {spec.liner} (k={liner['k_wall']:.0f} W/mK, "
        f"T_limit={liner['T_limit']:.0f} K), closeout "
        f"{spec.closeout_material}, process {spec.process} "
        f"({proc['note']})")

    # ---------------- mutable design state (repair rules act here) -------
    dp_budget = spec.dp_budget
    thrust_per_element = spec.thrust_per_element
    coolant = Fluid(fuel_name)
    oxidizer = Fluid(ox_name)
    T_cool_in = spec.coolant_inlet_T
    if T_cool_in is None:
        T_cool_in = coolant.T_sat(2e5) - 3.0
        log("0. setup", "coolant inlet temperature",
            f"{T_cool_in:.0f} K: 3 K subcooled below saturation at 2 bar "
            "(typical run-tank condition)")
    # feed pressure rule: chamber + injector drop (+ manifold allowance)
    # + jacket budget + 10% line margin
    def feed_pressure(dp_b):
        return (spec.chamber_pressure * (1.0 + spec.stiffness * 1.15)
                + dp_b) * 1.10

    P_inlet = feed_pressure(dp_budget)
    supercritical_forced = False

    last_error = ""
    max_iter = 6
    for iteration in range(1, max_iter + 1):
        ledger: list[LedgerItem] = []

        # ============ A. performance sizing ===============================
        Pc, Pa = spec.chamber_pressure, spec.ambient_pressure
        eps, why = _optimum_expansion(gas, Pc, Pa, spec.expansion_ratio_cap)
        log("A. performance", "expansion ratio", why)
        eta_cstar, eta_cf = 0.95, 0.97
        cstar = _c_star(gas) * eta_cstar
        Cf, Pe = _thrust_coefficient(gas, Pc, eps, Pa)
        Cf *= eta_cf
        if Pa > 5e3 and Pe < 0.4 * Pa:
            # Summerfield separation guard (only reachable via the cap)
            log("A. performance", "separation guard",
                f"Pe/Pa = {Pe/Pa:.2f} < 0.4 — shrinking expansion to the "
                "Summerfield limit")
            from scipy.optimize import brentq as _brentq
            eps = _brentq(lambda e: _thrust_coefficient(gas, Pc, e, Pa)[1]
                          - 0.4 * Pa, 1.5, eps)
            Cf, Pe = _thrust_coefficient(gas, Pc, eps, Pa)
            Cf *= eta_cf
        At = spec.thrust / (Cf * Pc)
        Rt = float(np.sqrt(At / np.pi))
        mdot = Pc * At / cstar
        Isp = spec.thrust / (mdot * G0)
        Cf_vac, _ = _thrust_coefficient(gas, Pc, eps, 0.0)
        Isp_vac = Cf_vac * eta_cf * cstar / G0
        log("A. performance", "throat sizing",
            f"eta_c*={eta_cstar}, eta_Cf={eta_cf} (Huzel & Huang typical) "
            f"-> c*={cstar:.0f} m/s, Cf={Cf:.3f}, At={At*1e4:.2f} cm^2 "
            f"(Dt {2e3*Rt:.1f} mm), mdot={mdot:.3f} kg/s, "
            f"Isp={Isp:.0f} s")
        cr, why = _contraction_ratio_rule(Rt)
        log("A. performance", "contraction ratio", why)
        Lstar, why = _l_star_rule(spec.propellants)
        log("A. performance", "L*", why)
        # cylinder holds ~80% of V_c (the convergent cone holds the rest)
        Lc = 0.8 * Lstar / cr
        log("A. performance", "chamber length",
            f"L_c = 0.8 L*/CR = {Lc*1e3:.0f} mm (cylindrical section; "
            "convergent cone supplies the remaining chamber volume)")
        contour = ChamberContour(
            throat_radius=Rt, contraction_ratio=cr, expansion_ratio=eps,
            chamber_length=Lc)
        mdot_fuel = mdot / (1.0 + of)
        mdot_ox = mdot - mdot_fuel

        # ============ B. thermal design ====================================
        if P_inlet > spec.max_feed_pressure:
            raise DesignError(
                f"required feed pressure {P_inlet/1e5:.0f} bar exceeds the "
                f"capability ceiling {spec.max_feed_pressure/1e5:.0f} bar",
                ledger, trace)
        log("B. thermal", "feed pressure",
            f"coolant feed {P_inlet/1e5:.0f} bar = (Pc + injector drop + "
            f"manifold allowance + jacket budget {dp_budget/1e5:.0f} bar) "
            "x 1.10 line margin"
            + (" [raised supercritical by repair rule]"
               if supercritical_forced else ""))
        bounds = {
            "channel_width": (proc["min_channel_width"], 3.0e-3),
            "t_wall": (proc["min_wall"], 1.5e-3),
        }
        cd = optimize_channels(
            contour, gas, coolant, Pc, mdot_fuel, T_cool_in, P_inlet,
            dp_budget=dp_budget, k_wall=liner["k_wall"], bounds=bounds,
            n_random=n_random, n_polish=n_polish,
            min_land=proc["min_land"])
        log("B. thermal", "channel search",
            f"{cd.n_evaluations} regen solves -> "
            f"{cd.channels.n_channels} channels "
            f"{cd.channels.channel_width*1e3:.2f}x"
            f"{cd.channels.channel_height*1e3:.2f} mm, wall "
            f"{cd.channels.t_wall*1e3:.2f} mm: peak T_wg {cd.peak_T_wg:.0f} "
            f"K, dP {cd.dp/1e5:.1f} bar")

        boiling = bool(cd.result.boiling_detected)
        ledger += [
            LedgerItem("peak hot-wall temperature",
                       cd.peak_T_wg <= liner["T_limit"],
                       f"{cd.peak_T_wg:.0f} K",
                       f"<= {liner['T_limit']:.0f} K ({spec.liner})"),
            LedgerItem("jacket pressure drop", cd.dp <= dp_budget * 1.05,
                       f"{cd.dp/1e5:.1f} bar",
                       f"<= {dp_budget/1e5:.0f} bar budget"),
            LedgerItem("single-phase coolant", not boiling,
                       "two-phase!" if boiling else "single-phase/"
                       "supercritical", "no boiling in the jacket"),
            LedgerItem("land at throat",
                       cd.land_at_throat >= proc["min_land"],
                       f"{cd.land_at_throat*1e3:.2f} mm",
                       f">= {proc['min_land']*1e3:.1f} mm "
                       f"({spec.process})"),
        ]

        # ---- repair rules for stage B ------------------------------------
        if boiling and not supercritical_forced:
            P_new = 1.2 * coolant.P_crit + dp_budget
            log("B. thermal", "REPAIR: supercritical feed",
                f"jacket coolant entered the vapor dome; raising feed to "
                f"{P_new/1e5:.0f} bar (1.2 x P_crit + budget) — pump-fed "
                "territory, same reason HYPROB feeds at ~160 bar")
            P_inlet = P_new
            supercritical_forced = True
            last_error = "two-phase coolant"
            continue
        if cd.peak_T_wg > liner["T_limit"]:
            if dp_budget < 2.0 * spec.dp_budget:
                dp_budget *= 1.5
                P_inlet = max(P_inlet, feed_pressure(dp_budget))
                log("B. thermal", "REPAIR: raise dp budget",
                    f"peak wall {cd.peak_T_wg:.0f} K over the "
                    f"{liner['T_limit']:.0f} K limit; raising the jacket "
                    f"budget to {dp_budget/1e5:.0f} bar for more coolant "
                    "velocity")
                last_error = "wall temperature"
                continue
            raise DesignError(
                f"cannot cool the wall: peak {cd.peak_T_wg:.0f} K > "
                f"{liner['T_limit']:.0f} K even at a doubled dp budget — "
                "consider a higher-k liner, lower Pc, or film-cooling "
                "beyond what this model credits", ledger, trace)

        # ============ C. injector ==========================================
        outlet = cd.result.coolant_outlet
        ox_state = oxidizer.state_TP(min(95.0, oxidizer.T_sat(2e5) - 3.0),
                                     P_inlet)
        log("C. injector", "injection states",
            f"ox: {ox_state.T:.0f} K, {ox_state.rho:.0f} kg/m^3; fuel from "
            f"regen outlet: {outlet.T:.0f} K, {outlet.rho:.0f} kg/m^3 at "
            f"{outlet.P/1e5:.0f} bar")
        inj = design_injector(
            Pc=Pc, thrust=spec.thrust, mdot_ox=mdot_ox, mdot_fuel=mdot_fuel,
            rho_ox=ox_state.rho, mu_ox=ox_state.mu, rho_fuel=outlet.rho,
            face_radius=contour.r[0], stiffness=spec.stiffness,
            spray_half_angle_deg=spec.spray_half_angle_deg,
            thrust_per_element=thrust_per_element,
            film_fraction=spec.film_fraction,
            min_orifice_d=proc["min_orifice"])
        for n in inj.notes:
            log("C. injector", "sizing", n)
        for w in inj.warnings:
            log("C. injector", "warning", w)
        port_ok = 2.0 * inj.element.r_tangential >= proc["min_orifice"]
        ledger.append(LedgerItem(
            "injector tangential ports", port_ok,
            f"d {2e3*inj.element.r_tangential:.2f} mm",
            f">= {proc['min_orifice']*1e3:.1f} mm ({spec.process})"))
        supply_ok = outlet.P >= Pc * (1.0 + spec.stiffness)
        ledger.append(LedgerItem(
            "injector supply pressure", supply_ok,
            f"regen outlet {outlet.P/1e5:.1f} bar",
            f">= Pc + drop = {Pc*(1+spec.stiffness)/1e5:.1f} bar"))
        if not port_ok:
            thrust_per_element *= 1.5
            log("C. injector", "REPAIR: fewer, larger elements",
                f"tangential ports below the {spec.process} floor; raising "
                f"thrust/element to {thrust_per_element/1e3:.1f} kN")
            last_error = "injector port size"
            continue
        if not supply_ok:
            P_inlet = (Pc * (1.0 + spec.stiffness * 1.15) + cd.dp) * 1.10
            log("C. injector", "REPAIR: raise feed pressure",
                f"regen outlet {outlet.P/1e5:.0f} bar cannot supply the "
                f"injector; feed raised to {P_inlet/1e5:.0f} bar")
            last_error = "injector supply pressure"
            continue

        # ============ D. manifolds =========================================
        inlet_state = coolant.state_TP(T_cool_in, P_inlet)
        man = design_manifolds(
            contour, cd.channels, mdot_fuel, cd.dp,
            rho_in=inlet_state.rho, mu_in=inlet_state.mu,
            rho_out=outlet.rho, mu_out=outlet.mu,
            mawp=P_inlet, coolant_name=coolant.name,
            material=spec.closeout_material)
        log("D. manifolds", "torus headers",
            f"inlet duct {man.inlet.duct_diameter*1e3:.1f} mm "
            f"({man.inlet.n_feeders} feeders), outlet "
            f"{man.outlet.duct_diameter*1e3:.1f} mm "
            f"({man.outlet.n_feeders}), clocking {man.clocking_deg:.0f} "
            f"deg -> maldistribution {man.maldistribution*100:.1f}%")
        ledger.append(LedgerItem(
            "channel-flow uniformity", man.maldistribution <= man.target,
            f"{man.maldistribution*100:.1f}%",
            f"<= {man.target*100:.0f}% (design target)"))

        # ============ E. structure =========================================
        S = MATERIALS[spec.closeout_material]["S_allow"]
        r_close = float(contour.r.max()) + cd.channels.t_wall \
            + cd.channels.channel_height
        t_close = max(1.25 * P_inlet * r_close / S, 1.0e-3)
        log("E. structure", "closeout shell",
            f"hoop stress at MAWP {P_inlet/1e5:.0f} bar on r "
            f"{r_close*1e3:.0f} mm vs {spec.closeout_material} allowable "
            f"{S/1e6:.0f} MPa (x1.25) -> t = {t_close*1e3:.2f} mm "
            "(1 mm print floor)")

        # ============ verdict ==============================================
        if all(item.ok for item in ledger):
            log("done", "converged",
                f"all {len(ledger)} ledger constraints satisfied after "
                f"{iteration} iteration(s)")
            if verbose:
                print(f"  ({time.perf_counter()-trace.t0:.1f} s)")
            return EngineDesign(
                spec=spec, gas=gas, mdot=mdot, mdot_ox=mdot_ox,
                mdot_fuel=mdot_fuel, of_ratio=of, expansion_ratio=eps,
                c_star=cstar, Cf=Cf, Isp_ambient=Isp, Isp_vac=Isp_vac,
                throat_radius=Rt, contraction_ratio=cr, chamber_length=Lc,
                L_star=Lstar, contour=contour, channel_design=cd,
                injector=inj, manifolds=man, t_closeout=t_close,
                P_coolant_inlet=P_inlet, ledger=ledger, trace=trace,
                iterations=iteration)
        # soft failures with no dedicated repair rule (e.g. uniformity):
        # nothing left to adjust deterministically -> fail loudly
        failed = [i.name for i in ledger if not i.ok]
        raise DesignError(
            f"constraints failed with no applicable repair rule: {failed}",
            ledger, trace)

    raise DesignError(
        f"rule budget exhausted after {max_iter} iterations "
        f"(last violation: {last_error})", [], trace)
