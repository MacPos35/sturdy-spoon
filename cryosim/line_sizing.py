"""Automatic feed-line (tube) sizing and fitting selection.

Given each line's service (liquid feed, pump discharge, pressurant gas,
vent), mass flow, fluid density, and maximum allowable working pressure,
this module selects:

* a **tube size** from the standard imperial seamless-tube catalog
  (1/8"-2" OD, standard walls 0.028"-0.120"): the smallest OD whose flow
  velocity stays under the service target, with the wall thickness picked
  from Barlow's thin-wall formula  t = P*OD / (2*S)  against the material
  allowable stress with a design factor (ASME B31.3-style approach), plus a
  minimum handling wall of 0.028";
* a **fitting type** per joint from cryo/pressure rules: 37 deg flare
  (AN/MS) or orbital welds for cryogenic lines (tapered-thread NPT joints
  leak at cryogenic temperature and are rejected), twin-ferrule compression
  for ambient-temperature gas and instrumentation lines, welded/flanged
  above 1" OD.

Velocity targets follow common liquid-rocket feed practice (Huzel & Huang
ch. 8: feed lines of order a few m/s to limit dynamic head and water-hammer;
gas lines tens of m/s):

===================  =================
service              target velocity
===================  =================
liquid feed          5 m/s
pump suction         3 m/s
pump discharge       8 m/s
pressurant gas       25 m/s
vent / relief gas    50 m/s
===================  =================

Assumptions / limitations
-------------------------
* Barlow thin-wall sizing with a single allowable stress per material
  (room-temperature ASME-style allowable — conservative for cryo service
  where austenitic steels get stronger, but NOT valid for aluminum braze
  joints etc.); no bend/flare thinning allowance, no external loads, no
  fatigue or water-hammer surge analysis. Verify against your team's
  structures process before ordering hardware.
* Pressure drop is reported per meter of straight tube (Haaland friction);
  actual routing (bends, fittings) adds equivalent length the user must
  budget separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regen_model import haaland_friction_factor

IN = 25.4e-3  # m per inch

#: Standard seamless tube outer diameters [in] and wall thicknesses [in].
TUBE_ODS_IN = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0]
TUBE_WALLS_IN = [0.028, 0.035, 0.049, 0.065, 0.083, 0.095, 0.120]

#: Allowable stress [Pa] (ASME B31.3-style room-temperature values) and
#: cryogenic suitability.
MATERIALS = {
    "316L": {"S_allow": 115e6, "cryo_ok": True,
             "note": "austenitic SS, tough at cryo"},
    "AL6061T6": {"S_allow": 65e6, "cryo_ok": True,
                 "note": "verify welded-zone knockdown"},
    "PTFE-lined flex": {"S_allow": None, "cryo_ok": False,
                        "note": "per-hose rating"},
}

#: Service velocity targets [m/s].
VELOCITY_TARGETS = {
    "liquid_feed": 5.0,
    "pump_suction": 3.0,
    "pump_discharge": 8.0,
    "pressurant_gas": 25.0,
    "vent_gas": 50.0,
}

#: Design factor applied on top of the allowable stress (schedule/tolerance
#: allowance).
DESIGN_FACTOR = 1.25
MIN_WALL_IN = 0.028


@dataclass
class LineSpec:
    """Selected tube + fitting for one line segment (SI units unless noted)."""

    name: str
    service: str
    fluid: str
    mdot: float                 # kg/s
    rho: float                  # kg/m^3
    mawp: float                 # Pa
    material: str
    od_in: float                # in
    wall_in: float              # in
    velocity: float             # m/s (at the selected ID)
    dp_per_m: float             # Pa/m straight-tube friction
    fitting: str
    cryogenic: bool
    notes: list = field(default_factory=list)

    @property
    def od(self) -> float:
        return self.od_in * IN

    @property
    def wall(self) -> float:
        return self.wall_in * IN

    @property
    def id_(self) -> float:
        return self.od - 2.0 * self.wall

    @property
    def dash(self) -> int:
        """AN dash size = tube OD in sixteenths of an inch."""
        return round(self.od_in * 16)

    def label(self) -> str:
        """Short spec label for drawing annotation."""
        return (f'{self.od_in:.3g}" x {self.wall_in:.3f}" {self.material} '
                f"(-{self.dash})")

    def describe(self) -> str:
        return (
            f"{self.name}: {self.od_in:.3g}\" OD x {self.wall_in:.3f}\" wall "
            f"{self.material} tube (ID {self.id_*1e3:.1f} mm, dash -{self.dash})"
            f" | {self.fluid}, {self.mdot*1e3:.0f} g/s -> {self.velocity:.1f} m/s"
            f" | MAWP {self.mawp/1e5:.0f} bar | dp {self.dp_per_m/1e5*100:.2f}"
            f" bar/100m | fittings: {self.fitting}"
            + ("".join(f"\n    note: {n}" for n in self.notes))
        )


def required_wall_in(mawp: float, od_in: float, material: str) -> float:
    """Barlow thin-wall thickness [in]: t = P*OD/(2*S), with design factor."""
    S = MATERIALS[material]["S_allow"]
    t_req_m = DESIGN_FACTOR * mawp * (od_in * IN) / (2.0 * S)
    return max(t_req_m / IN, MIN_WALL_IN)


def select_fitting(od_in: float, cryogenic: bool, service: str) -> tuple[str, list]:
    """Fitting-type rule set. Returns (fitting description, notes)."""
    notes = []
    if od_in >= 1.0:
        f = "butt-welded joints / bolted flanges"
        notes.append("above ~1\" OD, flare/compression torque becomes "
                     "impractical - weld and flange")
    elif cryogenic:
        f = "37 deg flare (AN/MS) or orbital-welded joints"
        notes.append("NPT tapered threads are rejected for cryogenic "
                     "service (leak on thermal cycling)")
    elif service in ("pressurant_gas", "vent_gas"):
        f = "twin-ferrule compression (rated) or 37 deg flare"
    else:
        f = "37 deg flare (AN/MS)"
    return f, notes


def size_line(
    name: str,
    service: str,
    fluid: str,
    mdot: float,
    rho: float,
    mawp: float,
    mu: float = 2.0e-4,
    material: str = "316L",
    cryogenic: bool = True,
) -> LineSpec:
    """Pick the smallest standard tube meeting velocity + Barlow criteria."""
    if service not in VELOCITY_TARGETS:
        raise ValueError(f"unknown service {service!r}; "
                         f"expected one of {list(VELOCITY_TARGETS)}")
    if not MATERIALS[material]["cryo_ok"] and cryogenic:
        raise ValueError(f"{material} is not rated for cryogenic service")
    v_max = VELOCITY_TARGETS[service]

    for od_in in TUBE_ODS_IN:
        t_req = required_wall_in(mawp, od_in, material)
        walls = [w for w in TUBE_WALLS_IN if w >= t_req - 1e-12]
        if not walls:
            continue  # OD can't take the pressure with catalog walls
        wall_in = walls[0]
        id_m = (od_in - 2 * wall_in) * IN
        if id_m <= 0:
            continue
        area = np.pi / 4 * id_m**2
        v = mdot / (rho * area) if mdot > 0 else 0.0
        if v <= v_max:
            Re = rho * v * id_m / mu if v > 0 else 0.0
            fD = haaland_friction_factor(Re, 1.5e-6 / id_m) if v > 0 else 0.0
            dp_m = fD / id_m * 0.5 * rho * v**2 if v > 0 else 0.0
            fitting, notes = select_fitting(od_in, cryogenic, service)
            spec = LineSpec(
                name=name, service=service, fluid=fluid, mdot=mdot, rho=rho,
                mawp=mawp, material=material, od_in=od_in, wall_in=wall_in,
                velocity=v, dp_per_m=dp_m, fitting=fitting,
                cryogenic=cryogenic, notes=notes,
            )
            if wall_in > 0.095:
                spec.notes.append("thick-wall tube: check bend radius and "
                                  "flare-ability; consider next OD up")
            return spec
    raise ValueError(
        f"no catalog tube satisfies {name}: mdot={mdot} kg/s at "
        f"MAWP={mawp/1e5:.0f} bar"
    )


# --------------------------------------------------------------------------
# Whole-feed-system sizing
# --------------------------------------------------------------------------
def size_feed_system(
    fluid_names: dict,
    mdots: dict,
    tank_meop: float,
    coolant_hp_mawp: float | None = None,
    pressurant_rho: float = 3.0,
    liquid_rhos: dict | None = None,
    pump_fed: bool = False,
) -> list[LineSpec]:
    """Size the canonical line set of the recommended feed system.

    Parameters
    ----------
    fluid_names : {'oxidizer': 'LOX', 'fuel': 'LCH4'}
    mdots : {'oxidizer': .., 'fuel': .., 'pressurant': ..} [kg/s]
    tank_meop : tank maximum expected operating pressure [Pa] (relief set
        point x margin) — used as feed/vent line MAWP.
    coolant_hp_mawp : MAWP of the high-pressure coolant segment (pump
        discharge -> jacket); defaults to tank_meop when pressure-fed.
    pressurant_rho : pressurant gas density at line conditions [kg/m^3].
    liquid_rhos : optional {'oxidizer': rho, 'fuel': rho}; CoolProp
        saturated-liquid values are used when omitted.
    """
    from .fluids import Fluid

    liquid_rhos = dict(liquid_rhos or {})
    for key in ("oxidizer", "fuel"):
        if key not in liquid_rhos:
            try:
                liquid_rhos[key] = Fluid(fluid_names[key]).sat_liquid(2e5).rho
            except Exception:
                liquid_rhos[key] = 1000.0

    def _cryo(name):
        try:
            return Fluid(name).T_sat(101325.0) < 200.0
        except Exception:
            return False

    specs = []
    for key in ("oxidizer", "fuel"):
        name = fluid_names[key]
        cryo = _cryo(name)
        specs.append(size_line(
            f"{name} feed line", "pump_suction" if (pump_fed and key == "fuel")
            else "liquid_feed", name, mdots[key], liquid_rhos[key],
            tank_meop, cryogenic=cryo,
        ))
        specs.append(size_line(
            f"{name} fill/drain line", "liquid_feed", name,
            0.5 * mdots[key], liquid_rhos[key], tank_meop, cryogenic=cryo,
        ))
        # vent sized for vapor at tank conditions; flow ~ pressurant demand
        specs.append(size_line(
            f"{name} vent/relief line", "vent_gas", name + " vapor",
            max(mdots.get("pressurant", 0.01), 0.01), pressurant_rho,
            tank_meop, cryogenic=cryo,
        ))
    specs.append(size_line(
        "pressurant header", "pressurant_gas", "pressurant",
        max(mdots.get("pressurant", 0.01), 0.005), pressurant_rho,
        tank_meop, cryogenic=False,
    ))
    if coolant_hp_mawp is not None and pump_fed:
        specs.append(size_line(
            "coolant HP line (pump -> jacket)", "pump_discharge",
            fluid_names["fuel"], mdots["fuel"], liquid_rhos["fuel"],
            coolant_hp_mawp, cryogenic=_cryo(fluid_names["fuel"]),
        ))
    return specs


def report(specs: list[LineSpec]) -> str:
    lines = ["Line sizing (velocity targets + Barlow wall, see docstring "
             "for assumptions)", "=" * 72]
    lines += [s.describe() for s in specs]
    lines.append("\nSizing is a starting point; verify surge/water-hammer, "
                 "bend thinning and vendor pressure ratings before ordering.")
    return "\n".join(lines)
