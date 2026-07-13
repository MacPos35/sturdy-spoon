"""Maximum-performance regen cooling-channel design optimization.

Searches the channel design space (channel count, width, height, hot-wall
thickness) for the layout that minimizes the peak hot-gas-side wall
temperature subject to a coolant pressure-drop budget — deliberately
allowing feature sizes that are HARD TO MANUFACTURE conventionally (down to
0.3 mm channels, 0.2 mm walls, sub-millimeter lands). Every dimension that
falls below conventional machining practice is flagged with the process it
would actually need (micro-milling/EDM slots, electroformed or AM closeout)
rather than being forbidden: the point is to show where the *physics*
optimum lies, and what manufacturability costs you relative to it.

Method: Latin-hypercube-style random exploration of the bounded design
space followed by a Nelder-Mead polish of the best point (the objective is
noisy-free but non-smooth at constraint edges; gradient-free is the robust
choice, and each evaluation is a full 1D regen solve, so budgets are kept
explicit).

Objective:  J = peak T_wg  +  w_dp * max(0, dP - budget)
with hard rejection (large J) for geometries whose lands close up at the
throat or whose regen solve fails (e.g., coolant pressure collapse).

Sanity anchor: on the shipped 5 kN LOX/CH4 example this optimizer converges
onto many narrow, deep channels — the classic High Aspect Ratio Cooling
Channel (HARCC) configuration whose wall-temperature *and* pressure-drop
benefit was demonstrated experimentally by NASA (Wadel & Meyer, AIAA
96-2584, 89 kN thrust chamber). The tool rediscovering HARCC from the
physics is the expected behavior, not a fluke.

Assumptions / limitations: inherits everything from
:mod:`cryosim.regen_model` (Bartz 1D, no boiling model, no thermal-stress
or LCF life constraint — a real design also limits wall dT for low-cycle
fatigue; that is reported but not constrained).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .chamber_geometry import ChamberContour, CoolingChannels
from .combustion import CombustionGas
from .fluids import Fluid
from .regen_model import RegenCoolingModel, RegenResult

#: Search bounds: (min, max) — deliberately below conventional machining.
DEFAULT_BOUNDS = {
    "n_channels": (20, 220),
    "channel_width": (0.3e-3, 3.0e-3),
    "channel_height": (0.5e-3, 6.0e-3),
    "t_wall": (0.2e-3, 1.5e-3),
}

#: Conventional-manufacturing floors used for WARNINGS (not constraints).
CONVENTIONAL_LIMITS = {
    "channel_width": (0.8e-3, "channel width < 0.8 mm: needs micro-milling/"
                              "wire-EDM slots or additive manufacturing"),
    "t_wall": (0.5e-3, "hot wall < 0.5 mm: needs electroformed or printed "
                       "closeout; verify burst and creep margins"),
    "land": (0.8e-3, "land < 0.8 mm at the throat: fin buckling and braze/"
                     "bond area become critical"),
}


@dataclass
class ChannelDesignResult:
    """Optimized channel layout and its predicted performance."""

    channels: CoolingChannels
    result: RegenResult
    peak_T_wg: float
    dp: float                     # Pa
    coolant_outlet_T: float
    land_at_throat: float
    objective: float
    n_evaluations: int
    manufacturability_warnings: list = field(default_factory=list)
    baseline: dict | None = None  # peak/dp of the user's starting design

    def report(self) -> str:
        ch = self.channels
        lines = [
            "Optimized cooling-channel design (physics optimum — check "
            "manufacturability warnings)",
            "=" * 72,
            f"  channels        : {ch.n_channels}",
            f"  width x height  : {ch.channel_width*1e3:.2f} x "
            f"{ch.channel_height*1e3:.2f} mm",
            f"  hot-wall thick. : {ch.t_wall*1e3:.2f} mm",
            f"  land at throat  : {self.land_at_throat*1e3:.2f} mm",
            f"  peak T_wg       : {self.peak_T_wg:.0f} K",
            f"  jacket dP       : {self.dp/1e5:.1f} bar",
            f"  coolant outlet  : {self.coolant_outlet_T:.0f} K",
            f"  regen solves    : {self.n_evaluations}",
        ]
        if self.baseline:
            lines.append(
                f"  vs baseline     : peak T_wg {self.baseline['peak']:.0f}"
                f" -> {self.peak_T_wg:.0f} K, dP "
                f"{self.baseline['dp']/1e5:.1f} -> {self.dp/1e5:.1f} bar"
            )
        if self.manufacturability_warnings:
            lines.append("  manufacturability:")
            lines += [f"    ! {w}" for w in self.manufacturability_warnings]
        else:
            lines.append("  manufacturability: within conventional practice")
        return "\n".join(lines)


def _manufacturability(ch: CoolingChannels, land: float) -> list[str]:
    warns = []
    lim, msg = CONVENTIONAL_LIMITS["channel_width"]
    if ch.channel_width < lim:
        warns.append(msg)
    lim, msg = CONVENTIONAL_LIMITS["t_wall"]
    if ch.t_wall < lim:
        warns.append(msg)
    lim, msg = CONVENTIONAL_LIMITS["land"]
    if land < lim:
        warns.append(msg)
    return warns


def optimize_channels(
    contour: ChamberContour,
    gas: CombustionGas,
    coolant: Fluid,
    Pc: float,
    mdot_coolant: float,
    T_inlet: float,
    P_inlet: float,
    dp_budget: float = 30e5,
    k_wall: float = 330.0,
    bounds: dict | None = None,
    n_random: int = 40,
    n_polish: int = 40,
    w_dp: float = 20.0 / 1e5,     # K of penalty per Pa over budget (20 K/bar)
    seed: int = 1,
    baseline: CoolingChannels | None = None,
) -> ChannelDesignResult:
    """Find the channel layout minimizing peak wall temperature.

    ``n_random`` scouting samples + up to ``n_polish`` Nelder-Mead
    evaluations; every evaluation is a full regen solve, so the total
    runtime is roughly (n_random + n_polish) x one solve.
    """
    b = dict(DEFAULT_BOUNDS)
    b.update(bounds or {})
    rng = np.random.default_rng(seed)
    evals = [0]
    min_land = 0.15e-3  # hard floor: below this the jacket doesn't exist

    def build(x) -> CoolingChannels | None:
        n, w, h, t = int(round(x[0])), x[1], x[2], x[3]
        try:
            ch = CoolingChannels(n_channels=n, channel_width=w,
                                 channel_height=h, t_wall=t, k_wall=k_wall)
        except ValueError:
            return None
        if ch.land_width(contour.Rt) < min_land:
            return None
        return ch

    def evaluate(x):
        ch = build(x)
        if ch is None:
            return 1e6, None, None
        evals[0] += 1
        try:
            model = RegenCoolingModel(contour, ch, gas, coolant)
            res = model.solve(Pc, mdot_coolant, T_inlet, P_inlet)
        except Exception:
            return 1e6, ch, None
        if (res.boiling_detected or res.pressure_collapsed
                or res.coolant_outlet.P < 0.3 * P_inlet):
            return 5e5 + res.peak_wall_temperature, ch, res
        J = res.peak_wall_temperature + w_dp * max(0.0, res.dP_total - dp_budget)
        return J, ch, res

    keys = ("n_channels", "channel_width", "channel_height", "t_wall")
    lo = np.array([b[k][0] for k in keys], float)
    hi = np.array([b[k][1] for k in keys], float)

    # -------- stage 1: stratified random scouting (LHS-style per axis)
    samples = lo + (hi - lo) * rng.random((n_random, 4))
    if baseline is not None:
        x0 = np.array([baseline.n_channels, baseline.channel_width,
                       baseline.channel_height, baseline.t_wall])
        samples = np.vstack([x0, samples])
    best = (np.inf, None, None, None)
    for x in samples:
        J, ch, res = evaluate(x)
        if res is not None and J < best[0]:
            best = (J, x.copy(), ch, res)

    if best[2] is None or best[3] is None:
        raise RuntimeError("no feasible channel design found in the sampled "
                           "space; relax bounds or dp budget")

    # -------- stage 2: Nelder-Mead polish (bounded via clipping)
    def penalized(x):
        x = np.clip(x, lo, hi)
        return evaluate(x)[0]

    sol = minimize(penalized, best[1], method="Nelder-Mead",
                   options={"maxfev": n_polish, "xatol": 1e-5,
                            "fatol": 0.5})
    x_fin = np.clip(sol.x, lo, hi)
    J_fin, ch_fin, res_fin = evaluate(x_fin)
    if res_fin is None or J_fin > best[0]:
        J_fin, x_fin, ch_fin, res_fin = best

    base_metrics = None
    if baseline is not None:
        try:
            bres = RegenCoolingModel(contour, baseline, gas, coolant).solve(
                Pc, mdot_coolant, T_inlet, P_inlet)
            base_metrics = {"peak": bres.peak_wall_temperature,
                            "dp": bres.dP_total}
        except Exception:
            base_metrics = None

    land = ch_fin.land_width(contour.Rt)
    return ChannelDesignResult(
        channels=ch_fin,
        result=res_fin,
        peak_T_wg=res_fin.peak_wall_temperature,
        dp=res_fin.dP_total,
        coolant_outlet_T=res_fin.coolant_outlet.T,
        land_at_throat=land,
        objective=J_fin,
        n_evaluations=evals[0],
        manufacturability_warnings=_manufacturability(ch_fin, land),
        baseline=base_metrics,
    )
