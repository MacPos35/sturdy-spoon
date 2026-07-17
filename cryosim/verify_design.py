"""Post-design independent verification (the generate->verify pattern).

Industry generative-design practice — and LEAP 71's Noyron workflow — does
not stop at generation: the produced geometry is handed to independent
analysis and the result checked back against the design intent. This
module is that step, scaled to cryosim's reduced-order tier. After the
ledger converges, the FINAL design is re-checked by:

1. **Thrust closure** — ``F = Cf Pc At`` must reproduce the requirement
   exactly (an identity of stage A; guards bookkeeping regressions).
2. **Regen re-march** — the cooling solution is re-solved directly on the
   final channel geometry (outside the optimizer) and must reproduce the
   stored peak wall temperature and pressure drop.
3. **Euler CFD mass-flow check** (closed nozzles) — the inviscid
   axisymmetric solver (:mod:`cryosim.cfd_nozzle`) runs on the actual
   designed contour; its mass flow must agree with the quasi-1D
   area-ratio assumption the sizing used, within 5% (the documented 2-D
   throat effect is ~3% on these grids). Skipped for aerospikes: the
   solver meshes closed axisymmetric ducts, not external plug flows.

This is verification of the *reduced-order tier against itself and one
independent flow solution* — not a substitute for the RANS/FEA campaign a
flight design gets (see the README fidelity-tier table).
"""

from __future__ import annotations

import numpy as np


def _reconstruct_film(design):
    """Rebuild the FilmCooling credit exactly as the pipeline did."""
    from .fluids import Fluid
    from .regen_model import FilmCooling
    spec = design.spec
    if not (spec.credit_film and spec.film_fraction > 0):
        return None
    fuel_name = spec.propellants.split("/", 1)[1]
    T_film = 600.0
    try:
        cp_film = Fluid(fuel_name).state_TP(T_film,
                                            spec.chamber_pressure).cp
    except ValueError:
        cp_film = 2500.0
    return FilmCooling(mdot=spec.film_fraction * design.mdot_fuel,
                       T_inject=T_film, cp=cp_film)


def verify_design(design, log) -> list:
    """Run the independent checks; returns ledger items (all must pass)."""
    from .engine_design import LedgerItem
    from .fluids import Fluid
    from .regen_model import RegenCoolingModel

    spec = design.spec
    Pc = spec.chamber_pressure
    items: list = []

    # ---- 1. thrust closure ------------------------------------------------
    At = np.pi * design.throat_radius ** 2
    F = design.Cf * Pc * At
    err_F = abs(F - spec.thrust) / spec.thrust
    items.append(LedgerItem(
        "verify: thrust closure", err_F < 1e-6,
        f"{F/1e3:.3f} kN ({err_F*100:.1e}% off)",
        "F = Cf Pc At reproduces the requirement"))
    log("V. verification", "thrust closure",
        f"Cf x Pc x At = {F/1e3:.3f} kN vs required "
        f"{spec.thrust/1e3:.3f} kN ({err_F*100:.1e}% error)")

    # ---- 2. regen re-march on the final geometry --------------------------
    ox_name, fuel_name = spec.propellants.split("/", 1)
    film = _reconstruct_film(design)
    circuits = [("cowl" if design.aerospike is not None else "jacket",
                 design.channel_design, Fluid(fuel_name),
                 design.mdot_fuel, film)]
    if design.aerospike is not None:
        circuits.append(("spike", design.spike_channel_design,
                         Fluid(ox_name), design.mdot_ox, None))
    for tag, cd, fluid, mdot_c, f_credit in circuits:
        contour = design.aerospike.cowl if tag == "cowl" else (
            design.aerospike.inner if tag == "spike" else design.contour)
        model = RegenCoolingModel(contour, cd.channels, design.gas, fluid,
                                  film=f_credit,
                                  bartz_factor=spec.bartz_factor)
        res = model.solve(Pc, mdot_c, cd.result.coolant_inlet.T,
                          cd.result.coolant_inlet.P)
        dT = abs(res.peak_wall_temperature - cd.peak_T_wg)
        ddp = abs(res.dP_total - cd.dp) / max(cd.dp, 1.0)
        ok = dT < 1.0 and ddp < 0.01
        items.append(LedgerItem(
            f"verify: {tag} regen re-march", ok,
            f"dT_peak {dT:.2f} K, d(dP) {ddp*100:.2f}%",
            "independent re-solve reproduces the optimizer's solution"))
        log("V. verification", f"{tag} regen re-march",
            f"peak T_wg {res.peak_wall_temperature:.1f} K vs stored "
            f"{cd.peak_T_wg:.1f} K; dP {res.dP_total/1e5:.2f} vs "
            f"{cd.dp/1e5:.2f} bar")

    # ---- 3. Euler CFD mass-flow check (closed nozzles only) ---------------
    if design.aerospike is None:
        from .cfd_nozzle import NozzleEulerCFD
        cfd = NozzleEulerCFD(design.contour, design.gas, Pc,
                             n_axial=100, n_radial=20)
        res = cfd.run(max_iter=6000)
        _, mdot_x = res.mdot_profile()
        mdot_cfd = float(np.mean(mdot_x[5:-5]))
        mdot_q1 = res.mdot_quasi1d()
        dev = (mdot_cfd - mdot_q1) / mdot_q1
        items.append(LedgerItem(
            "verify: Euler CFD mass flow", abs(dev) <= 0.05,
            f"{dev*100:+.1f}% vs quasi-1D",
            "<= 5% (2-D flow on the actual contour)"))
        log("V. verification", "Euler CFD",
            f"inviscid axisymmetric solve on the designed contour "
            f"({'converged' if res.converged else 'residual floor'}): "
            f"mass flow {mdot_cfd:.3f} vs quasi-1D {mdot_q1:.3f} kg/s "
            f"({dev*100:+.1f}%); first-order scheme, ~3% throat "
            "smearing is documented")
    else:
        log("V. verification", "Euler CFD",
            "skipped: the solver meshes closed axisymmetric nozzles; the "
            "aerospike gas path is covered by the regen re-march and the "
            "Angelino continuity closure")

    return items
