"""Deliverables for an autonomous engine design: STLs, figures, report.

Turns a converged :class:`~cryosim.engine_design.EngineDesign` into the
LEAP 71-style output package:

* ``chamber_jacket.stl`` / ``injector_head.stl`` / ``engine_assembly.stl``
  — watertight binary STL (mm units) meshed from the implicit model
  (:mod:`cryosim.voxel_geometry`);
* ``cross_section.png`` — annotated meridional section of the design;
* ``injector_face.png`` — element/film-hole layout on the face;
* ``wall_temperature.png`` — regen solution along the contour (reuses
  :func:`cryosim.plots.plot_regen_distribution`);
* ``trace.md`` — the full explainable decision record;
* ``report.md`` — predicted performance, constraint ledger, mesh QA and
  links to everything above.

Meshing fidelity is voxel-limited (stated per part in the QA lines);
``voxel_jacket_mm``/``voxel_injector_mm`` trade resolution for time/memory.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine_design import EngineDesign, LINER_MATERIALS, PROCESSES
from .plots import plot_regen_distribution
from .voxel_geometry import (MeshQA, build_chamber_jacket,
                             build_injector_head, concat_meshes, mesh_qa,
                             mesh_solid)

#: Bulk densities for the printed-mass estimate [kg/m^3].
PART_DENSITY = {"chamber_jacket": None, "injector_head": 7980.0}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def draw_cross_section(design: EngineDesign, path: str) -> None:
    """Meridional section: liner, channels band, closeout, manifolds."""
    c = design.contour
    ch = design.channel_design.channels
    man = design.manifolds
    x, r = c.x * 1e3, c.r * 1e3
    t_w, h = ch.t_wall * 1e3, ch.channel_height * 1e3
    t_c = design.t_closeout * 1e3

    fig, ax = plt.subplots(figsize=(9, 5), layout="tight")
    for sgn in (+1, -1):
        ax.fill_between(x, sgn * r, sgn * (r + t_w),
                        color="#b87333", label="liner (hot wall)"
                        if sgn > 0 else None)
        ax.fill_between(x, sgn * (r + t_w), sgn * (r + t_w + h),
                        color="#9ecae1",
                        label="cooling channels" if sgn > 0 else None)
        ax.fill_between(x, sgn * (r + t_w + h), sgn * (r + t_w + h + t_c),
                        color="#969696",
                        label="closeout" if sgn > 0 else None)
    for m, xc in ((man.inlet, x[-1]), (man.outlet, x[0])):
        R = m.r_centerline * 1e3
        d = m.duct_diameter * 1e3
        tw = m.wall_thickness * 1e3
        for sgn in (+1, -1):
            ax.add_patch(plt.Circle((xc, sgn * R), d / 2 + tw,
                                    color="#969696"))
            ax.add_patch(plt.Circle((xc, sgn * R), d / 2, color="white"))
    e = design.injector.element
    H = (e.L_vortex + e.L_nozzle) * 1e3 + 10.0
    face_r = design.injector.face_radius * 1e3
    ax.fill_between([-H, 0], -face_r - 4, face_r + 4, color="#d9d9d9",
                    label="injector head (envelope)")
    ax.axvline(x[np.argmin(r)], ls=":", c="k", lw=0.8)
    ax.set_xlabel("x from injector face [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(f"'{design.spec.name}' meridional section — "
                 f"{design.spec.thrust/1e3:.1f} kN {design.spec.propellants}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_injector_face(design: EngineDesign, path: str) -> None:
    inj = design.injector
    e, a = inj.element, inj.annulus
    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    ax.add_patch(plt.Circle((0, 0), inj.face_radius * 1e3, fill=False,
                            color="k", lw=1.2))
    for r_ring, n_on in inj.rings:
        for k in range(n_on):
            ang = 2 * np.pi * k / max(n_on, 1)
            cy, cz = r_ring * np.cos(ang) * 1e3, r_ring * np.sin(ang) * 1e3
            ax.add_patch(plt.Circle((cy, cz), a.r_outer * 1e3,
                                    color="#9ecae1"))
            ax.add_patch(plt.Circle((cy, cz), a.r_inner * 1e3,
                                    color="white"))
            ax.add_patch(plt.Circle((cy, cz), e.r_nozzle * 1e3,
                                    color="#b87333"))
    if inj.film is not None:
        f = inj.film
        for k in range(f.n_holes):
            ang = 2 * np.pi * k / f.n_holes
            ax.add_patch(plt.Circle((f.ring_radius * np.cos(ang) * 1e3,
                                     f.ring_radius * np.sin(ang) * 1e3),
                                    max(f.d_hole / 2 * 1e3, 0.3),
                                    color="#31a354"))
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"injector face: {inj.n_elements} coax-swirl elements"
                 + ("" if inj.film is None else
                    f" + {inj.film.n_holes} film holes"))
    lim = inj.face_radius * 1e3 * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------
# Package generation
# ----------------------------------------------------------------------

def generate_package(design: EngineDesign, outdir: str,
                     voxel_jacket_mm: float = 0.5,
                     voxel_injector_mm: float = 0.3,
                     with_geometry: bool = True,
                     verbose: bool = False) -> dict:
    """Write the full output package; returns {artifact: path}."""
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}
    qa: list[MeshQA] = []

    def say(msg):
        if verbose:
            print(f"  {msg}")

    # ---- figures ---------------------------------------------------------
    p = os.path.join(outdir, "cross_section.png")
    draw_cross_section(design, p)
    paths["cross_section"] = p
    p = os.path.join(outdir, "injector_face.png")
    draw_injector_face(design, p)
    paths["injector_face"] = p
    p = os.path.join(outdir, "wall_temperature.png")
    cd = design.channel_design
    x_throat = design.contour.x[np.argmin(design.contour.r)]
    plot_regen_distribution(cd.result, x_throat, p,
                            title=f"'{design.spec.name}' regen solution")
    paths["wall_temperature"] = p
    say("figures written")

    # ---- geometry ----------------------------------------------------------
    if with_geometry:
        liner_rho = LINER_MATERIALS[design.spec.liner]["rho"]
        say(f"meshing chamber jacket at {voxel_jacket_mm} mm voxels ...")
        solid, lo, hi = build_chamber_jacket(
            design.contour, cd.channels, design.t_closeout, design.manifolds)
        jacket = mesh_solid(solid, lo, hi, voxel_jacket_mm * 1e-3)
        p = os.path.join(outdir, "chamber_jacket.stl")
        jacket.save_stl(p, b"cryosim chamber jacket")
        paths["chamber_jacket"] = p
        qa.append(mesh_qa(jacket, "chamber_jacket.stl", rho=liner_rho))
        say(qa[-1].describe())

        say(f"meshing injector head at {voxel_injector_mm} mm voxels ...")
        solid_i, lo_i, hi_i = build_injector_head(design.injector)
        head = mesh_solid(solid_i, lo_i, hi_i, voxel_injector_mm * 1e-3)
        p = os.path.join(outdir, "injector_head.stl")
        head.save_stl(p, b"cryosim injector head")
        paths["injector_head"] = p
        qa.append(mesh_qa(head, "injector_head.stl",
                          rho=PART_DENSITY["injector_head"]))
        say(qa[-1].describe())

        say("assembling multi-shell engine ...")
        # combine the two already-watertight shells rather than re-meshing
        # the union at a compromise voxel (which would under-resolve the
        # injector's fine features and pinch the topology)
        asm = concat_meshes([jacket, head], voxel_jacket_mm * 1e-3)
        p = os.path.join(outdir, "engine_assembly.stl")
        asm.save_stl(p, b"cryosim engine assembly")
        paths["engine_assembly"] = p
        qa.append(mesh_qa(asm, "engine_assembly.stl", rho=None))
        say(qa[-1].describe())

    # ---- trace + report ----------------------------------------------------
    p = os.path.join(outdir, "trace.md")
    with open(p, "w") as fh:
        fh.write(design.trace.as_markdown())
    paths["trace"] = p

    p = os.path.join(outdir, "report.md")
    with open(p, "w") as fh:
        fh.write(render_report(design, qa, with_geometry))
    paths["report"] = p
    say("report written")
    return paths


#: Conventional-machining fine-feature thresholds [m] (below → needs a
#: fine-feature process such as LPBF; used only to frame the notes).
_CONVENTIONAL = {"channel width": 0.8e-3, "hot wall": 0.5e-3,
                 "throat land": 0.8e-3}

#: injector-warning substrings that are modeling caveats, not defects.
_MODEL_CAVEATS = ("approximate", "gas-like", "viscous losses", "Re ")


def _manufacturability(design: EngineDesign) -> tuple[list, list]:
    """Split manufacturability into genuine warnings vs process notes.

    Process-aware: a feature that meets the *selected* process's floor is
    manufacturable by that process, so it becomes an informational note
    ("requires LPBF") rather than a warning — the design deliberately targets
    LPBF, so flagging "needs additive manufacturing" would be spurious. Only a
    feature below the selected process's own floor is a real warning.
    """
    proc = PROCESSES[design.spec.process]
    pname = design.spec.process.upper()
    ch = design.channel_design.channels
    feats = [
        ("channel width", ch.channel_width, proc["min_channel_width"]),
        ("hot wall", ch.t_wall, proc["min_wall"]),
        ("throat land", design.channel_design.land_at_throat,
         proc["min_land"]),
    ]
    warnings, notes = [], []
    for name, val, floor in feats:
        if val < floor - 1e-9:
            warnings.append(
                f"{name} {val*1e3:.2f} mm is below the {pname} floor "
                f"({floor*1e3:.2f} mm) — infeasible as-is")
        elif val < _CONVENTIONAL[name]:
            notes.append(
                f"{name} {val*1e3:.2f} mm requires {pname} (below the "
                f"{_CONVENTIONAL[name]*1e3:.1f} mm conventional-machining "
                "limit — as intended for this process)")
    # injector: separate real issues from modeling caveats
    for w in design.injector.warnings:
        (notes if any(s in w for s in _MODEL_CAVEATS)
         else warnings).append(w)
    return warnings, notes


def render_report(design: EngineDesign, qa: list[MeshQA],
                  with_geometry: bool) -> str:
    s = design.spec
    lines = [
        f"# Engine design report — '{s.name}'",
        "",
        f"Autonomously generated by `cryosim design` from top-level "
        f"requirements only ({s.thrust/1e3:.1f} kN, {s.propellants}, "
        f"Pc {s.chamber_pressure/1e5:.0f} bar, ambient "
        f"{s.ambient_pressure/1e3:.1f} kPa). The design procedure is "
        "deterministic encoded engineering logic — every decision is in "
        "[trace.md](trace.md). Reduced-order models throughout; see the "
        "README validation table before trusting numbers.",
        "",
        "## Predicted performance",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in design.performance_table()]
    lines += ["", "## Constraint ledger", ""]
    for item in design.ledger:
        mark = "✅" if item.ok else "❌"
        lines.append(f"- {mark} **{item.name}**: {item.value} "
                     f"(requirement: {item.requirement})")
    warnings, notes = _manufacturability(design)
    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- ⚠️ {w}" for w in warnings]
    if notes:
        lines += ["", f"## Manufacturing & modeling notes "
                  f"(process: {design.spec.process.upper()})", ""]
        lines += [f"- {n}" for n in notes]
    if with_geometry:
        lines += ["", "## Generated geometry (binary STL, mm)", ""]
        for q in qa:
            lines.append(f"- `{q.name}` — {q.n_triangles} triangles, "
                         f"{'watertight' if q.watertight else '**NOT WATERTIGHT**'}, "
                         f"volume {q.volume_cm3:.0f} cm³"
                         + (f", est. mass {q.mass_kg:.2f} kg"
                            if q.mass_kg is not None else "")
                         + f", voxel {q.voxel_mm:.2f} mm")
        lines += ["",
                  "Voxel-limited fidelity: features under ~2 voxels are "
                  "rounded or closed; re-mesh finer for print prep.",
                  ""]
    lines += [
        "## Figures",
        "",
        "![cross section](cross_section.png)",
        "",
        "![injector face](injector_face.png)",
        "",
        "![wall temperature](wall_temperature.png)",
        "",
        "## Design trace",
        "",
        f"{len(design.trace.entries)} decisions recorded — see "
        "[trace.md](trace.md).",
        "",
    ]
    return "\n".join(lines)
