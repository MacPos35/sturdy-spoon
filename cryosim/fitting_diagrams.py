"""Fitting cross-section diagrams + line-spec table.

Renders a one-page figure with simplified cross-section drawings of the
joint types the line-sizing rules select — 37 deg flare (AN/MS),
twin-ferrule compression, full-penetration butt weld — plus the rejected
NPT tapered-thread joint (marked, with the cryogenic-leak reason), and a
table of every sized line (tube, ID, velocity, MAWP, fitting choice).

The drawings are educational schematics of how each joint seals, not
manufacturing drawings.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .line_sizing import LineSpec

INK = "#222222"
HATCH_KW = dict(facecolor="#e8e8e8", edgecolor=INK, lw=1.1, hatch="////")
OUTLINE_KW = dict(facecolor="white", edgecolor=INK, lw=1.1)


def _mirror(ax, pts, **kw):
    """Draw a polygon and its mirror about y=0 (half-section symmetry)."""
    pts = np.asarray(pts, float)
    ax.add_patch(Polygon(pts, closed=True, **kw))
    ax.add_patch(Polygon(pts * [1, -1], closed=True, **kw))


def _centerline(ax, x0, x1):
    ax.plot([x0, x1], [0, 0], color="#888", lw=0.7, ls=(0, (8, 3, 2, 3)))


def _label(ax, x, y, text, tx, ty):
    ax.annotate(text, (x, y), (tx, ty), fontsize=6.0,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#555"),
                ha="center", va="center")


def _panel_flare(ax):
    """37 deg flare (AN/MS) joint, half-section."""
    ri, ro = 0.8, 1.1
    # tube wall, flaring out at 37 deg at the end
    fl = np.tan(np.radians(37.0))
    xf, Lf = 4.0, 0.9  # flare start, axial length
    _mirror(ax, [(0, ri), (xf, ri), (xf + Lf, ri + Lf * fl),
                 (xf + Lf, ri + Lf * fl + 0.3), (xf - 0.05, ro), (0, ro)],
            **HATCH_KW)
    # union body with matching 37 deg cone nose
    _mirror(ax, [(xf + Lf, ri + Lf * fl), (xf, ri), (xf, 0.55), (9.0, 0.55),
                 (9.0, 2.0), (xf + Lf, 2.0)], **HATCH_KW)
    # union bore
    _mirror(ax, [(xf, 0.55), (9.0, 0.55), (9.0, ri), (xf + 0.4, ri)],
            facecolor="white", edgecolor="none")
    # external thread ticks on the union, right of the B-nut
    for xt in np.arange(7.0, 8.4, 0.28):
        ax.plot([xt, xt + 0.14], [2.0, 2.25], color=INK, lw=0.8)
        ax.plot([xt, xt + 0.14], [-2.0, -2.25], color=INK, lw=0.8)
    # sleeve + B-nut outline
    _mirror(ax, [(2.6, ro + 0.05), (xf + 0.4, ro + 0.05), (xf + 0.4, 1.75),
                 (2.6, 1.75)], **OUTLINE_KW)
    _mirror(ax, [(2.0, 1.75), (6.8, 1.75), (6.8, 2.45), (2.0, 2.45)],
            **OUTLINE_KW)
    _centerline(ax, -0.3, 9.3)
    # 37 deg callout
    ax.annotate("37° sealing cone", (xf + 0.5, ri + 0.5 * Lf * fl),
                (2.2, -3.3), fontsize=6.0,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"),
                ha="center")
    _label(ax, 1.0, 0.95, "tube (flared end)", 1.2, 3.3)
    _label(ax, 4.4, 2.4, "B-nut", 4.4, 3.4)
    _label(ax, 8.0, 1.3, "union body", 8.0, 3.3)
    ax.set_title("37° flare (AN/MS) — cryogenic lines", fontsize=7.5)


def _panel_compression(ax):
    """Twin-ferrule compression joint, half-section."""
    ri, ro = 0.8, 1.1
    _mirror(ax, [(0, ri), (6.2, ri), (6.2, ro), (0, ro)], **HATCH_KW)
    # body
    _mirror(ax, [(5.0, ro), (9.0, ro), (9.0, 2.1), (5.6, 2.1), (5.0, 1.6)],
            **HATCH_KW)
    # front + back ferrules (wedges biting on the tube)
    _mirror(ax, [(4.5, ro), (5.4, ro), (4.9, 1.55)],
            facecolor="#bfcfe0", edgecolor=INK, lw=1.0)
    _mirror(ax, [(3.8, ro), (4.45, ro), (4.3, 1.35)],
            facecolor="#bfcfe0", edgecolor=INK, lw=1.0)
    # nut
    _mirror(ax, [(2.9, ro + 0.02), (5.2, ro + 0.02), (5.6, 1.6), (5.6, 2.05),
                 (2.9, 2.05)], **OUTLINE_KW)
    _centerline(ax, -0.3, 9.3)
    _label(ax, 4.9, 1.35, "front ferrule", 2.4, 3.2)
    _label(ax, 4.3, 1.2, "back ferrule", 4.6, -3.2)
    _label(ax, 4.0, 1.95, "nut", 6.6, 3.2)
    _label(ax, 8.2, 1.5, "body", 8.2, 3.2)
    ax.set_title("twin-ferrule compression — gas / instrument lines",
                 fontsize=7.5)


def _panel_weld(ax):
    """Full-penetration butt weld, half-section."""
    ri, ro = 0.8, 1.1
    _mirror(ax, [(0, ri), (4.1, ri), (3.8, ro), (0, ro)], **HATCH_KW)
    _mirror(ax, [(4.9, ri), (9.0, ri), (9.0, ro), (5.2, ro)], **HATCH_KW)
    # weld bead filling the V-groove with cap
    _mirror(ax, [(4.1, ri), (4.9, ri), (5.3, ro + 0.18), (3.7, ro + 0.18)],
            facecolor="#9a9a9a", edgecolor=INK, lw=1.0)
    _centerline(ax, -0.3, 9.3)
    _label(ax, 4.5, 1.15, "full-penetration bead\n(orbital GTAW)", 4.5, 3.3)
    ax.set_title("butt weld — permanent runs, OD ≥ 1\"", fontsize=7.5)


def _panel_npt(ax):
    """NPT tapered thread — rejected for cryogenic service."""
    ri, ro = 0.8, 1.1
    taper = 0.10
    _mirror(ax, [(0, ri), (4.6, ri), (4.6, ro + 0.55 - taper * 4.6), (0, ro)],
            **HATCH_KW)
    _mirror(ax, [(4.0, ri), (9.0, ri), (9.0, 2.0), (4.0, 2.0)], **HATCH_KW)
    _mirror(ax, [(4.0, ri), (8.0, ri), (8.0, ri + 0.02), (4.0, ri + 0.02)],
            facecolor="white", edgecolor="none")
    for xt in np.arange(1.6, 4.5, 0.34):
        yt = ro + 0.55 - taper * xt
        ax.plot([xt, xt + 0.17], [yt, yt + 0.22], color=INK, lw=0.8)
        ax.plot([xt, xt + 0.17], [-yt, -yt - 0.22], color=INK, lw=0.8)
    _centerline(ax, -0.3, 9.3)
    ax.plot([1.2, 7.8], [-2.9, 2.9], color="#b03030", lw=2.2)
    ax.plot([1.2, 7.8], [2.9, -2.9], color="#b03030", lw=2.2)
    ax.set_title("NPT tapered thread — REJECTED for cryo\n"
                 "(sealant-dependent, leaks on thermal cycling)",
                 fontsize=7.0, color="#b03030")


def draw_fittings(specs: list[LineSpec], path: str) -> str:
    """Render fitting cross-sections + the sized-line table."""
    n_rows = len(specs)
    fig = plt.figure(figsize=(11.5, 4.4 + 0.24 * n_rows))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.55 + 0.11 * n_rows],
                          hspace=0.02, top=0.88, bottom=0.04)

    panels = (_panel_flare, _panel_compression, _panel_weld, _panel_npt)
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-4.2, 4.2)
        ax.set_aspect("equal")
        ax.axis("off")
        panel(ax)

    # ------------------------------------------------------------- table
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    cols = ["line", "tube (OD x wall)", "ID", "vel", "MAWP", "dash",
            "fitting"]
    rows = []
    for s in specs:
        fit_short = ("weld/flange" if "flange" in s.fitting
                     else "compression/flare" if "compression" in s.fitting
                     else "AN flare/weld")
        rows.append([
            s.name,
            f'{s.od_in:.3g}" x {s.wall_in:.3f}" {s.material}',
            f"{s.id_*1e3:.1f} mm",
            f"{s.velocity:.1f} m/s",
            f"{s.mawp/1e5:.0f} bar",
            f"-{s.dash}",
            fit_short,
        ])
    tab = axt.table(cellText=rows, colLabels=cols, loc="upper center",
                    cellLoc="left", colLoc="left",
                    colWidths=[0.24, 0.18, 0.08, 0.08, 0.08, 0.06, 0.20])
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.6)
    tab.scale(1.0, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f0f0f0")
    axt.set_title(
        "auto-sized lines (velocity targets + Barlow wall; "
        "verify surge, bends and vendor ratings before ordering)",
        fontsize=7.5,
    )
    fig.suptitle("Fitting selection & line sizing", fontsize=11,
                 fontweight="bold")
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return path
