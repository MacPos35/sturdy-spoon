"""ISA-5.1-flavored feed-system schematic renderer.

Draws the recommendation of :mod:`cryosim.pid_recommender` as a proper
piping & instrumentation style diagram: drawn valve/instrument symbols,
routed process lines (tank overhead manifolds with vent + relief + burst
disc, pressurant header with per-tank check valves, feed trains through
filter and main valve to the engine, regen coolant loop up the jacket to
the injector), dashed instrument signal leaders, a symbol legend, and a
title block.

Layout is deterministic per configuration (regulated / blowdown /
autogenous / pump-fed; N tanks; regen on/off) rather than auto-routed —
the recommender's rule set and this renderer describe the same canonical
architecture by construction.

Line color encodes the working fluid (oxidizer / fuel / pressurant), a
common modern P&ID practice; symbols and text stay monochrome. Still a
design/checklist aid — NOT a safety-reviewed P&ID (printed in the title
block).
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

INK = "#222222"
C_OX = "#0072B2"      # oxidizer lines
C_FUEL = "#E69F00"    # fuel / coolant lines
C_PRESS = "#6e6e6e"   # pressurant lines
C_SIG = "#999999"     # instrument signals
LW = 1.4


class Schematic:
    """Symbol library + canvas; collects legend entries as symbols are used."""

    def __init__(self, width=178.0, height=126.0):
        self.fig, self.ax = plt.subplots(figsize=(width / 12.5, height / 12.5))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.W, self.H = width, height
        self._legend_used: dict[str, str] = {}

    # ------------------------------------------------------------ plumbing
    def pipe(self, pts, color=INK, lw=LW):
        pts = np.asarray(pts, float)
        self.ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw,
                     solid_capstyle="round", zorder=1)

    def signal(self, pts):
        pts = np.asarray(pts, float)
        self.ax.plot(pts[:, 0], pts[:, 1], color=C_SIG, lw=0.9, ls=(0, (3, 2)),
                     zorder=1)

    def junction(self, x, y, color=INK):
        self.ax.add_patch(Circle((x, y), 0.55, color=color, zorder=3))

    def flow_arrow(self, x, y, ang_deg, color=INK, s=1.5):
        a = np.radians(ang_deg)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        p = np.array([x, y])
        tri = [p + d * s, p - d * s * 0.6 + n * s * 0.6,
               p - d * s * 0.6 - n * s * 0.6]
        self.ax.add_patch(Polygon(tri, closed=True, color=color, zorder=3))

    def vent_to_atm(self, x, y, ang_deg=90):
        """Open vent: three diverging strokes at a line end."""
        a = np.radians(ang_deg)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        p = np.array([x, y])
        for off in (-0.9, 0.0, 0.9):
            q = p + n * off
            self.ax.plot([q[0], q[0] + d[0] * 2.2], [q[1], q[1] + d[1] * 2.2],
                         color=INK, lw=1.0)
        self._legend_used["atm"] = "vent to atmosphere (safe direction)"

    def tag(self, x, y, text, size=5.2, ha="center", va="center", color=INK,
            weight="normal"):
        self.ax.text(x, y, text, fontsize=size, ha=ha, va=va, color=color,
                     fontweight=weight, zorder=5)

    # ------------------------------------------------------------- symbols
    def _bowtie(self, x, y, s, vertical=False):
        if vertical:
            t1 = [(x - s * 0.8, y - s), (x + s * 0.8, y - s), (x, y)]
            t2 = [(x - s * 0.8, y + s), (x + s * 0.8, y + s), (x, y)]
        else:
            t1 = [(x - s, y - s * 0.8), (x - s, y + s * 0.8), (x, y)]
            t2 = [(x + s, y - s * 0.8), (x + s, y + s * 0.8), (x, y)]
        for t in (t1, t2):
            self.ax.add_patch(Polygon(t, closed=True, facecolor="white",
                                      edgecolor=INK, lw=1.1, zorder=4))

    def valve_manual(self, x, y, tag=None, s=2.0, vertical=False):
        self._bowtie(x, y, s, vertical)
        if vertical:
            self.ax.plot([x, x + s * 1.4], [y, y], color=INK, lw=1.0, zorder=4)
            self.ax.plot([x + s * 1.4] * 2, [y - s * 0.7, y + s * 0.7],
                         color=INK, lw=1.0, zorder=4)
            self._tag_beside(x, y, tag, dx=s * 2.4)
        else:
            self.ax.plot([x, x], [y, y + s * 1.4], color=INK, lw=1.0, zorder=4)
            self.ax.plot([x - s * 0.7, x + s * 0.7], [y + s * 1.4] * 2,
                         color=INK, lw=1.0, zorder=4)
            self._tag_beside(x, y, tag, dy=s * 2.6)
        self._legend_used["valve_manual"] = \
            "manual valve (bowtie, T-handle) — fill/drain, isolation"

    def valve_remote(self, x, y, tag=None, s=2.0, vertical=False, act="P"):
        """Actuated valve; actuator letter: P pneumatic, M motor, S solenoid."""
        self._bowtie(x, y, s, vertical)
        if vertical:
            self.ax.plot([x, x + s * 1.2], [y, y], color=INK, lw=1.0, zorder=4)
            bx, by = x + s * 1.2 + s * 0.9, y
        else:
            self.ax.plot([x, x], [y, y + s * 1.2], color=INK, lw=1.0, zorder=4)
            bx, by = x, y + s * 1.2 + s * 0.75
        self.ax.add_patch(Rectangle((bx - s * 0.9, by - s * 0.7), s * 1.8,
                                    s * 1.4, facecolor="white", edgecolor=INK,
                                    lw=1.0, zorder=4))
        self.tag(bx, by, act, size=4.6)
        self._tag_beside(x, y, tag, dy=-s * 1.9 if not vertical else 0,
                         dx=0 if not vertical else -s * 2.6)
        self._legend_used["valve_remote"] = \
            "remote actuated valve (P=pneumatic) — mains, vents, isolation"

    def check_valve(self, x, y, tag=None, s=1.9, direction=(1, 0)):
        d = np.array(direction, float)
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        p = np.array([x, y])
        seat = p + d * s * 0.75
        self.ax.plot(*zip(seat + n * s * 0.85, seat - n * s * 0.85),
                     color=INK, lw=1.6, zorder=4)
        tri = [p - d * s * 0.9 + n * s * 0.8, p - d * s * 0.9 - n * s * 0.8,
               seat]
        self.ax.add_patch(Polygon(tri, closed=True, facecolor="white",
                                  edgecolor=INK, lw=1.1, zorder=4))
        self._tag_beside(x, y, tag, dy=s * 2.0 if abs(d[0]) > 0.5 else 0,
                         dx=s * 2.6 if abs(d[1]) > 0.5 else 0)
        self._legend_used["check_valve"] = \
            "check valve (flap onto seat bar; flow with the triangle)"

    def relief_valve(self, x, y, tag=None, s=2.0, outlet="left"):
        """Spring relief: inlet from below, outlet horizontal at valve body."""
        self._bowtie(x, y, s, vertical=True)
        # spring above body
        zz_x = x + np.array([0, -0.7, 0.7, -0.7, 0.7, 0]) * s * 0.5
        zz_y = y + s + np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0]) * s * 0.55
        self.ax.plot(zz_x, zz_y, color=INK, lw=1.0, zorder=4)
        dx = -1 if outlet == "left" else 1
        self.pipe([(x, y), (x + dx * s * 2.2, y)])
        self.vent_to_atm(x + dx * s * 2.2, y, ang_deg=180 if dx < 0 else 0)
        self._tag_beside(x, y, tag, dx=-dx * s * 2.8)
        self._legend_used["relief_valve"] = \
            "spring relief valve — primary overpressure protection"

    def burst_disc(self, x, y, tag=None, s=1.7):
        """Rupture disc on a vertical line: domed membrane in a square body."""
        self.ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s,
                                    facecolor="white", edgecolor=INK, lw=1.1,
                                    zorder=4))
        self.ax.add_patch(Arc((x, y - s * 0.45), 2 * s * 0.85, 2 * s * 0.9,
                              theta1=25, theta2=155, color=INK, lw=1.2,
                              zorder=5))
        self._tag_beside(x, y, tag, dx=s * 2.6)
        self._legend_used["burst_disc"] = \
            "burst disc — redundant, non-reclosing overpressure protection"

    def regulator(self, x, y, tag=None, s=2.0):
        """Pressure regulator on a vertical line: bowtie + diaphragm dome."""
        self._bowtie(x, y, s, vertical=True)
        self.ax.plot([x, x - s * 1.3], [y, y], color=INK, lw=1.0, zorder=4)
        self.ax.add_patch(Arc((x - s * 1.3 - s * 0.8, y), s * 1.6, s * 1.6,
                              theta1=90, theta2=270, color=INK, lw=1.1,
                              zorder=4))
        self.ax.plot([x - s * 1.3 - s * 0.8, x - s * 1.3 - s * 0.8],
                     [y - s * 0.8, y + s * 0.8], color=INK, lw=1.1, zorder=4)
        self._tag_beside(x, y, tag, dx=s * 2.8)
        self._legend_used["regulator"] = \
            "pressure regulator (diaphragm) — holds tank setpoint"

    def filter_(self, x, y, tag=None, s=2.0, vertical=True):
        pts = [(x, y + s), (x + s, y), (x, y - s), (x - s, y)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="white",
                                  edgecolor=INK, lw=1.1, zorder=4))
        for off in (-0.5, 0.0, 0.5):
            self.ax.plot([x - s * 0.45 + off, x + s * 0.45 + off],
                         [y + s * 0.45, y - s * 0.45], color=INK, lw=0.8,
                         zorder=5)
        self._tag_beside(x, y, tag, dx=s * 2.6)
        self._legend_used["filter"] = "in-line filter (hatched diamond)"

    def pump(self, x, y, tag=None, s=2.4, direction=(0, -1)):
        self.ax.add_patch(Circle((x, y), s, facecolor="white", edgecolor=INK,
                                 lw=1.2, zorder=4))
        d = np.array(direction, float)
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        p = np.array([x, y])
        tri = [p + d * s * 0.85, p - d * s * 0.5 + n * s * 0.7,
               p - d * s * 0.5 - n * s * 0.7]
        self.ax.add_patch(Polygon(tri, closed=True, color=INK, zorder=5))
        self._tag_beside(x, y, tag, dx=s * 2.2)
        self._legend_used["pump"] = \
            "pump (circle + flow triangle) — electric feed pump"

    def heat_exchanger(self, x, y, tag=None, s=2.4):
        self.ax.add_patch(Circle((x, y), s, facecolor="white", edgecolor=INK,
                                 lw=1.2, zorder=4))
        xx = np.linspace(x - s * 0.8, x + s * 0.8, 40)
        self.ax.plot(xx, y + s * 0.45 * np.sin((xx - x) / s * 3 * np.pi),
                     color=INK, lw=1.0, zorder=5)
        self._tag_beside(x, y, tag, dx=s * 2.2)
        self._legend_used["hx"] = \
            "heat exchanger / heated tap (autogenous pressurization source)"

    def quick_disconnect(self, x, y, tag=None, s=1.8, vertical=True):
        if vertical:
            for dy, sgn in ((-s * 0.5, 1), (s * 0.5, -1)):
                self.ax.plot([x - s * 0.9, x + s * 0.9], [y + dy, y + dy],
                             color=INK, lw=1.4, zorder=4)
                self.ax.plot([x - s * 0.9, x - s * 0.9],
                             [y + dy, y + dy + sgn * s * 0.5], color=INK,
                             lw=1.4, zorder=4)
                self.ax.plot([x + s * 0.9, x + s * 0.9],
                             [y + dy, y + dy + sgn * s * 0.5], color=INK,
                             lw=1.4, zorder=4)
        self._tag_beside(x, y, tag, dx=s * 2.8)
        self._legend_used["qd"] = \
            "quick disconnect (facing brackets) — GSE interface"

    def instrument(self, x, y, letters, tap=None, tag=None, r=2.1):
        """ISA instrument bubble; dashed leader to the tap point."""
        if tap is not None:
            self.signal([tap, (x, y)])
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK,
                                 lw=1.1, zorder=4))
        self.tag(x, y + r * 0.30, letters, size=4.6, weight="bold")
        if tag:
            self.tag(x, y - r * 0.42, tag.split("-")[-1], size=3.8)
        self._legend_used["instrument"] = \
            "instrument bubble (PT pressure, TC temperature) w/ signal leader"

    def _tag_beside(self, x, y, tag, dx=0.0, dy=0.0):
        if tag:
            self.tag(x + (dx or 0), y + (dy or 0), tag, size=4.2)

    # ------------------------------------------------------------- vessels
    def tank(self, x, y_bottom, w, h, label, color):
        """Vertical vessel with 2:1 elliptical heads. Returns port points."""
        from matplotlib.patches import Ellipse

        ry = w / 4
        self.ax.add_patch(Rectangle((x - w / 2, y_bottom + ry), w,
                                    h - 2 * ry, facecolor="#f7f7f7",
                                    edgecolor="none", zorder=1.8))
        for yy, t1, t2 in ((y_bottom + ry, 180, 360), (y_bottom + h - ry, 0, 180)):
            self.ax.add_patch(Ellipse((x, yy), w, 2 * ry, facecolor="#f7f7f7",
                                      edgecolor="none", zorder=1.8))
            self.ax.add_patch(Arc((x, yy), w, 2 * ry, theta1=t1, theta2=t2,
                                  edgecolor=INK, lw=1.4, zorder=2))
        self.ax.plot([x - w / 2, x - w / 2], [y_bottom + ry, y_bottom + h - ry],
                     color=INK, lw=1.4, zorder=2.1)
        self.ax.plot([x + w / 2, x + w / 2], [y_bottom + ry, y_bottom + h - ry],
                     color=INK, lw=1.4, zorder=2.1)
        # liquid level
        lvl = y_bottom + h * 0.62
        self.ax.plot([x - w / 2 + 0.6, x + w / 2 - 0.6], [lvl, lvl],
                     color=color, lw=1.0, ls=(0, (4, 2)), zorder=2.2)
        self.tag(x, y_bottom + h * 0.42, label, size=7.0, weight="bold",
                 color=color)
        self.tag(x, y_bottom + h * 0.28, "TANK", size=4.6)
        return {
            "top": (x, y_bottom + h),
            "top_left": (x - w * 0.28, y_bottom + h - ry * 0.25),
            "top_right": (x + w * 0.28, y_bottom + h - ry * 0.25),
            "bottom": (x, y_bottom),
            "side_right": (x + w / 2, y_bottom + h * 0.5),
            "side_left": (x - w / 2, y_bottom + h * 0.5),
        }

    def bottle(self, x, y_bottom, w, h, label):
        self.ax.add_patch(FancyBboxPatch(
            (x - w / 2, y_bottom), w, h,
            boxstyle=f"round,pad=0,rounding_size={w/2.2:.2f}",
            facecolor="#f0f0f0", edgecolor=INK, lw=1.3, zorder=2))
        self.tag(x, y_bottom + h / 2, label, size=5.0, weight="bold")
        return (x, y_bottom)

    def engine(self, x, y_exit, scale=1.0):
        """Regen thrust chamber. Returns port points."""
        s = scale
        r_c, r_t, r_e = 5.2 * s, 2.3 * s, 6.5 * s
        y_inj_b = y_exit + 15.5 * s     # injector bottom
        y_throat = y_exit + 6.0 * s
        prof = [(-r_c, y_inj_b), (-r_c, y_throat + 3.6 * s),
                (-r_t, y_throat), (-r_e, y_exit)]
        left = [(x + px, py) for px, py in prof]
        right = [(x - px, py) for px, py in prof]
        wall = left + right[::-1]
        self.ax.add_patch(Polygon(wall, closed=False, facecolor="none",
                                  edgecolor=INK, lw=1.5, zorder=3))
        # cooling jacket outline (offset)
        off = 1.15 * s
        projo = [(-r_c - off, y_inj_b), (-r_c - off, y_throat + 3.8 * s),
                 (-r_t - off, y_throat), (-r_e - off, y_exit)]
        lo = [(x + px, py) for px, py in projo]
        ro = [(x - px, py) for px, py in projo]
        for seg in (lo, ro):
            self.ax.add_patch(Polygon(seg, closed=False, facecolor="none",
                                      edgecolor=INK, lw=0.9, ls=(0, (2, 1.5)),
                                      zorder=3))
        # injector
        self.ax.add_patch(Rectangle((x - r_c - off, y_inj_b), 2 * (r_c + off),
                                    3.4 * s, facecolor="#f0f0f0",
                                    edgecolor=INK, lw=1.3, zorder=3))
        self.tag(x, y_inj_b + 1.7 * s, "INJECTOR", size=4.6)
        self.tag(x, y_throat + 1.4 * s, "REGEN\nCHAMBER", size=4.4)
        self._legend_used["jacket"] = \
            "dashed contour = regenerative cooling jacket (counter-flow)"
        return {
            "inj_left": (x - r_c - off, y_inj_b + 1.7 * s),
            "inj_right": (x + r_c + off, y_inj_b + 1.7 * s),
            "jacket_inlet": (x + r_e + off, y_exit + 0.6 * s),
            "jacket_outlet": (x + r_c + off, y_inj_b - 0.8 * s),
            "chamber_side": (x - r_c - off, y_inj_b - 2.6 * s),
            "y_inj_bottom": y_inj_b,
        }

    # -------------------------------------------------------------- frames
    def legend(self, x, y_top, width=44.0):
        entries = sorted(self._legend_used.items())
        row_h = 3.4
        h = row_h * len(entries) + 6.5
        self.ax.add_patch(Rectangle((x, y_top - h), width, h,
                                    facecolor="white", edgecolor=INK, lw=1.0,
                                    zorder=2))
        self.tag(x + width / 2, y_top - 2.8, "LEGEND", size=5.6, weight="bold")
        yy = y_top - 6.5
        mini = {
            "valve_manual": lambda cx, cy: self.valve_manual(cx, cy, s=1.1),
            "valve_remote": lambda cx, cy: self.valve_remote(cx, cy, s=1.0),
            "check_valve": lambda cx, cy: self.check_valve(cx, cy, s=1.1),
            "relief_valve": lambda cx, cy: self._bowtie(cx, cy, 1.1, True) or
                self.ax.plot([cx, cx + 0.6, cx - 0.6, cx],
                             [cy + 1.1, cy + 1.7, cy + 2.2, cy + 2.8],
                             color=INK, lw=0.9),
            "burst_disc": lambda cx, cy: self.burst_disc(cx, cy, s=1.0),
            "regulator": lambda cx, cy: self.regulator(cx, cy, s=1.0),
            "filter": lambda cx, cy: self.filter_(cx, cy, s=1.2),
            "pump": lambda cx, cy: self.pump(cx, cy, s=1.3),
            "hx": lambda cx, cy: self.heat_exchanger(cx, cy, s=1.3),
            "qd": lambda cx, cy: self.quick_disconnect(cx, cy, s=1.1),
            "instrument": lambda cx, cy: self.instrument(cx, cy, "PT", r=1.4),
            "atm": lambda cx, cy: (self.pipe([(cx - 1.2, cy), (cx + 0.4, cy)]),
                                   self.vent_to_atm(cx + 0.4, cy, 0)),
            "jacket": lambda cx, cy: self.ax.plot(
                [cx - 1.4, cx + 1.4], [cy, cy], color=INK, lw=0.9,
                ls=(0, (2, 1.5))),
        }
        saved = dict(self._legend_used)
        for key, desc in entries:
            if key in mini:
                mini[key](x + 3.6, yy)
            self.tag(x + 7.4, yy, desc, size=3.9, ha="left")
            yy -= row_h
        self._legend_used = saved  # legend minis shouldn't add entries

    def line_key(self, x, y, items):
        for i, (label, color, dashed) in enumerate(items):
            yy = y - i * 3.0
            self.ax.plot([x, x + 5.5], [yy, yy], color=color, lw=1.5,
                         ls=(0, (3, 2)) if dashed else "-")
            self.tag(x + 7.0, yy, label, size=3.9, ha="left")

    def title_block(self, x, y, w, h, title, subtitle):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                                    edgecolor=INK, lw=1.2, zorder=2))
        self.ax.plot([x, x + w], [y + h * 0.52, y + h * 0.52], color=INK,
                     lw=0.8)
        self.tag(x + w / 2, y + h * 0.74, title, size=5.6, weight="bold")
        self.tag(x + w / 2, y + h * 0.30, subtitle, size=3.9)
        self.tag(x + w / 2, y + h * 0.12,
                 "design checklist aid — NOT safety reviewed", size=3.6,
                 color="#8a1a1a")

    def save(self, path):
        self.fig.savefig(path, bbox_inches="tight", dpi=170)
        plt.close(self.fig)
        return path


# ==========================================================================
# Layout driver
# ==========================================================================
def draw_pid(rec, path: str, line_labels: dict | None = None) -> str:
    """Render a FeedSystemRecommendation as a P&ID-style schematic.

    ``line_labels``: optional short tube-spec strings from the line-sizing
    module, keyed 'oxidizer' / 'fuel' / 'pressurant' / 'coolant_hp' —
    drawn along the corresponding runs.
    """
    cfg = rec.config
    line_labels = line_labels or {}
    sc = Schematic()
    tanks = cfg.tanks[:3]
    n = len(tanks)

    def tcolor(tk):
        return C_OX if tk.oxidizer else C_FUEL

    def find_tag(subsystem, substr):
        for c in rec.components:
            if c.subsystem == subsystem and substr in c.name.lower():
                return c.tag
        return None

    # --------------------------------------------------- coordinates
    x_left_margin = 62.0
    span = 62.0 if n > 1 else 0.0
    xs = [x_left_margin + i * (span / max(n - 1, 1)) for i in range(n)]
    if n == 1:
        xs = [x_left_margin + 20]
    y_tank_bot, tank_h, tank_w = 56.0, 32.0, 17.0
    y_header = 110.0
    x_eng = (xs[0] + xs[-1]) / 2 if n > 1 else xs[0] + 14
    y_exit = 6.0

    # ------------------------------------------------------- engine
    eng = sc.engine(x_eng, y_exit)

    # ------------------------------------------------------- tanks
    ports = []
    for tk, x in zip(tanks, xs):
        ports.append(sc.tank(x, y_tank_bot, tank_w, tank_h, tk.name,
                             tcolor(tk)))

    # ------------------------------------------- pressurant section
    x_mid = (xs[0] + xs[-1]) / 2
    y_top = y_tank_bot + tank_h                # 88: tank top apex
    cv_y = y_top + 9.0                         # 97: check-valve row
    hdr_y = y_top + 14.0                       # 102: distribution header

    def drop_into_tank(x, tag=None):
        sc.pipe([(x, hdr_y), (x, y_top)], color=C_PRESS)
        sc.check_valve(x, cv_y, None, direction=(0, -1))
        if tag:
            sc.tag(x + 4.6, cv_y, tag, size=4.2)
        sc.flow_arrow(x, cv_y - 4.2, -90, C_PRESS)

    if cfg.pressurization == "regulated":
        sc.bottle(x_mid, y_header + 1.0, 6.5, 9.5, "He")
        sc.pipe([(x_mid, y_header + 1.0), (x_mid, hdr_y)], color=C_PRESS)
        sc.valve_remote(x_mid, y_header - 2.5, None, vertical=True)
        sc.tag(x_mid - 5.2, y_header - 2.5,
               find_tag("pressurant", "isolation"), size=4.2)
        sc.regulator(x_mid, y_header - 6.5, find_tag("pressurant",
                     "regulator"))
        # relief on the regulated (LP) side
        rv_x = x_mid + 13.0
        sc.junction(x_mid, hdr_y, color=C_PRESS)
        sc.pipe([(rv_x, hdr_y), (rv_x, hdr_y + 2.0)], color=C_PRESS)
        sc.relief_valve(rv_x, hdr_y + 4.0, None, s=1.7, outlet="right")
        sc.tag(rv_x + 4.0, hdr_y + 1.4, find_tag("pressurant", "relief"),
               size=4.2)
        sc.pipe([(xs[0], hdr_y), (xs[-1], hdr_y), (rv_x, hdr_y)],
                color=C_PRESS)
        cv_tag = find_tag("pressurant", "check")
        for x in xs:
            if n > 1:
                sc.junction(x, hdr_y, color=C_PRESS)
            drop_into_tank(x, cv_tag)
            cv_tag = None  # tag once
    elif cfg.pressurization in ("blowdown", "pump"):
        qd_y = y_header - 2.0
        sc.quick_disconnect(x_mid, qd_y, None)
        sc.tag(x_mid, qd_y + 4.0, "GSE pressurant charge  " +
               (find_tag("pressurant", "quick") or
                find_tag("pressurant", "charge") or ""), size=4.2)
        sc.pipe([(x_mid, qd_y - 1.5), (x_mid, hdr_y)], color=C_PRESS)
        sc.pipe([(xs[0], hdr_y), (xs[-1], hdr_y)], color=C_PRESS)
        for x in xs:
            if n > 1:
                sc.junction(x, hdr_y, color=C_PRESS)
            drop_into_tank(x)
    elif cfg.pressurization == "autogenous":
        # heated tap from the regen outlet routed to the coolant tank; other
        # tanks get a GSE pad charge.
        cool_idx = next((i for i, t in enumerate(tanks) if t.is_coolant), None)
        others = [x for i, x in enumerate(xs) if i != cool_idx]
        if others:
            x_qd = sum(others) / len(others)
            qd_y = y_header - 2.0
            sc.quick_disconnect(x_qd, qd_y, None)
            sc.tag(x_qd, qd_y + 4.0, "GSE pad charge", size=4.2)
            sc.pipe([(x_qd, qd_y - 1.5), (x_qd, hdr_y)], color=C_PRESS)
            for x in others:
                sc.pipe([(x_qd, hdr_y), (x, hdr_y)], color=C_PRESS)
                drop_into_tank(x)
        if cool_idx is not None:
            # autogenous branch drawn later from the regen outlet manifold
            sc._autogenous_x = xs[cool_idx]

    # ------------------------------ per-tank overhead + feed trains
    feed_ends = {}
    for tk, x, p in zip(tanks, xs, ports):
        col = tcolor(tk)
        sub = tk.name
        # vent (top-left port), routed up and out to the left
        vx = x - tank_w * 0.30
        vy0 = y_tank_bot + tank_h - 1.5
        vy1 = vy0 + 5.5
        sc.pipe([(vx, vy0), (vx, vy1), (vx - 9.0, vy1)], color=col)
        sc.valve_remote(vx - 4.5, vy1, None, s=1.6, vertical=False)
        sc.vent_to_atm(vx - 9.0, vy1, 180)
        sc.tag(vx - 4.5, vy1 - 3.0, find_tag(sub, "vent") or "VV", size=4.2)
        # relief + burst disc manifold (top-right port)
        rx = x + tank_w * 0.30
        ry0 = y_tank_bot + tank_h - 1.5
        tee_y = ry0 + 3.0
        sc.pipe([(rx, ry0), (rx, tee_y)], color=col)
        sc.junction(rx, tee_y, color=col)
        rvx = rx + 8.0
        sc.pipe([(rx, tee_y), (rvx, tee_y), (rvx, tee_y + 1.5)], color=col)
        sc.relief_valve(rvx, tee_y + 3.5, None, s=1.6, outlet="right")
        sc.tag(rvx + 3.8, tee_y + 0.6, find_tag(sub, "relief"), size=4.2)
        sc.pipe([(rx, tee_y), (rx, tee_y + 2.2)], color=col)
        sc.burst_disc(rx, tee_y + 3.6, None, s=1.4)
        sc.tag(rx + 4.3, tee_y + 3.6, find_tag(sub, "burst"), size=4.2)
        sc.pipe([(rx, tee_y + 5.0), (rx, tee_y + 6.5)], color=col)
        sc.vent_to_atm(rx, tee_y + 6.5, 90)
        # instruments (side)
        side = p["side_right"]
        sc.instrument(side[0] + 7.5, side[1] + 3.0, "PT",
                      tap=(side[0], side[1] + 3.0), tag=find_tag(sub, "pressure"))
        if tk.cryogenic:
            sc.instrument(side[0] + 7.5, side[1] - 3.5, "TC",
                          tap=(side[0], side[1] - 3.5),
                          tag=find_tag(sub, "temperature"))
        # fill/drain (bottom tee, manual valve, QD cap)
        b = p["bottom"]
        fd_y = b[1] - 4.0
        sc.pipe([b, (b[0], fd_y)], color=col)
        sc.junction(b[0], fd_y, color=col)
        sc.pipe([(b[0], fd_y), (b[0] - 9.5, fd_y)], color=col)
        sc.valve_manual(b[0] - 5.0, fd_y, None, s=1.7, vertical=False)
        sc.quick_disconnect(b[0] - 9.5, fd_y, None, s=1.4)
        sc.tag(b[0] - 5.0, fd_y - 3.6, find_tag(sub, "fill") or "FDV",
               size=4.2)
        # feed train down: filter -> main valve
        y_flt, y_mv = fd_y - 5.5, fd_y - 12.0
        sc.pipe([(b[0], fd_y), (b[0], y_mv - 3.0)], color=col)
        sc.filter_(b[0], y_flt, find_tag(sub, "filter"), s=1.8)
        sc.valve_remote(b[0], y_mv, None, s=1.9, vertical=True)
        sc.tag(b[0] - 5.4, y_mv, find_tag(sub, "main valve") or "MV",
               size=4.2)
        spec_key = "oxidizer" if tk.oxidizer else "fuel"
        if spec_key in line_labels:
            if tk.oxidizer:
                sc.tag(b[0] + 2.2, y_mv - 7.0, line_labels[spec_key],
                       size=3.6, ha="left", color=col)
            else:
                sc.tag(b[0] - 4.2, y_mv - 7.0, line_labels[spec_key],
                       size=3.6, ha="right", color=col)
        sc.flow_arrow(b[0], y_flt + 3.6, -90, col)
        y_end = y_mv - 3.0
        # oxidizer purge tee
        if tk.oxidizer:
            sc.pipe([(b[0], y_end), (b[0], y_end - 2.0)], color=col)
            y_end -= 2.0
            sc.junction(b[0], y_end, color=col)
            px = b[0] - 12.0
            sc.pipe([(b[0], y_end), (px, y_end)], color=C_PRESS)
            sc.check_valve(px + 5.0, y_end, None, s=1.6, direction=(1, 0))
            sc.tag(px - 0.2, y_end + 2.8,
                   find_tag(sub, "purge") or "PGV", size=4.2)
            sc.tag(px - 0.2, y_end - 2.6, "N$_2$ purge", size=4.0)
        feed_ends[tk.name] = (b[0], y_end, tk)

    # --------------------------------------- routing to the engine
    n_plain = 0
    for name, (fx, fy, tk) in feed_ends.items():
        col = tcolor(tk)
        if tk.is_coolant and cfg.regen_cooled:
            # coolant: (pump ->) jacket inlet at the nozzle exit,
            # jacket outlet -> injector (counter-flow regen)
            jx, jy = eng["jacket_inlet"]
            y0 = fy
            if cfg.pressurization == "pump":
                pump_y = fy - 5.0
                sc.pipe([(fx, fy), (fx, pump_y + 2.4)], color=col)
                sc.pump(fx, pump_y, find_tag("engine", "pump"),
                        direction=(0, -1))
                y0 = pump_y - 2.4
            # trapped-volume relief between main valve/pump and the jacket
            rl_y = 24.0
            sc.pipe([(fx, y0), (fx, jy), (jx, jy)], color=col)
            sc.junction(fx, rl_y, color=col)
            sc.pipe([(fx, rl_y), (fx + 9.0, rl_y), (fx + 9.0, rl_y + 1.5)],
                    color=col)
            sc.relief_valve(fx + 9.0, rl_y + 3.5, None, s=1.5, outlet="right")
            sc.tag(fx + 12.5, rl_y + 0.6, find_tag("engine", "relief"),
                   size=4.2)
            sc.flow_arrow(fx, (y0 + rl_y) / 2, -90, col)
            sc.flow_arrow((fx + jx) / 2, jy, 180, col)
            sc.tag(jx + 1.0, jy - 2.4, "jacket in", size=4.0, ha="left")
            if "coolant_hp" in line_labels:
                sc.tag((fx + jx) / 2 + 2.0, jy + 2.4,
                       line_labels["coolant_hp"], size=3.6, color=col)
            # jacket inlet manifold pressure on the vertical run
            sc.instrument(fx - 8.5, 12.5, "PT", tap=(fx, 12.5),
                          tag=find_tag("engine", "manifold inlet"))
            # jacket outlet to injector, with outlet TC
            ox_, oy_ = eng["jacket_outlet"]
            man_x = ox_ + 6.5
            inj_y = eng["inj_right"][1]
            sc.pipe([(ox_, oy_), (man_x, oy_), (man_x, inj_y),
                     (eng["inj_right"][0], inj_y)], color=col)
            sc.flow_arrow(man_x, (oy_ + inj_y) / 2 + 0.4, 90, col)
            sc.tag(man_x + 2.0, oy_ - 2.0, "jacket out = injector in",
                   size=4.0, ha="left")
            sc.instrument(man_x + 8.5, oy_ + 6.8, "TC",
                          tap=(man_x, oy_ + 2.8),
                          tag=find_tag("engine", "outlet temperature"))
            # autogenous pressurization branch from the outlet manifold
            if getattr(sc, "_autogenous_x", None) is not None:
                ax_ = sc._autogenous_x
                bx = max(fx, man_x) + 24.0
                take_y = oy_ + 1.3
                sc.pipe([(man_x, take_y), (bx, take_y),
                         (bx, hdr_y), (ax_, hdr_y), (ax_, y_top)],
                        color=C_PRESS)
                sc.junction(man_x, take_y, color=col)
                sc.heat_exchanger(bx, 55.0, find_tag("pressurant", "heat")
                                  or "HX")
                sc.valve_remote(bx, 78.0, None, s=1.7, vertical=True)
                sc.tag(bx - 5.6, 78.0, find_tag("pressurant", "flow-control")
                       or find_tag("pressurant", "control"), size=4.2)
                sc.check_valve(ax_, cv_y, None, direction=(0, -1))
                sc.flow_arrow(bx, 90.0, 90, C_PRESS)
                sc.tag(bx + 2.4, 64.0, "autogenous\npressurization",
                       size=4.0, ha="left")
        else:
            # straight to the injector; oxidizer takes the left port
            port = "inj_left" if (tk.oxidizer or n_plain == 0) else "inj_right"
            n_plain += 1
            ix, iy = eng[port]
            sc.pipe([(fx, fy), (fx, iy), (ix, iy)], color=col)
            sc.flow_arrow(fx, (fy + iy) / 2, -90, col)

    # chamber instruments + igniter
    cs = eng["chamber_side"]
    sc.instrument(cs[0] - 8.0, cs[1], "PT", tap=cs,
                  tag=find_tag("engine", "chamber pressure"))
    sc.instrument(cs[0] - 8.0, cs[1] - 6.0, "IGN",
                  tap=(cs[0], cs[1] - 4.0), tag=None)

    if "pressurant" in line_labels:
        sc.tag(x_mid + 9.0, hdr_y + 1.9, line_labels["pressurant"],
               size=3.6, ha="left", color=C_PRESS)

    # ------------------------------------------------ legend & frame
    sc.legend(2.0, sc.H - 2.0)
    sc.line_key(2.0, 34.0, [
        ("oxidizer line", C_OX, False),
        ("fuel / coolant line", C_FUEL, False),
        ("pressurant line", C_PRESS, False),
        ("instrument signal", C_SIG, True),
    ])
    sc.title_block(sc.W - 52.0, 2.0, 50.0, 12.0,
                   f"FEED SYSTEM — {cfg.pressurization.upper()}-FED",
                   "cryosim rule-based recommendation")
    sc.ax.add_patch(Rectangle((0.5, 0.5), sc.W - 1.0, sc.H - 1.0,
                              facecolor="none", edgecolor=INK, lw=1.0))
    return sc.save(path)
