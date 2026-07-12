"""Standard output plots for tank-only and coupled simulations.

Static matplotlib figures for CLI/report use: one quantity family per axis
(never dual axes), colorblind-safe Okabe-Ito series colors assigned in fixed
order, recessive grids, units on every axis.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, fixed assignment
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#7f7f7f"

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "font.size": 9.5,
    "lines.linewidth": 1.8,
})


def _save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- tank
def plot_tank_history(hist, path, title="Tank thermal history"):
    """4-panel tank plot from a TankHistory."""
    t = hist.t / 60.0  # minutes
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].plot(t, hist.P / 1e5, color=C_BLUE)
    ax[0, 0].set(xlabel="time [min]", ylabel="ullage pressure [bar]")

    ax[0, 1].plot(t, hist.T_ullage, color=C_ORANGE, label="ullage")
    ax[0, 1].plot(t, hist.T_surface, color=C_GREEN, label="surface layer")
    ax[0, 1].plot(t, hist.T_bulk, color=C_BLUE, label="bulk liquid")
    ax[0, 1].set(xlabel="time [min]", ylabel="temperature [K]")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(t, hist.boiloff_rate * 1e3, color=C_RED)
    ax[1, 0].set(xlabel="time [min]", ylabel="boil-off rate [g/s]")

    ax[1, 1].plot(t, hist.fill_fraction * 100, color=C_PURPLE)
    ax[1, 1].set(xlabel="time [min]", ylabel="fill level [%]")
    fig.suptitle(title)
    return _save(fig, path)


# ------------------------------------------------------------------ coupled
def plot_coupled_tank(a, path):
    t = a["t"]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].plot(t, a["P_ullage"] / 1e5, color=C_BLUE)
    ax[0, 0].set(xlabel="time [s]", ylabel="ullage pressure [bar]")

    ax[0, 1].plot(t, a["T_ullage"], color=C_ORANGE, label="ullage")
    ax[0, 1].plot(t, a["T_surface"], color=C_GREEN, label="surface layer")
    ax[0, 1].plot(t, a["T_bulk"], color=C_BLUE, label="bulk liquid")
    ax[0, 1].set(xlabel="time [s]", ylabel="tank temperature [K]")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(t, a["boiloff_rate"] * 1e3, color=C_RED)
    ax[1, 0].set(xlabel="time [s]", ylabel="boil-off rate [g/s]")

    ax[1, 1].plot(t, a["fill_fraction"] * 100, color=C_PURPLE)
    ax[1, 1].set(xlabel="time [s]", ylabel="fill level [%]")
    fig.suptitle("Coupled burn — tank state")
    return _save(fig, path)


def plot_coupled_slosh(a, path):
    t = a["t"]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].plot(t, a["slosh_freq"], color=C_BLUE)
    ax[0, 0].set(xlabel="time [s]", ylabel="1st slosh mode frequency [Hz]")

    ax[0, 1].plot(t, a["slosh_mass_fraction"] * 100, color=C_ORANGE)
    ax[0, 1].set(xlabel="time [s]", ylabel="slosh mass fraction m1/m [%]")

    ax[1, 0].plot(t, a["slosh_amplitude"] * 1e3, color=C_GREEN,
                  label="slosh mass displacement")
    ax[1, 0].plot(t, a["cg_lateral_offset"] * 1e3, color=C_RED,
                  label="liquid CG lateral offset")
    ax[1, 0].set(xlabel="time [s]", ylabel="lateral displacement [mm]")
    ax[1, 0].legend(frameon=False)

    ax[1, 1].plot(t, a["cg_axial"] * 100, color=C_PURPLE)
    ax[1, 1].set(xlabel="time [s]", ylabel="liquid CG height [cm]")
    fig.suptitle("Coupled burn — slosh & CG")
    return _save(fig, path)


def plot_coupled_regen(a, path):
    t = a["t"]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].plot(t, a["coolant_inlet_T"], color=C_BLUE, label="inlet (= tank outlet)")
    ax[0, 0].plot(t, a["coolant_outlet_T"], color=C_RED, label="outlet (= injector inlet)")
    ax[0, 0].set(xlabel="time [s]", ylabel="coolant temperature [K]")
    ax[0, 0].legend(frameon=False)

    ax[0, 1].plot(t, a["peak_wall_T"], color=C_ORANGE)
    ax[0, 1].set(xlabel="time [s]", ylabel="peak hot-wall temperature [K]")

    ax[1, 0].plot(t, a["regen_dP"] / 1e5, color=C_GREEN, label="jacket Δp")
    ax[1, 0].plot(t, a["injector_margin"] / 1e5, color=C_PURPLE,
                  label="injector margin")
    ax[1, 0].set(xlabel="time [s]", ylabel="pressure [bar]")
    ax[1, 0].legend(frameon=False)

    ax[1, 1].plot(t, a["q_throat"] / 1e6, color=C_RED)
    ax[1, 1].set(xlabel="time [s]", ylabel="throat heat flux [MW/m$^2$]")
    fig.suptitle("Coupled burn — regenerative cooling")
    return _save(fig, path)


def plot_regen_distribution(res, x_throat, path,
                            title="Regen cooling — axial distributions"):
    """Wall/coolant temperatures and heat flux along the chamber."""
    x = res.x * 100  # cm
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(x, res.r * 1e3, color=C_GRAY)
    ax[0].set_ylabel("wall radius [mm]")
    ax[0].set_ylim(bottom=0)

    ax[1].plot(x, res.T_wg, color=C_RED, label="hot-gas-side wall")
    ax[1].plot(x, res.T_wc, color=C_ORANGE, label="coolant-side wall")
    ax[1].plot(x, res.T_coolant, color=C_BLUE, label="coolant bulk")
    ax[1].set_ylabel("temperature [K]")
    ax[1].legend(frameon=False)

    ax[2].plot(x, res.q / 1e6, color=C_RED)
    ax[2].set(xlabel="axial position from injector [cm]",
              ylabel="heat flux [MW/m$^2$]")
    for a_ in ax:
        a_.axvline(x_throat * 100, color=C_GRAY, lw=0.8, ls="--")
    ax[0].text(x_throat * 100, ax[0].get_ylim()[1] * 0.05, " throat",
               fontsize=8, color=C_GRAY)
    fig.suptitle(title)
    return _save(fig, path)


def write_history_csv(a, path):
    keys = list(a.keys())
    data = np.column_stack([np.asarray(a[k], dtype=float) for k in keys])
    np.savetxt(path, data, delimiter=",", header=",".join(keys), comments="")
    return path
