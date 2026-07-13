"""Axisymmetric inviscid (Euler) CFD of the chamber/nozzle internal flow.

A small finite-volume solver for the compressible Euler equations on a
contour-fitted structured mesh, used to CHECK the quasi-1D isentropic
assumption behind the Bartz-based regen model — where does the real 2D
flow field deviate (throat curvature effects, wall vs centerline Mach), and
is the 1D mass flow right?

Scheme (deliberately simple and robust):

* conservative axisymmetric formulation:  d(U r)/dt + div(F r) = [0,0,p,0],
  discretized with radius-weighted face fluxes so mass/momentum/energy are
  conserved to machine precision on the mesh (the axis face has r=0 and
  drops out naturally);
* Rusanov (local Lax-Friedrichs) fluxes — first-order, very dissipative,
  unconditionally shock-safe; adequate for steady subsonic/transonic/
  supersonic nozzle flow on the coarse grids used here;
* explicit local time stepping to steady state, initialized from the
  quasi-1D solution;
* boundary conditions: stagnation inlet (p0 = Pc, T0 = T_c, axial inflow),
  supersonic extrapolation outlet, slip (mirror) walls.

Honest scope statement (this matters): **this is an INVISCID Euler
solver.** There are no boundary layers, no turbulence model, and therefore
NO wall heat transfer or skin friction — it cannot replace Bartz, RANS or
conjugate CFD for thermal design. What it CAN do, and is validated to do
here, is quantify how good the quasi-1D area-ratio flow assumption is
(mass flow, Mach/pressure distributions, 2D throat effects) on your actual
contour. First-order dissipation smears the throat gradient slightly on
coarse grids — refine before quoting numbers to better than a few percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chamber_geometry import ChamberContour
from .combustion import CombustionGas, mach_from_area_ratio


@dataclass
class CFDResult:
    """Converged (or best-effort) steady flow field, cell-centered."""

    x: np.ndarray          # (N, M) cell centers, axial
    r: np.ndarray          # (N, M) cell centers, radial
    rho: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    T: np.ndarray
    mach: np.ndarray
    residuals: np.ndarray  # density-residual history (L2, normalized)
    converged: bool
    gas: CombustionGas
    Pc: float
    contour: ChamberContour

    # ------------------------------------------------------------ analysis
    def mdot_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Axisymmetric mass flow through each axial station [kg/s]."""
        # radial extent of each cell at the station
        dr = np.gradient(self.r, axis=1)
        mdot = np.sum(self.rho * self.u * 2 * np.pi * self.r * dr, axis=1)
        return self.x[:, 0], mdot

    def mdot_quasi1d(self) -> float:
        return self.Pc * self.contour.At / self.gas.c_star

    def centerline_mach(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[:, 0], self.mach[:, 0]

    def wall_mach(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[:, -1], self.mach[:, -1]

    def quasi1d_mach(self) -> np.ndarray:
        """1D Mach at the cell-center stations (for comparison plots)."""
        g = self.gas.gamma
        ar = np.interp(self.x[:, 0], self.contour.x,
                       self.contour.area_ratio())
        sup = self.x[:, 0] > self.contour.x_throat
        return np.array([
            mach_from_area_ratio(a, g, s) for a, s in zip(ar, sup)
        ])


class NozzleEulerCFD:
    """Contour-fitted axisymmetric Euler solver."""

    def __init__(self, contour: ChamberContour, gas: CombustionGas,
                 Pc: float, n_axial: int = 120, n_radial: int = 24):
        self.contour = contour
        self.gas = gas
        self.Pc = Pc
        self.g = gas.gamma
        self.R = gas.R_specific
        N, M = n_axial, n_radial
        self.N, self.M = N, M

        # node coordinates: radial lines from axis to wall
        xn = np.linspace(contour.x[0], contour.x[-1], N + 1)
        Rw = np.interp(xn, contour.x, contour.r)
        j = np.arange(M + 1) / M
        self.xn = np.repeat(xn[:, None], M + 1, 1)          # (N+1, M+1)
        self.rn = Rw[:, None] * j[None, :]

        # cell geometry
        x00 = self.xn[:-1, :-1]; r00 = self.rn[:-1, :-1]
        x10 = self.xn[1:, :-1];  r10 = self.rn[1:, :-1]
        x11 = self.xn[1:, 1:];   r11 = self.rn[1:, 1:]
        x01 = self.xn[:-1, 1:];  r01 = self.rn[:-1, 1:]
        self.area = 0.5 * np.abs(
            (x10 - x00) * (r01 - r00) - (x01 - x00) * (r10 - r00)
        ) + 0.5 * np.abs(
            (x10 - x11) * (r01 - r11) - (x01 - x11) * (r10 - r11)
        )
        self.xc = 0.25 * (x00 + x10 + x11 + x01)
        self.rc = 0.25 * (r00 + r10 + r11 + r01)

        # I-faces (constant-i): normal ~ +x, area vector (dr, 0)
        self.nI = np.stack([self.rn[:, 1:] - self.rn[:, :-1],
                            np.zeros((N + 1, M))], axis=-1)   # (N+1, M, 2)
        self.rI = 0.5 * (self.rn[:, 1:] + self.rn[:, :-1])
        # J-faces (constant-j): normal ~ +r: (-dr_edge, dx_edge)
        dxe = self.xn[1:, :] - self.xn[:-1, :]
        dre = self.rn[1:, :] - self.rn[:-1, :]
        self.nJ = np.stack([-dre, dxe], axis=-1)              # (N, M+1, 2)
        self.rJ = 0.5 * (self.rn[1:, :] + self.rn[:-1, :])

        self._init_from_quasi1d()

    # ------------------------------------------------------------ states
    def _prim_to_cons(self, rho, u, v, p):
        E = p / (self.g - 1.0) + 0.5 * rho * (u**2 + v**2)
        return np.stack([rho, rho * u, rho * v, E], axis=-1)

    def _cons_to_prim(self, U):
        rho = np.maximum(U[..., 0], 1e-8)
        u = U[..., 1] / rho
        v = U[..., 2] / rho
        p = np.maximum(
            (self.g - 1.0) * (U[..., 3] - 0.5 * rho * (u**2 + v**2)), 10.0)
        return rho, u, v, p

    def _init_from_quasi1d(self):
        g = self.g
        ar = np.interp(self.xc[:, 0], self.contour.x,
                       self.contour.area_ratio())
        sup = self.xc[:, 0] > self.contour.x_throat
        M1 = np.array([mach_from_area_ratio(a, g, s)
                       for a, s in zip(ar, sup)])
        T = self.gas.T_c / (1 + (g - 1) / 2 * M1**2)
        p = self.Pc * (T / self.gas.T_c) ** (g / (g - 1))
        rho = p / (self.R * T)
        u = M1 * np.sqrt(g * self.R * T)
        shape = self.xc.shape
        self.U = self._prim_to_cons(
            np.repeat(rho[:, None], shape[1], 1),
            np.repeat(u[:, None], shape[1], 1),
            np.zeros(shape),
            np.repeat(p[:, None], shape[1], 1),
        )

    # ------------------------------------------------------------- fluxes
    def _normal_flux(self, UL, UR, n):
        """Rusanov flux through faces with area vectors n (last dim 2)."""
        g = self.g
        L = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2) + 1e-30
        nx, nr = n[..., 0] / L, n[..., 1] / L

        def flux(U):
            rho, u, v, p = self._cons_to_prim(U)
            Vn = u * nx + v * nr
            return np.stack([
                rho * Vn,
                rho * u * Vn + p * nx,
                rho * v * Vn + p * nr,
                (U[..., 3] + p) * Vn,
            ], axis=-1), Vn, np.sqrt(g * p / rho)

        FL, VnL, aL = flux(UL)
        FR, VnR, aR = flux(UR)
        lam = np.maximum(np.abs(VnL) + aL, np.abs(VnR) + aR)
        F = 0.5 * (FL + FR) - 0.5 * lam[..., None] * (UR - UL)
        return F * L[..., None]

    def _inlet_state(self):
        """Isentropic stagnation inlet from interior static pressure."""
        g = self.g
        _, _, _, p_int = self._cons_to_prim(self.U[0])
        pr = np.clip(self.Pc / p_int, 1.0 + 1e-9, None)
        M = np.sqrt(2 / (g - 1) * (pr ** ((g - 1) / g) - 1.0))
        M = np.clip(M, 1e-4, 0.99)
        T = self.gas.T_c / (1 + (g - 1) / 2 * M**2)
        p = self.Pc * (T / self.gas.T_c) ** (g / (g - 1))
        rho = p / (self.R * T)
        u = M * np.sqrt(g * self.R * T)
        return self._prim_to_cons(rho, u, np.zeros_like(u), p)

    def _wall_ghost(self):
        """Mirror the wall-adjacent cells across the local wall face."""
        Uw = self.U[:, -1].copy()
        n = self.nJ[:, -1]
        L = np.sqrt(n[:, 0] ** 2 + n[:, 1] ** 2) + 1e-30
        nx, nr = n[:, 0] / L, n[:, 1] / L
        rho, u, v, p = self._cons_to_prim(Uw)
        Vn = u * nx + v * nr
        return self._prim_to_cons(rho, u - 2 * Vn * nx, v - 2 * Vn * nr, p)

    # ---------------------------------------------------------------- run
    def run(self, max_iter: int = 5000, cfl: float = 0.4,
            tol: float = 1e-4, log_every: int = 200) -> CFDResult:
        N, M = self.N, self.M
        res_hist = []
        d_char = np.sqrt(self.area)          # cell length scale
        r_vol = np.maximum(self.rc, 1e-9) * self.area

        for it in range(max_iter):
            rho, u, v, p = self._cons_to_prim(self.U)

            # I-face fluxes, radius-weighted
            UL = np.concatenate([self._inlet_state()[None], self.U], axis=0)
            UR = np.concatenate([self.U, self.U[-1][None]], axis=0)
            FI = self._normal_flux(UL, UR, self.nI) * self.rI[..., None]

            # J-face fluxes (axis face has r=0 -> zero weight)
            ULj = np.concatenate([self.U[:, :1], self.U], axis=1)
            URj = np.concatenate([self.U, self._wall_ghost()[:, None]],
                                 axis=1)
            FJ = self._normal_flux(ULj, URj, self.nJ) * self.rJ[..., None]

            div = (FI[1:] - FI[:-1]) + (FJ[:, 1:] - FJ[:, :-1])
            src = np.zeros_like(div)
            src[..., 2] = p * self.area
            dUdt = (-div + src) / r_vol[..., None]

            a = np.sqrt(self.g * p / rho)
            dt = cfl * d_char / (np.sqrt(u**2 + v**2) + a)
            self.U = self.U + dt[..., None] * dUdt

            res = np.sqrt(np.mean((dt * dUdt[..., 0]) ** 2)) / \
                max(np.mean(rho), 1e-12)
            res_hist.append(res)
            if it > 100 and res_hist[0] > 0 and res < tol * res_hist[0]:
                break

        rho, u, v, p = self._cons_to_prim(self.U)
        T = p / (self.R * rho)
        mach = np.sqrt(u**2 + v**2) / np.sqrt(self.g * p / rho)
        res_arr = np.array(res_hist)
        return CFDResult(
            x=self.xc, r=self.rc, rho=rho, u=u, v=v, p=p, T=T, mach=mach,
            residuals=res_arr,
            converged=bool(res_arr[-1] < 1e-2 * res_arr[:50].max()),
            gas=self.gas, Pc=self.Pc, contour=self.contour,
        )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_cfd(result: CFDResult, path: str) -> str:
    """Mach contour (mirrored about the axis) + centerline/wall comparison
    with quasi-1D + residual history."""
    from . import plots  # house style

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 0.8])

    ax0 = fig.add_subplot(gs[0])
    for sgn in (1, -1):
        pc = ax0.pcolormesh(result.x * 100, sgn * result.r * 100,
                            result.mach, cmap="viridis",
                            shading="gouraud", vmin=0.0)
    fig.colorbar(pc, ax=ax0, label="Mach")
    ax0.plot(result.contour.x * 100, result.contour.r * 100, color="k", lw=1.2)
    ax0.plot(result.contour.x * 100, -result.contour.r * 100, color="k", lw=1.2)
    ax0.set(xlabel="x [cm]", ylabel="r [cm]",
            title="Mach field (axisymmetric Euler, first-order Rusanov)")
    ax0.set_aspect("equal")

    ax1 = fig.add_subplot(gs[1])
    xc, mc = result.centerline_mach()
    xw, mw = result.wall_mach()
    ax1.plot(xc * 100, result.quasi1d_mach(), color=plots.C_GRAY,
             ls="--", label="quasi-1D (regen model assumption)")
    ax1.plot(xc * 100, mc, color=plots.C_BLUE, label="CFD centerline")
    ax1.plot(xw * 100, mw, color=plots.C_RED, label="CFD wall")
    ax1.set(xlabel="x [cm]", ylabel="Mach")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[2])
    ax2.semilogy(result.residuals / result.residuals[0], color=plots.C_GREEN)
    ax2.set(xlabel="iteration", ylabel="density residual (norm.)")
    fig.suptitle("Nozzle Euler CFD vs quasi-1D — INVISCID: flow-field check "
                 "only, no wall heat transfer", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
