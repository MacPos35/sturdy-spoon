"""Automatic gimbal (TVC) stabilization of a slosh-coupled rocket.

Planar (pitch-plane), small-angle, linear model of an ascending rocket
stabilized by a gimbaled engine, INCLUDING the first lateral slosh mode as
the equivalent pendulum from :mod:`cryosim.slosh_model` — the classic
vehicle/slosh/TVC interaction problem (NASA SP-8009 ch. 4; Dodge 2000
ch. 5; Wie, "Space Vehicle Dynamics and Control", slosh chapters).

Degrees of freedom q = [z, theta, psi]: lateral translation of the
composite CG, pitch attitude, and slosh-pendulum angle relative to the
body axis. With the rigid (non-sloshing) mass m0 at station x0, slosh mass
m1 at station xm (stations measured from the composite CG, +toward nose),
pendulum length L, axial acceleration a = F/m_total, small-angle lateral
positions are  z0 = z + x0*theta  and  z1 = z + xm*theta + L*psi, giving
(Lagrange):

    M = [[m0+m1,        0,             m1*L     ],
         [0,     m0*x0^2+I0+m1*xm^2,   m1*xm*L  ],
         [m1*L,  m1*xm*L,              m1*L^2   ]]
    K = diag(0, 0, m1*a*L),   C = diag(0, 0, 2*zeta*omega_s*m1*L^2)

(the z-theta coupling vanishes because m0*x0 + m1*xm = 0 by the CG
definition). Generalized forces from gimbaled thrust F at station xe with
deflection delta and a lateral disturbance d(t) at station xd:

    Q_z = F*(theta + delta) + d,   Q_theta = F*xe*delta + xd*d,   Q_psi = 0

(the axial thrust component acts along the vehicle centerline and produces
no moment; only the gimbal component F*delta at the engine station does).

**Automatic controller**: a PD attitude law delta = -(Kp*theta + Kd*theta_dot)
whose gains are computed from the vehicle itself — plant gain
k = F*|xe|/I_theta, so  Kp = wc^2/k, Kd = 2*zc*wc/k  place the rigid-body
poles at the requested bandwidth wc and damping zc. The actuator is a
first-order lag with angle and rate limits. Closed-loop eigenvalues of the
full slosh-coupled linear system are computed and checked (the honest
stability statement), and the tool warns when the control bandwidth
approaches the slosh frequency with a significant slosh mass fraction —
the configuration that historically destabilizes vehicles.

Assumptions / limitations
-------------------------
* Linear, small angles, planar; constant mass/inertia and thrust over the
  run (quasi-static snapshots — re-run at several burn times for a sweep).
* No aerodynamics (add the unstable aero moment before trusting margins in
  atmosphere), no bending modes, one slosh mode per tank, rigid engine.
* This is a stability/interaction study tool for an *imaginary rigid
  rocket*, not a flight-certification 6-DOF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .slosh_model import SloshParameters

G0 = 9.80665


@dataclass
class VehicleModel:
    """Planar vehicle with one slosh pendulum (SI; stations from composite
    CG, positive toward the nose)."""

    m_rigid: float          # kg, non-sloshing mass (dry + rigid propellant)
    I_rigid: float          # kg m^2, about the rigid part's own CG
    x_rigid: float          # m, rigid-part CG station
    m_slosh: float          # kg
    x_slosh: float          # m, slosh-mass station
    L_slosh: float          # m, pendulum length
    zeta_slosh: float       # -
    x_engine: float         # m, gimbal station (negative: below CG)
    thrust: float           # N
    x_cg_datum: float = 0.0  # composite-CG station in the user's datum
                             # (set by from_components; for station conversion)

    def __post_init__(self):
        cg = (self.m_rigid * self.x_rigid + self.m_slosh * self.x_slosh) \
            / self.m_total
        if abs(cg) > 1e-6:
            raise ValueError(
                f"stations must be measured from the composite CG "
                f"(residual {cg:.2e} m); use from_components()"
            )

    @property
    def m_total(self) -> float:
        return self.m_rigid + self.m_slosh

    @property
    def accel(self) -> float:
        return self.thrust / self.m_total

    @property
    def I_theta(self) -> float:
        """Pitch inertia entry of the mass matrix."""
        return (self.m_rigid * self.x_rigid**2 + self.I_rigid
                + self.m_slosh * self.x_slosh**2)

    @property
    def omega_slosh(self) -> float:
        return np.sqrt(self.accel / self.L_slosh)

    # ------------------------------------------------------------ assembly
    def mass_matrix(self) -> np.ndarray:
        m1, L, xm = self.m_slosh, self.L_slosh, self.x_slosh
        return np.array([
            [self.m_total, 0.0, m1 * L],
            [0.0, self.I_theta, m1 * xm * L],
            [m1 * L, m1 * xm * L, m1 * L**2],
        ])

    def stiffness_damping(self) -> tuple[np.ndarray, np.ndarray]:
        m1, L = self.m_slosh, self.L_slosh
        K = np.diag([0.0, 0.0, m1 * self.accel * L])
        C = np.diag([0.0, 0.0, 2 * self.zeta_slosh * self.omega_slosh
                     * m1 * L**2])
        return K, C

    @classmethod
    def from_components(
        cls,
        dry_mass: float,
        dry_cg: float,
        dry_inertia: float,
        tank_bottom: float,
        slosh: SloshParameters,
        engine_station: float,
        thrust: float,
    ) -> "VehicleModel":
        """Build from dry vehicle + a tank's slosh parameters.

        Stations in any common datum (+toward nose). The rigid propellant
        mass is treated as a point mass (its own inertia neglected —
        conservative for TVC authority studies, stated assumption).
        """
        m_lr = slosh.m_rigid
        x_lr = tank_bottom + slosh.rigid_height
        m1 = slosh.first.mass
        x_m = tank_bottom + slosh.first.height_above_bottom
        m0 = dry_mass + m_lr
        x0 = (dry_mass * dry_cg + m_lr * x_lr) / m0
        I0 = (dry_inertia + dry_mass * (dry_cg - x0) ** 2
              + m_lr * (x_lr - x0) ** 2)
        x_cg = (m0 * x0 + m1 * x_m) / (m0 + m1)
        return cls(
            m_rigid=m0, I_rigid=I0, x_rigid=x0 - x_cg,
            m_slosh=m1, x_slosh=x_m - x_cg,
            L_slosh=slosh.first.pendulum_length,
            zeta_slosh=slosh.zeta_viscous,
            x_engine=engine_station - x_cg,
            thrust=thrust,
            x_cg_datum=x_cg,
        )


@dataclass
class GimbalController:
    """Auto-tuned PD attitude controller + first-order gimbal actuator."""

    Kp: float
    Kd: float
    tau: float = 0.05           # s, actuator lag
    delta_max: float = np.radians(6.0)
    rate_max: float = np.radians(30.0)
    bandwidth: float = 0.0      # rad/s (informational)
    warnings: list = field(default_factory=list)

    @classmethod
    def auto_tune(
        cls,
        vehicle: VehicleModel,
        bandwidth_hz: float = 1.0,
        damping: float = 0.7,
        **kwargs,
    ) -> "GimbalController":
        """Pole placement on the rigid-body attitude plant.

        The SIGNED plant gain k = F*xe/I_theta is negative for an engine
        below the CG, and the gains inherit that sign so the law
        delta = -(Kp*theta + Kd*theta_dot) always closes the loop the
        stabilizing way:  theta_ddot = k*delta = -wc^2 theta - 2 zc wc
        theta_dot. Also raises the classic slosh-interaction warnings.
        """
        wc = 2 * np.pi * bandwidth_hz
        k = vehicle.thrust * vehicle.x_engine / vehicle.I_theta
        ctl = cls(Kp=wc**2 / k, Kd=2 * damping * wc / k, bandwidth=wc,
                  **kwargs)
        ws = vehicle.omega_slosh
        mf = vehicle.m_slosh / vehicle.m_total
        if 0.33 * ws < wc < 3.0 * ws and mf > 0.03:
            ctl.warnings.append(
                f"control bandwidth ({wc/2/np.pi:.2f} Hz) is within 3x of "
                f"the slosh frequency ({ws/2/np.pi:.2f} Hz) with slosh mass "
                f"fraction {mf*100:.0f}% - classic slosh/TVC interaction "
                "region; add baffles (raise zeta) or move the bandwidth"
            )
        if vehicle.x_engine >= 0:
            ctl.warnings.append("engine station is above the CG - check the "
                                "sign conventions of your stations")
        return ctl


@dataclass
class GimbalHistory:
    t: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    delta: np.ndarray
    delta_cmd: np.ndarray
    slosh_displacement: np.ndarray   # L * psi (bob lateral vs body) [m]
    max_theta_deg: float
    settled: bool
    stable: bool
    eigenvalues: np.ndarray
    warnings: list


def closed_loop_eigenvalues(vehicle: VehicleModel,
                            ctl: GimbalController) -> np.ndarray:
    """Eigenvalues of the full linear closed loop (7 states:
    q, q_dot, actuator delta)."""
    M = vehicle.mass_matrix()
    K, C = vehicle.stiffness_damping()
    Minv = np.linalg.inv(M)
    F, xe = vehicle.thrust, vehicle.x_engine

    # generalized force distribution: Q = Bq*[theta] + Bd*delta
    # Q_z = F*theta + F*delta ; Q_theta = F*xe*delta ; Q_psi = 0
    B_theta = np.array([F, 0.0, 0.0])      # multiplies theta (state 1)
    B_delta = np.array([F, F * xe, 0.0])   # multiplies delta

    A = np.zeros((7, 7))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -Minv @ K
    A[3:6, 3:6] = -Minv @ C
    A[3:6, 1] += Minv @ B_theta
    A[3:6, 6] = Minv @ B_delta
    # actuator: delta_dot = (delta_cmd - delta)/tau,
    # delta_cmd = -Kp*theta - Kd*theta_dot
    A[6, 1] = -ctl.Kp / ctl.tau
    A[6, 4] = -ctl.Kd / ctl.tau
    A[6, 6] = -1.0 / ctl.tau
    return np.linalg.eigvals(A)


def simulate_gimbal(
    vehicle: VehicleModel,
    ctl: GimbalController,
    t_end: float = 20.0,
    disturbance=None,
    x_disturbance: float | None = None,
    theta0: float = 0.0,
    n_out: int = 1000,
) -> GimbalHistory:
    """Time response with actuator saturation and rate limiting.

    ``disturbance``: lateral force [N] vs time (e.g., a wind-gust pulse);
    applied at station ``x_disturbance`` (default: the CG).
    """
    dist = disturbance or (lambda t: 0.0)
    xd = 0.0 if x_disturbance is None else x_disturbance
    M = vehicle.mass_matrix()
    K, C = vehicle.stiffness_damping()
    Minv = np.linalg.inv(M)
    F, xe = vehicle.thrust, vehicle.x_engine

    def rhs(t, y):
        q = y[0:3]
        qd = y[3:6]
        delta = y[6]
        theta, theta_dot = q[1], qd[1]
        d = float(dist(t))
        Q = np.array([
            F * (theta + delta) + d,
            F * xe * delta + xd * d,
            0.0,
        ])
        qdd = Minv @ (Q - K @ q - C @ qd)
        cmd = float(np.clip(-ctl.Kp * theta - ctl.Kd * theta_dot,
                            -ctl.delta_max, ctl.delta_max))
        ddot = float(np.clip((cmd - delta) / ctl.tau,
                             -ctl.rate_max, ctl.rate_max))
        return np.concatenate([qd, qdd, [ddot]])

    y0 = np.zeros(7)
    y0[1] = theta0
    t_eval = np.linspace(0, t_end, n_out)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, rtol=1e-8,
                    atol=1e-11, max_step=0.01)
    if not sol.success:
        raise RuntimeError(f"gimbal simulation failed: {sol.message}")

    theta = sol.y[1]
    delta = sol.y[6]
    cmd = np.clip(-ctl.Kp * theta - ctl.Kd * sol.y[4],
                  -ctl.delta_max, ctl.delta_max)
    eig = closed_loop_eigenvalues(vehicle, ctl)
    stable = bool(np.all(eig.real < 1e-9))
    tail = theta[sol.t > 0.8 * t_end]
    settled = bool(np.all(np.abs(tail) < np.radians(0.2)))
    warnings = list(ctl.warnings)
    if not stable:
        warnings.append("closed loop UNSTABLE (eigenvalue with positive "
                        "real part) - reduce bandwidth or add slosh damping")
    return GimbalHistory(
        t=sol.t, z=sol.y[0], theta=theta, psi=sol.y[2], delta=delta,
        delta_cmd=cmd,
        slosh_displacement=vehicle.L_slosh * sol.y[2],
        max_theta_deg=float(np.degrees(np.abs(theta).max())),
        settled=settled, stable=stable, eigenvalues=eig, warnings=warnings,
    )


def plot_gimbal(hist: GimbalHistory, path: str) -> str:
    """4-panel response plot."""
    from . import plots  # applies house style

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6))
    ax[0, 0].plot(hist.t, np.degrees(hist.theta), color=plots.C_BLUE)
    ax[0, 0].set(xlabel="time [s]", ylabel="pitch attitude θ [deg]")

    ax[0, 1].plot(hist.t, np.degrees(hist.delta_cmd), color=plots.C_GRAY,
                  label="command")
    ax[0, 1].plot(hist.t, np.degrees(hist.delta), color=plots.C_RED,
                  label="actuator")
    ax[0, 1].set(xlabel="time [s]", ylabel="gimbal angle δ [deg]")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(hist.t, hist.slosh_displacement * 1e3, color=plots.C_GREEN)
    ax[1, 0].set(xlabel="time [s]",
                 ylabel="slosh mass displacement [mm]")

    ax[1, 1].plot(hist.t, hist.z * 1e3, color=plots.C_PURPLE)
    ax[1, 1].set(xlabel="time [s]", ylabel="lateral CG drift z [mm]")
    status = "STABLE" if hist.stable else "UNSTABLE"
    fig.suptitle(f"Gimbal TVC response — closed loop {status}, "
                 f"max |θ| = {hist.max_theta_deg:.2f}°")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
