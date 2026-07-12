"""Coupled tank (slosh + thermal) -> regen-cooling -> injector simulation.

Couples the three sub-models over a user-specified burn profile:

* The **tank thermal model** is integrated in time with the propellant
  outflow demanded by the engine (mdot = Pc At / c* x coolant fraction) and
  the flight axial acceleration, giving ullage pressure, outlet (bulk)
  temperature and boil-off.
* The **slosh analog** parameters are updated quasi-statically with fill
  level and acceleration; the first mode is integrated concurrently under
  the lateral-acceleration input. Slosh amplitude feeds back into the
  thermal model through the mixing factor
  Phi = 1 + c_mix * |x1| / R   (destratification / enhanced interfacial
  transfer). c_mix is a parametric knob — the slosh->thermal link has no
  public quantitative validation data (README, "validation status").
* The **regen model** is solved quasi-steadily (regen thermal time
  constants are ~ms, tank time scales ~s) at a configurable interval, with
  the coolant inlet at the *current tank outlet state*: T = bulk liquid
  temperature, P = ullage pressure + hydrostatic head under acceleration -
  feed-line loss. Its outlet state is the **injector inlet** — tracked over
  the burn instead of assuming constant properties.

The modeled tank IS the coolant tank (default demo: LCH4). The oxidizer
side is represented only through the mixture ratio when converting chamber
mass flow to coolant flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .chamber_geometry import ChamberContour, CoolingChannels
from .combustion import CombustionGas
from .fluids import Fluid
from .regen_model import RegenCoolingModel, RegenResult
from .slosh_model import SloshModel
from .tank_geometry import TankGeometry
from .thermal_model import G0, TankThermalModel


@dataclass
class BurnProfile:
    """Flight/burn inputs. All callables accept time [s].

    ``pc_of_t``: chamber pressure [Pa]; zero or negative = engine off.
    ``axial_accel_of_t``: axial acceleration (thrust + gravity) [m/s^2],
    used for slosh frequency and hydrostatic head.
    ``lateral_accel_of_t``: lateral excitation for the slosh mode [m/s^2].
    """

    pc_of_t: object
    axial_accel_of_t: object = lambda t: G0
    lateral_accel_of_t: object = lambda t: 0.0
    t_end: float = 30.0


@dataclass
class Engine:
    """Engine definition tying chamber, channels and combustion gas."""

    contour: ChamberContour
    channels: CoolingChannels
    gas: CombustionGas
    mixture_ratio: float = 3.4
    coolant_is_fuel: bool = True
    injector_dp_fraction: float = 0.2   # injector stiffness (of Pc), reporting

    @property
    def coolant_fraction(self) -> float:
        """Coolant share of total chamber mass flow."""
        mr = self.mixture_ratio
        return 1.0 / (1.0 + mr) if self.coolant_is_fuel else mr / (1.0 + mr)

    def mdot_total(self, Pc: float) -> float:
        return Pc * self.contour.At / self.gas.c_star


@dataclass
class CoupledConfig:
    dt: float = 0.25              # outer time step [s]
    regen_interval: float = 1.0   # re-solve regen every ... [s]
    c_mix: float = 5.0            # slosh->thermal mixing gain (parametric!)
    feed_dp: float = 0.5e5        # feed-line loss [Pa]
    pump_dp: float = 0.0          # feed pump pressure rise [Pa] (0 = pressure-fed)
    fill_cutoff: float = 0.03     # stop when fill drops below this
    min_pump_margin_warn: float = 0.0


@dataclass
class CoupledHistory:
    """Time histories of the coupled run (SI units)."""

    t: list = field(default_factory=list)
    # tank
    P_ullage: list = field(default_factory=list)
    T_bulk: list = field(default_factory=list)
    T_surface: list = field(default_factory=list)
    T_ullage: list = field(default_factory=list)
    fill_fraction: list = field(default_factory=list)
    boiloff_rate: list = field(default_factory=list)
    m_liquid: list = field(default_factory=list)
    # slosh
    slosh_freq: list = field(default_factory=list)
    slosh_mass_fraction: list = field(default_factory=list)
    slosh_zeta: list = field(default_factory=list)
    slosh_amplitude: list = field(default_factory=list)
    cg_axial: list = field(default_factory=list)
    cg_lateral_offset: list = field(default_factory=list)
    mixing_factor: list = field(default_factory=list)
    # engine/regen
    Pc: list = field(default_factory=list)
    mdot_coolant: list = field(default_factory=list)
    coolant_inlet_T: list = field(default_factory=list)
    coolant_inlet_P: list = field(default_factory=list)
    coolant_outlet_T: list = field(default_factory=list)
    coolant_outlet_P: list = field(default_factory=list)
    regen_dP: list = field(default_factory=list)
    peak_wall_T: list = field(default_factory=list)
    q_throat: list = field(default_factory=list)
    injector_margin: list = field(default_factory=list)
    boiling: list = field(default_factory=list)
    # snapshots of full axial distributions (time, RegenResult)
    regen_snapshots: list = field(default_factory=list)

    def asarrays(self) -> dict:
        return {
            k: np.asarray(v)
            for k, v in self.__dict__.items()
            if k != "regen_snapshots"
        }


class CoupledSimulator:
    """Drives tank + slosh + regen through a burn profile."""

    def __init__(
        self,
        fluid: Fluid,
        tank: TankGeometry,
        tank_model: TankThermalModel,
        slosh_model: SloshModel,
        engine: Engine,
        config: CoupledConfig | None = None,
        n_regen_points: int = 100,
    ):
        self.fluid = fluid
        self.tank = tank
        self.tank_model = tank_model
        self.slosh_model = slosh_model
        self.engine = engine
        self.cfg = config or CoupledConfig()
        self.regen = RegenCoolingModel(
            engine.contour, engine.channels, engine.gas, fluid
        )

    # ------------------------------------------------------------------ run
    def run(self, profile: BurnProfile, P0: float, fill0: float,
            T_liquid0: float | None = None) -> CoupledHistory:
        cfg = self.cfg
        y = self.tank_model.initial_state(P0, fill0, T_liquid=T_liquid0)
        slosh_xv = np.array([0.0, 0.0])
        hist = CoupledHistory()
        t = 0.0
        last_regen_t = -1e9
        regen_res: RegenResult | None = None

        while t < profile.t_end - 1e-9:
            dt = min(cfg.dt, profile.t_end - t)
            Pc = max(float(profile.pc_of_t(t)), 0.0)
            accel = max(float(profile.axial_accel_of_t(t)), 0.01)
            mdot_cool = (
                self.engine.mdot_total(Pc) * self.engine.coolant_fraction
                if Pc > 0 else 0.0
            )

            # --- current tank state / geometry
            m_u, U_u, m_l, T_b, T_s, T_wd, T_ww = y
            rho_l = self.fluid.sat_liquid(
                max(self.fluid.P_sat(min(T_b, self.fluid.T_crit - 0.5)), 1e3)
            ).rho
            V_l = m_l / rho_l
            fill = V_l / self.tank.V_total
            if fill < cfg.fill_cutoff:
                break
            h_fill = self.tank.height_at_volume(V_l)

            # --- ullage pressure for feed calc
            V_u = max(self.tank.V_total - V_l, 1e-6)
            P_u, T_u = self.tank_model._ullage_PT(m_u, U_u, V_u)

            # --- slosh parameters + one outer step of the first mode
            sp = self.slosh_model.params_at(V_l, P_u, accel)
            w, z = sp.first.omega, sp.zeta_viscous

            def slosh_rhs(tt, xv):
                return [
                    xv[1],
                    -2 * z * w * xv[1] - w**2 * xv[0]
                    - float(profile.lateral_accel_of_t(tt)),
                ]

            sol = solve_ivp(slosh_rhs, (t, t + dt), slosh_xv, rtol=1e-7,
                            atol=1e-10, max_step=min(dt, 0.05))
            slosh_xv = sol.y[:, -1]
            x1 = float(slosh_xv[0])

            # --- slosh -> thermal mixing factor (parametric knob, see README)
            Phi = 1.0 + cfg.c_mix * min(abs(x1) / self.tank.R, 1.0)

            # --- regen quasi-steady solve at the current tank outlet state
            if Pc > 0 and (t - last_regen_t) >= cfg.regen_interval - 1e-9:
                P_inlet = P_u + rho_l * accel * h_fill - cfg.feed_dp + cfg.pump_dp
                regen_res = self.regen.solve(
                    Pc, mdot_cool, T_inlet=float(T_b),
                    P_inlet=float(max(P_inlet, 1.2 * Pc)),
                )
                last_regen_t = t
                hist.regen_snapshots.append((t, regen_res))

            # --- record
            hist.t.append(t)
            hist.P_ullage.append(P_u)
            hist.T_bulk.append(float(T_b))
            hist.T_surface.append(float(T_s))
            hist.T_ullage.append(float(T_u))
            hist.fill_fraction.append(fill)
            hist.m_liquid.append(float(m_l))
            hist.slosh_freq.append(sp.first.frequency)
            hist.slosh_mass_fraction.append(sp.slosh_mass_fraction)
            hist.slosh_zeta.append(sp.zeta_viscous)
            hist.slosh_amplitude.append(x1)
            hist.cg_axial.append(self.tank.liquid_cg_height(h_fill))
            hist.cg_lateral_offset.append(sp.first.mass * x1 / sp.m_liquid)
            hist.mixing_factor.append(Phi)
            hist.Pc.append(Pc)
            hist.mdot_coolant.append(mdot_cool)
            deriv = self.tank_model.rhs(t, y, mdot_cool, accel, Phi)
            hist.boiloff_rate.append(float(-deriv[2] - mdot_cool))
            if regen_res is not None and Pc > 0:
                hist.coolant_inlet_T.append(regen_res.coolant_inlet.T)
                hist.coolant_inlet_P.append(regen_res.coolant_inlet.P)
                hist.coolant_outlet_T.append(regen_res.coolant_outlet.T)
                hist.coolant_outlet_P.append(regen_res.coolant_outlet.P)
                hist.regen_dP.append(regen_res.dP_total)
                hist.peak_wall_T.append(regen_res.peak_wall_temperature)
                hist.q_throat.append(
                    float(regen_res.q[self.engine.contour.i_throat])
                )
                hist.injector_margin.append(
                    regen_res.coolant_outlet.P
                    - Pc * (1.0 + self.engine.injector_dp_fraction)
                )
                hist.boiling.append(regen_res.boiling_detected)
            else:
                for key in ("coolant_inlet_T", "coolant_inlet_P",
                            "coolant_outlet_T", "coolant_outlet_P",
                            "regen_dP", "peak_wall_T", "q_throat",
                            "injector_margin"):
                    getattr(hist, key).append(np.nan)
                hist.boiling.append(False)

            # --- advance the tank ODEs over this step
            sol = solve_ivp(
                lambda tt, yy: self.tank_model.rhs(tt, yy, mdot_cool, accel, Phi),
                (t, t + dt), y, method="RK45", rtol=1e-6,
                atol=1e-8 * np.maximum(np.abs(y), 1.0),
            )
            if not sol.success:
                raise RuntimeError(f"coupled tank step failed at t={t}: {sol.message}")
            y = sol.y[:, -1]
            t += dt

        return hist
