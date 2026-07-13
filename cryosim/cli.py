"""Command-line interface.

Usage::

    cryosim run examples/lch4_coupled_burn.yaml -o output/
    cryosim run examples/lox_tank_selfpress.yaml -o output/
    cryosim pid examples/lch4_coupled_burn.yaml -o output/

``run`` executes either a coupled burn simulation (``mode: coupled``) or a
tank-only thermal/slosh simulation (``mode: tank_only``) from a YAML config
and writes CSV histories + PNG plots. ``pid`` renders the rule-based
feed-system recommendation for the config's ``feed_system`` section.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from .chamber_geometry import ChamberContour, CoolingChannels
from .combustion import CombustionGas, gas_preset
from .coupling import BurnProfile, CoupledConfig, CoupledSimulator, Engine
from .fluids import Fluid
from .pid_recommender import (
    FeedSystemConfig,
    TankSpec,
    draw_pid,
    recommend_feed_system,
)
from .slosh_model import SloshModel
from .tank_geometry import TankGeometry
from .thermal_model import G0, TankThermalConfig, TankThermalModel
from . import plots


def _table_or_scalar(spec):
    """YAML value -> callable(t). Accepts a scalar or [[t, v], ...] table."""
    if isinstance(spec, (int, float)):
        v = float(spec)
        return lambda t: v
    tab = np.asarray(spec, dtype=float)
    return lambda t: float(np.interp(t, tab[:, 0], tab[:, 1]))


def build_tank(cfg: dict) -> tuple[Fluid, TankGeometry, TankThermalModel]:
    fluid = Fluid(cfg.get("fluid", "LOX"))
    tk = cfg["tank"]
    geom = TankGeometry(
        radius=float(tk["radius"]),
        cyl_length=float(tk["cyl_length"]),
        bottom_dome=tk.get("bottom_dome", "elliptical"),
        top_dome=tk.get("top_dome", "elliptical"),
        dome_aspect=float(tk.get("dome_aspect", 2.0)),
    )
    tcfg = TankThermalConfig()
    for key, val in tk.get("thermal", {}).items():
        if not hasattr(tcfg, key):
            raise KeyError(f"unknown tank.thermal option {key!r}")
        setattr(tcfg, key, float(val) if val is not None else None)
    model = TankThermalModel(
        fluid, geom,
        wall_mass=float(tk["wall_mass"]),
        heat_flux=float(tk.get("heat_flux", 0.0)),
        config=tcfg,
    )
    return fluid, geom, model


def build_engine(cfg: dict) -> Engine:
    en = cfg["engine"]
    contour = ChamberContour(
        throat_radius=float(en["throat_radius"]),
        contraction_ratio=float(en.get("contraction_ratio", 8.0)),
        expansion_ratio=float(en.get("expansion_ratio", 4.0)),
        chamber_length=float(en.get("chamber_length", 0.15)),
        n_points=int(en.get("n_points", 100)),
    )
    ch = en["channels"]
    channels = CoolingChannels(
        n_channels=int(ch["n_channels"]),
        channel_width=float(ch["channel_width"]),
        channel_height=float(ch["channel_height"]),
        t_wall=float(ch["t_wall"]),
        k_wall=float(ch.get("k_wall", 350.0)),
        helix_angle_deg=float(ch.get("helix_angle_deg", 0.0)),
        roughness=float(ch.get("roughness", 3.0e-6)),
    )
    gas_spec = en.get("gas", "lox/ch4")
    if isinstance(gas_spec, str):
        gas = gas_preset(gas_spec)
    else:
        gas = CombustionGas(
            T_c=float(gas_spec["T_c"]),
            gamma=float(gas_spec["gamma"]),
            molar_mass=float(gas_spec["molar_mass"]),
            mu=float(gas_spec["mu"]) if "mu" in gas_spec else None,
            Pr=float(gas_spec["Pr"]) if "Pr" in gas_spec else None,
        )
    return Engine(
        contour=contour, channels=channels, gas=gas,
        mixture_ratio=float(en.get("mixture_ratio", 3.4)),
        coolant_is_fuel=bool(en.get("coolant_is_fuel", True)),
        injector_dp_fraction=float(en.get("injector_dp_fraction", 0.2)),
    )


def run_coupled(cfg: dict, outdir: str) -> None:
    fluid, geom, tmodel = build_tank(cfg)
    engine = build_engine(cfg)
    slosh = SloshModel(fluid, geom,
                       zeta_override=cfg.get("slosh", {}).get("zeta_override"))

    cc = cfg.get("coupled", {})
    ccfg = CoupledConfig(
        dt=float(cc.get("dt", 0.25)),
        regen_interval=float(cc.get("regen_interval", 1.0)),
        c_mix=float(cc.get("c_mix", 5.0)),
        feed_dp=float(cc.get("feed_dp", 0.5e5)),
        pump_dp=float(cc.get("pump_dp", 0.0)),
        fill_cutoff=float(cc.get("fill_cutoff", 0.03)),
    )

    burn = cfg["burn"]
    lat = burn.get("lateral_accel", {})
    amp = float(lat.get("amplitude", 0.0))
    freq = float(lat.get("frequency", 1.0))
    profile = BurnProfile(
        pc_of_t=_table_or_scalar(burn["pc"]),
        axial_accel_of_t=(lambda f=_table_or_scalar(
            burn.get("axial_accel_g", 1.0)): lambda t: f(t) * G0)(),
        lateral_accel_of_t=lambda t: amp * np.sin(2 * np.pi * freq * t),
        t_end=float(burn["t_end"]),
    )

    init = cfg["initial"]
    sim = CoupledSimulator(fluid, geom, tmodel, slosh, engine, ccfg)
    print(f"running coupled burn: {profile.t_end:.1f} s ...")
    hist = sim.run(
        profile,
        P0=float(init["pressure"]),
        fill0=float(init["fill_fraction"]),
        T_liquid0=float(init["T_liquid"]) if init.get("T_liquid") else None,
    )
    a = hist.asarrays()

    plots.write_history_csv(a, os.path.join(outdir, "coupled_history.csv"))
    plots.plot_coupled_tank(a, os.path.join(outdir, "tank.png"))
    plots.plot_coupled_slosh(a, os.path.join(outdir, "slosh.png"))
    plots.plot_coupled_regen(a, os.path.join(outdir, "regen.png"))
    if hist.regen_snapshots:
        t_mid, res_mid = hist.regen_snapshots[len(hist.regen_snapshots) // 2]
        plots.plot_regen_distribution(
            res_mid, engine.contour.x_throat,
            os.path.join(outdir, "regen_axial.png"),
            title=f"Regen cooling — axial distributions at t = {t_mid:.1f} s",
        )
        if any(r.boiling_detected for _, r in hist.regen_snapshots):
            print("WARNING: two-phase coolant detected in the jacket — "
                  "single-phase correlations invalid there (no boiling model)")
        if any(r.pressure_collapsed for _, r in hist.regen_snapshots):
            print("WARNING: coolant pressure collapsed in the jacket — feed "
                  "pressure cannot sustain the flow; results downstream of "
                  "the collapse are not physical")
    print(f"done: {len(a['t'])} steps -> {outdir}/")
    print(f"  fill {a['fill_fraction'][0]*100:.0f}% -> {a['fill_fraction'][-1]*100:.0f}%, "
          f"ullage P {a['P_ullage'][-1]/1e5:.2f} bar")
    ok = ~np.isnan(a["coolant_outlet_T"])
    if ok.any():
        print(f"  injector inlet T {np.nanmin(a['coolant_outlet_T']):.0f}-"
              f"{np.nanmax(a['coolant_outlet_T']):.0f} K, "
              f"peak wall {np.nanmax(a['peak_wall_T']):.0f} K, "
              f"min injector margin {np.nanmin(a['injector_margin'])/1e5:.1f} bar")


def run_tank_only(cfg: dict, outdir: str) -> None:
    fluid, geom, tmodel = build_tank(cfg)
    slosh = SloshModel(fluid, geom,
                       zeta_override=cfg.get("slosh", {}).get("zeta_override"))
    sim = cfg.get("sim", {})
    t_end = float(sim.get("t_end", 3600.0))
    accel = float(sim.get("accel_g", 1.0)) * G0
    mdot = _table_or_scalar(sim.get("mdot_out", 0.0))

    init = cfg["initial"]
    y0 = tmodel.initial_state(
        float(init["pressure"]), float(init["fill_fraction"]),
        T_liquid=float(init["T_liquid"]) if init.get("T_liquid") else None,
    )
    print(f"running tank-only simulation: {t_end:.0f} s ...")
    hist = tmodel.simulate((0.0, t_end), y0, mdot_out=mdot, accel=accel,
                           n_out=int(sim.get("n_out", 150)),
                           rtol=float(sim.get("rtol", 1e-5)))
    plots.plot_tank_history(
        hist, os.path.join(outdir, "tank.png"),
        title=f"{fluid.name} tank self-pressurization",
    )
    a = {k: getattr(hist, k) for k in
         ("t", "P", "T_ullage", "T_surface", "T_bulk", "m_liquid", "m_ullage",
          "fill_fraction", "boiloff_rate")}
    plots.write_history_csv(a, os.path.join(outdir, "tank_history.csv"))

    # slosh parameter sweep over the run (quasi-static)
    import matplotlib.pyplot as plt
    fr, mf, zt = [], [], []
    for i in range(len(hist.t)):
        V = hist.m_liquid[i] / fluid.sat_liquid(hist.P[i]).rho
        p = slosh.params_at(V, hist.P[i], accel)
        fr.append(p.first.frequency); mf.append(p.slosh_mass_fraction)
        zt.append(p.zeta_viscous)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(hist.t / 60, fr, color=plots.C_BLUE)
    ax[0].set(xlabel="time [min]", ylabel="1st mode frequency [Hz]")
    ax[1].plot(hist.t / 60, np.array(mf) * 100, color=plots.C_ORANGE)
    ax[1].set(xlabel="time [min]", ylabel="slosh mass fraction [%]")
    ax[2].plot(hist.t / 60, np.array(zt) * 100, color=plots.C_GREEN)
    ax[2].set(xlabel="time [min]", ylabel="viscous damping ratio [%]")
    fig.suptitle("Slosh analog parameters (quasi-static)")
    fig.savefig(os.path.join(outdir, "slosh_params.png"), dpi=150)
    plt.close(fig)
    print(f"done -> {outdir}/  (P: {hist.P[0]/1e5:.2f} -> {hist.P[-1]/1e5:.2f} bar)")


def run_pid(cfg: dict, outdir: str) -> None:
    fs = cfg.get("feed_system")
    if fs is None:
        raise SystemExit("config has no feed_system section")
    if fs.get("pressurization") == "optimal":
        # let the rule set pick the architecture from the sim config
        from .pid_recommender import select_optimal_architecture

        burn = cfg.get("burn", {})
        pc_spec = burn.get("pc", 20e5)
        pc = float(pc_spec) if isinstance(pc_spec, (int, float)) else \
            max(float(v) for _, v in pc_spec)
        fcfg, rationale = select_optimal_architecture(
            oxidizer=fs.get("oxidizer", "LOX"),
            fuel=cfg.get("fluid", "LCH4"),
            coolant_is_fuel=bool(cfg.get("engine", {}).get("coolant_is_fuel",
                                                           True)),
            chamber_pressure=pc,
            burn_time=float(burn.get("t_end", 20.0)),
        )
        print("selected architecture:", fcfg.pressurization)
        for r in rationale:
            print("  -", r)
    else:
        fcfg = FeedSystemConfig(
            pressurization=fs.get("pressurization", "regulated"),
            tanks=[TankSpec(**t) for t in fs.get("tanks", [])] or
                  FeedSystemConfig().tanks,
            regen_cooled=bool(fs.get("regen_cooled", True)),
        )
    rec = recommend_feed_system(fcfg)

    # ---- automatic line sizing + fitting selection (needs engine + burn)
    line_labels, specs = {}, None
    if "engine" in cfg and "burn" in cfg:
        specs, line_labels = _size_lines(cfg, fcfg)

    txt_path = os.path.join(outdir, "feed_system.txt")
    with open(txt_path, "w") as fh:
        fh.write(rec.as_text() + "\n")
        if specs:
            from .line_sizing import report

            fh.write("\n\n" + report(specs) + "\n")
    draw_pid(rec, os.path.join(outdir, "feed_system_pid.svg"),
             line_labels=line_labels)
    written = [txt_path, f"{outdir}/feed_system_pid.svg"]
    if specs:
        from .fitting_diagrams import draw_fittings

        draw_fittings(specs, os.path.join(outdir, "fittings.png"))
        written.append(f"{outdir}/fittings.png")
    print(rec.as_text())
    if specs:
        from .line_sizing import report

        print("\n" + report(specs))
    print(f"\nwritten: {', '.join(written)}")


def _size_lines(cfg: dict, fcfg) -> tuple[list, dict]:
    """Size the feed lines from the sim config (heuristic MAWPs, see
    line_sizing docstring)."""
    from .fluids import Fluid
    from .line_sizing import size_feed_system

    engine = build_engine(cfg)
    burn = cfg.get("burn", {})
    pc_spec = burn.get("pc", 20e5)
    pc = float(pc_spec) if isinstance(pc_spec, (int, float)) else \
        max(float(v) for _, v in pc_spec)
    mdot_tot = engine.mdot_total(pc)
    mr = engine.mixture_ratio
    mdot_f = mdot_tot / (1.0 + mr)
    mdot_ox = mdot_tot - mdot_f

    ox_name = next((t.name for t in fcfg.tanks if t.oxidizer), "LOX")
    fuel_name = next((t.name for t in fcfg.tanks if not t.oxidizer),
                     cfg.get("fluid", "LCH4"))

    init_p = float(cfg.get("initial", {}).get("pressure", 3e5))
    setp = cfg.get("tank", {}).get("thermal", {}).get("pressurant_setpoint")
    tank_p = max(init_p, float(setp) if setp else 0.0)
    tank_meop = 1.5 * tank_p  # relief margin heuristic, documented

    pump_dp = float(cfg.get("coupled", {}).get("pump_dp", 0.0))
    pump_fed = fcfg.pressurization == "pump" or pump_dp > 0
    coolant_hp_mawp = 1.5 * (tank_p + pump_dp) if pump_fed else None

    # pressurant density at line conditions; He for regulated, else
    # autogenous fuel vapor
    if fcfg.pressurization == "regulated":
        rho_press = tank_p * 0.004 / (8.314 * 288.0)
    else:
        try:
            rho_press = Fluid(fuel_name).state_TP(250.0, tank_p).rho
        except Exception:
            rho_press = 3.0
    rho_ox = Fluid(ox_name).sat_liquid(max(tank_p, 1.2e5)).rho
    rho_f = Fluid(fuel_name).sat_liquid(max(tank_p, 1.2e5)).rho
    mdot_press = (mdot_ox / rho_ox + mdot_f / rho_f) * rho_press

    specs = size_feed_system(
        fluid_names={"oxidizer": ox_name, "fuel": fuel_name},
        mdots={"oxidizer": mdot_ox, "fuel": mdot_f,
               "pressurant": mdot_press},
        tank_meop=tank_meop,
        coolant_hp_mawp=coolant_hp_mawp,
        pressurant_rho=rho_press,
        liquid_rhos={"oxidizer": rho_ox, "fuel": rho_f},
        pump_fed=pump_fed,
    )
    labels = {}
    for s in specs:
        if s.name == f"{ox_name} feed line":
            labels["oxidizer"] = s.label()
        elif s.name == f"{fuel_name} feed line":
            labels["fuel"] = s.label()
        elif s.name == "pressurant header":
            labels["pressurant"] = s.label()
        elif s.name.startswith("coolant HP"):
            labels["coolant_hp"] = s.label()
    return specs, labels


def _pc_of(cfg: dict) -> float:
    pc_spec = cfg.get("burn", {}).get("pc", 20e5)
    return float(pc_spec) if isinstance(pc_spec, (int, float)) else \
        max(float(v) for _, v in pc_spec)


def _coolant_inlet(cfg: dict) -> tuple[float, float]:
    """(T, P) of the regen coolant inlet per the coupled-config heuristics."""
    init = cfg.get("initial", {})
    T_in = float(init.get("T_liquid") or 111.0)
    P_tank = float(init.get("pressure", 3e5))
    pump_dp = float(cfg.get("coupled", {}).get("pump_dp", 0.0))
    return T_in, P_tank + pump_dp


def run_optimize(cfg: dict, outdir: str) -> None:
    """Max-performance channel design search around the config's engine."""
    from .optimize import optimize_channels

    engine = build_engine(cfg)
    fluid = Fluid(cfg.get("fluid", "LCH4"))
    Pc = _pc_of(cfg)
    mdot_cool = engine.mdot_total(Pc) * engine.coolant_fraction
    T_in, P_in = _coolant_inlet(cfg)
    oc = cfg.get("optimize", {})
    print("optimizing channel design (this runs many regen solves)...")
    result = optimize_channels(
        engine.contour, engine.gas, fluid, Pc, mdot_cool, T_in, P_in,
        dp_budget=float(oc.get("dp_budget", 30e5)),
        k_wall=engine.channels.k_wall,
        n_random=int(oc.get("n_random", 40)),
        n_polish=int(oc.get("n_polish", 40)),
        seed=int(oc.get("seed", 1)),
        baseline=engine.channels,
    )
    print(result.report())
    with open(os.path.join(outdir, "optimized_channels.txt"), "w") as fh:
        fh.write(result.report() + "\n")
    plots.plot_regen_distribution(
        result.result, engine.contour.x_throat,
        os.path.join(outdir, "optimized_regen_axial.png"),
        title="Optimized channel design — axial distributions",
    )
    print(f"written: {outdir}/optimized_channels.txt, "
          f"{outdir}/optimized_regen_axial.png")


def run_gimbal(cfg: dict, outdir: str) -> None:
    """Slosh-coupled TVC stabilization of the configured (imaginary) rocket."""
    from .gimbal_control import (GimbalController, VehicleModel,
                                 plot_gimbal, simulate_gimbal)
    from .slosh_model import SloshModel

    fluid, geom, _ = build_tank(cfg)
    gc = cfg.get("gimbal", {})
    engine = build_engine(cfg) if "engine" in cfg else None
    Pc = _pc_of(cfg)
    thrust = float(gc.get("thrust", 1.6 * Pc * engine.contour.At
                          if engine else 5000.0))
    fill = float(gc.get("fill_fraction",
                        cfg.get("initial", {}).get("fill_fraction", 0.7)))
    accel_g = float(gc.get("accel_g", 4.0))
    P_tank = float(cfg.get("initial", {}).get("pressure", 3e5))
    sp = SloshModel(fluid, geom).params_at(fill * geom.V_total, P_tank,
                                           accel=accel_g * G0)
    vehicle = VehicleModel.from_components(
        dry_mass=float(gc.get("dry_mass", 60.0)),
        dry_cg=float(gc.get("dry_cg", 1.7)),
        dry_inertia=float(gc.get("dry_inertia", 45.0)),
        tank_bottom=float(gc.get("tank_bottom", 0.9)),
        slosh=sp,
        engine_station=float(gc.get("engine_station", 0.0)),
        thrust=thrust,
    )
    ctl = GimbalController.auto_tune(
        vehicle,
        bandwidth_hz=float(gc.get("bandwidth_hz", 1.0)),
        damping=float(gc.get("damping", 0.7)),
    )
    gust = gc.get("gust", {"force": 120.0, "t0": 1.0, "duration": 0.4})
    dist = (lambda t: float(gust["force"])
            if float(gust["t0"]) < t < float(gust["t0"]) + float(gust["duration"])
            else 0.0)
    # gust applied at a station (default: nose-ish, 1 m above the dry CG)
    gust_station = float(gust.get("station",
                                  float(gc.get("dry_cg", 1.7)) + 1.0))
    hist = simulate_gimbal(vehicle, ctl,
                           t_end=float(gc.get("t_end", 20.0)),
                           disturbance=dist,
                           x_disturbance=gust_station - vehicle.x_cg_datum)
    plot_gimbal(hist, os.path.join(outdir, "gimbal.png"))
    print(f"auto-tuned gains: Kp={ctl.Kp:.3f}, Kd={ctl.Kd:.3f} "
          f"(bandwidth {ctl.bandwidth/2/np.pi:.2f} Hz); slosh mode "
          f"{vehicle.omega_slosh/2/np.pi:.2f} Hz, "
          f"slosh mass {vehicle.m_slosh/vehicle.m_total*100:.1f}%")
    print(f"closed loop {'STABLE' if hist.stable else 'UNSTABLE'}; "
          f"max |theta| = {hist.max_theta_deg:.2f} deg; "
          f"settled = {hist.settled}")
    for w in hist.warnings:
        print(f"  ! {w}")
    print(f"written: {outdir}/gimbal.png")


def run_manifold(cfg: dict, outdir: str) -> None:
    """Automatic torus-manifold design for the configured jacket."""
    from .manifold_design import design_manifolds, draw_manifolds

    engine = build_engine(cfg)
    fluid = Fluid(cfg.get("fluid", "LCH4"))
    Pc = _pc_of(cfg)
    mdot_cool = engine.mdot_total(Pc) * engine.coolant_fraction
    T_in, P_in = _coolant_inlet(cfg)
    from .regen_model import RegenCoolingModel

    print("solving the jacket to get channel dp and outlet state...")
    res = RegenCoolingModel(engine.contour, engine.channels, engine.gas,
                            fluid).solve(Pc, mdot_cool, T_in, P_in)
    mc = cfg.get("manifold", {})
    design = design_manifolds(
        engine.contour, engine.channels, mdot_cool,
        dp_channel=res.dP_total,
        rho_in=res.coolant_inlet.rho, mu_in=res.coolant_inlet.mu,
        rho_out=res.coolant_outlet.rho, mu_out=res.coolant_outlet.mu,
        mawp=float(mc.get("mawp", 1.5 * P_in)),
        coolant_name=cfg.get("fluid", "LCH4"),
        target=float(mc.get("target_maldistribution", 0.03)),
        max_feeders=int(mc.get("max_feeders", 2)),
    )
    print(design.report())
    with open(os.path.join(outdir, "manifold_design.txt"), "w") as fh:
        fh.write(design.report() + "\n")
    draw_manifolds(design, engine.contour,
                   os.path.join(outdir, "manifolds.png"))
    print(f"written: {outdir}/manifold_design.txt, {outdir}/manifolds.png")


def run_cfd(cfg: dict, outdir: str) -> None:
    """Inviscid Euler check of the quasi-1D nozzle-flow assumption."""
    from .cfd_nozzle import NozzleEulerCFD, plot_cfd

    engine = build_engine(cfg)
    Pc = _pc_of(cfg)
    cc = cfg.get("cfd", {})
    solver = NozzleEulerCFD(engine.contour, engine.gas, Pc,
                            n_axial=int(cc.get("n_axial", 120)),
                            n_radial=int(cc.get("n_radial", 24)))
    print("running axisymmetric Euler solver (inviscid — flow-field check "
          "only, no wall heat transfer)...")
    res = solver.run(max_iter=int(cc.get("max_iter", 6000)))
    x, mdot = res.mdot_profile()
    m1d = res.mdot_quasi1d()
    mc = res.centerline_mach()[1]
    m1 = res.quasi1d_mach()
    print(f"mass flow: CFD {np.mean(mdot[5:-5]):.3f} kg/s vs quasi-1D "
          f"{m1d:.3f} kg/s ({(np.mean(mdot[5:-5])-m1d)/m1d*100:+.1f}%)")
    print(f"exit Mach: CFD centerline {mc[-1]:.2f} vs quasi-1D {m1[-1]:.2f}")
    if not res.converged:
        print("note: residual not fully converged — treat as qualitative; "
              "raise max_iter/refine grid for quantitative use")
    plot_cfd(res, os.path.join(outdir, "cfd_nozzle.png"))
    print(f"written: {outdir}/cfd_nozzle.png")


def run_design(args) -> None:
    """Autonomous requirements-to-engine design (see engine_design.py)."""
    from .design_report import generate_package
    from .engine_design import DesignError, EngineSpec, design_engine

    spec = EngineSpec.from_yaml(args.config)
    print(f"designing '{spec.name}': {spec.thrust/1e3:.1f} kN "
          f"{spec.propellants} at Pc {spec.chamber_pressure/1e5:.0f} bar "
          "(deterministic encoded-rule pipeline — full trace below)")
    n = 20 if args.fast else 40
    try:
        design = design_engine(spec, n_random=n, n_polish=n, verbose=True)
    except DesignError as e:
        raise SystemExit(f"DESIGN FAILED\n{e}") from None
    print()
    print(design.describe())
    print()
    paths = generate_package(
        design, args.outdir, voxel_jacket_mm=args.voxel_jacket,
        voxel_injector_mm=args.voxel_injector,
        with_geometry=not args.no_geometry, verbose=True)
    print("\nwritten:")
    for name, p in paths.items():
        print(f"  {name:16s} {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="cryosim",
        description="Coupled slosh + thermal + regen-cooling simulator "
                    "for small cryogenic bi-propellant rockets",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("run", "run a simulation from a YAML config"),
        ("pid", "feed-system P&ID recommendation + line sizing"),
        ("optimize", "max-performance cooling-channel design search"),
        ("gimbal", "slosh-coupled TVC stabilization study"),
        ("cfd", "axisymmetric Euler check of the quasi-1D assumption"),
        ("manifold", "automatic torus-manifold design for the jacket"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="YAML configuration file")
        p.add_argument("-o", "--outdir", default="output",
                       help="output directory (default: output/)")
    pb = sub.add_parser("benchmark",
                        help="computed vs literature/reference value table")
    pb.add_argument("-o", "--outdir", default="output")
    pb.add_argument("--full", action="store_true",
                    help="add model-level solves (regen/CFD/K-site; slower)")
    pd = sub.add_parser(
        "design",
        help="autonomous requirements-to-engine design: contour, cooling, "
             "injector, manifolds + 3D-printable STLs (Noyron-style)")
    pd.add_argument("config", help="engine requirements YAML")
    pd.add_argument("-o", "--outdir", default="output")
    pd.add_argument("--voxel-jacket", type=float, default=0.5,
                    help="chamber-jacket voxel size [mm] (default 0.5)")
    pd.add_argument("--voxel-injector", type=float, default=0.3,
                    help="injector-head voxel size [mm] (default 0.3)")
    pd.add_argument("--no-geometry", action="store_true",
                    help="skip STL meshing (design + report only)")
    pd.add_argument("--fast", action="store_true",
                    help="smaller channel search for a quick look")
    args = ap.parse_args(argv)

    if args.cmd == "design":
        run_design(args)
        return

    if args.cmd == "benchmark":
        from .benchmarks import run_benchmarks, to_markdown, to_text

        os.makedirs(args.outdir, exist_ok=True)
        rows = run_benchmarks(full=args.full)
        print(to_text(rows))
        path = os.path.join(args.outdir, "benchmarks.md")
        with open(path, "w") as fh:
            fh.write(to_markdown(rows) + "\n")
        print(f"\nwritten: {path}")
        return

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(args.outdir, exist_ok=True)

    if args.cmd == "pid":
        run_pid(cfg, args.outdir)
    elif args.cmd == "optimize":
        run_optimize(cfg, args.outdir)
    elif args.cmd == "gimbal":
        run_gimbal(cfg, args.outdir)
    elif args.cmd == "cfd":
        run_cfd(cfg, args.outdir)
    elif args.cmd == "manifold":
        run_manifold(cfg, args.outdir)
    elif cfg.get("mode", "coupled") == "tank_only":
        run_tank_only(cfg, args.outdir)
    else:
        run_coupled(cfg, args.outdir)


if __name__ == "__main__":
    main()
