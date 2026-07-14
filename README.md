# cryosim — coupled slosh + thermal + regen-cooling simulator

A reduced-order Python tool for **small cryogenic bi-propellant rockets**
(student-team scale: tank diameters ~0.2–0.5 m, engines of a few kN) that
couples, over a user-specified burn/flight profile:

1. **Mechanical slosh dynamics** — NASA SP-8009 equivalent-pendulum analog:
   slosh frequency, slosh mass fraction, damping, and CG shift vs. fill level,
   tank radius, and axial acceleration;
2. **Lumped-parameter tank thermal/pressurization** — ullage gas, stratified
   surface layer, bulk liquid, and wall nodes with mass/energy conservation:
   self-pressurization and boil-off over time;
3. **1D regenerative-cooling model** — Bartz hot-gas correlation +
   Dittus-Boelter coolant-side convection + 1D wall conduction marched along
   the chamber/throat/nozzle contour: wall temperatures, coolant enthalpy
   rise, and pressure drop;
4. **Coupling** — slosh-induced mixing perturbs tank stratification and
   pressure; the propellant drawn from the tank (at its current outlet state)
   **is** the regen coolant; the heated regen outlet state is tracked as the
   **injector inlet condition** through the burn;
5. A **rule-based feed-system P&ID recommender**: pass a scheme (regulated /
   blowdown / autogenous / pump-fed) or `pressurization: optimal` to let a
   transparent rule set pick the architecture from the fluids and chamber
   pressure (e.g., methane coolant at Pc ≥ ~20 bar needs supercritical
   channel pressure above what a saturated-ullage tank can supply →
   pump-fed, with the reasoning printed). The output is an ISA-5.1-flavored
   schematic with drawn valve/instrument symbols, routed process and
   instrument lines (vent + relief + burst-disc manifolds, pressurant
   header with per-tank check valves, feed trains, regen coolant loop to
   the injector), a described symbol legend, fluid-colored line key, and a
   title block — plus the component list with one-line rationales.
   The `pid` command also performs **automatic line sizing and fitting
   selection** (`cryosim/line_sizing.py`): each line (feed, fill/drain,
   vent/relief, pressurant header, pump-discharge coolant run) gets the
   smallest standard tube (1/8"–2" OD, standard walls) meeting a
   per-service velocity target (liquid ~5 m/s, pump suction 3 m/s,
   discharge 8 m/s, gas 25 m/s, vent 50 m/s — Huzel & Huang ch. 8 practice)
   with the wall from Barlow's formula against the material allowable
   stress (ASME B31.3-style, ×1.25 design factor, 0.028" minimum handling
   wall), reporting velocity, AN dash size, and friction Δp/m. Fitting
   types follow cryo/pressure rules (37° flare or orbital weld for cryo —
   NPT rejected; twin-ferrule for ambient gas; weld/flange ≥1" OD) and a
   **fitting cross-section diagram sheet** (`fittings.png`) is generated
   with the sized-line table; tube specs are annotated on the P&ID runs.
   Sizing heuristics and their limits (no surge/water-hammer or bend
   analysis) are documented in the module docstring and the output itself.

**Design choice, stated up front: the design models are reduced-order, not
CFD.** Every design sub-model is the fast, analytical/empirical model that
student teams (and industry, for early design iterations) actually use
before committing to CFD: mechanical slosh analogs, lumped-node tank
thermodynamics, and Bartz-class 1D cooling analysis. It trades local
fidelity for speed, transparency, and coverage of coupled system behavior.
One genuine CFD solver IS included (`cryosim cfd`) — an **inviscid
axisymmetric Euler** finite-volume solver whose only job is to *verify* the
quasi-1D flow assumption on your actual contour; being inviscid it has no
boundary layers, no turbulence, and predicts **no wall heat transfer** — it
does not, and cannot, replace the Bartz model or RANS for thermal design.

Beyond the core coupled simulation, an **autonomous engine-design pipeline**
and three design-study tools are included:

* **`cryosim design`** — a requirements-to-hardware generative design
  pipeline in the spirit of **LEAP 71's Noyron** computational model: you
  state *only* top-level requirements (thrust, propellants, chamber
  pressure, ambient) and deterministic encoded engineering rules derive the
  whole engine — nozzle/contour sizing, the cooling-channel layout (via the
  optimizer below), a coaxial-swirl injector head fed with the *regen-outlet*
  fuel state, torus manifolds, and the structural closeout — iterating
  against an explicit constraint ledger with ordered repair rules (e.g.
  two-phase coolant in the jacket → supercritical pump-fed feed, the same
  conclusion the coupled demo reaches by simulation). Like Noyron, the "AI"
  is **not** machine learning: it is traceable, reproducible engineering
  logic, and every decision is written to `trace.md`. The output package is
  LEAP 71-style too: **watertight, 3D-printable STL geometry** (chamber
  jacket with the helical channels and manifolds printed in, plus the
  injector head with swirlers, annuli, tangential ports and film-cooling
  ring) generated by a small PicoGK-like implicit/voxel kernel
  (`cryosim/voxel_geometry.py`: signed-distance solids → marching cubes →
  closed meshes with per-part watertightness/volume/mass QA), a
  cross-section drawing, the regen solution, and a performance report.

* **`cryosim optimize`** — max-performance cooling-channel search
  (channel count/width/height/wall thickness) minimizing peak wall
  temperature under a pressure-drop budget, with manufacturability floors
  deliberately treated as *warnings*, not constraints: the optimizer is
  allowed to find hard-to-manufacture geometry, and on the shipped 5 kN
  LOX/CH4 example it independently rediscovers the NASA **high-aspect-ratio
  cooling channel (HARCC)** configuration (many narrow, deep channels:
  peak wall 1014 → 711 K *and* lower Δp; cf. Wadel & Meyer, AIAA 96-2584),
  flagged with the manufacturing process each feature actually needs.
* **`cryosim cfd`** — the Euler flow-field check described above (first-
  order Rusanov scheme; on the demo contour: mass flow within ~3% of
  quasi-1D, exit Mach −7%, both shrinking under grid refinement).
* **`cryosim manifold`** — automatic design of the jacket's torus
  manifolds: the inlet (dividing) and outlet (combining) headers are
  marched with classical 1D manifold theory (momentum regain + friction;
  Bajura & Jones 1976 / Shah & Sekulić ch. 12), the duct diameters are the
  smallest meeting a channel-flow uniformity target under a 30 m/s header
  velocity cap, feeder count and inlet/outlet port clocking are chosen
  automatically, and walls are sized with the toroidal-shell inner-crotch
  stress factor. Motivated by the measured effect of manifolds on LOX/CH4
  regen wall temperatures (J. Thermal Science 29, 2020). Smooth-shell
  sizing only — bosses/welds/cutouts need separate stress analysis.
* **`cryosim gimbal`** — automatic TVC stabilization of an imaginary rigid
  rocket **including the slosh pendulum** from the slosh model: a PD
  attitude controller auto-tuned by pole placement on the vehicle's own
  inertia/thrust/geometry (signed plant gain), actuator lag + angle/rate
  limits, closed-loop eigenvalue stability verdict, and the classic
  slosh/TVC interaction warning when the control bandwidth approaches the
  slosh frequency. Linear, planar, no aerodynamics — an interaction-study
  tool, not a 6-DOF (stated in the module docstring).

---

**On your phone?** The full input-to-output walkthrough with every plot
is in [docs/RESULTS.md](docs/RESULTS.md) — GitHub renders it inline.

## Install & run

```bash
pip install -e .           # needs Python >= 3.10; installs CoolProp, scipy, ...
pip install -e ".[dev]"    # + pytest

# end-to-end coupled burn (LOX/LCH4, methane-cooled, pump-fed):
cryosim run examples/lch4_coupled_burn.yaml -o output/lch4_burn

# tank-only LOX pad-hold self-pressurization + slosh parameters:
cryosim run examples/lox_tank_selfpress.yaml -o output/lox_pad

# feed-system component recommendation + schematic:
cryosim pid examples/lch4_coupled_burn.yaml -o output/pid

# autonomous requirements-to-engine design (report + printable STLs):
cryosim design examples/engine_5kn_methalox.yaml -o output/engine

# tests (fast unit tests + slower validation regressions):
pytest -q                  # everything
pytest -q -m "not validation"   # fast unit tests only
```

Outputs: CSV time histories and PNG plots of ullage pressure, tank node
temperatures, boil-off rate, slosh frequency/mass/CG shift, coolant outlet
(= injector inlet) temperature, wall-temperature and heat-flux distributions,
and coolant pressure drop / injector pressure margin.

All fluid properties come from **CoolProp** through one wrapper
(`cryosim/fluids.py`) used by both the tank and the coolant side, so the
working fluid is swappable by name: LOX (tank default), liquid methane
(coupled-demo default), ethanol, LH2 (used by the validation cases), N2, ...

---

## Physics and sources

### Slosh (`cryosim/slosh_model.py`)

Linear lateral slosh in an upright rigid cylinder, radius R, equivalent
flat-bottom depth h, axial acceleration a; `xi_n` = roots of J1'(xi)=0:

| Quantity | Expression |
|---|---|
| frequency | `omega_n^2 = (xi_n a / R) tanh(xi_n h/R)` |
| slosh mass | `m_n = m_liq * 2 tanh(xi_n h/R) / [(h/R) xi_n (xi_n^2 - 1)]` |
| pendulum length | `L_n = R / (xi_n tanh(xi_n h/R))` |
| mass depth below surface | `d_n = (R/xi_n) tanh(xi_n h/2R)` |
| smooth-wall damping | Mikishev–Dorozhkin: `zeta = 0.79 sqrt(nu/sqrt(a R^3)) * [depth corr.]` |

Sources: NASA SP-8009 (Dodge, 1968); NASA SP-106 (Abramson, 1966); Dodge,
*The New Dynamic Behavior of Liquids in Moving Containers* (SwRI, 2000).
Dome ends are handled by the SP-8009 equal-volume flat-bottom depth; the
model flags (`in_dome`) states where the free surface is inside a dome.

### Tank thermal / self-pressurization (`cryosim/thermal_model.py`)

Nodes: real-vapor ullage (CoolProp (rho,u) flash for pressure), saturated
interface at `T_sat(P)`, thin stratified surface layer, bulk liquid, dry/wet
wall (cryogenic cp(T), k(T) tables; axial liquid-line conduction). Interface
energy balance sets evaporation/condensation:
`mdot_evap = (Q_ullage->int + Q_surface->int)/h_fg`. Free convection via
Churchill–Chu / McAdams correlations (Incropera & DeWitt ch. 9). Optional
autogenous pressurant injection to a setpoint. The share of wetted-wall heat
routed to the surface layer (`chi`, default 0.25) is a **calibrated**
stratification parameter — see validation.

### Regenerative cooling (`cryosim/regen_model.py`, `cryosim/combustion.py`)

Per axial station, solved implicitly for the hot-wall temperature:

```
q = h_g (T_aw - T_wg) = (k_w/t_w)(T_wg - T_wc) = eta_fin h_c (T_wc - T_coolant)
```

* `h_g` — Bartz (1957) with the sigma property correction and turbulent
  recovery `r = Pr^(1/3)`;
* `h_c` — Dittus-Boelter `Nu = 0.023 Re^0.8 Pr^0.4` (+ optional Sieder-Tate
  viscosity-ratio and helix curvature corrections); channel lands as straight
  fins (Huzel & Huang ch. 4; NASA SP-8087);
* coolant march: enthalpy update per station; Darcy pressure drop with the
  Haaland friction factor + momentum (acceleration) term; local CoolProp
  properties capture transcritical methane behavior.

Combustion-gas properties (T_c, gamma, M) are computed by the built-in
equilibrium solver (`combustion_equilibrium.py`, above) for LOX +
CH4/H2/ethanol/propane; nominal presets remain for other fuels, and a
`combustion_gas` override accepts your own CEA/RPA values.

### Injector (`cryosim/injector_design.py`)

Ideal (inviscid) open swirl-injector theory via the principle of maximum
flow (Abramovich/Kliachko, as presented by Bazarov, Yang & Puri in *Liquid
Rocket Thrust Chambers*, AIAA Prog. vol. 200, 2004, ch. 2):

| Quantity | Expression |
|---|---|
| geometric characteristic | `A = R_in r_n / (n_t r_t^2)` |
| maximum-flow condition | `A = (1 - phi) sqrt(2) / (phi sqrt(phi))` |
| discharge coefficient | `mu = phi sqrt(phi) / sqrt(2 - phi)` |
| spray half-angle | `tan a = 2 mu A / sqrt((1 + sqrt(1-phi))^2 - 4 mu^2 A^2)` |

Element sizing follows Bazarov's classical algorithm (spray-angle target →
`A, phi, mu` → nozzle radius from the flow → tangential ports from `A`);
the fuel annulus is a plain orifice at the regen-outlet state; elements are
packed on concentric rings with pitch/wall-clearance rules; injector
stiffness defaults to 20 % Pc (chug-margin heuristic, Huzel & Huang ch. 4 /
NASA SP-8089). **No viscous-loss correction, no atomization/mixing/stability
prediction** — hydraulic sizing only, warnings flag gas-like (supercritical)
fuel where the incompressible orifice equation degrades.

### Autonomous design pipeline (`cryosim/engine_design.py`)

Stage A performance sizing (real **equilibrium-combustion** chamber gas —
see below — then ideal `c*`/`Cf` from the isentropic relations × fixed
efficiencies — `c*` 0.95, and a nozzle factor split into friction 0.987 ×
**divergence efficiency λ** so a bell earns a separable gain — perfect-
expansion area ratio with a Summerfield separation guard and vacuum cap,
contraction-ratio and L* rules; a **thrust-optimized bell** contour by
default) → stage B thermal (`optimize_channels` under the Δp budget with
process feature floors) → stage C injector (fed the stage-B regen outlet
state) → stage D manifolds → stage E closeout hoop stress (Barlow ×1.25 vs
`line_sizing.MATERIALS`). A constraint ledger (wall temperature, Δp,
single-phase coolant, land width, port size, supply pressure, flow
uniformity) drives ordered repair rules; failure is loud and carries the
ledger. Every decision lands in the design trace.

### Equilibrium combustion (`cryosim/combustion_equilibrium.py`)

Instead of hand-entered gas presets, the chamber state is computed from the
actual propellants by **Gibbs free-energy minimization** (element-potential
method, NASA CEA-style; Gordon & McBride RP-1311): for a CxHyOz fuel + LOX
at a given O/F and Pc it solves the equilibrium composition over 8 C/H/O
species (CO2, CO, H2O, H2, O2, OH, H, O) with NASA-7 thermodynamic
polynomials, closes the adiabatic energy balance for the flame temperature,
and returns T_c, frozen γ, mean molar mass and c*. A damped-Newton solve on
the element potentials makes it deterministic.

**Shifting-equilibrium expansion** (`shifting_c_star`, `shifting_isp`): the
chamber gas is expanded isentropically to the sonic throat (for c*) and on
to the exit area ratio (for Isp) with the composition **re-equilibrated at
each pressure** — so the recombination of dissociated radicals as the gas
cools is credited (the CEA "rocket" problem). This is what makes the
performance and the *optimum mixture ratio* correct: `optimize_of` on peak
vacuum Isp lands at O/F ≈ 3.2 for LOX/CH4 and ≈ 1.85 for LOX/ethanol —
matching CEA's mildly-rich optimum. (A *frozen* model, missing recombination,
biases the optimum too fuel-rich; the shifting expansion fixes it.)

Validated against published CEA points (LOX/CH4, LOX/H2, LOX/ethanol) to
within ~2–3% on T_c, c* and Isp, with the peak-Isp O/F in the CEA band —
see `validation/test_equilibrium_cea.py`. **Limits, stated plainly:** the
8-species set omits condensed carbon, so very-rich soot-forming mixtures
(LOX/CH4 below O/F ~1.6) are not captured; reactants are referenced at 298 K
(real cryo injection is colder → slightly lower T_c). LOX/H2's optimum comes
out a little rich (~3.8 vs CEA ~4.3), the hardest case for the reduced set.

### Nozzle contour (`cryosim/chamber_geometry.py`)

The divergent section is either a straight cone or, by default, a
**thrust-optimized bell** built by Rao's parabolic-approximation method: a
downstream throat arc swept to the parabola-start angle θ_n, then a
quadratic-Bézier parabola to the exit lip at angle θ_e (θ_n/θ_e from a
documented fit to Rao's 80%-bell chart — the exact chart isn't reproducible
offline). The bell is ~80% the length of a 15° cone and recovers most of
its divergence loss: divergence efficiency `λ = 0.5(1 + cos θ_exit)` (Sutton
eq. 3-34) rises from 0.983 (15° cone) to ~0.99–1.00, a ~0.7–1.5% Cf/Isp
gain that the pipeline credits. This is the standard preliminary-design
angularity correction, **not** a method-of-characteristics contour.

### Voxel geometry (`cryosim/voxel_geometry.py`)

Implicit solids (signed-distance-style fields: revolved profiles, helical
channel bands, tori, arbitrary cylinders, spheres, hole rings, half-spaces)
composed with hard **or smooth (filleted) booleans** — Quílez's polynomial
smooth-min gives the organic, "grown" transitions where the torus manifolds
blend into the chamber wall and the injector's spherical propellant **dome**
blends onto its barrel. The jacket also carries the practical hardware detail
real printed engines have — a ring of instrumentation/igniter bosses, rounded
mounting feet, and flared inlet stubs, all smooth-unioned on. Sampled on a
padded voxel grid, meshed with marching
cubes (scikit-image), written as binary STL with watertightness
(edge-pairing), volume (divergence theorem) and mass QA. Voxel-limited
fidelity: features under ~2 voxels round off or close — stated per part in
the report.

### Coupling (`cryosim/coupling.py`)

Outer time loop: tank ODEs integrated with the engine-demanded outflow
(`mdot = Pc At / c*` × coolant fraction) and flight acceleration; slosh
first mode integrated under the lateral-acceleration input with
quasi-static parameters; regen solved **quasi-steadily** (regen thermal time
constants ~ms ≪ tank time scales ~s) at a configurable interval with
coolant inlet = current tank outlet state (bulk T; ullage pressure +
hydrostatic head − feed loss + optional pump rise). Slosh feeds back into
the tank via a mixing factor `Phi = 1 + c_mix |x1|/R` that enhances
interfacial exchange and destratifies the surface layer.

---

## Method fidelity vs industry practice

Rocket design has two tiers of tools: fast **reduced-order / preliminary**
methods used to iterate a design, and high-fidelity **final-design**
solvers (full NASA CEA, method-of-characteristics nozzles, RANS/conjugate
CFD, FEA, then hot-fire) used to qualify it. This tool sits squarely in the
first tier — and, importantly, so does the *generation* loop of the
industry system it emulates (LEAP 71's Noyron generates from analytical
models + engineering logic, then hands off to external simulation to
verify). Every sub-model here is the *same method* the industry uses for
that first tier:

| Sub-model | Industry-standard method | cryosim | Standing |
|---|---|---|---|
| Combustion | NASA CEA — Gibbs-min equilibrium, shifting/frozen | 8-species Gibbs-min + shifting expansion | **Same method**, within ~2–3% of CEA (fewer species, no soot) |
| Nozzle contour | Rao thrust-optimized parabola (prelim); MOC (final) | Rao parabolic approximation | **Standard preliminary**; MOC is the higher tier |
| Nozzle performance | Cf × angularity(λ) × friction efficiencies | identical | **Standard** |
| Regen cooling | Bartz + Dittus–Boelter 1D (prelim); conjugate CFD (final) | identical | **Standard preliminary** |
| Injector | Bazarov swirl theory + empirical Cd | identical | **Standard preliminary** (no atomization/stability) |
| Manifolds | Bajura–Jones header theory | identical | **Standard** |
| Structures | Barlow/hoop stress + ASME allowables | identical | **Standard** |
| Slosh / TVC | NASA SP-8009 pendulum analog + linear control | identical | **Standard** (the accepted linear reference) |
| AM geometry | Implicit/voxel kernel (PicoGK-class) | small PicoGK-like kernel | **Emerging industry standard** (LEAP 71/nTop lineage) |

So: **the methods are industry-standard for preliminary design, correctly
implemented and cross-checked** (`cryosim benchmark` puts 22 computed
numbers next to NIST/CEA/textbook references; all within tolerance/band).
Where it is deliberately *not* "better than industry" is fidelity: it does
not replace full CEA, MOC, CFD or FEA, and it should not be used to qualify
flight hardware. It is faster and more transparent than those tools, which
is the point of the tier — not a substitute for them.

## Validation status — read this before trusting numbers

Each sub-model ships with regression tests (`validation/`, run via
`pytest -m validation`). **What was and wasn't possible is stated
explicitly**; the development environment could not download full NASA/paper
PDFs (network restricted to package registries), so validation anchors to
quantitative statements in report abstracts, exactly-computable analytical
references, and documented operating bands — never to silently invented
numbers.

| Sub-model | Reference | What is reproduced | Status |
|---|---|---|---|
| Slosh frequencies, masses, analog | NASA SP-106 ch. 2 / SP-8009 analytical solution (experimentally confirmed in those programs); Bessel eigenvalues via scipy | Dimensionless frequency table values, slosh-mass limits, CG conservation, exact resonant magnification | **Validated** against the published linear theory (the accepted design reference for linear slosh) |
| Slosh damping | Mikishev–Dorozhkin correlation as given in Dodge (2000) | Hand-evaluated correlation points, viscosity/depth scaling | **Correlation reproduced**; the correlation itself is empirical (±~30% scatter). No digitized TN D-1367 traces were accessible |
| Tank self-pressurization | NASA K-site flightweight 4.89 m³ LH2 tank: Hasan/Lin/Van Dresar (NTRS 19910011011), Van Dresar & Lin TM-105411 (NTRS 19920009200) | Quasi-steady dP/dt vs the analytic homogeneous rate falls in the report-stated band per fill level (≤~2× at 29/49%, >3× at 83% at 3.5 W/m²); homogeneous reference cross-checked by two independent implementations | **Band-validated**. Known limitation: the reports' non-monotonic fill trend (49% slowest) is NOT reproduced; mid-fill over-predicted ~20–30% (conservative for vent sizing). `chi` = 0.25 is calibrated to this dataset |
| Regen: friction/coolant side | Colebrook–White equation; Incropera & DeWitt | Haaland vs implicit Colebrook <2% over Re, roughness grid; Dittus-Boelter hand values | **Validated** (exact) |
| Regen: hot-gas side | Bartz (1957) | Independent hand-assembled evaluation of the published equation + scaling laws | **Implementation-verified**. Bartz itself over-predicts LOX/CH4 heat flux by ~20–30% (ODREC, Appl. Sci. 14(1):71, 2024; JAXA EUCASS 2017-381) — conservative; no silent correction applied |
| Regen: system level | CIRA HYPROB 30 kN LOX/LCH4 demonstrator class (96-channel methane-cooled jacket; the dataset ODREC validated against) | Model lands in the documented operating bands: throat flux tens of MW/m², copper wall peak at throat <1000 K, transcritical CH4 outlet 350–550 K, tens of bar jacket drop | **Band-validated only** — full tabulated HYPROB data not accessible offline; stated, not hidden |
| Slosh → thermal coupling | Ludwig & Dreyer, Cryogenics 63 (2014) — qualitative | Mixing knob direction only (slosh → destratification → pressure effect) | **UNVALIDATED**. `c_mix` is a parametric knob. No public quantitative dataset was found at this scale |
| Channel optimizer | NASA HARCC demonstration (Wadel & Meyer, AIAA 96-2584) | Optimizer independently converges on the high-aspect-ratio channel configuration known experimentally to cut wall temperature and Δp | **Concept-anchored** (the optimum's *character* matches the published result; magnitudes inherit the regen model's validation status) |
| Euler CFD | Quasi-1D isentropic theory (exact for smooth C-D nozzles; Anderson ch. 5) | Mass-flow bias ≤ ~3%, exit Mach ≤ ~7% low, both shrinking under refinement (first-order scheme); transonic at the geometric throat; mass conservation along the duct | **Verified vs the 1D exact reference**; inviscid — NOT validated (or usable) for heat transfer |
| Gimbal/TVC | Linear control theory + slosh-vehicle equations (SP-8009 ch. 4 / Dodge 2000 ch. 5 formulation) | Closed-loop eigen-structure (slosh pair at ω_s, pole placement at requested bandwidth), gust recovery, gain scaling with inertia | **Verified vs linear theory**; no experimental TVC dataset used — no aero, planar, rigid body |
| Manifold design | Bajura & Jones (1976) header theory; toroidal-shell membrane stress (Roark) | Dividing/combining pressure-profile shapes and symmetry, uniformity improving with duct area and channel stiffness (the literature's area-ratio trend), torus wall → cylinder limit as R/r → ∞ | **Verified vs the classical theory's trends**; k_m = 0.7 is the literature mid-range, not calibrated to a rocket dataset |
| Injector (swirl theory) | Bazarov/Yang/Puri ch. 2 closed forms; classical `mu(A)`/spray-angle charts | Hand-evaluated relation set, exact maximum-flow round-trip, chart anchor at A = 1 (mu ≈ 0.44, half-angle ≈ 33°), monotonic swirl trends, per-element flow closure | **Implementation-verified vs the ideal theory** — inviscid; real injectors run a few % lower Cd / narrower cone; NOT hot-fire validated, no stability prediction |
| Nozzle (bell contour) | Rao TOP parabolic-approximation method; angularity efficiency `0.5(1+cosθ)` (Sutton eq. 3-34) | Exit radius = Rt√ε and exit area ratio exact, bell shorter than the 15° cone, θ_exit and λ monotonic in ε, λ(cone)=0.983 exact, bell λ>cone | **Verified vs the parabolic method + angularity model**; θ_n/θ_e are a documented chart *fit*, not MOC — geometry and the ~1% Cf credit are representative, not a point design |
| Equilibrium combustion | NASA CEA (Gordon & McBride) published points for LOX/CH4, LOX/H2, LOX/ethanol | T_c, c* and shifting Isp within ~2–3% of CEA; element conservation exact; **peak-Isp O/F in the CEA band** (CH4 ~3.2, EtOH ~1.85) via shifting-equilibrium expansion; higher Pc→hotter | **Band-validated vs CEA**. 8-species, no condensed carbon (soot-forming O/F<~1.6 not captured); LOX/H2 optimum a touch rich |
| Design pipeline | Sutton/Huzel & Huang closed-form performance relations | c*/Cf hand checks, thrust closure `F = Cf Pc At` (exact), mass/mixture balance, plausibility bands for the 5 kN demo, exact determinism of two identical runs | **Rule-level verified**; the rules encode kN-class practice — outputs inherit each sub-model's validation status; the printed engine is a *concept*, not a qualified design |
| Voxel geometry | Analytic volumes (sphere, torus, tube, oblique cylinder, helical channel band) | Volumes within ~2–3 % at test resolution, closed-mesh (edge-pairing) invariant on every part incl. the full jacket and injector head, STL byte-level round-trip | **Verified vs analytic references**; fidelity is voxel-limited (stated per part) |

## Assumptions & limitations (per module)

* **Global**: axisymmetric rigid tanks; single-species tank fluid (no helium
  pressurant — an autogenous setpoint mode is provided instead; injected
  pressurant is treated as an *external* mass/energy source, i.e. it is not
  deducted from the injector flow, so system mass is not closed while it
  operates); no viscous/turbulent CFD (the included Euler solver is
  flow-field verification only). Deep blowdown below the liquid's vapor
  pressure would flash-boil — not modeled.
* **Slosh**: linear small-amplitude lateral slosh only — no swirl/rotary or
  breaking-wave regimes, no baffle model (pass your own damping ratio),
  large Bond number assumed, analog degraded when the surface is in a dome.
* **Thermal**: single-slab stratified layer (no continuous profile); free-
  convection closures at flat-plate level; calibrated `chi`;
  wall property tables are curve-fit approximations (NIST cryo data).
* **Regen**: quasi-1D; no injector-region boundary-layer development; frozen
  gas composition, calorically-perfect isentropic relations; adiabatic
  closeout; radiation neglected; **no boiling model** — two-phase coolant
  states are flagged (`boiling_detected`), not resolved, and transcritical
  heat-transfer deterioration near the pseudo-critical point is not modeled.
  The regen heat-transfer march is quasi-1D and contour-shape agnostic, so it
  applies equally to the conical and bell divergent options.
* **Coupling**: quasi-steady regen; mixing knob unvalidated (above);
  oxidizer tank represented only through the mixture ratio (model the ox
  tank by running a second instance with `coolant_is_fuel: false`).
* **Injector**: ideal swirl theory (no viscous losses), incompressible
  orifice equation on a supercritical (gas-like) fuel flagged not resolved;
  no atomization, mixing-efficiency or combustion-stability model — the
  stiffness heuristic is a chug margin, not a stability proof.
* **Design pipeline**: deterministic rules encoding kN-class practice;
  film cooling is sized hydraulically but its thermal benefit is NOT
  credited in the regen solution (conservative); chamber gas and c*/Isp are
  from the built-in equilibrium solver with shifting-equilibrium expansion
  (8-species, no soot — see its limitations above), not a full CEA/kinetics
  run; the bell contour is Rao's
  parabolic
  *approximation* with chart-fit angles (not a method-of-characteristics
  design), and its divergence-efficiency credit is the exit-angle angularity
  factor only; the organic fillets/dome are geometry for the printable model,
  not stress-optimized features; the STLs are voxel-limited visual/print
  concepts, not toleranced CAD.
* **P&ID recommender**: a checklist aid encoding common student-team
  practice; the schematic uses simplified ISA-5.1-style symbology and the
  architecture selector is transparent rules, not an optimizer over a cost
  function — explicitly **not** a safety-reviewed P&ID (also printed in
  the drawing's title block).

## A finding worth knowing (from the coupled demo)

Full-flow methane regen cooling of a **small** engine is marginal at
subcritical feed pressure: heated CH4 crosses the two-phase dome inside the
jacket (the tool flags this) and pressure-fed tank pressures can't exceed
methane's 46 bar critical pressure with a saturated ullage. The shipped
demo therefore uses a **pump-fed** architecture (3 bar autogenous tank,
~90 bar supercritical channels) — the same reason HYPROB feeds its jacket
at ~160 bar.

## Repository layout

```
cryosim/            the package (one module per sub-model; see docstrings)
tests/              fast unit tests
validation/         regression tests vs published references (pytest -m validation)
examples/           YAML configs + notebook
```

## References

* Abramson, H.N. (ed.), *The Dynamic Behavior of Liquids in Moving
  Containers*, NASA SP-106, 1966.
* Dodge, F.T., *Propellant Slosh Loads*, NASA SP-8009, 1968; and *The New
  Dynamic Behavior of Liquids in Moving Containers*, SwRI, 2000.
* Hasan, Lin & Van Dresar, *Self-pressurization of a flightweight liquid
  hydrogen storage tank subjected to low heat flux*, NASA TM (NTRS
  19910011011), 1991; Van Dresar & Lin, NASA TM-105411 (NTRS 19920009200), 1992.
* Bartz, D.R., *A Simple Equation for Rapid Estimation of Rocket Nozzle
  Convective Heat Transfer Coefficients*, Jet Propulsion 27(1), 1957.
* NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, 1972.
* Huzel & Huang, *Modern Engineering for Design of Liquid-Propellant Rocket
  Engines*, AIAA, 1992.
* Kose & Celik, *Regenerative Cooling Comparison of LOX/LCH4 and LOX/LC3H8
  Rocket Engines Using ODREC*, Appl. Sci. 14(1):71, 2024 (and the CIRA
  HYPROB experimental papers by Ricci, Battista et al. cited therein).
* Ludwig & Dreyer, *Investigations on thermodynamic phenomena of the
  active-pressurization process of a cryogenic propellant tank*, Cryogenics 63, 2014.
* Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer* (convection
  correlations); Bell et al., CoolProp (doi:10.1021/ie4033999).
* Bazarov, Yang & Puri, "Design and Dynamics of Jet and Swirl Injectors",
  in *Liquid Rocket Thrust Chambers: Aspects of Modeling, Analysis, and
  Design*, AIAA Progress in Astronautics and Aeronautics vol. 200, 2004.
* Sutton & Biblarz, *Rocket Propulsion Elements* (performance relations).
* Gordon, S. & McBride, B.J., *Computer Program for Calculation of Complex
  Chemical Equilibrium Compositions and Applications* (NASA CEA), NASA
  RP-1311, 1994 (equilibrium method + thermodynamic data basis).
* LEAP 71's Noyron / PicoGK (leap71.com) — the inspiration for the
  `cryosim design` requirements-to-printable-hardware pipeline and the
  voxel geometry approach (this repo's kernel is an independent,
  far smaller Python implementation, not derived from PicoGK code).

---

*Personal portfolio project (DARE Stratos V / Axolotl context) — not
official DARE work. MIT license.*
