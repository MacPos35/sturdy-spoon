# Computed vs literature/reference values

Generated live by `cryosim benchmark` — every computed number next to its reference, with source and deviation. *Kinds*: **data** = NIST/table point; **exact** = mathematical reference; **band** = experiment-class range (the honest claim for a reduced-order model); **correlation** = empirical fit with quoted scatter.

| Topic | Quantity | Computed | Reference | Unit | Deviation | Kind | Source |
|---|---|---|---|---|---|---|---|
| fluids | O2 normal boiling point | **90.1878** | 90.19 | K | -0.00% | data | NIST WebBook |
| fluids | O2 sat. liquid density @ 1 atm | **1141** | 1141 | kg/m3 | +0.02% | data | NIST WebBook |
| fluids | O2 latent heat @ 1 atm | **213.0559** | 213 | kJ/kg | +0.03% | data | NIST WebBook |
| fluids | O2 surface tension @ 90 K | **13.1927** | 13.2 | mN/m | -0.06% | data | NIST WebBook |
| fluids | CH4 normal boiling point | **111.6672** | 111.67 | K | -0.00% | data | NIST WebBook |
| fluids | CH4 critical pressure | **45.992** | 45.99 | bar | +0.00% | data | NIST WebBook |
| slosh | 1st Bessel root xi_1 of J1' | **1.8412** | 1.8412 | - | -0.00% | exact | Abramson NASA SP-106 tables |
| slosh | slosh-mass coefficient 2/(xi(xi^2-1)) | **0.4545** | 0.454545 | - | -0.01% | exact | published m1 = mF(R/2.2h)tanh(1.84h/R) form |
| slosh | dimensionless freq. w^2R/g @ h/R=1 | **1.7508** | 1.7506 | - | +0.01% | exact | SP-106 ch.2 (xi_1 tanh xi_1) |
| slosh | M-D damping, deep tank (R=0.5 m, water) | **0.0751** | 0.0750792 | % | +0.00% | correlation | Mikishev-Dorozhkin corr. (±~30% scatter) |
| nozzle flow | Mach @ A/A*=2, g=1.4 (supersonic) | **2.1972** | 2.1972 | - | -0.00% | exact | Anderson, Modern Compressible Flow, App. A |
| nozzle flow | Mach @ A/A*=2, g=1.4 (subsonic) | **0.3059** | 0.3059 | - | +0.00% | exact | Anderson, Modern Compressible Flow, App. A |
| nozzle flow | divergence eff. lambda, 15-deg cone | **0.983** | 0.982963 | - | +0.00% | exact | Sutton eq. 3-34 / Huzel & Huang: 0.5(1+cos a) |
| friction | Darcy f, smooth pipe, Re=1e5 | **0.0178** | 0.018 | - | -0.97% | exact | Moody chart / Colebrook-White |
| combustion | ideal c*, LOX/CH4 preset (MR~3.3) | **1818** | 1750 – 1880 | m/s | in band | band | CEA class values (RocketCEA / Braeunig charts) |
| combustion (equil.) | flame temp T_c, LOX/CH4 O/F3.2 @20bar | **3433** | 3400 – 3560 | K | in band | band | NASA CEA equil. ~3500 K; 8-species model runs ~2% cool (documented) |
| combustion (equil.) | shifting c*, LOX/CH4 O/F3.2 @20bar | **1864** | 1800 – 1880 | m/s | in band | band | NASA CEA shifting equilibrium |
| combustion (equil.) | vac Isp, LOX/CH4 O/F3.4 eps40 (shifting) | **373.299** | 358 – 378 | s | in band | band | NASA CEA shifting equilibrium |
| combustion (equil.) | peak-Isp O/F, LOX/CH4 (shifting) | **3.25** | 3 – 3.6 | - | in band | band | NASA CEA optimum (mildly rich of stoich 3.99) |
| combustion | Bartz viscosity SI constant | **1.184e-07** | 1.1841e-07 | Pa s (g/mol)^-0.5 K^-0.6 | -0.01% | exact | unit conversion of Bartz 1957 / H&H eq. 4-16 |
| structures | torus wall / cylinder wall as R/r->inf | **1.0017** | 1 | - | +0.17% | exact | Flugge toroidal membrane -> cylinder limit |
| structures | 316L allowable stress | **115** | 115 | MPa | +0.00% | data | ASME B31.3 basis: 2/3 x 25 ksi L-grade yield |
| nozzle (MOC) | theta_max vs nu(Me)/2 | **33.5311** | 33.5311 | deg | +0.00% | exact | Anderson ch. 11: θmax = ν(Me)/2 for the MLN |
| nozzle (MOC) | divergence eff. lambda (axial exit) | **1** | 1 | - | +0.00% | exact | uniform axial exit → no angularity loss |
| thermo-structural | hot-wall thermal stress @ΔT=200K | **344.697** | 344.697 | MPa | +0.00% | exact | Eα ΔT/(2(1−ν)) — Huzel & Huang / NASA CR-72 |
| thermo-structural | LCF life @ Δε=1% (CuCrZr class) | **1004** | 300 – 3000 | cycles | in band | band | Manson–Coffin; copper-liner regen chambers ~1e2–1e3 |
| stability | 1T acoustic mode root J'_1 | **1.8412** | 1.8412 | - | -0.00% | exact | first-tangential mode: root of J'_1 (SP-194) |
| combustion (equil.) | LOX/CH4 soot-onset O/F | **1.25** | 1.1 – 1.7 | - | in band | band | CEA condensed-carbon boundary (Boudouard); φ~2.4–3.6 |
| regen (HYPROB class) | throat heat flux, 30 kN @ 55 bar | **51.1136** | 30 – 80 | MW/m2 | in band | band | HYPROB LOX/CH4 literature band (Bartz ~20-30% high for CH4 per ODREC) |
| regen (HYPROB class) | peak hot-wall temperature | **893.5227** | 700 – 1000 | K | in band | band | HYPROB copper-liner analyses |
| regen (HYPROB class) | CH4 outlet temperature | **476.1418** | 350 – 550 | K | in band | band | HYPROB / methane regen literature |
| CFD | Euler mass flow vs quasi-1D exact | **1.9314** | 1.87189 | kg/s | +3.18% | exact | quasi-1D isentropic (exact reference); first-order scheme bias |
| tank thermal (K-site) | dP/dt ratio to homogeneous, 49% fill @ 3.5 W/m2 | **2.4211** | 1 – 3 | x hom. | in band | band | Van Dresar & Lin TM-105411: measured <~2; model over-predicts ~20-30% (documented) |

**33/33 within tolerance/band.**
