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
| friction | Darcy f, smooth pipe, Re=1e5 | **0.0178** | 0.018 | - | -0.97% | exact | Moody chart / Colebrook-White |
| combustion | ideal c*, LOX/CH4 preset (MR~3.3) | **1818** | 1750 – 1880 | m/s | in band | band | CEA class values (RocketCEA / Braeunig charts) |
| combustion | Bartz viscosity SI constant | **1.184e-07** | 1.1841e-07 | Pa s (g/mol)^-0.5 K^-0.6 | -0.01% | exact | unit conversion of Bartz 1957 / H&H eq. 4-16 |
| structures | torus wall / cylinder wall as R/r->inf | **1.0017** | 1 | - | +0.17% | exact | Flugge toroidal membrane -> cylinder limit |
| structures | 316L allowable stress | **115** | 115 | MPa | +0.00% | data | ASME B31.3 basis: 2/3 x 25 ksi L-grade yield |
| regen (HYPROB class) | throat heat flux, 30 kN @ 55 bar | **51.1136** | 30 – 80 | MW/m2 | in band | band | HYPROB LOX/CH4 literature band (Bartz ~20-30% high for CH4 per ODREC) |
| regen (HYPROB class) | peak hot-wall temperature | **893.5227** | 700 – 1000 | K | in band | band | HYPROB copper-liner analyses |
| regen (HYPROB class) | CH4 outlet temperature | **476.1418** | 350 – 550 | K | in band | band | HYPROB / methane regen literature |
| CFD | Euler mass flow vs quasi-1D exact | **1.9314** | 1.87189 | kg/s | +3.18% | exact | quasi-1D isentropic (exact reference); first-order scheme bias |
| tank thermal (K-site) | dP/dt ratio to homogeneous, 49% fill @ 3.5 W/m2 | **2.4211** | 1 – 3 | x hom. | in band | band | Van Dresar & Lin TM-105411: measured <~2; model over-predicts ~20-30% (documented) |

**22/22 within tolerance/band.**
