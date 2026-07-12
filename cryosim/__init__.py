"""cryosim — coupled slosh + thermal/pressurization + regenerative-cooling simulator.

A reduced-order (lumped-parameter / 1D) engineering tool for small cryogenic
bi-propellant rockets, scoped for student-team tank diameters of ~0.2-0.5 m.

Design choice, stated up front: this is deliberately NOT a CFD tool. Every
sub-model is an analytical/empirical reduced-order model of the kind used for
early design iteration (NASA SP-8009 mechanical slosh analogs, lumped-node tank
thermodynamics, Bartz + Dittus-Boelter 1D regenerative-cooling marching). See
README.md for the full list of assumptions and validation status of each part.
"""

__version__ = "0.1.0"
