"""Fluid property interface backed by CoolProp.

A single :class:`Fluid` class serves BOTH the tank side (saturation properties,
latent heat, surface tension, liquid/vapor states) and the regen coolant side
(single-phase transport properties at local T, P), so any CoolProp fluid can be
swapped in by name: LOX (default oxidizer), liquid methane, ethanol, liquid
hydrogen (used by the validation cases), nitrogen, ...

Assumptions / limitations
-------------------------
* Single-species working fluid per tank: no dissolved pressurant, no
  multi-component mixtures. Ullage is pure propellant vapor.
* Properties come from CoolProp's HEOS backend (multi-parameter reference
  equations of state, e.g. Schmidt & Wagner for O2); accuracy is that of the
  underlying EOS.
* Surface tension is only available along the saturation curve.
"""

from __future__ import annotations

from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

#: Convenience aliases -> CoolProp canonical fluid names.
FLUID_ALIASES = {
    "lox": "Oxygen",
    "o2": "Oxygen",
    "oxygen": "Oxygen",
    "lch4": "Methane",
    "ch4": "Methane",
    "methane": "Methane",
    "lh2": "Hydrogen",
    "h2": "Hydrogen",
    "hydrogen": "Hydrogen",
    "ln2": "Nitrogen",
    "n2": "Nitrogen",
    "nitrogen": "Nitrogen",
    "ethanol": "Ethanol",
    "water": "Water",
    "helium": "Helium",
    # RP-1 has no CoolProp EOS; n-dodecane is the standard single-component
    # surrogate (density ~750 vs RP-1's ~810 kg/m^3 at 288 K, similar cp/mu
    # trends) — slightly conservative for regen cooling capacity.
    "rp1": "n-Dodecane",
    "rp-1": "n-Dodecane",
    "kerosene": "n-Dodecane",
}


def canonical_name(name: str) -> str:
    """Resolve a user-facing fluid name (e.g. 'LOX') to a CoolProp name."""
    return FLUID_ALIASES.get(name.strip().lower(), name)


@dataclass(frozen=True)
class FluidState:
    """Thermodynamic + transport state at a point (SI units).

    ``two_phase`` marks states inside the vapor dome: thermodynamic values
    are mixture values, but transport properties (cp, mu, k) are those of
    the saturated liquid — single-phase correlations using them are NOT
    valid there (the regen model flags this as ``boiling_detected``).
    """

    T: float          # K
    P: float          # Pa
    rho: float        # kg/m^3
    h: float          # J/kg  (specific enthalpy)
    u: float          # J/kg  (specific internal energy)
    cp: float         # J/(kg K)
    mu: float         # Pa s
    k: float          # W/(m K)
    two_phase: bool = False

    @property
    def Pr(self) -> float:
        return self.cp * self.mu / self.k


class Fluid:
    """Thin, unit-consistent (SI) wrapper around CoolProp for one fluid."""

    def __init__(self, name: str = "LOX"):
        self.input_name = name
        self.name = canonical_name(name)
        # Fail fast on unknown fluids.
        self.T_crit = PropsSI("Tcrit", self.name)
        self.P_crit = PropsSI("pcrit", self.name)
        self.T_triple = PropsSI("Ttriple", self.name)
        self.molar_mass = PropsSI("molemass", self.name)  # kg/mol

    def __repr__(self) -> str:  # pragma: no cover
        return f"Fluid({self.name!r})"

    # ---------------------------------------------------------------- states
    def state_TP(self, T: float, P: float) -> FluidState:
        """Single-phase state from temperature and pressure."""
        args = ("T", T, "P", P, self.name)
        return FluidState(
            T=T, P=P,
            rho=PropsSI("Dmass", *args),
            h=PropsSI("Hmass", *args),
            u=PropsSI("Umass", *args),
            cp=PropsSI("Cpmass", *args),
            mu=PropsSI("viscosity", *args),
            k=PropsSI("conductivity", *args),
        )

    def state_PH(self, P: float, h: float) -> FluidState:
        """State from pressure and specific enthalpy (regen coolant marching).

        Handles two-phase (boiling) states: thermodynamics are mixture
        values, transport properties fall back to the saturated liquid and
        the state is flagged ``two_phase`` (see FluidState docstring).
        """
        args = ("P", P, "Hmass", h, self.name)
        T = PropsSI("T", *args)
        rho = PropsSI("Dmass", *args)
        u = PropsSI("Umass", *args)
        try:
            return FluidState(
                T=T, P=P, rho=rho, h=h, u=u,
                cp=PropsSI("Cpmass", *args),
                mu=PropsSI("viscosity", *args),
                k=PropsSI("conductivity", *args),
            )
        except ValueError:
            # inside the vapor dome: transport from saturated liquid
            sl = self.sat_liquid(P)
            return FluidState(
                T=T, P=P, rho=rho, h=h, u=u,
                cp=sl.cp, mu=sl.mu, k=sl.k, two_phase=True,
            )

    def flash_rho_u(self, rho: float, u: float) -> tuple[float, float]:
        """(P, T) from density and internal energy — the ullage-node closure."""
        P = PropsSI("P", "Dmass", rho, "Umass", u, self.name)
        T = PropsSI("T", "Dmass", rho, "Umass", u, self.name)
        return P, T

    # ------------------------------------------------------------ saturation
    def T_sat(self, P: float) -> float:
        return PropsSI("T", "P", P, "Q", 0, self.name)

    def P_sat(self, T: float) -> float:
        return PropsSI("P", "T", T, "Q", 0, self.name)

    def h_fg(self, P: float) -> float:
        """Latent heat of vaporization at pressure P [J/kg]."""
        hg = PropsSI("Hmass", "P", P, "Q", 1, self.name)
        hf = PropsSI("Hmass", "P", P, "Q", 0, self.name)
        return hg - hf

    def sat_liquid(self, P: float) -> FluidState:
        return self._sat_state(P, 0.0)

    def sat_vapor(self, P: float) -> FluidState:
        return self._sat_state(P, 1.0)

    def _sat_state(self, P: float, Q: float) -> FluidState:
        args = ("P", P, "Q", Q, self.name)
        return FluidState(
            T=PropsSI("T", *args), P=P,
            rho=PropsSI("Dmass", *args),
            h=PropsSI("Hmass", *args),
            u=PropsSI("Umass", *args),
            cp=PropsSI("Cpmass", *args),
            mu=PropsSI("viscosity", *args),
            k=PropsSI("conductivity", *args),
        )

    def surface_tension(self, T: float) -> float:
        """Saturation-line surface tension [N/m]."""
        return PropsSI("surface_tension", "T", T, "Q", 0, self.name)

    # ------------------------------------------- liquid thermal-expansion etc.
    def beta(self, T: float, P: float) -> float:
        """Isobaric thermal expansion coefficient [1/K]."""
        return PropsSI("isobaric_expansion_coefficient", "T", T, "P", P, self.name)
