"""Aerospike: Angelino contour, truncation loss, chamber, pipeline."""

import numpy as np
import pytest

from cryosim.aerospike import (
    AerospikeSurface,
    annular_stability_screen,
    design_aerospike_chamber,
    design_spike_contour,
    truncation_cf_loss,
)
from cryosim.combustion import gas_preset, mach_from_area_ratio
from cryosim.engine_design import EngineSpec, design_engine
from cryosim.moc_nozzle import prandtl_meyer

FAST = dict(n_random=8, n_polish=8)
GAMMA, EPS = 1.22, 3.5


def test_ideal_spike_closes_on_axis():
    sp = design_spike_contour(GAMMA, EPS, A_t=1e-3, length_fraction=1.0)
    assert sp.r[-1] == pytest.approx(0.0, abs=1e-6)
    # design closure: pi R_lip^2 = eps A_t
    assert np.pi * sp.R_lip**2 == pytest.approx(EPS * 1e-3, rel=1e-9)
    # exit Mach honors the area ratio
    Me = mach_from_area_ratio(EPS, GAMMA, supersonic=True)
    assert sp.exit_mach == pytest.approx(Me, rel=1e-9)
    assert sp.nu_exit == pytest.approx(prandtl_meyer(Me, GAMMA), rel=1e-12)
    # monotone: radius falls, x grows
    assert np.all(np.diff(sp.r) <= 1e-12)
    assert np.all(np.diff(sp.x) > 0)


def test_truncated_spike_geometry():
    sp = design_spike_contour(GAMMA, EPS, A_t=1e-3, length_fraction=0.3)
    assert sp.length == pytest.approx(0.3 * sp.length_full, rel=1e-6)
    assert sp.r_base > 0
    # surface pressure falls monotonically along the expansion
    assert np.all(np.diff(sp.p_ratio) <= 1e-15)


def test_truncation_loss_behavior():
    # full spike: no loss; shorter spikes lose more; vacuum loses more
    # than sea level (no ambient back-pressure credit on the tip)
    assert truncation_cf_loss(GAMMA, EPS, 0.05, 1.0) == 0.0
    l30 = truncation_cf_loss(GAMMA, EPS, 0.05, 0.3)
    l50 = truncation_cf_loss(GAMMA, EPS, 0.05, 0.5)
    assert 0 < l50 < l30
    assert truncation_cf_loss(GAMMA, EPS, 0.0, 0.3) > l30
    # magnitude sanity: a 30% spike loses a few % of Cf, not tens
    assert l30 < 0.1


def test_chamber_geometry_consistency():
    ch = design_aerospike_chamber(1e-3, EPS, 6.0, 1.1, GAMMA, 0.3)
    # gas path contracts from CR at the face to 1 at the throat annulus
    assert ch.cowl.area_ratio()[0] == pytest.approx(6.0, rel=1e-6)
    assert ch.cowl.area_ratio()[ch.cowl.i_throat] == pytest.approx(1.0)
    # walls straddle the mean radius
    assert ch.cowl.r[0] > ch.R_mean > ch.inner.r[0]
    # spike side: subsonic to the shoulder, supersonic beyond
    assert not ch.inner.supersonic[ch.inner.i_throat]
    assert ch.inner.supersonic[-1]
    # the land-check radius is the true minimum of each wall
    assert ch.cowl.Rt == pytest.approx(float(ch.cowl.r.min()))
    assert ch.inner.Rt == pytest.approx(float(ch.inner.r.min()))
    # Bartz scale is the annulus hydraulic diameter
    assert ch.cowl.Dt == pytest.approx(2.0 * ch.gap_throat)


def test_chamber_rejects_closing_center():
    with pytest.raises(ValueError, match="contraction"):
        design_aerospike_chamber(1e-3, EPS, 60.0, 1.1, GAMMA, 0.3)


def test_annular_stability_modes():
    gas = gas_preset("lox/ch4")
    s = annular_stability_screen(gas, R_mean=0.08, L_gas=0.12,
                                 stiffness=0.20)
    a = s.sound_speed
    assert s.modes["1T-ann"] == pytest.approx(a / (2 * np.pi * 0.08))
    assert s.modes["2T-ann"] == pytest.approx(2 * s.modes["1T-ann"])
    assert s.modes["1L"] == pytest.approx(a / (2 * 0.12))
    assert s.stiffness_ok
    assert any("screening" in n for n in s.notes)


# ---------------------------------------------------------------- pipeline

@pytest.fixture(scope="module")
def aspike():
    spec = EngineSpec(thrust=20e3, propellants="lox/ch4",
                      chamber_pressure=20e5, nozzle_type="aerospike",
                      credit_film=True, bartz_factor=0.8, name="as-20kN")
    return design_engine(spec, **FAST)


def test_aerospike_pipeline_converges(aspike):
    assert all(i.ok for i in aspike.ledger)
    assert aspike.aerospike is not None
    assert aspike.spike_channel_design is not None
    # both circuits in the ledger
    names = " ".join(i.name for i in aspike.ledger)
    assert "cowl:" in names and "spike:" in names


def test_aerospike_thrust_closure(aspike):
    At = np.pi * aspike.throat_radius**2      # equivalent throat area
    assert aspike.Cf * aspike.spec.chamber_pressure * At == pytest.approx(
        aspike.spec.thrust, rel=1e-6)
    assert aspike.Isp_vac > aspike.Isp_ambient
    # 20 kN LOX/CH4 at 20 bar SL: same class band as the bell pipeline
    assert 220.0 <= aspike.Isp_ambient <= 285.0


def test_aerospike_injector_is_annular(aspike):
    inj = aspike.injector
    assert inj.face_r_inner > 0
    # no center element on an annular face; all rings clear the inner wall
    assert all(r > inj.face_r_inner for r, _ in inj.rings)


def test_aerospike_dual_circuit_states(aspike):
    # fuel cools the cowl, LOX the spike; both stay single-phase
    assert not aspike.channel_design.result.boiling_detected
    assert not aspike.spike_channel_design.result.boiling_detected
    # LOX spike outlet feeds the injector: it must be above Pc + drop
    ox_out = aspike.spike_channel_design.result.coolant_outlet
    Pc = aspike.spec.chamber_pressure
    assert ox_out.P >= Pc * (1.0 + aspike.spec.stiffness)


def test_aerospike_deterministic():
    spec = EngineSpec(thrust=8e3, propellants="lox/ch4",
                      chamber_pressure=20e5, nozzle_type="aerospike",
                      name="as-det")
    a = design_engine(spec, **FAST)
    b = design_engine(spec, **FAST)
    assert a.throat_radius == b.throat_radius
    assert str(a.trace) == str(b.trace)


def test_surface_duck_type_complete():
    """The regen/optimizer/manifold modules read exactly these attributes;
    keep the contract explicit."""
    ch = design_aerospike_chamber(1e-3, EPS, 6.0, 1.1, GAMMA, 0.3)
    for s in (ch.cowl, ch.inner):
        assert isinstance(s, AerospikeSurface)
        for attr in ("x", "r", "area", "At", "Dt", "r_curv_throat", "Rt",
                     "x_throat", "i_throat"):
            assert hasattr(s, attr)
        assert len(s.x) == len(s.r) == len(s.area)
        assert s.area_ratio().shape == s.x.shape
        assert s.is_supersonic().dtype == bool


# ---------------------------------------------------------------- geometry

def test_aerospike_parts_mesh_watertight(aspike):
    """All three printed parts + the assembly must be watertight (coarse
    voxels for test speed; report uses finer)."""
    from cryosim.voxel_geometry import (build_aerospike_body,
                                        build_annular_injector_head,
                                        build_spike, concat_meshes,
                                        mesh_solid)
    meshes = []
    for builder, args, vox in (
            (build_aerospike_body, (aspike,), 1.5e-3),
            (build_spike, (aspike,), 1.0e-3),
            (build_annular_injector_head, (aspike.injector,), 0.8e-3)):
        solid, lo, hi = builder(*args)
        m = mesh_solid(solid, lo, hi, vox)
        assert m.is_watertight(), builder.__name__
        assert m.volume() > 0
        meshes.append(m)
    asm = concat_meshes(meshes, 1.5e-3)
    assert asm.is_watertight()
    # cowl mesh must be larger in every bbox direction than the spike
    lo_c, hi_c = meshes[0].bounds()
    lo_s, hi_s = meshes[1].bounds()
    assert (hi_c[1] - lo_c[1]) > (hi_s[1] - lo_s[1])


def test_aerospike_package(tmp_path, aspike):
    """generate_package produces the aerospike deliverable set."""
    from cryosim.design_report import generate_package
    paths = generate_package(aspike, str(tmp_path), with_geometry=False)
    for key in ("cross_section", "injector_face", "wall_temperature",
                "wall_temperature_spike", "report", "trace"):
        assert key in paths
    md = open(paths["report"]).read()
    assert "annular throat" in md and "spike" in md


def test_aerospike_verification_items(aspike):
    """Both circuits re-marched; the Euler CFD check is (documented as)
    skipped for external plug flows."""
    v = [i for i in aspike.ledger if i.name.startswith("verify")]
    assert len(v) == 3 and all(i.ok for i in v)
    names = " ".join(i.name for i in v)
    assert "cowl regen re-march" in names
    assert "spike regen re-march" in names
    assert "Euler CFD" not in names
    assert "skipped" in str(aspike.trace)
