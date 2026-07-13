"""Tests for automatic tube sizing and fitting selection."""

import os

import pytest

from cryosim.line_sizing import (
    IN,
    LineSpec,
    required_wall_in,
    select_fitting,
    size_feed_system,
    size_line,
    report,
)


def test_barlow_wall_hand_value():
    # t = DF * P * OD / (2 S): 1.25 * 135e5 * (0.625*25.4e-3) / (2*115e6)
    t = required_wall_in(135e5, 0.625, "316L")
    assert t == pytest.approx(1.25 * 135e5 * 0.625 * IN / (2 * 115e6) / IN,
                              rel=1e-9)
    assert t == pytest.approx(0.0459, abs=5e-4)


def test_min_handling_wall():
    assert required_wall_in(2e5, 0.25, "316L") == 0.028


def test_velocity_target_picks_od():
    # LOX feed 1.44 kg/s: 3/4" x 0.028 gives 5.2 m/s > 5 target -> 1" tube
    spec = size_line("LOX feed", "liquid_feed", "LOX", 1.44, 1134.0, 6e5)
    assert spec.od_in == 1.0
    assert spec.velocity < 5.0
    # halved flow fits in the next size down
    spec2 = size_line("LOX feed", "liquid_feed", "LOX", 0.7, 1134.0, 6e5)
    assert spec2.od_in < 1.0


def test_high_pressure_thickens_wall():
    lo = size_line("l", "pump_discharge", "LCH4", 0.1, 400.0, 10e5)
    hi = size_line("l", "pump_discharge", "LCH4", 0.1, 400.0, 300e5)
    assert hi.wall_in > lo.wall_in
    assert hi.wall_in >= required_wall_in(300e5, hi.od_in, "316L")


def test_dash_size():
    spec = size_line("x", "liquid_feed", "LOX", 0.05, 1134.0, 5e5)
    assert spec.dash == round(spec.od_in * 16)


def test_fitting_rules():
    f, _ = select_fitting(0.5, cryogenic=True, service="liquid_feed")
    assert "flare" in f and "NPT" not in f
    f, notes = select_fitting(1.0, cryogenic=True, service="liquid_feed")
    assert "weld" in f or "flange" in f
    f, _ = select_fitting(0.25, cryogenic=False, service="pressurant_gas")
    assert "compression" in f
    # cryo rejection note present
    _, notes = select_fitting(0.5, cryogenic=True, service="liquid_feed")
    assert any("NPT" in n for n in notes)


def test_unsizable_line_raises():
    with pytest.raises(ValueError, match="no catalog tube"):
        size_line("x", "liquid_feed", "LOX", 1.0, 1134.0, 2500e5)


def test_non_cryo_material_rejected():
    with pytest.raises(ValueError, match="not rated"):
        size_line("x", "liquid_feed", "LOX", 0.1, 1134.0, 5e5,
                  material="PTFE-lined flex", cryogenic=True)


@pytest.fixture(scope="module")
def demo_specs():
    return size_feed_system(
        fluid_names={"oxidizer": "LOX", "fuel": "LCH4"},
        mdots={"oxidizer": 1.44, "fuel": 0.43, "pressurant": 0.006},
        tank_meop=4.5e5,
        coolant_hp_mawp=135e5,
        pressurant_rho=2.4,
        pump_fed=True,
    )


def test_feed_system_line_set(demo_specs):
    names = [s.name for s in demo_specs]
    for expected in ("LOX feed line", "LCH4 feed line", "LOX vent/relief line",
                     "pressurant header", "coolant HP line (pump -> jacket)"):
        assert expected in names
    # pump-fed: fuel feed sized as pump suction (slower target => bigger ID)
    fuel = next(s for s in demo_specs if s.name == "LCH4 feed line")
    assert fuel.service == "pump_suction"
    hp = next(s for s in demo_specs if s.name.startswith("coolant HP"))
    assert hp.mawp == pytest.approx(135e5)
    assert hp.wall_in >= 0.049 - 1e-12


def test_no_hp_line_when_pressure_fed():
    specs = size_feed_system(
        fluid_names={"oxidizer": "LOX", "fuel": "Ethanol"},
        mdots={"oxidizer": 0.5, "fuel": 0.3, "pressurant": 0.01},
        tank_meop=60e5, pressurant_rho=8.0, pump_fed=False,
    )
    assert not any(s.name.startswith("coolant HP") for s in specs)


def test_report_and_labels(demo_specs):
    txt = report(demo_specs)
    assert "Barlow" in txt and "verify" in txt.lower()
    lab = demo_specs[0].label()
    assert '"' in lab and "-" in lab


def test_draw_fittings_creates_file(demo_specs, tmp_path):
    from cryosim.fitting_diagrams import draw_fittings

    out = draw_fittings(demo_specs, str(tmp_path / "fittings.png"))
    assert os.path.getsize(out) > 30000


def test_pid_accepts_line_labels(demo_specs, tmp_path):
    from cryosim.pid_recommender import (FeedSystemConfig, draw_pid,
                                         recommend_feed_system)

    rec = recommend_feed_system(FeedSystemConfig(pressurization="pump"))
    labels = {"oxidizer": demo_specs[0].label(), "fuel": "x", "pressurant": "y",
              "coolant_hp": "z"}
    out = draw_pid(rec, str(tmp_path / "pid.png"), line_labels=labels)
    assert os.path.getsize(out) > 20000
