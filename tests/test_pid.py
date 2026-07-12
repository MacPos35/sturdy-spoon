"""Tests for the rule-based feed-system P&ID recommender."""

import os

import pytest

from cryosim.pid_recommender import (
    FeedSystemConfig,
    TankSpec,
    draw_pid,
    recommend_feed_system,
)


def names(rec):
    return " | ".join(c.name.lower() for c in rec.components)


def test_regulated_cryo_baseline():
    rec = recommend_feed_system(FeedSystemConfig(pressurization="regulated"))
    txt = names(rec)
    assert "regulator" in txt
    assert "check valve" in txt
    # every tank gets redundant overpressure protection + vent (cryo)
    for tank in ("lox", "lch4"):
        assert f"{tank} relief valve" in txt
        assert f"{tank} burst disc" in txt
        assert f"{tank} remote vent valve" in txt
    # oxidizer line purge present
    assert "purge" in txt
    assert "igniter" in txt


def test_blowdown_has_no_regulator_but_warns():
    rec = recommend_feed_system(FeedSystemConfig(pressurization="blowdown"))
    assert "regulator" not in names(rec)
    assert any("blowdown" in w for w in rec.warnings)


def test_autogenous_has_heat_exchanger():
    rec = recommend_feed_system(FeedSystemConfig(pressurization="autogenous"))
    assert "autogenous" in names(rec)
    assert any("start-up" in w for w in rec.warnings)


def test_pump_fed_adds_pump_and_npsh_warning():
    rec = recommend_feed_system(FeedSystemConfig(pressurization="pump"))
    assert "pump" in names(rec)
    assert any("npsh" in w.lower() for w in rec.warnings)


def test_storable_tank_skips_vent():
    cfg = FeedSystemConfig(
        pressurization="regulated",
        tanks=[TankSpec("LOX", cryogenic=True, oxidizer=True),
               TankSpec("Ethanol", cryogenic=False, is_coolant=True)],
    )
    rec = recommend_feed_system(cfg)
    txt = names(rec)
    assert "lox remote vent valve" in txt
    assert "ethanol remote vent valve" not in txt


def test_regen_without_coolant_tank_warns():
    cfg = FeedSystemConfig(
        tanks=[TankSpec("LOX", oxidizer=True)], regen_cooled=True
    )
    rec = recommend_feed_system(cfg)
    assert any("is_coolant" in w for w in rec.warnings)


def test_regen_instrumentation_present():
    rec = recommend_feed_system(FeedSystemConfig())
    txt = names(rec)
    assert "coolant outlet temperature" in txt
    assert "coolant manifold inlet pressure" in txt


def test_unique_tags():
    rec = recommend_feed_system(FeedSystemConfig())
    tags = [c.tag for c in rec.components]
    assert len(tags) == len(set(tags))


def test_text_report_mentions_disclaimer():
    rec = recommend_feed_system(FeedSystemConfig())
    assert "NOT a safety-reviewed" in rec.as_text()


def test_draw_pid_creates_file(tmp_path):
    rec = recommend_feed_system(FeedSystemConfig(pressurization="pump"))
    out = draw_pid(rec, str(tmp_path / "pid.svg"))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5000
