"""Tests for the computed-vs-reference benchmark table (fast set)."""

import pytest

from cryosim.benchmarks import run_benchmarks, to_markdown, to_text


@pytest.fixture(scope="module")
def rows():
    return run_benchmarks(full=False)


def test_all_fast_benchmarks_pass(rows):
    failed = [r.quantity for r in rows if not r.ok]
    assert not failed, f"benchmarks out of tolerance: {failed}"


def test_every_row_cites_a_source(rows):
    for r in rows:
        assert r.source and len(r.source) > 8
        assert r.kind in ("data", "exact", "band", "correlation")


def test_covers_the_core_topics(rows):
    topics = {r.topic for r in rows}
    for t in ("fluids", "slosh", "nozzle flow", "friction", "combustion",
              "structures"):
        assert t in topics


def test_formatters(rows):
    txt = to_text(rows)
    assert "reference" in txt and "source:" in txt
    assert f"{sum(r.ok for r in rows)}/{len(rows)}" in txt
    md = to_markdown(rows)
    assert md.count("|") > 8 * len(rows)  # a real table
    assert "NIST" in md
