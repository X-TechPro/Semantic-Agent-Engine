"""Re-ingesting a source is a no-op (D-038).

Measured 2026-10-22: the hand-tester's browser reconnected to a restarted
server and replayed the sidebar's fixture click. Units survived (they dedupe
by id) but every source re-linked against the NOW-FULL store: the demo corpus
went from 18 edges to 37, minting reverse-direction pairs (atlas-1 -> atlas-3
only exists because atlas-3 was already there), and every unit's supersede
stamp was cleared by INSERT OR REPLACE and rewritten mid-pass. The same
source_id means the same content (unit ids derive from it), so the honest
second pass changes nothing — asserted here on counts AND on the stamps.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Source

pytestmark = pytest.mark.unit

FIXTURE = repo_root() / "tests" / "golden" / "demo_scenarios.json"


def _sources():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = []
    for raw in sorted(data["sources"], key=lambda r: r["source_date"]):
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        out.append(Source(**raw))
    return out


@pytest.fixture()
def eng(tmp_path):
    db = str(tmp_path / "i.db")
    e = Engine(load_config(overrides={"store": {"path": db}}), path=db)
    yield e
    e.close()


def test_first_pass_is_the_normal_pass(eng):
    reps = [eng.ingest(s) for s in _sources()]
    assert sum(r.n_stored for r in reps) == 22
    assert eng.store.count()["edges"] == 18


def test_second_ingest_changes_nothing(eng):
    srcs = _sources()
    for s in srcs:
        eng.ingest(s)
    counts0 = eng.store.count()
    stamps0 = {u.unit_id: (u.valid_to, u.valid_from, u.superseded_by)
               for u in eng.store.all_units()}
    for s in srcs:
        rep = eng.ingest(s)
        assert rep.n_stored == 0 and rep.n_edges == 0 and rep.n_superseded == 0, rep
    assert eng.store.count() == counts0, "a re-ingest moved the counts"
    stamps1 = {u.unit_id: (u.valid_to, u.valid_from, u.superseded_by)
               for u in eng.store.all_units()}
    assert stamps1 == stamps0, "a re-ingest moved a supersede stamp"
