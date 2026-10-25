"""Session-first ingest (D-055): link the session, commit once, then memory pass.

Pins: the batch path is off by default; it reproduces the online graph on the
demo corpus (18 edges, 4 supersede stamps, the budget chain); and a second
ingest is still a no-op.
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


def _engine(tmp_path, name, *, batch):
    db = str(tmp_path / f"{name}.db")
    over = {"store": {"path": db}}
    if batch:
        over["link"] = {"session_batch": True}
    return Engine(load_config(overrides=over), path=db)


def test_batch_ships_off():
    assert load_config()["link"]["session_batch"] is False


def test_batch_reproduces_the_online_graph(tmp_path):
    online = _engine(tmp_path, "online", batch=False)
    batch = _engine(tmp_path, "batch", batch=True)
    for eng in (online, batch):
        for s in _sources():
            eng.ingest(s)
    assert batch.store.count()["units"] == online.store.count()["units"] == 22
    assert batch.store.count()["edges"] == online.store.count()["edges"] == 18
    q = ("SELECT count(*) FROM units WHERE valid_to IS NOT NULL")
    assert batch.store.db.execute(q).fetchone()[0] == 4
    # the budget chain survived the session boundary (phase 3 memory pass)
    current = [u for u in batch.store.all_units()
               if "budget" in u.text.lower() and u.is_current]
    assert any("2,4" in u.text for u in current)
    assert any("1,5" in u.text and not u.is_current for u in batch.store.all_units())
    online.close()
    batch.close()


def test_batch_second_ingest_is_still_a_no_op(tmp_path):
    eng = _engine(tmp_path, "idem", batch=True)
    srcs = _sources()
    for s in srcs:
        eng.ingest(s)
    before = eng.store.count()
    for s in srcs:
        rep = eng.ingest(s)
        assert rep.n_stored == 0 and rep.n_edges == 0
    assert eng.store.count() == before
    eng.close()
