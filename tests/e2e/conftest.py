"""Shared engine for the §13 demo scenarios.

Built from the hand-written fixture (`tests/golden/demo_scenarios.json`) rather
than the generator, so the demo assertions do not move when the ledger RNG or
renderer changes. Hash embedder + temp store: no model download, no GPU.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from engine.config import load_config
from engine.engine import Engine
from engine.schemas import Source

FIXTURE = Path(__file__).resolve().parents[1] / "golden" / "demo_scenarios.json"
CONTACTS = Path(__file__).resolve().parents[2] / "data" / "contacts.json"


@pytest.fixture(scope="session")
def demo_engine(tmp_path_factory):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root = tmp_path_factory.mktemp("demo_store")
    cfg = load_config(overrides={
        "store": {"path": str(root / "engine.db")},
        "embed": {"backend": "hash", "dim": 256},
        "agent": {"contacts_path": str(CONTACTS)},
    })
    eng = Engine(cfg, path=root / "engine.db")
    sources = []
    for raw in data["sources"]:
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        sources.append(Source(**raw))
    for s in sorted(sources, key=lambda x: (x.source_date, x.source_id)):
        eng.ingest(s)
    yield eng
    eng.close()


@pytest.fixture(scope="session")
def demo_contacts():
    return json.loads(CONTACTS.read_text(encoding="utf-8"))
