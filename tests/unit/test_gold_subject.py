"""GoldSubjectModel — the annotation supplier for the real-data demos.

It extracts NOTHING and must look like that everywhere: lookup is exact-text,
an unlabelled sentence gets None (never a guess), a missing gold directory is
a startup error, and the resolver name lands in stats(). Tests are offline:
tmp dirs, synthetic JSON, no models.
"""
from __future__ import annotations

import json

import pytest

from engine.config import load_config
from engine.slots.gold_subject import GoldSubjectModel

pytestmark = pytest.mark.unit


def _fixture(tmp_path, keys_047=("buffet", None)):
    meetings = {"047a_EDRH": {"segments": [
        {"speaker": 1, "text": "le buffet sera italien"},
        {"speaker": 1, "text": "bon allez bonne journée"}]}}
    (tmp_path / "meetings.json").write_text(json.dumps(meetings), encoding="utf-8")
    gold = {"meeting": "047a_EDRH", "split": "train", "n_segments": 2,
            "keys": {"0": keys_047[0], "1": keys_047[1]}}
    (tmp_path / "summre_subjects_047a_EDRH.json").write_text(
        json.dumps(gold), encoding="utf-8")
    return tmp_path / "summre_subjects_047a_EDRH.json", tmp_path / "meetings.json"


def test_hit_miss_and_stats(tmp_path):
    _fixture(tmp_path)
    m = GoldSubjectModel(tmp_path, tmp_path / "meetings.json")
    assert m.subject_for("le buffet sera italien", "047a_EDRH") == "047a_edrh.buffet"
    assert m.subject_for("bon allez bonne journée", "047a_EDRH") is None  # gold null
    assert m.subject_for("texte jamais labelle", "047a_EDRH") is None    # miss, no guess
    assert m.subject_for("le buffet sera italien", "autre_projet") is None  # per-meeting
    assert m.stats == {"asks": 4, "hit": 1, "miss": 3}


def test_owner_slot_key_shape(tmp_path):
    _fixture(tmp_path)
    m = GoldSubjectModel(tmp_path, tmp_path / "meetings.json")
    assert m.subject_for("le buffet sera italien", "047a_EDRH",
                         owner_slot=True) == "047a_edrh.owner.buffet"


def test_empty_gold_dir_is_a_startup_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="summre_subjects"):
        GoldSubjectModel(tmp_path, tmp_path / "meetings.json")


def test_pipeline_accepts_gold_resolver(tmp_path):
    from engine.ingest import IngestPipeline
    from engine.store.sqlite_store import SQLiteStore

    _fixture(tmp_path)
    cfg = load_config(overrides={"slots": {
        "subject_resolver": "gold", "subject_model_dir": str(tmp_path)}})
    pipe = IngestPipeline(SQLiteStore(tmp_path / "t.db", embed_version="stub"),
                          embedder=None, cfg=cfg)
    assert pipe.subject_model is not None
    assert pipe.subject_resolver == "gold"


def test_unknown_resolver_still_rejected(tmp_path):
    from engine.ingest import IngestPipeline
    from engine.store.sqlite_store import SQLiteStore

    cfg = load_config(overrides={"slots": {"subject_resolver": "telepathy"}})
    with pytest.raises(ValueError, match="measured and removed"):
        IngestPipeline(SQLiteStore(tmp_path / "t.db", embed_version="stub"),
                       embedder=None, cfg=cfg)
    for i, gone in enumerate(("embed_lookup", "llm", "supervised", "drawer",
                              "graph")):
        cfg = load_config(overrides={"slots": {"subject_resolver": gone}})
        with pytest.raises(ValueError, match="only 'rules'"):
            # one store per removed name: six SQLite handles on one path is a
            # table-exists collision, not the behaviour under test
            IngestPipeline(SQLiteStore(tmp_path / f"gone{i}.db",
                                        embed_version="stub"),
                           embedder=None, cfg=cfg)
