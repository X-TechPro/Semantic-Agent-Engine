"""The rerank branch must actually execute (BUILD_GUIDE 6.9, 10 rows 3/8/9/11).

Nothing tested this path, and it cost the project its reranker: `FusedHit` is
constructed inside the branch while `pipeline.py` imported only `rrf`, so the
first-ever run of `--rerank overlap` died with `NameError: name 'FusedHit' is not
defined`. A stage that has never executed cannot be reported as measured — that
is how 10's rerank rows stayed empty while the code implied a backend existed.

Both directions are asserted:

* with a reranker, a hit carries a `rerank` score part and the stats record
  `rerank_s`;
* without one, `rerank_s` must be ABSENT. The timing used to be written
  unconditionally, so a latency table would have shown a 0.0 ms stage that never
  ran — a measured-looking number for work that did not happen.

Graph expansion can append hits that were never reranked, so the positive case
asserts SOME hit carries the score part, not every one.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.config import load_config
from engine.engine import Engine
from engine.schemas import Query, Source, Turn

pytestmark = pytest.mark.unit


def _engine(tmp_path, backend: str) -> Engine:
    db = tmp_path / f"rr-{backend}.db"
    for stale in tmp_path.glob(f"rr-{backend}.db*"):
        stale.unlink()
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": "hash"},
                                 "retrieve": {"rerank": {"backend": backend}}})
    eng = Engine(cfg, path=db)
    for i, (who, text, d) in enumerate([
        ("Marie", "Le budget d'Atlas est proposé à 1,2 M€.", date(2026, 9, 1)),
        ("Marie", "Le budget d'Atlas est validé à 1,5 M€.", date(2026, 10, 1)),
        ("Karim", "Le prestataire d'Atlas est Nova Cloud.", date(2026, 10, 2)),
    ]):
        eng.ingest(Source(source_id=f"m{i}", source_type="meeting", source_date=d,
                          project="Atlas", default_speaker=who,
                          turns=[Turn(speaker=who, text=text, turn_index=0,
                                      project="Atlas")]))
    return eng


class TestRerankExecutes:
    def test_overlap_reranker_runs_and_is_recorded(self, tmp_path):
        eng = _engine(tmp_path, "overlap")
        try:
            assert eng.rerank_backend == "overlap", (
                "Engine.rerank_backend must report the object that was built")
            hits = eng.retrieve(Query(text="le budget d'Atlas validé"))
            assert hits, "expected retrieval hits"
            with_part = [h for h in hits if "rerank" in h.score_parts]
            assert with_part, (
                f"no hit carries a rerank score part ({[h.score_parts for h in hits][:2]}) "
                f"— the rerank branch did not run")
            timings = eng.pipeline.last_stats.timings
            assert "rerank_s" in timings, (
                f"rerank_s missing from {sorted(timings)} — the rerank branch did "
                f"not run")
        finally:
            eng.close()

    def test_none_backend_records_no_rerank_timing(self, tmp_path):
        eng = _engine(tmp_path, "none")
        try:
            assert eng.rerank_backend == "none"
            hits = eng.retrieve(Query(text="le budget d'Atlas validé"))
            assert hits
            timings = eng.pipeline.last_stats.timings
            assert "rerank_s" not in timings, (
                f"a 0.0 ms rerank stage was timed although it never ran: {timings}")
            assert all("rerank" not in h.score_parts for h in hits)
        finally:
            eng.close()

    def test_engine_builds_one_pipeline(self, tmp_path):
        """Two construction sites is how an engine drifts from its own config."""
        eng = _engine(tmp_path, "overlap")
        try:
            first = eng.pipeline
            eng.retrieve(Query(text="le budget d'Atlas"))
            assert eng.pipeline is first, "retrieve() must reuse the built pipeline"
        finally:
            eng.close()

    def test_unknown_backend_raises_instead_of_silently_disabling(self, tmp_path):
        cfg = load_config(overrides={"store": {"path": str(tmp_path / "x.db")},
                                     "embed": {"backend": "hash"},
                                     "retrieve": {"rerank": {"backend": "typo"}}})
        eng = Engine(cfg, path=tmp_path / "x.db")
        try:
            with pytest.raises(ValueError, match="rerank backend"):
                _ = eng.pipeline
        finally:
            eng.close()
