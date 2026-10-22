"""The frbench artefact plumbing — collectors, unwrappers, the report's §6d.

The verdicts from external French data are only as trustworthy as the code that
finds, unwraps and renders them, and every part here is pure enough to test with
temp files (no torch, no network). The specific failures these pin:

* **`_frbench` must not confuse the rules-fallback artefact with the model one.**
  Both exist on disk (`massive_reader-model_*` from the CUDA-OOM run and
  `massive_model-reader-*` from the re-run); their contents differ completely
  (rules: 9.4% intent accuracy, model: 16.2%), and the report's row is only honest
  if the *model* artefact wins — which is why the fallback got its own key.
* **The miracl slice and the pooled run must never collide** — one directory is
  evidence of an invalid instrument (3 scoreable questions), the other the real
  verdict (343); the collector distinguishes them by the `-pool` tag.
* **The docid-slice parse** in `read_passages_pool` trades a json.loads per line
  for a string prefix/suffix read — fast, but it silently depends on exact
  line shape, so a fixture test covers both the found and the missing-shard paths.
* **`build()` renders §6d from a collected dict**, so the fallback case, the
  model case and the "nothing fetched yet" case can all be checked offline.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import pytest

from eval import frbench
from eval.report import _fbres, _frbench, build

pytestmark = pytest.mark.unit


def _art(tmp: Path, name: str, payload: dict) -> Path:
    d = tmp / "runs" / "frbench" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


# --------------------------------------------------------------- collect + unwrap

def test_fbres_handles_envelope_and_bare_forms():
    env = {"component": "xnli", "result": {"pairclassifier": {"auc": 0.5}}}
    bare = {"corpus": "miracl", "arms": {}}
    assert _fbres(env) == {"pairclassifier": {"auc": 0.5}}
    assert _fbres(bare) is bare
    assert _fbres(None) == {}
    # a result key that is NOT a dict must pass the whole artefact through, not None
    assert _fbres({"result": 7, "other": 1}) == {"result": 7, "other": 1}


def test_frbench_picks_the_model_artifact_not_the_fallback(tmp_path):
    _art(tmp_path, "massive_reader-model_test_full",
         {"component": "massive", "result": {"reader_backend": "rules[fallback]",
                                             "intent_accuracy_in_scope": 0.0939}})
    _art(tmp_path, "massive_model-reader-_test_full",
         {"component": "massive", "result": {"reader_backend": "model(reader)",
                                             "intent_accuracy_in_scope": 0.1621}})
    _art(tmp_path, "miracl_intfloat-pool_dev_full", {"arms": {"dense": {}}, "n_scoreable": 343})
    _art(tmp_path, "miracl_intfloat_dev_n5000", {"arms": {}, "n_scoreable": 3})
    _art(tmp_path, "xnli-nli_camembert-head_test_full",
         {"component": "xnli-nli", "result": {"test": {"accuracy": 0.798}}})
    fb = _frbench(tmp_path / "runs")
    assert fb["massive_model"]["result"]["intent_accuracy_in_scope"] == 0.1621
    assert fb["massive_fallback"]["result"]["reader_backend"].startswith("rules")
    assert fb["miracl_pool"]["n_scoreable"] == 343
    assert fb["miracl_slice_invalid"]["n_scoreable"] == 3
    assert fb["xnli_nli_head"]["result"]["test"]["accuracy"] == 0.798


def test_frbench_newest_wins_per_family(tmp_path):
    old = _art(tmp_path, "xnli_pairclassifier_test_full", {"marker": "old"})
    (old.parent / "xnli_pairclassifier_dev_full").mkdir()
    newer = old.parent / "xnli_pairclassifier_dev_full" / "result.json"
    newer.write_text(json.dumps({"marker": "new"}))
    past = time.time() - 500
    os.utime(old / "result.json", (past, past))
    fb = _frbench(tmp_path / "runs")
    assert fb["xnli_pairclassifier"]["marker"] == "new", \
        "stale split artefacts must not outrank the fresh one"


def test_frbench_missing_dir_is_empty_not_crash(tmp_path):
    assert _frbench(tmp_path / "runs") == {}


# ------------------------------------------------------------------ pool loader

def _shard(d: Path, idx: int, docs: list[tuple[str, str]]):
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"docs-{idx}.jsonl.gz", "wt") as f:
        for docid, text in docs:
            f.write(json.dumps({"docid": docid, "text": text}) + "\n")


def test_read_passages_pool_finds_all_and_names_the_missing(tmp_path):
    corpus = tmp_path / "corpus"
    _shard(corpus, 0, [("1#0", "un"), ("2#0", "deux")])
    _shard(corpus, 1, [("3#0", "trois")])
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1\tQ0\t1#0\t1\nq1\tQ0\t2#0\t0\nq2\tQ0\t3#0\t1\nq2\tQ0\t9#9\t1\n")
    keep = frbench.qrels_candidate_ids(qrels)
    assert keep == {"1#0", "2#0", "3#0", "9#9"}
    with pytest.raises(RuntimeError, match="9#9"):
        frbench.read_passages_pool(corpus, keep)
    found = frbench.read_passages_pool(corpus, {"1#0", "2#0", "3#0"})
    assert [p["docid"] for p in found] == ["1#0", "2#0", "3#0"]
    # the slice optimisation must still yield full text
    assert found[1]["text"] == "deux"


# ------------------------------------------------------------- report §6d render

def _minimal_collected(**over):
    c = {"git": "t", "date": "2026-01-01", "config_hash": "h"}
    c.update(over)
    return c


def test_report_renders_external_section_from_artifacts():
    fb = {
        "massive_model": {"result": {
            "reader_backend": "model(reader)", "kind_query_rate": 0.1221,
            "intent_accuracy_in_scope": 0.1621, "n_in_scope": 1598,
            "macro_f1_in_scope": 0.0904, "oov_rejection_rate": 0.9448,
            "mean_intent_confidence": 0.9601, "abstained_rate": 0.0}},
        "miracl_pool": {"arms": {"dense": {"recall@1": 0.3016, "recall@5": 0.7649,
                                           "recall@10": 0.9808, "mrr@10": 0.6965,
                                           "ndcg@10": 0.7388},
                                 "dense_refined": {"recall@1": 0.2829, "recall@5": 0.7659,
                                                   "recall@10": 0.9706, "mrr@10": 0.6774,
                                                   "ndcg@10": 0.7212}},
                        "n_scoreable": 343, "n_queries": 343, "n_passages": 3407,
                        "concept_refine": {"anchored": True,
                                           "cosine_to_base_mean": 0.9853,
                                           "drift_p95": 0.1763,
                                           "max_drift_allowed": 0.25,
                                           "n_refined": 3407, "status": None}},
        "miracl_slice_invalid": {"n_scoreable": 3, "n_queries": 343, "n_passages": 5000},
        "xnli_nli_head": {"result": {
            "head": {"n_train_rows": 60000, "val_macro_f1_at_selection": 0.8102},
            "chance_accuracy": 0.3333,
            "test": {"accuracy": 0.798, "macro_f1": 0.7986,
                     "per_class": {"supports": {"f1": 0.7995},
                                   "unrelated": {"f1": 0.7655},
                                   "contradicts": {"f1": 0.8307}},
                     "confusion": {}}}},
    }
    text = build(_minimal_collected(frbench=fb))
    assert "## 6d" in text
    assert "0.1621" in text and "0.1221" in text, "model reader numbers must appear"
    assert "0.798" in text and "60000" in text
    assert "`dense_refined`" in text and "0.9853" in text
    assert "could judge only 3/343" in text, "the invalid slice must stay visible"
    assert "not measured — run `make frbench`" not in text


def test_report_external_section_says_not_measured_when_nothing_fetched():
    text = build(_minimal_collected(frbench={}))
    assert "run `make frbench`" in text
    # and the §10 real-corpus bullet must switch wording to the un-run form
    assert "downloads not performed" in text


def test_report_renders_fallback_reader_honestly_when_only_that_ran():
    fb = {"massive_fallback": {"result": {
        "reader_backend": "rules[fallback: OOM]", "kind_query_rate": 0.0652,
        "intent_accuracy_in_scope": 0.0939, "n_in_scope": 1598,
        "macro_f1_in_scope": 0.0821, "oov_rejection_rate": 0.9731,
        "mean_intent_confidence": None, "abstained_rate": 0.0}}}
    text = build(_minimal_collected(frbench=fb))
    assert "rules[fallback" in text, "the report must show which backend actually ran"


# ------------------------------------------------------------------- fetch tool

def test_fetch_is_idempotent_and_updates_provenance(tmp_path, monkeypatch):
    from engine.tools import fetch_french as ff

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"sha": "deadbeefcafe1234"}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    # _fetch resolves repo_root() at call time -- patch THAT, or the test writes
    # provenance into the real repo (the first draft reached the hub for b.json).
    monkeypatch.setattr(ff, "repo_root", lambda: tmp_path)
    folder = tmp_path / "data" / "external" / "fake__corp"
    folder.mkdir(parents=True)
    # unit tests never hit the hub: both files pre-exist, so the "cached" branch
    # is what runs; the download path lives behind `make frdata`
    (folder / "a.json").write_text("cached")
    (folder / "b.json").write_text("cached")
    got = ff._fetch("fake__corp", "fake/corp", ["a.json", "b.json"], "cc-by-4.0")
    assert got["revision"] == "deadbeefca", "return value is the 10-char short sha"
    assert [state for _, _, state in got["files"]] == ["cached", "cached"]
    prov = json.loads((folder / "PROVENANCE.json").read_text())
    assert prov["revision"] == "deadbeefcafe1234", "the file keeps the full pin"
    assert prov["licence_card"] == "cc-by-4.0"
    assert set(prov["files"]) == {"a.json", "b.json"}


# ---------------------------------------------------------- CLI plumbing (frbench)

def test_cmd_frbench_passes_pool_and_concept_to_the_harness(monkeypatch):
    """`--pool`/`--concept` were added to the parser before the call site forwarded
    them -- a knob accepted at the CLI and silently discarded there is the D-011
    inert-knob class, and the concept arm would be unreachable from `sae`."""
    import io
    from contextlib import redirect_stdout

    import engine.cli as cli
    import eval.frbench as FB

    seen = {}

    def fake_run(comp, **kw):
        seen[comp] = kw
        return {"component": comp, "result": {"ok": True}}

    monkeypatch.setattr(FB, "run", fake_run)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["frbench", "--component", "xnli-nli", "--pool", "qrels",
                         "--concept", "models/concept/concept-e5.pt"])
    assert code == 0
    assert seen["xnli-nli"]["pool"] == "qrels"
    assert seen["xnli-nli"]["concept"] == "models/concept/concept-e5.pt"
    assert seen["xnli-nli"]["rerank"] is True


def test_cmd_frbench_reports_a_failing_component_as_nonzero(capsys, monkeypatch):
    import engine.cli as cli
    import eval.frbench as FB

    def boom(comp, **kw):
        raise RuntimeError("qrels docids absent from corpus shards")

    monkeypatch.setattr(FB, "run", boom)
    code = cli.main(["frbench", "--component", "miracl"])
    err = capsys.readouterr().err
    assert code == 1, "a component that raised must not exit green"
    assert "miracl FAILED" in err and "docids" in err


def test_cmd_frdata_prints_pinned_revisions(capsys, monkeypatch):
    import engine.cli as cli
    from engine.tools import fetch_french as ff

    monkeypatch.setattr(ff, "fetch",
                        lambda only=None: {"xnli": {"revision": "deadbeefca",
                                                   "files": []}})
    code = cli.main(["frdata", "--only", "xnli"])
    assert code == 0
    assert "deadbeefca" in capsys.readouterr().out


# ------------------------------------------------- DATASETS generator (no lies)

def test_datasets_generator_reports_external_use_from_disk(tmp_path):
    """The generated file must never carry the old hardcoded "Nothing external"
    sentence once provenance files exist -- a claim that survives regeneration is
    exactly how that line went stale in the first place."""
    from engine.tools.verify_datasets import Row, _usage_state, render

    ext = tmp_path / "data" / "external" / "miracl__miracl"
    ext.mkdir(parents=True)
    (ext / "PROVENANCE.json").write_text(json.dumps(
        {"repo": "miracl/miracl", "revision": "abc1234567", "files": ["t.tsv"],
         "licence_card": "apache-2.0"}))
    fbdir = tmp_path / "runs" / "frbench" / "miracl_pool_dev_full"
    fbdir.mkdir(parents=True)
    (fbdir / "result.json").write_text("{}")
    rows = [Row(asserted="MIRACL-fr", tried=["miracl/miracl"], resolved="miracl/miracl",
                licence="apache-2.0", licence_source="card front-matter",
                guide_claim="not stated", agrees=None,
                note="guide stated no licence; card states apache-2.0")]
    text = render(rows, root=tmp_path)
    assert "1 external corpus/corpora fetched" in text
    assert "miracl/miracl" in text and "1 real-French component verdict" in text
    assert "Nothing external" not in text
    # the honest empty state still exists
    assert "Nothing external" in "\n".join(_usage_state(tmp_path / "missing"))
