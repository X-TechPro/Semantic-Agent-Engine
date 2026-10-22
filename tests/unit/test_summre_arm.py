"""The B-13 SUMM-RE arm: semantics, wiring, report rows, DATASETS states.

`eval_summre_temporal` talks to torch and a store, so its wiring is exercised
here with fakes — what must be true offline is: the one-read guard fires before
any model loads, the alignment table is what makes a pair scoreable, arm
semantics (A stale-first vs B demoted) flow into the numbers, and 0.0 is a
measurement, not a missing value (the falsy-zero trap). The GPU body itself is
run by hand and lands as `runs/frbench/summre_*_test_full` (e5 arms cannot be
unit-tier).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.frbench import build_meeting_source, summre_pair_scores
from eval.report import summre_section

pytestmark = pytest.mark.unit


# ------------------------------------------------------------- pure semantics

def test_pair_scores_win_demotion_and_falsy_zero():
    rows = [{"rank_old": 0, "rank_new": None},        # stale on top, current absent
            {"rank_old": None, "rank_new": 0},        # demoted out = a win
            {"rank_old": 4, "rank_new": 1},           # current above stale
            {"rank_old": 1, "rank_new": 4}]            # stale above current
    got = summre_pair_scores(rows)
    assert got["n_pairs"] == 4
    assert got["new_above_old"] == 0.5
    assert got["stale_above_new"] == 0.5
    assert got["mrr_new_at_k"] == round((1.0 + 0.5 + 0.2) / 4, 4)
    assert got["both_in_pool"] == 2
    # a measured all-wrong arm keeps its 0.0 (never None, never skipped)
    zero = summre_pair_scores([{"rank_old": 0, "rank_new": None}])
    assert zero["new_above_old"] == 0.0, "0.0 is a measurement, not an absence"
    assert summre_pair_scores([])["new_above_old"] is None   # no denominator = None


def test_meeting_source_carries_turn_indices_and_relative_chronology():
    segs = [{"speaker": "01", "start_min": 1.5, "text": "on part sur openstack d'abord"},
            {"speaker": "02", "start_min": 9.0, "text": "finalement la salle au palais"}]
    src = build_meeting_source("mX", segs)
    assert src.project == "mX" and src.source_id == "summre:mX"
    assert [t.turn_index for t in src.turns] == [0, 1]
    assert src.turns[1].timestamp > src.turns[0].timestamp
    assert src.language == "fr"


# ------------------------------------------------------------------ wiring

class _Unit:
    def __init__(self, uid, ti, source):
        self.unit_id, self.turn_index = uid, ti
        self.source_id, self.status, self.superseded_by = source, "mention", None


class _Store:
    def __init__(self, units, edges):
        self._units, self._edges = units, edges

    def units_of_source(self, sid):
        return [u for u in self._units if u.source_id == sid]

    def edges_for(self, uid, direction="from"):
        return [e for e in self._edges if e.frm == uid]


class _FakeEngine:
    """Two arms, one behaviour difference: linker_enabled — like the real bench."""
    def __init__(self, cfg, path=None, linker_enabled=False, embedder=None,
                 roster=None, reranker=None, _shared=None):
        self.linker_enabled = linker_enabled
        self.reader_backend = "rules"
        self.ingest_pipeline = SimpleNamespace(linker_backend="rules")
        self.store = _Store(_shared["units"],
                            _shared["edges"] if linker_enabled else [])
        self._shared = _shared
        self.ingested = []

    def ingest(self, src):
        self.ingested.append(src.source_id)
        if self.linker_enabled:
            # simulate the one edge the gold says the linker should find
            u = {x.unit_id: x for x in self._shared["units"]}
            u["u-old"].superseded_by = "u-new"

    def retrieve(self, q, linker_enabled=True, scoping=True):
        u = {x.unit_id: x for x in self._shared["units"]}
        order = (["u-new", "u-other", "u-old"] if linker_enabled
                 else ["u-old", "u-new", "u-other"])
        return [SimpleNamespace(unit=u[uid]) for uid in order]

    def close(self):
        pass


@pytest.fixture()
def summre_tmp(tmp_path, monkeypatch):
    meetings = {"mA": [
        {"speaker": "01", "start_min": 1.0, "text": "on part sur openstack pour la salle"},
        {"speaker": "02", "start_min": 9.0, "text": "finalement la salle sera au palais"},
        {"speaker": "01", "start_min": 12.0, "text": "et le budget passe par la mairie"},
    ]}
    gold = {"meetings": {"mA": {"sentences_reviewed": 3, "threads": [
        {"topic": "salle.choix", "relations": [
            {"old": 0, "new": 1, "relation": "supersedes"},
            {"old": 0, "new": 2, "relation": "supports"}]}]}}}
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "external" / "summre_test").mkdir(parents=True)
    (tmp_path / "data/gold/summre_temporal_test.json").write_text(json.dumps(gold))
    (tmp_path / "data/external/summre_test/meetings.json").write_text(
        json.dumps(meetings))
    from eval import frbench
    monkeypatch.setattr(frbench, "repo_root", lambda: tmp_path)
    shared = {"units": [_Unit("u-old", 0, "summre:mA"),
                        _Unit("u-new", 1, "summre:mA"),
                        _Unit("u-other", 2, "summre:mA")],
              "edges": [SimpleNamespace(frm="u-old", relation="supports")]}
    engines = []

    def fake_engine(cfg, **kw):
        eng = _FakeEngine(cfg, _shared=shared, **{k: v for k, v in kw.items()
                                                  if k != "path"})
        engines.append(eng)
        return eng

    monkeypatch.setattr("engine.engine.Engine", fake_engine)
    embed_calls: list = []

    def fake_build(cfg):
        e = SimpleNamespace(version="e5-fake-v1")
        embed_calls.append(e)
        return e

    monkeypatch.setattr("engine.embed.base.build_embedder", fake_build)
    return tmp_path, frbench, engines, embed_calls


def test_eval_summre_wiring_scores_arms_and_writes_artefact(summre_tmp):
    tmp_path, frbench, engines, _embed = summre_tmp
    res = frbench.eval_summre_temporal(device="cpu")
    # arms differ by exactly the flags, and the semantics land
    A, B = res["arms"]["baseline_A"], res["arms"]["system_B"]
    assert A["n_pairs"] == 1 and B["n_pairs"] == 1
    assert A["new_above_old"] == 0.0 and A["stale_above_new"] == 1.0
    assert B["new_above_old"] == 1.0 and B["stale_above_new"] == 0.0
    # mechanism + instrument columns computed, not invented
    mech = res["mechanism_B"]
    assert mech["marked"] == 1 and mech["marked_superseded_rate"] == 1.0
    assert mech["direct"] == 1          # superseded_by == the gold-new unit
    assert mech["supports"] == {"found": 1, "n": 1, "rate": 1.0}
    assert res["instrument"]["turns_without_unit"] == 0
    assert res["instrument"]["segments_vs_units"]["mA"] == [3, 3]
    assert res["reader_backend"] == "rules" and res["linker_backend"] == "rules"
    # artefact written under the tmp root, backend-named
    arts = list((tmp_path / "runs" / "frbench").glob("summre_*_test_full/result.json"))
    assert len(arts) == 1
    assert json.loads(arts[0].read_text())["n_supersedes_pairs"] == 1
    # §9.5: the SAME read may not be spent twice
    with pytest.raises(RuntimeError, match="already spent"):
        frbench.eval_summre_temporal(device="cpu")


def test_summre_arm_shares_one_embedder_across_arms(summre_tmp):
    """The 8 GB fairness rule: ONE embedder instance injected into both arms
    (sae-bench-arms-share-models). Building it per arm stacked 568M-parameter
    models before and OOMed the §10 row -- and would make the arms differ by
    allocator pressure rather than by `linker_enabled`."""
    tmp_path, frbench, engines, embed_calls = summre_tmp
    frbench.eval_summre_temporal(device="cpu")
    assert [e.linker_enabled for e in engines] == [False, True], "A then B"
    assert len(embed_calls) == 1, "build_embedder called once for both arms"
    assert len({id(e.store) for e in engines}) == 2, "two stores, one model"


def test_run_dispatches_summre_with_the_device(monkeypatch):
    """The dispatch branch is part of the wired surface (D-011 class): a run()
    that silently dropped `summre` or its device flag would make the spent-read
    arm unreachable from `sae frbench --component summre`."""
    from eval import frbench

    seen = {}

    def fake_eval(device="cuda"):
        seen["device"] = device
        return {"ok": True}

    monkeypatch.setattr(frbench, "eval_summre_temporal", fake_eval)
    out = frbench.run("summre", device="cpu")
    assert seen == {"device": "cpu"}
    assert out == {"component": "summre", "result": {"ok": True}}


def test_all_sweep_excludes_the_one_shot_summre_read(monkeypatch):
    """`--component all` is the tunable sweep; summre is one-read test-only and
    must never be re-spent by a forgotten flag (9.5)."""
    from eval import frbench
    seen = []
    monkeypatch.setattr(frbench, "run",
                        lambda c, **kw: seen.append(c) or {"component": c, "result": {}})
    rc = frbench.main(["--component", "all"])
    assert rc == 0 and "summre" not in seen and "miracl" in seen
    rc = frbench.main(["--component", "summre"])
    assert rc == 0 and seen[-1] == "summre"


# ----------------------------------------------------- report rows, computed

def test_summre_section_orders_mechanism_before_delta_and_survives_zeros():
    st = {"n_supersedes_pairs": 28, "reader_backend": "rules",
          "linker_backend": "rules", "top_k": 30,
          "arms": {"baseline_A": {"n_pairs": 28, "new_above_old": 0.25,
                                  "stale_above_new": 0.61, "mrr_new_at_k": 0.3},
                   "system_B": {"n_pairs": 28, "new_above_old": 0.43,
                                "stale_above_new": 0.29, "mrr_new_at_k": 0.45}},
          "mechanism_B": {"marked": 4, "n_pairs": 28,
                          "marked_superseded_rate": 0.1429, "direct_chain_rate": 0.0,
                          "contradicts": {"found": 0, "n": 8, "rate": 0.0},
                          "supports": {"found": 2, "n": 29, "rate": 0.069}},
          "instrument": {"status_mix_B": {"mention": 900, "confirmed": 4}}}
    rows = summre_section(st)
    joined = "\n".join(rows)
    mech_i = next(i for i, r in enumerate(rows) if "mechanism reach" in r)
    delta_i = next(i for i, r in enumerate(rows) if "newer claim above" in r)
    assert mech_i < delta_i, "the ceiling is stated before the delta is read"
    assert "chain direct 0" in joined, \
        "a measured 0.0 chain rate must print as a zero, never vanish (falsy-zero); " \
        "the exact digits belong to _fmt, the fact that it prints belongs here"
    assert "0.069" in joined
    assert summre_section({}) == []


def test_summre_section_warns_on_empty_denominator():
    st = {"arms": {"baseline_A": {"n_pairs": 0}, "system_B": {"n_pairs": 0}},
          "mechanism_B": {}, "instrument": {}}
    rows = summre_section(st)
    assert any("nothing was scoreable" in r for r in rows)


# ------------------------------------------------- DATASETS three-state claim

def test_usage_state_computes_the_summre_claim(tmp_path):
    from engine.tools.verify_datasets import _usage_state
    (tmp_path / "data" / "external" / "x").mkdir(parents=True)
    (tmp_path / "data" / "external" / "x" / "PROVENANCE.json").write_text(
        json.dumps({"repo": "x", "revision": "r", "files": ["a"], "licence_card": ""}))
    (tmp_path / "runs" / "frbench").mkdir(parents=True)
    (tmp_path / "runs" / "frbench" / "xnli_a_test_full").mkdir()
    (tmp_path / "runs/frbench/xnli_a_test_full/result.json").write_text("{}")

    out = "\n".join(_usage_state(tmp_path))
    assert "stays synthetic: no hand-labelled" in out        # no gold, no artefact

    (tmp_path / "data" / "gold").mkdir()
    (tmp_path / "data/gold/summre_temporal_test.json").write_text("{}")
    out = "\n".join(_usage_state(tmp_path))
    assert "BUILT" in out and "has not been run" in out      # gold, unread

    d = tmp_path / "runs" / "frbench" / "summre_e5_test_full"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(
        {"arms": {"baseline_A": {"new_above_old": 0.25},
                  "system_B": {"new_above_old": 0.43, "n_pairs": 28}}}))
    out = "\n".join(_usage_state(tmp_path))
    assert "25%" in out and "43%" in out and "28 hand-labelled" in out


# ------------------------------------------------------- the gold file itself

GOLD = Path(__file__).resolve().parents[2] / "data/gold/summre_temporal_test.json"
EXTRACT = (Path(__file__).resolve().parents[2]
           / "data/external/summre_test/meetings.json")


@pytest.mark.skipif(not (GOLD.exists() and EXTRACT.exists()),
                    reason="hand-labelled set lives with the local extract")
def test_gold_labels_stay_in_range_ordered_and_in_vocab():
    """The one read is only as honest as the labels: this re-checks EVERY index,
    ordering and relation against the extract it points into, so a later edit
    to either file cannot quietly make the spent read wrong."""
    g = json.loads(GOLD.read_text())
    m = json.loads(EXTRACT.read_text())
    vocab = {"supersedes", "contradicts", "supports"}
    for mid, mm in g["meetings"].items():
        segs = m[mid]
        assert len(segs) == mm["sentences_reviewed"]
        for th in mm["threads"]:
            for r in th["relations"]:
                assert r["relation"] in vocab
                assert 0 <= r["old"] < len(segs) and 0 <= r["new"] < len(segs)
                assert r["old"] < r["new"], f"{mid} {th['topic']}"
    sup = sum(1 for mm in g["meetings"].values() for th in mm["threads"]
              for r in th["relations"] if r["relation"] == "supersedes")
    assert sup == 28, "the spent read's denominator; changing it is a new experiment"
