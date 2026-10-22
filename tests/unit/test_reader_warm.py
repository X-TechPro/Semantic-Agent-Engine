"""Reader warm-up safety machinery, tested without torch or GPU (B-12 / D-024).

The heavy path runs by hand once; what belongs in the unit tier is the part
where a silent mistake would corrupt evidence rather than crash:

* one decision table: `eval.frbench` must import the engine's mapping, never
  carry a copy (the reader-gold duplication failure, one level up);
* guard-input heads masked at assembly: `type`/`status` gradients must be
  structurally impossible, so their label columns are checked here;
* measured zeros stay numbers: `none_share == 0.0` and an accuracy of exactly
  0.0 must survive the signal computation (the falsy-zero trap);
* the refusal guards: base==out overwrite, and the second ledger-test read;
* artefact identity: the warm family must never become the report's baseline
  row, and a `_rwarm` stage run must not hijack the default stage line;
* CLI plumbing: `--reader-model`/`--reader` must reach the harness, not be
  accepted and discarded (the D-011 inert-knob class, tested the same way as
  the `--pool`/`--concept` rows).
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from engine.read import massive as MA
from engine.read.dataset import QUERY_KIND, QUERY_STATUS, QUERY_TYPE
from engine.read.model import HEADS, MASK

pytestmark = pytest.mark.unit

FIXTURE = [
    {"id": 1, "text": "Où est la gare ?", "label": "transport_traffic"},
    {"id": 2, "text": "Mets un réveil à 7h", "label": "alarm_set"},
    {"id": 3, "text": "Envoie un mail à Marion", "label": "email_sendemail"},
    {"id": 4, "text": "Un truc marrant", "label": "general_quirky"},
    {"id": 5, "text": "Qui a écrit Les Misérables ?", "label": "qa_factoid"},
]


def _write_massive(root, split: str, rows: list[dict]) -> None:
    d = root / "data" / "external" / MA.MASSIVE_DIR_NAME / split
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "fr.json.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------- one table

def test_frbench_reads_the_engines_table_not_a_copy():
    from eval import frbench
    assert frbench.map_massive_intent is MA.map_massive_intent
    assert frbench.MAP_RULES is MA.MAP_RULES
    assert frbench.load_massive is MA.load_massive


def test_mapping_is_the_auditable_substring_rule():
    assert MA.map_massive_intent("qa_maths") == "answer"
    assert MA.map_massive_intent("weather_query") == "answer"
    assert MA.map_massive_intent("calendar_set_date") == "schedule_event"
    assert MA.map_massive_intent("lists_createoradd") == "create_task"
    assert MA.map_massive_intent("email_sendemail") == "draft_email"
    assert MA.map_massive_intent("general_quirky") is None
    assert MA.map_massive_intent("play_radio") is None


# ------------------------------------------------------------- row assembly

def test_massive_rows_labels_none_share_and_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(MA, "repo_root", lambda: tmp_path)
    _write_massive(tmp_path, "train", FIXTURE)
    got = MA.massive_rows("train")
    assert [r["intent"] for r in got["rows"]] == \
        ["answer", "schedule_event", "draft_email", "none", "answer"]
    assert got["n"] == 5 and got["n_mapped"] == 4
    assert got["none_share"] == 0.2
    assert got["intent_mix"]["answer"] == 2
    # a measured 0.0 none-share must stay 0.0 (falsy-zero trap)
    _write_massive(tmp_path, "allmapped", [FIXTURE[0], FIXTURE[4]])
    assert MA.massive_rows("allmapped")["none_share"] == 0.0
    # wrong layout raises, it does not train on invented labels
    _write_massive(tmp_path, "broken", [{"text": "pas de label"}])
    with pytest.raises(ValueError, match="lack fields"):
        MA.massive_rows("broken")


def test_massive_rows_limit_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(MA, "repo_root", lambda: tmp_path)
    big = [dict(r, id=i) for i in range(40) for r in FIXTURE]
    _write_massive(tmp_path, "train", big)
    a = MA.massive_rows("train", limit=10, seed=7)
    b = MA.massive_rows("train", limit=10, seed=7)
    assert a["n"] == 10
    assert [r["text"] for r in a["rows"]] == [r["text"] for r in b["rows"]]


def test_row_label_arrays_uses_the_query_convention_not_literals():
    rows = [{"intent": "answer"}, {"intent": "none"}]
    la = MA.row_label_arrays(rows)
    assert list(la["kind"]) == [HEADS["kind"].index(QUERY_KIND)] * 2
    assert list(la["type"]) == [HEADS["type"].index(QUERY_TYPE)] * 2
    assert list(la["status"]) == [HEADS["status"].index(QUERY_STATUS)] * 2
    assert list(la["intent"]) == [HEADS["intent"].index("answer"),
                                  HEADS["intent"].index("none")]


def test_warm_packing_masks_the_guard_input_heads():
    from engine.read import warm as W
    packed = W._massive_packed({"rows": [{"text": "a", "intent": "answer"}], "n": 1})
    assert list(packed["labels"]["type"]) == [MASK]
    assert list(packed["labels"]["status"]) == [MASK]
    assert list(packed["labels"]["kind"]) != [MASK]
    assert list(packed["labels"]["intent"]) != [MASK]


def test_freeze_masks_guard_heads_across_the_whole_train_set():
    """The exact regression the first run hit: replay rows carried live
    type/status gold while massive rows were masked, so the heads feeding the
    guards trained -- caught by the drift assertion, now pinned here."""
    from engine.read import warm as W
    labels = {"kind": np.array([0, 1, 1]), "type": np.array([2, 3, 0]),
              "status": np.array([5, 5, 5]), "intent": np.array([0, 7, -1])}
    out = W._freeze_guard_heads(labels, 3)
    assert list(out["type"]) == [MASK, MASK, MASK]
    assert list(out["status"]) == [MASK, MASK, MASK]
    assert list(out["kind"]) == [0, 1, 1], "trainable heads pass through untouched"
    assert list(out["intent"]) == [0, 7, -1]
    assert list(labels["type"]) == [2, 3, 0], "the caller's dict is not mutated"


def test_signal_keeps_a_measured_zero_in_the_mean():
    from engine.read import warm as W
    gold_mv = {"kind": np.array([1, 1]), "intent": np.array([0, 0])}
    pred_mv = {"kind": np.array([0, 0]), "intent": np.array([0, 0])}   # kind all wrong
    gold_ld = {"kind": np.array([0, 1]), "intent": np.array([0, 0])}
    pred_ld = {"kind": np.array([0, 1]), "intent": np.array([0, 0])}
    sig, rep = W._signal({"labels": gold_mv, "preds": pred_mv, "probs": {}},
                         {"labels": gold_ld, "preds": pred_ld, "probs": {}})
    assert rep["massive_kind_acc"] == 0.0, "a zero accuracy is a measured zero, not absent"
    assert rep["ledger_kind_f1"] == 1.0
    assert 0.0 < sig < 1.0


# ----------------------------------------------------------------- refusals

def test_train_refuses_to_overwrite_its_own_base(tmp_path):
    from engine.read import warm as W
    with pytest.raises(ValueError, match="refusing to overwrite"):
        W.train(base=tmp_path, out=tmp_path)
    with pytest.raises(FileNotFoundError, match="no trained reader"):
        W.train(base=tmp_path / "absent", out=tmp_path / "warm")


def test_revalidate_refuses_the_second_ledger_test_read(tmp_path, monkeypatch):
    from engine.read import warm as W
    monkeypatch.setattr(W, "RUN_DIR", tmp_path)
    (tmp_path / "ledger_test_once.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already read once"):
        W.revalidate()


def test_frbench_refuses_the_second_warm_test_read(tmp_path, monkeypatch):
    from eval import frbench
    monkeypatch.setattr(frbench, "repo_root", lambda: tmp_path)
    d = frbench.run_dir_for("massive", frbench.embed_tag("model(reader_warm)")[:32],
                            None, "test")
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already read once"):
        frbench.eval_reader_massive("test", model_dir="models/reader_warm")


# ---------------------------------------------------------------- artefact identity

def test_frbench_family_keeps_warm_out_of_the_baseline_row(tmp_path):
    runs = tmp_path / "runs" / "frbench"
    base = runs / "massive_model-reader_test_full"
    base.mkdir(parents=True)
    (base / "result.json").write_text(json.dumps(
        {"result": {"reader_backend": "model(reader)",
                    "intent_accuracy_in_scope": 0.1621}}))
    warm = runs / "massive_model-reader_warm_test_full"
    warm.mkdir(parents=True)
    (warm / "result.json").write_text(json.dumps(
        {"result": {"reader_backend": "model(reader_warm)",
                    "intent_accuracy_in_scope": 0.6}}))
    # a VALIDATION run that finishes last must not retag the headline row:
    # newest-by-mtime inside `massive_model*` would otherwise swap in tuning-split
    # numbers under a test-split label
    vdir = runs / "massive_model-reader_validation_full"
    vdir.mkdir(parents=True)
    (vdir / "result.json").write_text(json.dumps(
        {"result": {"reader_backend": "model(reader)",
                    "intent_accuracy_in_scope": 0.5555}}))
    from eval.report import _frbench
    fb = _frbench(tmp_path / "runs")
    assert fb["massive_model"]["result"]["intent_accuracy_in_scope"] == 0.1621, \
        "the baseline row must not be silently replaced by the newer warm or val run"
    assert fb["massive_model_warm"]["result"]["intent_accuracy_in_scope"] == 0.6
    assert fb["massive_model_val"]["result"]["intent_accuracy_in_scope"] == 0.5555


def test_stage_row_refuses_warm_reader_ablations(tmp_path):
    from eval.report import _stage
    d = tmp_path / "stage_linker_test_42_rwarm"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"reader_backend": "model(reader_warm)",
                                               "marker": "warm-row"}))
    base = tmp_path / "stage_linker_test_42"
    base.mkdir()
    (base / "result.json").write_text(json.dumps({"marker": "rules-row"}))
    assert _stage(tmp_path, "linker")["marker"] == "rules-row"


def test_stage_helpers_map_reader_choice_to_config_and_dirs():
    from eval.run_stage import _dir_suffix, _reader_override
    assert _reader_override(None) is None
    assert _reader_override("rules") == {"backend": "rules"}
    assert _reader_override("model") == {"backend": "model"}
    o = _reader_override("warm")
    assert o["backend"] == "model" and o["model_dir"].endswith("reader_warm"), \
        "warm selects the trained backend at the warm dir -- NOT a new config " \
        "backend value (which would move the config hash and force a gate flip)"
    assert _dir_suffix(None, None) == ""
    assert _dir_suffix("nli", None) == "_nli"
    assert _dir_suffix(None, "warm") == "_rwarm"
    assert _dir_suffix("nli", "warm") == "_nli_rwarm"


# ----------------------------------------------------------------- CLI plumbing

def test_cli_parser_accepts_the_new_surfaces():
    import engine.cli as cli
    p = cli.build_parser()
    a = p.parse_args(["frbench", "--component", "massive",
                      "--reader-model", "models/reader_warm"])
    assert a.reader_model == "models/reader_warm"
    b = p.parse_args(["stage", "reader", "--reader", "warm", "--split", "dev"])
    assert b.reader == "warm" and b.split == "dev"
    c = p.parse_args(["reader-warm", "--train-only"])
    assert c.train_only is True and c.fn is cli.cmd_reader_warm
    assert c.max_train == 4000


def test_cmd_frbench_passes_reader_model_to_the_harness(monkeypatch):
    """Same D-011 class as the --pool/--concept rows: a flag the parser accepts
    and the call site discards makes the warm checkpoint unreachable from `sae`
    while `python -m eval.frbench` had it."""
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
        code = cli.main(["frbench", "--component", "massive",
                         "--reader-model", "models/reader_warm"])
    assert code == 0
    assert seen["massive"]["reader_model"] == "models/reader_warm"


# ------------------------------------------------- orchestration with a fake model

def _fake_parts():
    """Ledger-shaped replay rows with REAL type/status gold -- exactly what the
    first run left unmasked and the drift assertion caught (613/613)."""
    def pack(texts, kinds, types, statuses, intents):
        return {"texts": texts,
                "speakers": [None] * len(texts),
                "contexts": [None] * len(texts),
                "labels": {"kind": np.array(kinds), "type": np.array(types),
                           "status": np.array(statuses), "intent": np.array(intents)}}
    return {"train": pack(["a", "b", "c", "d"], [0, 1, 0, 1], [5, 5, 5, 5],
                          [5, 5, 5, 5], [-1, 0, -1, 2]),
            "dev": pack(["e", "f"], [0, 1], [5, 5], [5, 5], [-1, 0]),
            "test": pack(["g", "h"], [0, 1], [5, 5], [5, 5], [-1, 0])}


def _fake_massive(split="train", *, limit=None, seed=42):
    rows = [{"text": f"q{i}", "intent": "answer"} for i in range(4)]
    rows.append({"text": "oov", "intent": "none"})
    return {"rows": rows, "n": len(rows), "n_mapped": 4, "none_share": 0.2,
            "intent_mix": {"answer": 4, "none": 1}}


class _StubModel:
    """ReaderModel stand-in: no torch, constant logits, records every fit call.
    `shift_after_fit` makes the STATUS head logits move once any fit happened --
    the refusal path's injected violation."""
    instances: list = []
    fitted_once = False
    shift_after_fit = False

    def __init__(self):
        self.temperatures = {"kind": 0.9, "type": 0.9, "status": 0.9, "intent": 0.9}
        self.train_meta = {"origin": "base-checkpoint"}
        self.max_length = 96
        self.fits: list[dict] = []
        _StubModel.instances.append(self)

    @classmethod
    def load(cls, directory, device=None):
        return cls()

    def logits_batch(self, texts, speakers=None, contexts=None, batch_size=32):
        out = {h: np.zeros((len(texts), len(HEADS[h]))) for h in HEADS}
        if _StubModel.shift_after_fit and _StubModel.fitted_once:
            out["status"] = out["status"] + 1.0
        return out

    def predict_batch(self, texts, speakers=None, contexts=None, batch_size=32):
        logits = self.logits_batch(texts, speakers, contexts, batch_size)
        probs = {h: np.full_like(l, 1.0 / l.shape[1]) for h, l in logits.items()}
        return {h: l.argmax(-1) for h, l in logits.items()}, probs

    def fit(self, texts, labels, **kw):
        self.fits.append({"n": len(texts), "kw": kw, "labels": labels})
        _StubModel.fitted_once = True

    def calibrate(self, logits, gold):
        return {h: {"temperature": 0.5} for h in logits}

    def save(self, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(
            {"temperatures": self.temperatures, "train_meta": self.train_meta}))


def _prep_stub_train(tmp_path, monkeypatch):
    from engine.read import warm as W
    _StubModel.instances = []
    _StubModel.fitted_once = False
    base = tmp_path / "base"
    base.mkdir()
    (base / "meta.json").write_text("{}")
    out = tmp_path / "warm"
    monkeypatch.setattr(W, "ReaderModel", _StubModel)
    monkeypatch.setattr(W, "_ledger_parts", lambda seed, projects: (_fake_parts(), None))
    monkeypatch.setattr(W, "massive_rows", lambda split, **kw: _fake_massive())
    monkeypatch.setattr(W, "RUN_DIR", tmp_path / "runs")
    return W, base, out


def test_train_orchestration_masks_replay_and_saves_lineage(tmp_path, monkeypatch):
    W, base, out = _prep_stub_train(tmp_path, monkeypatch)
    try:
        art = W.train(epochs=2, max_train=None, batch=8, device="cpu",
                      base=base, out=out, projects=2)
    finally:
        _StubModel.shift_after_fit = False
    fitting = _StubModel.instances[0]
    assert len(fitting.fits) == 2
    for f in fitting.fits:
        assert f["kw"]["freeze_encoder"] is True
        # THE regression pin: the replay rows carry type/status gold (5s), and
        # the assembled training set must still show them fully masked --
        # gradient to a guard-input head is structurally impossible
        assert (f["labels"]["type"] == MASK).all()
        assert (f["labels"]["status"] == MASK).all()
        assert (f["labels"]["kind"] != MASK).any()
        assert (f["labels"]["intent"] != MASK).any()
    assert art["frozen_head_drift"] == {"type": 0.0, "status": 0.0}
    assert art["selected"]["epoch"] == 1, "a tie must keep the first improvement"
    saved = json.loads((out / "meta.json").read_text())
    wu = saved["train_meta"]["warmup"]
    assert wu["base"] == str(base)
    assert wu["base_train_meta"] == {"origin": "base-checkpoint"}, \
        "lineage is the BASE checkpoint's meta, not this run's own fit echo"
    assert wu["replay"]["n_ledger_train"] == 4
    assert (tmp_path / "runs" / "warm_result.json").exists()
    assert not (tmp_path / "runs" / "best").exists(), "scratch checkpoint cleaned"


def test_train_refuses_to_save_when_guard_heads_move(tmp_path, monkeypatch):
    W, base, out = _prep_stub_train(tmp_path, monkeypatch)
    _StubModel.shift_after_fit = True
    try:
        with pytest.raises(RuntimeError, match="guard-input heads moved"):
            W.train(epochs=1, max_train=None, batch=8, device="cpu",
                    base=base, out=out, projects=2)
        assert not (out / "meta.json").exists(), \
            "the violation must be caught BEFORE the shipped checkpoint is written"
    finally:
        _StubModel.shift_after_fit = False
