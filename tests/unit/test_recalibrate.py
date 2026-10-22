"""B-11: the offline recalibration audit, tested without a GPU.

The stub model reuses the real `ReaderModel.calibrate` (pure numpy + CPU torch)
so the accept/refuse arithmetic is the shipped one; only the logits are fake.
Empty denominators stay None, a measured 0.0 stays 0.0 (the falsy-zero trap),
and `apply` may only ever write `models/<name>_recal` — never a shipped
checkpoint — which is what the tmp-repo_root tests assert on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.offline import (abstain_scan, recal_accepts,
                            recalibrate_thresholds, rebuild_dev_rows)
from engine.read.model import ReaderModel

N = 8
KIND_GOLD = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _one_hot(gold, n_classes, sign=5.0):
    lg = np.zeros((len(gold), n_classes))
    for i, g in enumerate(gold):
        lg[i, int(max(g, 0))] = sign
        lg[i] -= sign / n_classes
    return lg


def _labels():
    return {"kind": KIND_GOLD,
            "type": np.array([1, 1, -1, -1, 2, 2, -1, -1]),
            "status": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "intent": np.full(N, -1)}


def _dev():
    return {"texts": [f"t{i}" for i in range(N)],
            "speakers": [None] * N, "contexts": [None] * N,
            "labels": _labels()}


def _logits(labels, flip_kind=False):
    out = {"kind": _one_hot(labels["kind"], 2),
           "type": _one_hot(labels["type"], 9),
           "status": _one_hot(labels["status"], 6),
           "intent": np.zeros((N, 8))}
    if flip_kind:                      # confident and WRONG — temperature cannot fix
        out["kind"] = -out["kind"]
    return out


class StubModel:
    """Duck-typed ReaderModel: injected logits, the REAL calibrate on self."""

    calibrate = ReaderModel.calibrate

    def __init__(self, temperatures, logits):
        self.temperatures = dict(temperatures)
        self._logits = {k: np.asarray(v, dtype="float64")
                        for k, v in logits.items()}
        self.train_meta = {"n_train": 100, "encoder": "stub", "seed": 42}

    def logits_batch(self, texts, speakers=None, contexts=None, batch_size=32):
        return {k: v.copy() for k, v in self._logits.items()}

    def save(self, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(
            {"temperatures": self.temperatures}), encoding="utf-8")


class _Eng:
    reader_backend = "model(stub)"
    reader = None
    cfg = {}


def test_abstain_scan_empty_denominators_none_measured_zero_kept():
    probs = {"kind": np.array([[0.99, 0.01]] * 2),
             "type": np.array([[0.99, 0.01] * 2] * 2)[:2],
             "status": np.array([[0.99, 0.01], [0.99, 0.01]])}
    gold = {"kind": np.array([0, 0]), "type": np.array([0, 0]),
            "status": np.array([0, 0])}
    rows = abstain_scan(probs, gold, grid=(0.5, 0.999))
    by_t = {r["threshold"]: r for r in rows}
    assert by_t[0.5]["abstain_rate"] == 0.0          # a measured zero, stays zero
    assert by_t[0.999]["abstain_rate"] == 1.0        # nothing kept at that floor
    assert by_t[0.999]["accuracy_kept"] is None      # empty denominator is None
    assert abstain_scan({}, {}) == []


def test_recal_accepts_accepts_improvement_and_refuses_each_failure():
    good = {"kind": {"ece": 0.02, "confidence_gap": 0.01},
            "type": {"ece": 0.02, "confidence_gap": 0.01},
            "status": {"ece": 0.02, "confidence_gap": 0.01}}
    refit = {h: {"ece_after": 0.01, "confidence_gap": 0.0} for h in good}
    assert recal_accepts(good, refit) == []
    worse = {"kind": {"ece_after": 0.05, "confidence_gap": 0.0},
             "type": {"ece_after": 0.01, "confidence_gap": 0.0},
             "status": {"ece_after": 0.01, "confidence_gap": 0.0}}
    assert any("worse" in r for r in recal_accepts(good, worse))
    breach = {h: {"ece_after": 0.30, "confidence_gap": 0.20} for h in good}
    reasons = recal_accepts(good, breach)
    assert len(reasons) == 9                        # worse + breach + gap, 3/head
    unmeas = dict(good); unmeas.pop("status")
    assert any("status" in r and "unmeasurable" in r
               for r in recal_accepts(unmeas, refit))
    # intent is not a gated head — its absence must never refuse a refit
    assert recal_accepts({**good, "intent": {}}, refit) == []


def test_rules_reader_reports_skipped_with_a_reason():
    class RulesEng:
        reader_backend = "rules"
        reader = None
        cfg = {}
    out = recalibrate_thresholds(RulesEng())
    assert "skipped" in out and "rules" in out["skipped"]
    assert out["model_name"] is None


def test_stale_calibration_accepts_and_writes_only_a_candidate(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr("engine.offline.repo_root", lambda: tmp_path)
    labels = _labels()
    # T=6.0 flattens confident logits: accuracy 1.0 at mean confidence ~0.90
    # — stale by the trainer's own tolerance, and the refit can only improve.
    model = StubModel({"kind": 6.0, "type": 6.0, "status": 6.0}, _logits(labels))
    out = recalibrate_thresholds(_Eng(), model=model, dev=_dev(),
                                 model_dir="models/reader", apply=False)
    assert out["stale"] is True and out["accept"] is True, out["refuse_reasons"]
    assert out["model_name"] == "reader"
    assert "apply" not in out                       # report-only writes nothing
    assert not (tmp_path / "models" / "reader_recal").exists()
    kept = {r["threshold"]: r for r in out["abstain_scan"]}
    # the shipped flat T=6.0 leaves min gated confidence around 0.2, so at the
    # 0.5 floor a stale calibration abstains EVERYTHING — visible in the report,
    # which is half the point of the scan.
    assert kept[0.5]["abstain_rate"] == 1.0
    assert kept[0.5]["accuracy_kept"] is None       # empty denominator is None
    out2 = recalibrate_thresholds(_Eng(), model=StubModel(
        {"kind": 6.0, "type": 6.0, "status": 6.0}, _logits(labels)),
        dev=_dev(), model_dir="models/reader", apply=True)
    cand = tmp_path / "models" / "reader_recal" / "meta.json"
    assert out2["accept"] and cand.exists()
    assert json.loads(cand.read_text())["temperatures"]


def test_confidently_wrong_head_refuses_and_writes_nothing(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr("engine.offline.repo_root", lambda: tmp_path)
    labels = _labels()
    model = StubModel({"kind": 0.05}, _logits(labels, flip_kind=True))
    out = recalibrate_thresholds(_Eng(), model=model, dev=_dev(),
                                 model_dir="models/reader", apply=True)
    assert out["accept"] is False
    assert any("breaches" in r or "worse" in r for r in out["refuse_reasons"])
    assert isinstance(out["apply"], str) and "refused" in out["apply"]
    assert not (tmp_path / "models" / "reader_recal").exists()
    # temperature scaling cannot move argmax: the guard-input labels are safe
    assert out["current"]["kind"]["accuracy"] == 0.0     # a zero, kept
    assert out["refit"]["kind"]["accuracy"] == 0.0


def test_rebuild_dev_rows_writes_splits(tmp_path):
    rows = rebuild_dev_rows(2, 42, tmp_path)
    assert rows["texts"] and rows["labels"]["kind"].dtype.kind in "iu"
    assert (tmp_path / "ledger_splits.json").exists()
