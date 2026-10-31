#!/usr/bin/env python3
"""Evaluate the synthetic-trained relation classifier on transcribed hard cases."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import memory_engine as me
import run_french_lab as lab
from relation_engine import predict_relation

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def main():
    engine = me.MeetingMemoryEngine(device="cpu", batch_size=64)
    ds = engine._get_dataset("transcribed")
    records = ds["records"]
    clf = joblib.load(ART / "relation_classifier.joblib")
    cases = json.loads((DATA / "relation_hard_cases.json").read_text(encoding="utf-8"))
    X, y, rows = [], [], []
    for case in cases:
        a, b = records[case["source"]], records[case["target"]]
        X.append(lab.pair_features(a["vector"], b["vector"], a, b))
        y.append(case["label"])
        rows.append(case)
    learned = clf.predict(np.asarray(X, dtype=np.float32))
    rule = [predict_relation(records[case["source"]], records[case["target"]])["relation"] for case in cases]
    labels = ["supports", "contradicts", "creates_action", "unrelated"]
    def block(pred):
        report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
        return {"accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "per_class": {label: {"precision": float(report[label]["precision"]),
                                      "recall": float(report[label]["recall"]),
                                      "f1": float(report[label]["f1-score"])} for label in labels},
                "confusion": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()}}
    payload = {
        "corpus": "transcribed hard cases",
        "pairs": len(cases),
        "label_counts": dict(Counter(y)),
        "learned": block(learned),
        "structural": block(rule),
        "predictions": [{"pair_id": case["pair_id"], "gold": case["label"],
                         "learned": str(pred), "structural": rule_pred,
                         "source": case["source"], "target": case["target"]}
                        for case, pred, rule_pred in zip(rows, learned, rule)],
        "synthetic_reference": json.loads((ART / "french_results.json").read_text())["relation_metrics"],
    }
    (ART / "relation_hard_cases_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": payload["pairs"], "label_counts": payload["label_counts"],
                      "learned": {"accuracy": payload["learned"]["accuracy"], "macro_f1": payload["learned"]["macro_f1"]},
                      "structural": {"accuracy": payload["structural"]["accuracy"], "macro_f1": payload["structural"]["macro_f1"]}},
                     ensure_ascii=False, indent=2))
    print("\nSaved", ART / "relation_hard_cases_eval.json")


if __name__ == "__main__":
    main()
