"""Turn-level change ranker: text base + structured/7B base + meta, LOMO.

Not a shipped detector. Reads data/gold/summre_change_labels.json and the
per-turn 7B scores, and evaluates the requested gate: top-20 turns per meeting
must contain at least half of that meeting's changes.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from engine.config import repo_root
from engine.link.rules import _lex_overlap  # noqa: F401  (kept for parity)
from engine.link.thread import ThreadIndex, detect_topic
from engine.schemas import Span, Unit

ROOT = repo_root()
LABELS = ROOT / "data/gold/summre_change_labels.json"
SCORES = ROOT / "runs/llm_judge/turn_scores_fast_2026-10-27.json"
MEETINGS = ["024c_EBPH", "047a_EDRH", "060a_EDRH", "072c_EEPH",
            "016c_EBPZ", "047c_EDPH"]

CUES = {
    "finalement": r"\bfinalement\b", "du_coup": r"\bdu coup\b",
    "en_fait": r"\ben fait\b", "plutot": r"\bplutot\b",
    "on_part": r"\bon part", "on_reste": r"\bon reste\b",
    "on_prend": r"\bon prend\b", "on_choisit": r"\bon choisit\b",
    "on_valide": r"\bon valide\b", "on_annule": r"\bon annule\b",
    "on_repousse": r"\bon repousse\b", "on_change": r"\bon change\b",
    "ce_sera": r"\bce sera\b|\bca sera\b", "on_fait": r"\bon fait\b",
    "on_pourrait": r"\bon pourrait\b", "il_faudrait": r"\bil faudrait\b",
    "on_devrait": r"\bon devrait\b", "je_pense": r"\bje pense\b",
    "sinon": r"\bsinon\b", "par_contre": r"\bpar contre\b",
    "au_lieu": r"\bau lieu\b", "non_pas": r"\bnon pas\b",
    "c_est_acte": r"\bc est acte\b", "on_decale": r"\bon decale\b",
    "on_abandonne": r"\bon abandonne\b", "on_passe": r"\bon passe\b",
}
CUES = {k: re.compile(v) for k, v in CUES.items()}
NUM = re.compile(r"\d+(?:[.,]\d+)?")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.replace("_", " "))


def U(mid: str, i: int, text: str) -> Unit:
    return Unit(unit_id=f"{mid}#{i}", source_id=f"summre:{mid}",
                source_type="meeting", source_date=date(2021, 10, 5),
                ingested_at="2021-10-05T00:00:00Z", text=text,
                sentences=[Span.whole(text)], speaker=None,
                project=mid.split("_")[0], turn_index=i, kind="info")


@dataclass
class Row:
    mid: str
    i: int
    text: str
    label: int
    cues: list[int]
    cue_count: int
    thread_size: int
    gap: int
    topic_start: int
    topic_id: int
    n_words: int
    p7b: float
    has7b: int


def load_scores() -> dict[tuple[str, int], float]:
    if not SCORES.exists():
        return {}
    out = {}
    for r in json.loads(SCORES.read_text()):
        if r.get("change") is None:
            continue
        out[(r["mid"], r["i"])] = (r["confidence"] if r["change"]
                                   else 1.0 - r["confidence"])
    return out


def load_rows() -> list[Row]:
    labels = json.loads(LABELS.read_text())["meetings"]
    raw = json.loads((ROOT / "data/external/summre_test/meetings.json").read_text())
    scores = load_scores()
    # stable topic ids over the six meetings
    seen: dict[str, int] = {}
    rows = []
    for mid in MEETINGS:
        turns = raw[mid]
        y = labels[mid]["labels"]
        index = ThreadIndex(window=8, topic_window=12)
        for i, turn in enumerate(turns):
            text = turn["text"]
            u = U(mid, i, text)
            mates = [x for x in index.context(u) if int(x.turn_index) < i]
            gap = i - max((int(x.turn_index) for x in mates), default=i)
            topic = index.topic_of(u) or "none"
            if topic not in seen:
                seen[topic] = len(seen)
            f = fold(text)
            hits = [1 if p.search(f) else 0 for p in CUES.values()]
            p7b = scores.get((mid, i))
            rows.append(Row(
                mid=mid, i=i, text=f, label=int(y[i]), cues=hits,
                cue_count=sum(hits), thread_size=len(mates), gap=gap,
                topic_start=1 if detect_topic(text) else 0,
                topic_id=seen[topic], n_words=len(WORD.findall(f)),
                p7b=0.5 if p7b is None else float(p7b),
                has7b=0 if p7b is None else 1))
            index.add(u)
    return rows


def numeric_vec(r: Row) -> list[float]:
    return r.cues + [r.cue_count, r.thread_size, r.gap, r.topic_start,
                     r.topic_id, r.n_words, r.p7b, r.has7b, float(r.i)]


def fit_base(train: list[Row], test: list[Row]):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=8000)
    A = vec.fit_transform([r.text for r in train])
    B = vec.transform([r.text for r in test])
    y = np.asarray([r.label for r in train])
    text = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                              solver="liblinear").fit(A, y)
    scaler = StandardScaler().fit(np.asarray([numeric_vec(r) for r in train]))
    Xn = scaler.transform(np.asarray([numeric_vec(r) for r in train]))
    struct = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000,
                                solver="liblinear").fit(Xn, y)
    return {
        "vec": vec, "text": text, "scaler": scaler, "struct": struct,
        "pt": text.predict_proba(B)[:, 1],
        "ps": struct.predict_proba(scaler.transform(
            np.asarray([numeric_vec(r) for r in test])))[:, 1],
    }


def meta_features(pt, ps, rows: list[Row]):
    return np.asarray([[pt[i], ps[i], r.p7b, r.has7b, r.cue_count,
                        r.thread_size, r.gap, r.topic_start, r.n_words]
                       for i, r in enumerate(rows)])


def run() -> dict:
    rows = load_rows()
    by_mid = {m: [r for r in rows if r.mid == m] for m in MEETINGS}
    result = {"model": "text-base + struct-base + meta (LOMO)",
              "labels": str(LABELS), "scores": str(SCORES), "folds": []}
    pooled = []
    top20_hits = top20_gold = 0
    for held in MEETINGS:
        train_mids = [m for m in MEETINGS if m != held]
        train = [r for m in train_mids for r in by_mid[m]]
        test = by_mid[held]
        y = np.asarray([r.label for r in train])
        # inner OOF for the meta learner
        oof_pt, oof_ps = [], []
        for inner in train_mids:
            tr = [r for m in train_mids if m != inner for r in by_mid[m]]
            te = by_mid[inner]
            base = fit_base(tr, te)
            oof_pt.extend(base["pt"]); oof_ps.extend(base["ps"])
        # the OOF arrays are in inner-meeting order; rebuild the matching rows
        oof_rows = [r for inner in train_mids for r in by_mid[inner]]
        meta = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear")
        meta.fit(meta_features(np.asarray(oof_pt), np.asarray(oof_ps), oof_rows), y)
        base = fit_base(train, test)
        p = meta.predict_proba(meta_features(base["pt"], base["ps"], test))[:, 1]
        yy = np.asarray([r.label for r in test])
        ap = average_precision_score(yy, p)
        auc = roc_auc_score(yy, p)
        order = np.argsort(-p)[:20]
        hit = int(yy[order].sum())
        result["folds"].append({"held": held, "n": len(test),
                                "positives": int(yy.sum()), "ap": round(ap, 4),
                                "auc": round(auc, 4), "top20_hits": hit,
                                "top20_recall": round(hit / max(1, int(yy.sum())), 4)})
        top20_hits += hit; top20_gold += int(yy.sum())
        pooled.extend(zip(p.tolist(), yy.tolist()))
    p = [a for a, _ in pooled]; y = [b for _, b in pooled]
    result["pooled_ap"] = round(average_precision_score(y, p), 4)
    result["pooled_auc"] = round(roc_auc_score(y, p), 4)
    result["top20_hits"] = top20_hits
    result["top20_gold"] = top20_gold
    result["top20_recall"] = round(top20_hits / top20_gold, 4)
    result["gate_top20_ge_0.50"] = bool(top20_hits / top20_gold >= 0.50)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/change_ranker/result.json"))
    args = ap.parse_args(argv)
    out = run()
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
