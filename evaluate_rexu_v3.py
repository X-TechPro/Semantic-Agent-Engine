"""Measure Rexu v3 on clean vs noise-injected corpora.

Noise axes: acronym formatting/injection, named-entity corruption, diarization
errors (see make_noisy_corpora.py). Gold ids are unchanged, so clean and noisy
runs are directly comparable.

Protocol matches the existing evaluations:
  * v8/v9 - candidates are the query's own meeting (60 messages);
  * v10   - candidates are all 275 messages across the 5 meetings.

Usage:
    python3 evaluate_rexu_v3.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pplx_rexu import RexuV3Retriever

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "noisy_eval"
SCOPED = {"v8", "v9"}  # v10 is cross-meeting, as in evaluate_v10.py


class _Eng:
    def __init__(self, ds: dict, device: str):
        self.device = device
        self._ds = ds

    def _get_dataset(self, slug: str) -> dict:
        return self._ds


def build(name: str, meeting_file: Path, query_file: Path, slug: str) -> dict:
    payload = json.loads(meeting_file.read_text(encoding="utf-8"))
    meetings = payload["meetings"] if isinstance(payload, dict) else payload
    qpayload = json.loads(query_file.read_text(encoding="utf-8"))
    queries = qpayload["queries"] if isinstance(qpayload, dict) else qpayload
    records, by_meeting = {}, {}
    for m in meetings:
        mid = m["meeting_id"]
        by_meeting[mid] = []
        for x in m["messages"]:
            cid = (x.get("id")
                   or f"{mid}:{int(x.get('message_id', 0)):03d}")
            by_meeting[mid].append(cid)
            records[cid] = {
                "chunk_id": cid,
                "meeting_id": mid,
                "encoder_text": (f"Réunion : {m.get('title', mid)}. "
                                 f"Intervenant : {x.get('speaker', 'Inconnu')}. "
                                 f"Intervention : {x.get('text', '')}"),
            }
    return {"slug": slug, "records": records, "by_meeting": by_meeting,
            "meetings": {m["meeting_id"]: m for m in meetings}, "queries": queries}


def evaluate(name: str, variant: str, device: str,
             noisy_dir: Path = ROOT / "data/noisy") -> dict:
    base = noisy_dir if variant == "noisy" else ROOT / "data"
    slug = f"{variant}_{name}" if variant == "clean" else f"{variant}_{name}_{noisy_dir.name}"
    ds = build(name, base / f"{name}_meetings.json",
               base / f"{name}_queries.json", slug)
    eng = _Eng(ds, device)
    t0 = time.perf_counter()
    retr = RexuV3Retriever(eng, ds["slug"])
    CACHE.mkdir(parents=True, exist_ok=True)
    retr._cache = CACHE / f"{ds['slug']}.pt"
    retr._ensure()
    build_s = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ranks, ev, allh, lat = [], [], [], []
    qs = [q for q in ds["queries"] if q.get("split") == "test"]
    all_ids = list(ds["records"])
    for q in qs:
        cand = (ds["by_meeting"].get(q.get("meeting_id"), [])
                if name in SCOPED else all_ids)
        t0 = time.perf_counter()
        scores = retr._score(q["query"], "rexu_v3", cand)
        lat.append(time.perf_counter() - t0)
        order = [cand[i] for i in np.argsort(-scores)]
        gold = set(q.get("target_chunk_ids") or q.get("evidence_chunk_ids") or [])
        pos = [order.index(c) + 1 for c in gold if c in order]
        r = min(pos) if pos else len(order) + 1
        ranks.append(r)
        found = set(order[:5]) & gold
        ev.append(len(found) / len(gold) if gold else 0.0)
        allh.append(1.0 if found == gold else 0.0)

    a = np.asarray(ranks)
    return {
        "corpus": name,
        "variant": variant,
        "n": len(qs),
        "R@1": float((a <= 1).mean()),
        "R@5": float((a <= 5).mean()),
        "MRR": float(np.where(a <= 10, 1 / a, 0).mean()),
        "EvidenceRecall@5": float(np.mean(ev)),
        "AllHops@5": float(np.mean(allh)),
        "resources": {
            "doc_build_s": round(build_s, 2),
            "query_ms_mean": round(1000 * statistics.mean(lat), 1),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if device == "cuda" else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/noisy_eval.json")
    ap.add_argument("--noisy-dir", type=Path, default=ROOT / "data/noisy")
    args = ap.parse_args()
    out = []
    for name in ("v8", "v9", "v10"):
        for variant in ("clean", "noisy"):
            r = evaluate(name, variant, args.device, args.noisy_dir)
            out.append(r)
            print(f"{name:3s} {variant:5s} n={r['n']:2d}  R@1 {r['R@1']:.3f}  "
                  f"R@5 {r['R@5']:.3f}  MRR {r['MRR']:.3f}  "
                  f"Ev@5 {r['EvidenceRecall@5']:.3f}  AllHops {r['AllHops@5']:.3f}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (args.noisy_dir / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
