"""A/B: jina-embeddings-v3 vs the current e5, on the steps where an embedding belongs.

Step 4 (link): given the new turn, does cosine prefer the true antecedent over
the same-point/different-value decoy and over the other-thread decoy? Two
conditions: the sentence alone, and the sentence inside its last-3-turn window.

Step 6 (search): with the whole meeting as the pool, where does the true
antecedent rank?

Step 3 (zero-shot only): can label-description similarity classify kind/status
without any fine-tuning? This is the honest "should we ditch Laya" probe; a
zero-shot embedding classifier is not expected to beat the trained head, and the
number says whether it is even close.

Both embedders are run in this process on the same texts, so the comparison does
not inherit a prefix or pooling difference from somewhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine.config import repo_root  # noqa: E402
from engine.eval.mlm_link import auc, decoys  # noqa: E402
from engine.eval.real_pairs import load_meetings  # noqa: E402
from engine.eval.v4_data import citation_set, load_edges, texts_of  # noqa: E402

JINA = "jinaai/jina-embeddings-v3"
JINA_V5 = "jinaai/jina-embeddings-v5-text-small"
E5 = "intfloat/multilingual-e5-base"


class Jina:
    name = "jina-v3"

    def __init__(self, device: str = "cuda", task: str = "text-matching"):
        from transformers import AutoModel

        self.task = task
        self.device = device
        # no torch_dtype: the remote config does hasattr(torch, torch_dtype)
        # and only works when that is a string
        self.m = AutoModel.from_pretrained(
            JINA, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def emb(self, texts: list[str], task: str | None = None) -> np.ndarray:
        # an empty ASR turn tokenizes to zero tokens and their LoRA matmul
        # raises on the empty dimension rather than returning a vector
        texts = [t if t and t.strip() else "." for t in texts]
        out = self.m.encode(list(texts), task=task or self.task,
                            max_length=256, batch_size=32)
        if torch.is_tensor(out):
            out = out.float().cpu().numpy()
        return np.asarray(out, dtype=np.float32)


class V5:
    """jina-embeddings-v5-text-small: Qwen3-0.6B, last-token pooling, 32k ctx."""

    name = "jina-v5-small"

    def __init__(self, device: str = "cuda", task: str = "text-matching"):
        from transformers import AutoModel

        self.task = task
        self.m = AutoModel.from_pretrained(
            JINA_V5, trust_remote_code=True, dtype=torch.bfloat16
        ).to(device).eval()

    @torch.no_grad()
    def emb(self, texts: list[str], task: str | None = None,
            chunk: int = 32) -> np.ndarray:
        # v5's encode() has no batch_size: it runs the whole list in one
        # forward, so a meeting-sized call allocates the entire pool at once
        texts = [t if t and t.strip() else "." for t in texts]
        parts = []
        for i in range(0, len(texts), chunk):
            out = self.m.encode(texts=texts[i:i + chunk], task=task or self.task)
            parts.append(np.asarray(torch.as_tensor(out).float().cpu(),
                                    dtype=np.float32))
        return np.concatenate(parts, axis=0)


class E5Model:
    name = "e5-base"

    def __init__(self, device: str = "cuda", role: str = "passage"):
        from sentence_transformers import SentenceTransformer

        self.role = role
        self.m = SentenceTransformer(E5, device=device)

    def emb(self, texts: list[str], task: str | None = None) -> np.ndarray:
        pref = "query: " if self.role == "query" else "passage: "
        texts = [t if t and t.strip() else "." for t in texts]
        return np.asarray(self.m.encode([pref + t for t in texts],
                                        normalize_embeddings=True,
                                        batch_size=64), dtype=np.float32)


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


def _window(texts: list[str], i: int, k: int = 3) -> str:
    return " ".join(t for t in texts[max(0, i - k):i + 1] if t.strip())


def pair_set(root: Path):
    meetings = load_meetings(root)
    edges = load_edges(root)
    rows = []
    for mid in sorted(edges):
        texts = texts_of(meetings, mid)
        gold_olds: dict[int, set[int]] = {}
        for e in edges[mid]:
            gold_olds.setdefault(e.new, set()).add(e.old)
        for e in edges[mid]:
            pick = decoys(texts, e.old, e.new, gold_olds.get(e.new))
            if not pick:
                continue
            rows.append({"meeting": mid, "old": e.old, "new": e.new, **pick})
    return meetings, rows


def eval_pairs(model, meetings: dict, rows: list[dict], *, window: int = 0,
               task: str | None = None) -> dict:
    """AUC over gold-vs-decoy, conditioned on the new turn."""
    cache: dict[int, str] = {}
    texts_by: dict[str, list[str]] = {}
    for mid in {r["meeting"] for r in rows}:
        texts_by[mid] = texts_of(meetings, mid)

    def text(mid: str, i: int, cache_key: int) -> str:
        if cache_key not in cache:
            t = texts_by[mid]
            cache[cache_key] = _window(t, i, window) if window else t[i]
        return cache[cache_key]

    # unique-keyed texts, so the same turn is embedded once per meeting
    def key(mid: str, i: int) -> int:
        return hash((mid, i))

    def build(local_rows):
        todo, ids = [], []
        for r in local_rows:
            for i in [r["new"], r["old"]] + ([r["slot"]] if "slot" in r else []) \
                    + ([r["topic"]] if "topic" in r else []):
                kk = key(r["meeting"], i)
                if kk not in ids:
                    ids.append(kk)
                    todo.append(text(r["meeting"], i, kk))
        return ids, todo

    local = [{"meeting": r["meeting"], "new": r["new"], "old": r["old"],
              **({"slot": r["slot"]} if "slot" in r else {}),
              **({"topic": r["topic"]} if "topic" in r else {})} for r in rows]
    ids, todo = build(local)
    embs = _norm(model.emb(todo, task=task))
    pos = {kk: embs[j] for j, kk in enumerate(ids)}

    def cos(mid, a, b):
        return float(pos[key(mid, a)] @ pos[key(mid, b)])

    out = {}
    for decoy in ("slot", "topic"):
        sub = [r for r in local if decoy in r]
        if not sub:
            continue
        gold = [cos(r["meeting"], r["new"], r["old"]) for r in sub]
        neg = [cos(r["meeting"], r["new"], r[decoy]) for r in sub]
        out[decoy] = {"n": len(sub),
                      "gold": round(float(np.mean(gold)), 4),
                      "decoy": round(float(np.mean(neg)), 4),
                      "auc": round(auc(gold, neg), 4)}
    return out


def eval_retrieval(model, meetings: dict, edges: dict, *, k: int = 20) -> dict:
    hits = {5: 0, 10: 0, 20: 0}
    total = 0
    for mid in sorted(edges):
        texts = texts_of(meetings, mid)
        embs = _norm(model.emb([t if t.strip() else "." for t in texts]))
        for e in edges[mid]:
            pool = [i for i in range(min(e.new, len(texts))) if texts[i].strip()]
            if not pool:
                continue
            sims = embs[pool] @ embs[e.new]
            order = [pool[j] for j in np.argsort(-sims)]
            valid = citation_set(meetings, mid, e.old, e.new)
            total += 1
            for cut in hits:
                if any(i in valid for i in order[:cut]):
                    hits[cut] += 1
    return {"n": total, **{f"recall@{c}": f"{h}/{total}" for c, h in hits.items()}}


def classify_zero_shot(model, texts: list[str], labels: dict[str, str],
                       task: str = "classification") -> np.ndarray:
    lab = _norm(model.emb(list(labels.values()), task=task))
    emb = _norm(model.emb(texts, task=task))
    return emb @ lab.T


def _prf(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))
    return {"macro_f1": round(float(np.mean(f1s)), 3), "accuracy": round(acc, 3)}


def step3_zero_shot(model, root: Path) -> dict:
    from engine.read.laya_tags import TASKS

    out = {}
    test = [json.loads(l) for l in
            (root / "data/synth/tags/test.jsonl").read_text().splitlines() if l.strip()]
    for task_name in ("kind", "status"):
        rows = [r for r in test if r.get(task_name) and task_name in r.get("tasks", ())]
        if not rows:
            continue
        labels = TASKS[task_name]
        probs = classify_zero_shot(model, [r["text"] for r in rows],
                                   labels["options"])
        pred = [list(labels["options"])[int(j)] for j in probs.argmax(1)]
        out[task_name] = _prf([r[task_name] for r in rows], pred,
                              list(labels["options"]))
        out[task_name]["n"] = len(rows)

    # real: is the turn a state-bearing one, judged by label similarity
    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())
    meetings = load_meetings(root)
    labels = TASKS["status"]
    names = list(labels["options"])
    state = [n for n in names if n != "unknown"]
    scores, ys = [], []
    for mid, doc in gold["meetings"].items():
        if mid not in meetings:
            continue
        texts = texts_of(meetings, mid)
        ys_ = doc["labels"][:len(texts)]
        texts = texts[:len(ys_)]
        sims = classify_zero_shot(model, texts, labels["options"])
        idx = {n: i for i, n in enumerate(names)}
        for row in sims:
            scores.append(float(max(row[idx[s]] for s in state) - row[idx["unknown"]]))
        ys += list(ys_)
    out["real_state_auc"] = round(auc([s for s, y in zip(scores, ys) if y == 1],
                                      [s for s, y in zip(scores, ys) if y == 0]), 3)
    out["real_n"] = len(ys)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v4/jina_ab.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args(argv)
    root = repo_root()
    meetings, rows = pair_set(root)
    edges = load_edges(root)

    import gc

    def free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # one model resident at a time. Holding two was measured to fail: the
    # second forward errors inside the activation / LoRA matmul, not with OOM.
    results = {}
    plan = [
        ("jina-v3", lambda: Jina(args.device, task="text-matching")),
        ("e5-base", lambda: E5Model(args.device)),
        ("jina-v5-small", lambda: V5(args.device, task="text-matching")),
        ("jina-v5-small::classification", lambda: V5(args.device, task="classification")),
        ("jina-v5-small::retrieval", lambda: V5(args.device, task="retrieval")),
    ]
    for name, make in plan:
        model = make()
        entry = {}
        entry["pairs_sentence"] = eval_pairs(model, meetings, rows, window=0)
        entry["pairs_window3"] = eval_pairs(model, meetings, rows, window=3)
        if not args.skip_retrieval:
            entry["retrieval"] = eval_retrieval(model, meetings, edges)
        entry["step3_zero_shot"] = step3_zero_shot(model, root)
        results[name] = entry
        print(json.dumps({name: entry}, indent=1), flush=True)
        del model
        free()

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
