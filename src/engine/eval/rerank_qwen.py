"""Step 6 with a real reranker: Qwen3-Reranker-0.6B on the pipeline's candidates.

The pipeline already returns the answering turn in the top 10 for ~89% of
questions, but first only ~36%. This measures whether a cross-encoder can order
those candidates better, before anything is wired.

Candidate lists are gathered first with the e5 embedder, the embedder is freed,
then the reranker scores (question, candidate) pairs — one model resident at a
time, as the 8 GB card requires.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from engine.config import repo_root
from engine.embed.hf_embedder import SentenceTransformerEmbedder
from engine.eval.meetings_eval import load_meetings, load_questions
from engine.eval.steps_eval import ingest
from engine.link.v4_rules import V4RuleLinker
from engine.query.v4_search import V4Search


def collect(meetings, questions, device: str, k: int) -> list[dict]:
    emb = SentenceTransformerEmbedder("intfloat/multilingual-e5-base",
                                      device=device)
    rows = []
    for mid, msgs in meetings.items():
        store = ingest(mid, msgs, linker=V4RuleLinker())
        search = V4Search(store, linker=V4RuleLinker(),
                          encoder=lambda t: emb.embed(list(t)))
        for q in questions.get(mid, []):
            hits = search.retrieve(q["text"], k=k, project=mid)
            rows.append({
                "query_id": q["query_id"], "meeting": mid, "style": q["style"],
                "text": q["text"], "must_abstain": q["must_abstain"],
                "expected": q["expected_message_ids"],
                "candidates": [{"uid": h.unit_id,
                                "mid": store.unit(h.unit_id)["message_id"],
                                "text": store.unit(h.unit_id)["text"]}
                               for h in hits],
            })
    del emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def ranks(row: dict, order: list[int], key: str = "mid") -> dict:
    ids = [row["candidates"][i][key] for i in order]
    expected = set(row["expected"])
    return {f"hit@{c}": any(m in expected for m in ids[:c]) for c in (1, 3, 5)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/meetings/rerank_qwen.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(sorted(root.glob("data/mine/*.meeting.json")))
    questions = load_questions(sorted(root.glob("data/mine/*.questions.json")))

    rows = collect(meetings, questions, args.device, args.k)

    from sentence_transformers import CrossEncoder

    INSTRUCTIONS = {
        "default": None,
        "meeting": ("Given a question about a meeting, retrieve the turn that "
                    "states the value or decision that answers it"),
        "french": ("Étant donné une question sur une réunion, retrouve le tour "
                   "qui énonce la valeur ou la décision qui répond"),
    }
    models = {}
    for name, ins in INSTRUCTIONS.items():
        kw = {}
        if ins:
            kw = {"prompts": {name: ins}, "default_prompt_name": name}
        models[name] = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512,
                                    device=args.device, **kw)

    variants = ["base"] + [f"rerank:{n}" for n in INSTRUCTIONS] + \
               [f"blend:{n}" for n in INSTRUCTIONS]
    summary = {name: {f"hit@{c}": 0 for c in (1, 3, 5)}
               for name in variants}
    summary["n"] = 0
    summary["by_style"] = {}
    detail = []
    for row in rows:
        if row["must_abstain"] or not row["candidates"]:
            continue
        pairs = [(row["text"], c["text"]) for c in row["candidates"]]
        n = len(pairs)
        results = {}
        for name, ce in models.items():
            with torch.no_grad():
                scores = ce.predict(pairs, batch_size=args.batch,
                                    show_progress_bar=False)
            order = list(torch.argsort(torch.as_tensor(scores),
                                       descending=True).tolist())
            results[f"rerank:{name}"] = ranks(row, order)
            base_rank = {i: i for i in range(n)}
            blend = sorted(range(n), key=lambda i: -(
                1.0 / (60 + base_rank[i] + 1) +
                1.0 / (60 + order.index(i) + 1)))
            results[f"blend:{name}"] = ranks(row, blend)
            results.setdefault("_tops", {})[name] = [
                row["candidates"][i]["mid"] for i in order[:5]]
        results["base"] = {f"hit@{c}": any(
            m in set(row["expected"]) for m in
            [cand["mid"] for cand in row["candidates"][:c]])
            for c in (1, 3, 5)}
        summary["n"] += 1
        for name in variants:
            if name == "base":
                continue
            for key in ("hit@1", "hit@3", "hit@5"):
                summary[name][key] += int(results[name][key])
        for key in results["base"]:
            summary["base"][key] += int(results["base"][key])
        style = summary["by_style"].setdefault(
            row["style"], {"n": 0, **{k: 0 for k in variants}})
        style["n"] += 1
        for name in variants:
            src = results["base"] if name == "base" else results[name]
            style[name] += int(src["hit@1"])
        detail.append({"query_id": row["query_id"], "style": row["style"],
                       "base": results["base"], "variants": results,
                       "expected": row["expected"]})

    n = max(1, summary["n"])
    for name in variants:
        summary[name] = {k: round(v / n, 3) for k, v in summary[name].items()}
    for style in summary["by_style"].values():
        for name in variants:
            style[name] = round(style[name] / max(1, style["n"]), 3)
    out = {"summary": summary, "detail": detail}
    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
