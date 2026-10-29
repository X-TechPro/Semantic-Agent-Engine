"""Run the v4 pipeline on hand-written meetings and score the questions.

Each meeting is ingested on its own (one project = one meeting), the questions
are answered through the shipped `V4Search` → `answer` path, and every question
is scored three ways:

* retrieval: does the top-k contain a message that states the expected value;
* citation: does the answer quote one of those messages;
* refusal: does a tricky question get "I don't have enough information".

The citation set is the primary score, because the same value can be stated in
more than one turn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.answer.v4 import answer
from engine.config import repo_root
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_rules import V4RuleLinker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store


def load_meetings(paths: list[Path]) -> dict[str, list[dict]]:
    out = {}
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        mid = doc.get("meeting_id") or p.name.split(".")[0]
        out[mid] = doc["messages"]
    return out


def load_questions(paths: list[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        for q in doc["questions"]:
            out.setdefault(q.get("meeting_id") or "", []).append(q)
    return out


def ingest(mid: str, messages: list[dict], tag_fn=None,
           long_split: bool = False) -> V4Store:
    store = V4Store()
    ing = V4Ingestor(store, linker=V4RuleLinker(), tag_fn=tag_fn,
                     long_split=long_split)
    for m in messages:
        ing.ingest(MessageRecord(
            message_id=m["message_id"], source_id=mid, project_id=mid,
            speaker=m["speaker"], timestamp=m["timestamp"], text=m["text"],
            source_type="meeting", origin="synthetic"))
    return store


def evaluate(meetings, questions, encoder, reranker=None, answer_fn=None) -> dict:
    per_question = []
    for mid, msgs in meetings.items():
        store = ingest(mid, msgs)
        search = V4Search(store, encoder=encoder, reranker=reranker)
        for q in questions.get(mid, []):
            hits = search.retrieve(q["text"], k=10, project=mid)
            msg_of = [store.unit(h.unit_id)["message_id"] for h in hits]
            expected = set(q["expected_message_ids"])
            ans = (answer_fn(q["text"], hits, store=store) if answer_fn
                   else answer(q["text"], hits, store=store))
            cited = {store.unit(c.unit_id)["message_id"]
                     for c in ans.citations if store.unit(c.unit_id)}
            row = {
                "query_id": q["query_id"], "meeting": mid, "style": q["style"],
                "text": q["text"], "must_abstain": q["must_abstain"],
                "hit1": msg_of[:1] and msg_of[0] in expected,
                "hit3": any(m in expected for m in msg_of[:3]),
                "hit10": any(m in expected for m in msg_of[:10]),
                "citation_hit": bool(cited & expected),
                "abstained": bool(ans.unknown),
                "answer": ans.text,
                "cited_messages": sorted(cited),
                "expected": sorted(expected),
                "rank": next((i + 1 for i, m in enumerate(msg_of)
                              if m in expected), None),
            }
            row["value_hit"] = bool(
                q.get("expected_value") and not ans.unknown
                and str(q["expected_value"]).lower() in ans.text.lower())
            row["refusal_ok"] = row["abstained"] == q["must_abstain"]
            if q["must_abstain"]:
                row["correct"] = row["refusal_ok"]
            else:
                # a refusal that happens to cite the right turn is not an answer
                row["correct"] = (not row["abstained"]) and (
                    row["citation_hit"] or row["value_hit"])
            per_question.append(row)
    return {"questions": per_question, "summary": summarize(per_question)}


def _rate(rows, key) -> float:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 3) if rows else 0.0


def summarize(rows: list[dict]) -> dict:
    asked = [r for r in rows if not r["must_abstain"]]
    tricky = [r for r in rows if r["must_abstain"]]
    by_style: dict[str, dict] = {}
    for style in sorted({r["style"] for r in rows}):
        sub = [r for r in rows if r["style"] == style]
        by_style[style] = {
            "n": len(sub), "correct": _rate(sub, "correct"),
            "hit@1": _rate([r for r in sub if not r["must_abstain"]], "hit1"),
            "hit@10": _rate([r for r in sub if not r["must_abstain"]], "hit10"),
            "citation_hit": _rate([r for r in sub if not r["must_abstain"]],
                                  "citation_hit"),
        }
    return {
        "n": len(rows),
        "correct": _rate(rows, "correct"),
        "answerable": {
            "n": len(asked),
            "hit@1": _rate(asked, "hit1"),
            "hit@3": _rate(asked, "hit3"),
            "hit@10": _rate(asked, "hit10"),
            "citation_hit": _rate(asked, "citation_hit"),
            "value_hit": _rate(asked, "value_hit"),
        },
        "tricky": {"n": len(tricky), "refused": _rate(tricky, "refusal_ok")},
        "by_style": by_style,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default="data/mine/*.meeting.json")
    ap.add_argument("--questions", default="data/mine/*.questions.json")
    ap.add_argument("--out", default="runs/meetings/eval.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--answer", choices=("rules", "llm", "hybrid"),
                    default="hybrid")
    ap.add_argument("--answer-model", default="qwen2.5:7b-instruct")
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(sorted(root.glob(args.meetings)))
    questions = load_questions(sorted(root.glob(args.questions)))

    from engine.embed.hf_embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(args.model, device=args.device)

    def encoder(texts):
        return emb.embed(list(texts))

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder

        instruction = ("Étant donné une question sur une réunion, retrouve le "
                       "tour qui énonce la valeur ou la décision qui répond")
        ce = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512,
                          device=args.device,
                          prompts={"fr": instruction},
                          default_prompt_name="fr")

        def reranker(query, texts):
            import torch

            with torch.no_grad():
                return ce.predict([(query, t) for t in texts], batch_size=16,
                                  show_progress_bar=False).tolist()

    answer_fn = None
    if args.answer == "llm":
        import functools

        from engine.answer.llm import answer_llm

        answer_fn = functools.partial(answer_llm, model=args.answer_model)
    elif args.answer == "hybrid":
        import functools

        from engine.answer.llm import answer_llm

        ask_llm = functools.partial(answer_llm, model=args.answer_model)

        def answer_fn(text, hits, store=None):
            # the deterministic state answer when it has proof; the model only
            # where the rules abstain
            a = answer(text, hits, store=store)
            return a if not a.unknown else ask_llm(text, hits, store=store)

    result = evaluate(meetings, questions, encoder, reranker, answer_fn)
    result["answerer"] = args.answer
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    s = result["summary"]
    print(f"questions {s['n']}  correct {s['correct']}")
    print(f"answerable {s['answerable']['n']}  hit@1 {s['answerable']['hit@1']}"
          f"  hit@3 {s['answerable']['hit@3']}  hit@10 {s['answerable']['hit@10']}"
          f"  citation {s['answerable']['citation_hit']}"
          f"  value {s['answerable']['value_hit']}")
    print(f"tricky {s['tricky']['n']}  refused {s['tricky']['refused']}")
    for style, v in s["by_style"].items():
        print(f"  {style:8} n={v['n']:2}  correct {v['correct']:.3f}"
              f"  hit@1 {v['hit@1']:.3f}  hit@10 {v['hit@10']:.3f}"
              f"  citation {v['citation_hit']:.3f}")
    missed = [r for r in result["questions"] if not r["correct"]]
    if missed:
        print("\nmissed:")
        for r in missed:
            print(f"  {r['query_id']:14} [{r['style']:7}] rank={r['rank']} "
                  f"abstained={r['abstained']} :: {r['answer'][:90]}")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
