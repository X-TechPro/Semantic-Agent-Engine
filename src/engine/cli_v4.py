"""Ask the v4 pipeline a question about one hand-written meeting.

Manual counterpart to `meetings_eval`: it prints the retrieval table, the
memory state, the answer, and the citations, so the pipeline can be inspected
rather than scored.

    python -m engine.cli_v4 data/mine/nova.meeting.json "Quel budget a été validé ?"
    python -m engine.cli_v4 data/mine/incident.meeting.json "Pourquoi la panne ?" --answer llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.answer.v4 import answer
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_rules import V4RuleLinker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store


def build(meeting_path: Path, use_rerank: bool, device: str):
    doc = json.loads(meeting_path.read_text(encoding="utf-8"))
    mid = doc.get("meeting_id") or meeting_path.name.split(".")[0]
    store = V4Store()
    ing = V4Ingestor(store, linker=V4RuleLinker())
    for m in doc["messages"]:
        ing.ingest(MessageRecord(
            message_id=m["message_id"], source_id=mid, project_id=mid,
            speaker=m["speaker"], timestamp=m["timestamp"], text=m["text"],
            source_type="meeting", origin="synthetic"))

    from engine.embed.hf_embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder("intfloat/multilingual-e5-base",
                                      device=device)

    def encoder(texts):
        return emb.embed(list(texts))

    reranker = None
    if use_rerank:
        from sentence_transformers import CrossEncoder

        ins = ("Étant donné une question sur une réunion, retrouve le tour qui "
               "énonce la valeur ou la décision qui répond")
        ce = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512,
                          device=device, prompts={"fr": ins},
                          default_prompt_name="fr")

        def reranker(query, texts):
            import torch

            with torch.no_grad():
                return ce.predict([(query, t) for t in texts], batch_size=16,
                                  show_progress_bar=False).tolist()

    return store, V4Search(store, linker=V4RuleLinker(), encoder=encoder,
                           reranker=reranker), mid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--answer", choices=("rules", "llm", "hybrid"),
                    default="hybrid")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    path = Path(args.meeting)
    if not path.exists():
        print(f"no such meeting file: {path}", file=sys.stderr)
        return 2
    store, search, mid = build(path, not args.no_rerank, args.device)
    hits = search.retrieve(args.question, k=args.k, project=mid)

    print(f"\n== {mid}: {args.question}\n")
    print(f"{'#':>3}  {'unit':<24} {'slot':<9} {'point':<14} {'status':<10} text")
    for h in hits:
        u = store.unit(h.unit_id)
        print(f"{h.rank:>3}  {u['unit_id']:<24} {u['slot']:<9} "
              f"{(u.get('point') or '-'):<14} {u['status']:<10} {u['text'][:60]}")

    if args.answer == "rules":
        ans = answer(args.question, hits, store=store)
    elif args.answer == "llm":
        from engine.answer.llm import answer_llm

        ans = answer_llm(args.question, hits, store=store, model=args.model)
    else:
        ans = answer(args.question, hits, store=store)
        if ans.unknown:
            from engine.answer.llm import answer_llm

            ans = answer_llm(args.question, hits, store=store, model=args.model)

    print("\n-- answer ----------------------------------------------------")
    print(ans.text)
    print(f"unknown={ans.unknown} verified={ans.verified} status={ans.status}")
    if ans.citations:
        print("\n-- proof -----------------------------------------------------")
        for c in ans.citations:
            u = store.unit(c.unit_id) or {}
            who = (store.message(u.get("message_id", "")) or {}).get("speaker")
            print(f"  [{c.unit_id}] {who or ''}: {c.text.strip()[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
