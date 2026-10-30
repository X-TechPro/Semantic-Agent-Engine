"""Does indexing the metadata help "when / who" questions?

The store keeps the date and the speaker next to each message, but the index
only ever saw the text. Three variants, same store, same questions:

  text      what ships today
  meta      "speaker · date · text"
  meta_id   "[message_id] speaker · date · text"

Scored on the 26 answerable questions plus ten meta questions written for this
probe (when was it decided, who decided it).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python experiments/meta_embed.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from engine.ingest_v4 import V4Ingestor
from engine.link.v4_rules import V4RuleLinker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store


def load_meetings(root: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(str(root / "data/mine/*.meeting.json"))):
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        out[doc["meeting_id"]] = doc
    return out


def build(meetings: dict[str, dict]) -> V4Store:
    store = V4Store()
    ing = V4Ingestor(store, linker=V4RuleLinker())
    for mid, doc in meetings.items():
        for m in doc["messages"]:
            ing.ingest(MessageRecord(
                message_id=m["message_id"], source_id=mid, project_id=mid,
                speaker=m["speaker"], timestamp=m["timestamp"], text=m["text"],
                source_type="meeting", origin="synthetic"))
    return store


ENC = None


def variant_texts(store: V4Store, mode: str):
    def fn(u):
        msg = store.message(u["message_id"]) or {}
        who = msg.get("speaker") or ""
        when = str(msg.get("timestamp") or "")[:10]
        head = " · ".join(x for x in ([f"[{u['unit_id']}]"] if mode == "meta_id" else [])
                          + [who, when] if x)
        return f"{head} · {u['text']}" if head else u["text"]
    return fn


def questions(root: Path) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(root / "data/mine/*.questions.json"))):
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        for q in doc["questions"]:
            if q.get("must_abstain"):
                continue
            out.append(q)
    return out


def score(hits, expected: list[str], k: int) -> bool:
    seen = [h.message_id for h in hits[:k]]
    return any(e in seen for e in expected)


def main() -> int:
    global ENC
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.root)

    from engine.embed.hf_embedder import SentenceTransformerEmbedder

    ENC = SentenceTransformerEmbedder("intfloat/multilingual-e5-base",
                                      device=args.device)
    store = build(load_meetings(root))
    qs = questions(root)
    meta = [q for q in qs if q["query_id"].startswith("meta-")]
    base = [q for q in qs if not q["query_id"].startswith("meta-")]
    print(f"{len(base)} answerable questions · {len(meta)} meta questions · "
          f"{len(store.all_units())} units\n")

    print(f"{'variant':<9} {'layer':<7} {'slice':<10} {'hit@1':>6} {'hit@3':>6} {'hit@10':>6}")
    for mode in ("text", "meta", "meta_id"):
        search = V4Search(store, encoder=lambda texts: ENC(list(texts)),
                          text_fn=variant_texts(store, mode))
        for name, qset in (("answerable", base), ("meta", meta)):
            for layer in ("dense", "pipeline"):
                h1 = h3 = h10 = 0
                for q in qset:
                    project = q["expected_message_ids"][0].split(":")[0]
                    if layer == "dense":
                        search._scope = project
                        ids = search._dense(q["text"])
                        mids = [(store.unit(u) or {}).get("message_id", "") for u in ids]
                        ok = lambda k: any(e in mids[:k] for e in q["expected_message_ids"])
                        h1 += ok(1); h3 += ok(3); h10 += ok(10)
                        continue
                    hits = search.retrieve(q["text"], k=10, project=project)
                    h1 += score(hits, q["expected_message_ids"], 1)
                    h3 += score(hits, q["expected_message_ids"], 3)
                    h10 += score(hits, q["expected_message_ids"], 10)
                n = len(qset)
                print(f"{mode:<9} {layer:<7} {name:<10} {h1/n:>6.3f} {h3/n:>6.3f} {h10/n:>6.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
