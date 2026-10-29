"""Measure each of the seven steps on the hand-written meetings.

Every step gets the metric that decides whether it is doing its job, not a
downstream score:

1. store      a write cannot be changed, a re-ingest is a no-op
2. chunk      every span indexes its own message text
3. label      does it flag the turns that change something, and does it store a
              value rather than a sentence
4. link       is the replaced turn among the candidates, and is it the one linked
5. memory     does the current state carry the new value, and from the new turn
6. search     is the answer's turn in the top-k
7. answer     cited, verified, and refused when unanswerable
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from engine.answer.v4 import answer
from engine.config import repo_root
from engine.embed.hf_embedder import SentenceTransformerEmbedder
from engine.eval.meetings_eval import load_meetings, load_questions
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_rules import V4RuleLinker
from engine.link.v4_linker import V4Linker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store
from engine.threads.state_v4 import event_key

STATE = {"proposed", "confirmed", "cancelled", "completed"}
SETTLED = {"confirmed", "cancelled", "completed"}
NUM = re.compile(r"\d[\d\s.,]*")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _has_value(value: str, gold: str) -> bool:
    return _flat(gold) in _flat(value)


def ingest(mid: str, messages: list[dict], linker=None) -> V4Store:
    store = V4Store()
    ing = V4Ingestor(store, linker=linker or V4RuleLinker())
    for m in messages:
        ing.ingest(MessageRecord(
            message_id=m["message_id"], source_id=mid, project_id=mid,
            speaker=m["speaker"], timestamp=m["timestamp"], text=m["text"],
            source_type="meeting", origin="synthetic"))
    return store


def step1_store(mid: str, messages: list[dict]) -> dict:
    store = ingest(mid, messages)
    before = len(store.all_units())
    # re-ingest the same messages: is it a no-op?
    reingest_ok, reingest_error = True, None
    ing = V4Ingestor(store)
    try:
        for m in messages:
            ing.ingest(MessageRecord(
                message_id=m["message_id"], source_id=mid, project_id=mid,
                speaker=m["speaker"], timestamp=m["timestamp"], text=m["text"],
                source_type="meeting", origin="synthetic"))
    except Exception as exc:
        reingest_ok, reingest_error = False, type(exc).__name__
    after = len(store.all_units())
    # an in-place edit must be refused by the append-only table
    refused = False
    try:
        store.db.execute("UPDATE messages SET text='x' WHERE message_id=?",
                        (messages[0]["message_id"],))
        store.db.commit()
    except Exception:
        refused = True
    text_after = store.message(messages[0]["message_id"])["text"]
    return {"duplicate_units": after - before, "update_refused": refused,
            "text_intact": text_after == messages[0]["text"],
            "reingest_ok": reingest_ok, "reingest_error": reingest_error}


def step2_chunk(store: V4Store, mid: str, messages: list[dict]) -> dict:
    bad = 0
    chunks = 0
    multi = 0
    for m in messages:
        units = store.units_for_message(m["message_id"])
        if len(units) > 1:
            multi += 1
        for u in units:
            for c in store.chunks_for_unit(u["unit_id"]):
                chunks += 1
                if m["text"][c["start"]:c["end"]] != c["text"]:
                    bad += 1
    return {"chunks": chunks, "offset_mismatches": bad,
            "messages": len(messages), "messages_split": multi}


def step3_label(store: V4Store, gold: dict) -> dict:
    hits = sum(1 for c in gold["changes"]
               if any(u.get("status") in STATE
                      for u in store.units_for_message(c["new"])))
    # a proposal is open by design: not a false positive on a non-change
    false_pos = sum(1 for m in gold["non_changes"]
                    if any(u.get("status") in SETTLED
                           for u in store.units_for_message(m)))
    # a unit "carries a state" only when it has a value, not when the reader
    # fell back to the clause
    valued = [u for u in store.all_units()
              if u.get("status") in STATE and u.get("value")
              and str(u["value"]).strip() != str(u["text"]).strip()]
    short = [u for u in valued if len(str(u["value"]).split()) <= 3]
    return {"change_recall": round(hits / max(1, len(gold["changes"])), 3),
            "nonchange_flagged": f"{false_pos}/{len(gold['non_changes'])}",
            "state_units": len(valued),
            "value_is_short": f"{len(short)}/{len(valued)}"}


def step4_link(store: V4Store, search: V4Search, mid: str,
               gold: dict, k: int = 10) -> dict:
    rows = []
    for c in gold["changes"]:
        olds = c.get("olds") or []
        if not olds:
            continue
        old_uids = {store.units_for_message(o)[0]["unit_id"] for o in olds
                    if store.units_for_message(o)}
        new_units = store.units_for_message(c["new"])
        if not old_uids or not new_units:
            continue
        new_uid = new_units[0]["unit_id"]
        hits = search.retrieve(store.unit(new_uid)["text"], k=k, project=mid,
                               mode="link")
        order = [h.unit_id for h in hits]
        ranks = [order.index(u) + 1 for u in old_uids if u in order]
        targets = {l["to_unit_id"] for l in store.links_of(new_uid, "supersedes")}
        rows.append({
            "new": c["new"], "old": olds,
            "candidate_rank": min(ranks) if ranks else None,
            "linked": bool(targets & old_uids),
        })
    if not rows:
        return {"pairs": 0}
    return {"pairs": len(rows),
            "candidate_hit": sum(r["candidate_rank"] is not None for r in rows),
            "linked": sum(r["linked"] for r in rows),
            "detail": rows}


def step5_memory(store: V4Store, mid: str, gold: dict) -> dict:
    right = 0
    checked = 0
    own = 0
    open_ok = 0
    open_checked = 0
    for c in gold["changes"]:
        if not c.get("value"):
            continue          # a decision without a tracked value is not a state
        units = store.units_for_message(c["new"])
        if not units:
            continue
        unit = units[0]
        slot = unit.get("slot") or c["slot"]
        point = unit.get("point") or ""
        if unit.get("status") == "proposed":
            # a proposal is open, not current: check it was recorded as open
            open_checked += 1
            rows = store.history(event_key(mid, slot, point), slot)
            if any(r["status"] == "proposed" and r["evidence_unit"] == unit["unit_id"]
                   and _has_value(r["value"], c["value"]) for r in rows):
                open_ok += 1
            continue
        cur = store.current_state(event_key(mid, slot, point), slot)
        checked += 1
        if cur and _has_value(cur.get("value", ""), c["value"]):
            right += 1
            if cur.get("evidence_unit") == unit["unit_id"]:
                own += 1
    leaked = 0
    for u in store.all_units():
        st = store.current_for_unit(u["unit_id"])
        if st and st.get("evidence_unit") != u["unit_id"]:
            leaked += 1
    return {"current_value_correct": f"{right}/{checked}",
            "current_evidence_is_new_turn": f"{own}/{checked}",
            "proposal_open": f"{open_ok}/{open_checked}",
            "units_carrying_another_units_state": leaked,
            "units": len(store.all_units())}


def step6_search(store, search, mid, questions) -> dict:
    rows = []
    for q in questions:
        expected = set(q["expected_message_ids"])
        hits = search.retrieve(q["text"], k=10, project=mid)
        msgs = [store.unit(h.unit_id)["message_id"] for h in hits]
        rows.append({"q": q["query_id"], "style": q["style"],
                     "hit1": bool(msgs[:1] and msgs[0] in expected),
                     "hit3": any(m in expected for m in msgs[:3]),
                     "hit10": any(m in expected for m in msgs[:10])})
    asked = [r for r, q in zip(rows, questions) if q["expected_message_ids"]]
    def rate(k):
        return round(sum(r[k] for r in asked) / len(asked), 3) if asked else 0.0
    return {"n": len(asked), "hit@1": rate("hit1"), "hit@3": rate("hit3"),
            "hit@10": rate("hit10")}


def step7_answer(store, search, mid, questions) -> dict:
    cited = value = refused = 0
    tricky = 0
    for q in questions:
        hits = search.retrieve(q["text"], k=10, project=mid)
        a = answer(q["text"], hits, store=store)
        expected = set(q["expected_message_ids"])
        if q["must_abstain"]:
            tricky += 1
            refused += int(a.unknown)
            continue
        cm = {store.unit(c.unit_id)["message_id"] for c in a.citations
              if store.unit(c.unit_id)}
        cited += int(bool(cm & expected))
        value += int(bool(q.get("expected_value") and not a.unknown
                          and _flat(q["expected_value"]) in _flat(a.text)))
    asked = len(questions) - tricky
    return {"answerable": asked,
            "citation_hit": round(cited / max(1, asked), 3),
            "value_hit": round(value / max(1, asked), 3),
            "tricky": tricky, "refused": f"{refused}/{tricky}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/meetings/steps.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--linker", default="models/flaubert_v4/fold_024c_EBPH")
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(sorted(root.glob("data/mine/*.meeting.json")))
    questions = load_questions(sorted(root.glob("data/mine/*.questions.json")))
    gold = json.loads((root / "data/mine/gold.json").read_text())["meetings"]

    emb = SentenceTransformerEmbedder("intfloat/multilingual-e5-base",
                                      device=args.device)
    linker = V4RuleLinker()
    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder

        instruction = ("Étant donné une question sur une réunion, retrouve le "
                       "tour qui énonce la valeur ou la décision qui répond")
        ce = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512,
                          device=args.device, prompts={"fr": instruction},
                          default_prompt_name="fr")

        def reranker(query, texts):
            import torch as _t

            with _t.no_grad():
                return ce.predict([(query, x) for x in texts], batch_size=16,
                                  show_progress_bar=False).tolist()

    out: dict = {"per_meeting": {}, "totals": {}}
    agg = {"s1": [], "s2": [], "s3": [], "s4": [], "s5": [], "s6": [], "s7": []}
    for mid, msgs in meetings.items():
        store = ingest(mid, msgs, linker=linker)
        search = V4Search(store, linker=linker,
                          encoder=lambda t: emb.embed(list(t)),
                          reranker=reranker)
        qs = questions.get(mid, [])
        r = {
            "1_store": step1_store(mid, msgs),
            "2_chunk": step2_chunk(store, mid, msgs),
            "3_label": step3_label(store, gold[mid]),
            "4_link": step4_link(store, search, mid, gold[mid]),
            "5_memory": step5_memory(store, mid, gold[mid]),
            "6_search": step6_search(store, search, mid, qs),
            "7_answer": step7_answer(store, search, mid, qs),
        }
        out["per_meeting"][mid] = r
        for key, bucket in (("1_store", "s1"), ("2_chunk", "s2"),
                            ("3_label", "s3"), ("4_link", "s4"),
                            ("5_memory", "s5"), ("6_search", "s6"),
                            ("7_answer", "s7")):
            agg[bucket].append(r)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    out["totals"] = {
        "1_store_immutable": all(m["1_store"]["update_refused"]
                                  and m["1_store"]["text_intact"] for m in agg["s1"]),
        "1_reingest_ok": all(m["1_store"]["reingest_ok"] for m in agg["s1"]),
        "1_duplicate_units": sum(m["1_store"]["duplicate_units"] for m in agg["s1"]),
        "2_offset_mismatches": sum(m["2_chunk"]["offset_mismatches"] for m in agg["s2"]),
        "3_change_recall": mean([m["3_label"]["change_recall"] for m in agg["s3"]]),
        "3_nonchanges_flagged": sum(int(m["3_label"]["nonchange_flagged"].split("/")[0])
                                    for m in agg["s3"]),
        "3_value_short": sum(int(m["3_label"]["value_is_short"].split("/")[0])
                             for m in agg["s3"]),
        "3_state_units": sum(m["3_label"]["state_units"] for m in agg["s3"]),
        "4_pairs": sum(m["4_link"].get("pairs", 0) for m in agg["s4"]),
        "4_candidate_hit": sum(m["4_link"].get("candidate_hit", 0) for m in agg["s4"]),
        "4_linked": sum(m["4_link"].get("linked", 0) for m in agg["s4"]),
        "5_correct": sum(int(m["5_memory"]["current_value_correct"].split("/")[0])
                         for m in agg["s5"]),
        "5_checked": sum(int(m["5_memory"]["current_value_correct"].split("/")[1])
                         for m in agg["s5"]),
        "5_proposals_open": sum(int(m["5_memory"]["proposal_open"].split("/")[0])
                                for m in agg["s5"]),
        "5_proposals": sum(int(m["5_memory"]["proposal_open"].split("/")[1])
                           for m in agg["s5"]),
        "5_leaked_units": sum(m["5_memory"]["units_carrying_another_units_state"]
                              for m in agg["s5"]),
        "6_hit@1": mean([m["6_search"]["hit@1"] for m in agg["s6"]]),
        "6_hit@3": mean([m["6_search"]["hit@3"] for m in agg["s6"]]),
        "6_hit@10": mean([m["6_search"]["hit@10"] for m in agg["s6"]]),
        "7_citation": mean([m["7_answer"]["citation_hit"] for m in agg["s7"]]),
        "7_refused": sum(int(m["7_answer"]["refused"].split("/")[0])
                         for m in agg["s7"]),
        "7_tricky": sum(m["7_answer"]["tricky"] for m in agg["s7"]),
    }
    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["totals"], indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
