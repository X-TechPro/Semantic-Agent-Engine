"""Step 7, layer by layer: where each answerable question stops.

Layers, in the order the answer path uses them:

1. parse    — the question's slot and point;
2. retrieve — is the expected turn in the top 10;
3. state    — does that turn carry a state;
4. match    — does the state pass slot, point, value, support, completion;
5. select   — is the matching turn the one chosen, or is a wrong one earlier;
6. direct   — if no state, can the turn still be quoted as an answer;
7. answer   — cited, verified, refused.

Usage: python -m engine.eval.step7_debug
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from engine.answer.v4 import (_COMPLETION_Q, _answers, _query_slot, _supported,
                              answer)
from engine.config import repo_root
from engine.embed.hf_embedder import SentenceTransformerEmbedder
from engine.eval.meetings_eval import load_meetings, load_questions
from engine.eval.steps_eval import ingest
from engine.link.v4_rules import V4RuleLinker
from engine.query.v4_search import V4Search
from engine.read.v4_tags import point_of


def _why(query: str, query_slot: str, hit, state: dict) -> str:
    """The first rule that rejects a state, in the same order as `_answers`."""
    if not state or not str(state.get("value") or "").strip():
        return "no_state"
    slot = str(state.get("slot") or "")
    if query_slot and slot and slot != query_slot:
        return "slot"
    value = str(state["value"]).lower()
    ev = ((hit.evidence or {}).get("text") if isinstance(hit.evidence, dict)
          else "") or ""
    text = (ev + " " + (getattr(hit, "text", "") or "")).lower()
    if value not in text:
        return "value_not_in_evidence"
    q_point = point_of(query)
    eid = str(state.get("event_id") or "")
    s_point = eid.split(":", 2)[2] if eid.count(":") >= 2 else ""
    if q_point:
        if s_point and s_point != q_point:
            return "point_mismatch"
        if not s_point and q_point not in text:
            if _COMPLETION_Q.search(query):
                return "completion_point"
            if not (query_slot and slot and slot == query_slot):
                return "point_unknown_slot_mismatch"
            return "ok_fallback"
        return "ok"
    return "ok" if _supported(query, hit) else "no_support"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/meetings/step7.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerank", action="store_true", default=True)
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(sorted(root.glob("data/mine/*.meeting.json")))
    questions = load_questions(sorted(root.glob("data/mine/*.questions.json")))

    emb = SentenceTransformerEmbedder("intfloat/multilingual-e5-base",
                                      device=args.device)
    from sentence_transformers import CrossEncoder

    ins = ("Étant donné une question sur une réunion, retrouve le tour qui "
           "énonce la valeur ou la décision qui répond")
    ce = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512,
                      device=args.device, prompts={"fr": ins},
                      default_prompt_name="fr")

    def reranker(query, texts):
        with torch.no_grad():
            return ce.predict([(query, t) for t in texts], batch_size=16,
                              show_progress_bar=False).tolist()

    rows = []
    for mid, msgs in meetings.items():
        store = ingest(mid, msgs, linker=V4RuleLinker())
        search = V4Search(store, linker=V4RuleLinker(),
                          encoder=lambda t: emb.embed(list(t)),
                          reranker=reranker)
        for q in questions.get(mid, []):
            if q["must_abstain"]:
                continue
            hits = search.retrieve(q["text"], k=10, project=mid)
            expected = set(q["expected_message_ids"])
            exp_hits = [h for h in hits
                        if store.unit(h.unit_id)["message_id"] in expected]
            reasons = {}
            for h in hits:
                u = store.unit(h.unit_id)
                key = "expected" if u["message_id"] in expected else "other"
                reasons.setdefault(key, []).append(
                    (h.rank, u["unit_id"], _why(q["text"], _query_slot(q["text"]),
                                                h, h.state) if h.state else "no_state",
                     u["kind"], u["point"], u["slot"], u["value"][:12]))
            chosen = answer(q["text"], hits, store=store)
            cited = {store.unit(c.unit_id)["message_id"] for c in chosen.citations
                     if store.unit(c.unit_id)}
            exp_state = [h for h in exp_hits if h.state]
            strict = [h for h in exp_state
                      if _why(q["text"], _query_slot(q["text"]), h, h.state) == "ok"]
            fallback = [h for h in exp_state
                        if _why(q["text"], _query_slot(q["text"]), h, h.state)
                        in ("ok", "ok_fallback")]
            rows.append({
                "query_id": q["query_id"], "style": q["style"],
                "slot": _query_slot(q["text"]), "point": point_of(q["text"]),
                "expected": sorted(expected),
                "retrieved": [h.rank for h in exp_hits],
                "expected_has_state": bool(exp_state),
                "expected_state_reasons": [
                    _why(q["text"], _query_slot(q["text"]), h, h.state)
                    for h in exp_state],
                "strict_ok": bool(strict), "fallback_ok": bool(fallback),
                "expected_kinds": [store.unit(h.unit_id)["kind"] for h in exp_hits],
                "chosen_cited": sorted(cited), "answered": not chosen.unknown,
                "correct": bool(cited & expected),
                "reasons": reasons,
            })

    funnel = {
        "answerable": len(rows),
        "retrieved_top10": sum(bool(r["retrieved"]) for r in rows),
        "expected_has_state": sum(r["expected_has_state"] for r in rows),
        "expected_state_passes_strict": sum(r["strict_ok"] for r in rows),
        "expected_state_passes_fallback": sum(r["fallback_ok"] for r in rows),
        "answered": sum(r["answered"] for r in rows),
        "cites_expected": sum(r["correct"] for r in rows),
    }
    blocks = Counter()
    for r in rows:
        if not r["retrieved"]:
            blocks["1_not_retrieved"] += 1
        elif not r["expected_has_state"]:
            blocks["2_no_state_on_expected"] += 1
        elif not r["fallback_ok"]:
            blocks["3_state_rejected:" + ",".join(sorted(set(r["expected_state_reasons"])))] += 1
        elif not r["correct"]:
            blocks["4_beaten_by_another_turn"] += 1
    out = {"funnel": funnel, "blocks": dict(blocks), "rows": rows}
    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(funnel, indent=1))
    print(json.dumps(blocks, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
