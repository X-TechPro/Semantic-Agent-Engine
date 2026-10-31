#!/usr/bin/env python3
"""Evaluate the v6 agent's deterministic decisions and one LLM smoke test.

Scores intent, clarification/abstention, recipient resolution, dossier sections
and citation coverage, action safety, latency, and the local Ollama call.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

import memory_engine as me
from agent_engine import AgentEngine

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def dossier_citation_coverage(action: dict) -> float:
    if not action or not action.get("sections"):
        return 0.0
    rows = [row for section, items in action["sections"].items() if section != "references"
            for row in items if isinstance(row, dict)]
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("chunk_id")) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--llm-smoke", action="store_true", default=True)
    args = ap.parse_args()

    engine = me.MeetingMemoryEngine(device=args.device, batch_size=args.batch_size)
    agent = AgentEngine(engine)
    cases = json.loads((DATA / "agent_eval.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = agent.run(case["query"], slug=case.get("dataset", "transcribed"),
                           meeting_id=case.get("meeting_id"), top_k=8, use_llm=False)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        action = result.get("action") or {}
        sections = action.get("sections") or {}
        required = case.get("required_sections", [])
        coverage = (sum(1 for name in required if sections.get(name)) / len(required)) if required else None
        rows.append({
            "query_id": case["query_id"], "query": case["query"],
            "expected_intent": case["expected_intent"], "predicted_intent": result["intent"],
            "intent_ok": result["intent"] == case["expected_intent"],
            "action": action.get("intent"), "action_ok": (
                (action.get("intent") is None) if case.get("expect_action") == "none"
                else (action.get("intent") == case.get("expect_action"))) if case.get("expect_action") else True,
            "recipient": (action.get("recipient") or {}).get("name"),
            "recipient_ok": (action.get("recipient") or {}).get("name") == case.get("expected_recipient")
            if case.get("expected_recipient") else None,
            "email_missing": (action.get("recipient") or {}).get("email") is None
            if case.get("expect_email_missing") else None,
            "clarify": result["intent"] == "clarify",
            "required_section_coverage": coverage,
            "citation_coverage": dossier_citation_coverage(action) if sections else None,
            "timeline_length": len(action.get("timeline", [])) if action.get("timeline") is not None else None,
            "citations": len(result.get("citations", [])),
            "evidence": len(result.get("evidence", [])),
            "executed": bool(action.get("executed", False)),
            "missing_information": action.get("missing_information"),
            "latency_ms": elapsed,
            "retrieval_ms": next((s["detail"].get("elapsed_ms") for s in result.get("steps", [])
                                  if s.get("step") == "retrieval" and s.get("detail")), None),
        })

    intent_acc = sum(r["intent_ok"] for r in rows) / len(rows)
    recipient_rows = [r for r in rows if r["recipient_ok"] is not None]
    recipient_acc = sum(r["recipient_ok"] for r in recipient_rows) / len(recipient_rows) if recipient_rows else None
    email_rows = [r for r in rows if r["email_missing"] is not None]
    email_safe = sum(bool(r["email_missing"]) for r in email_rows) / len(email_rows) if email_rows else None
    dossier_rows = [r for r in rows if r["required_section_coverage"] is not None]
    section_acc = sum(r["required_section_coverage"] for r in dossier_rows) / len(dossier_rows) if dossier_rows else None
    citation_rows = [r for r in dossier_rows if r["citation_coverage"] is not None]
    citation_acc = sum(r["citation_coverage"] for r in citation_rows) / len(citation_rows) if citation_rows else None
    clarify_rows = [r for r in rows if any(c.get("expect_clarify") or c.get("expect_ambiguous")
                                          for c in cases if c["query_id"] == r["query_id"])]
    clarify_acc = sum(r["clarify"] for r in clarify_rows) / len(clarify_rows) if clarify_rows else None
    action_safety = sum(1 for r in rows if not r["executed"]) / len(rows)
    no_action_cases = [r for r in rows if any(c.get("expect_no_action") for c in cases if c["query_id"] == r["query_id"])]
    no_action = sum(1 for r in no_action_cases if r["action"] in (None, "answer", "clarify"))
    citation_rate = sum(1 for r in rows if r["citations"] > 0) / len(rows)

    llm = {"status": "skipped"}
    if args.llm_smoke:
        smoke = agent.run("Quelle décision a été prise sur la migration des données et qui l'a formulée ?",
                          slug="transcribed", top_k=8, use_llm=True)
        llm = {**smoke.get("llm", {}), "answer_chars": len(smoke.get("answer", "")),
               "citations": len(smoke.get("citations", []))}

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "intent_accuracy": intent_acc,
        "recipient_accuracy": recipient_acc,
        "email_missing_rate": email_safe,
        "dossier_section_coverage": section_acc,
        "dossier_citation_coverage": citation_acc,
        "clarification_accuracy": clarify_acc,
        "no_action_safety": no_action / len(no_action_cases) if no_action_cases else None,
        "all_actions_unexecuted": action_safety,
        "answer_citation_rate": citation_rate,
        "latency_ms_mean": sum(r["latency_ms"] for r in rows) / len(rows),
        "retrieval_ms_mean": sum(r["retrieval_ms"] or 0 for r in rows) / len(rows),
        "peak_vram_gb": round(peak_vram, 2),
        "max_rss_mb": round(rss, 1),
        "llm": llm,
    }
    payload = {"summary": summary, "cases": rows}
    ART.mkdir(exist_ok=True)
    (ART / "agent_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFailures:")
    for r in rows:
        if not r["intent_ok"] or (r["action_ok"] is False) or (r["recipient_ok"] is False):
            print(" ", r["query_id"], "expected", r["expected_intent"], "got", r["predicted_intent"],
                  "| action", r["action"], "| recipient", r["recipient"])
    print("\nSaved", ART / "agent_eval.json")


if __name__ == "__main__":
    main()
