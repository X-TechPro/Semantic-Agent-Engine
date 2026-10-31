#!/usr/bin/env python3
"""Integrity and benchmark smoke tests for the French synthetic meeting memory."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
DB = ART / "french_meeting_memory_v2.sqlite3"


def fail(message: str) -> None:
    raise AssertionError(message)


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    corpus = json.loads((DATA / "french_synthetic_meetings.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA / "french_retrieval_queries.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "french_dataset_audit.json").read_text(encoding="utf-8"))
    results = json.loads((ART / "french_results.json").read_text(encoding="utf-8"))
    rankings = json.loads((ART / "french_feature_aware_test_rankings.json").read_text(encoding="utf-8"))

    all_meetings = [m for rows in corpus.values() for m in rows]
    if len(all_meetings) != 60:
        fail(f"Expected 60 meetings, got {len(all_meetings)}")
    styles = Counter(m["style"] for m in all_meetings)
    if styles != Counter({"formelle": 30, "messy": 30}):
        fail(f"Unexpected style counts: {styles}")

    split_projects = defaultdict(set)
    all_chunks = {}
    for split, meetings in corpus.items():
        for meeting in meetings:
            if len(meeting["messages"]) < 50:
                fail(f"{meeting['meeting_id']} has fewer than 50 messages")
            speakers = {x["username"] for x in meeting["messages"]}
            if len(speakers) < 3:
                fail(f"{meeting['meeting_id']} has too few distinct speakers")
            split_projects[meeting["project_id"]].add(split)
            for msg in meeting["messages"]:
                if set(msg) != {"username", "message_id", "timestamp", "message"}:
                    fail(f"Incorrect source JSON schema in {meeting['meeting_id']}: {set(msg)}")
                if not isinstance(msg["message_id"], int) or not isinstance(msg["timestamp"], str):
                    fail(f"Incorrect source field types in {meeting['meeting_id']}")
                cid = f"{meeting['meeting_id']}:{msg['message_id']:03d}"
                all_chunks[cid] = msg
    leaked = {p: s for p, s in split_projects.items() if len(s) > 1}
    if leaked:
        fail(f"Project leakage: {leaked}")
    if audit.get("issues"):
        fail(f"Dataset audit has issues: {audit['issues']}")
    if audit.get("project_leakage") != 0:
        fail("Audit reports project leakage")

    if len(queries) != 1260:
        fail(f"Expected 1260 query/evidence pairs, got {len(queries)}")
    for q in queries:
        target = q["target_chunk_id"]
        if target not in all_chunks:
            fail(f"Unknown target chunk {target}")
        if not target.startswith(q["meeting_id"] + ":"):
            fail(f"Query target crosses meeting boundary: {q['query_id']}")
        if q.get("split") not in corpus:
            fail(f"Unknown query split: {q['split']}")

    if not DB.exists():
        fail(f"SQLite memory not found: {DB}")
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    integrity_check = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity_check != "ok":
        fail(f"SQLite integrity_check: {integrity_check}")
    fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_errors:
        fail(f"Foreign-key errors: {fk_errors[:5]}")

    counts = {}
    for table in ("meetings", "messages", "message_embeddings", "feature_nodes", "feature_edges", "semantic_units", "semantic_relations"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if counts["meetings"] != 60 or counts["messages"] != len(all_chunks):
        fail(f"Unexpected source row counts: {counts}")
    if counts["message_embeddings"] != counts["messages"]:
        fail("Embedding/message count mismatch")
    if counts["feature_nodes"] < 1 or counts["feature_edges"] < 1:
        fail("Feature graph is empty")

    source_rows = conn.execute("SELECT chunk_id, immutable_json, sha256 FROM messages").fetchall()
    hash_mismatches = 0
    json_mismatches = 0
    for cid, stored, digest in source_rows:
        if hashlib.sha256(stored.encode("utf-8")).hexdigest() != digest:
            hash_mismatches += 1
        if cid not in all_chunks or stored != canonical(all_chunks[cid]):
            json_mismatches += 1
    if hash_mismatches or json_mismatches:
        fail(f"Source integrity mismatch: sha={hash_mismatches}, json={json_mismatches}")

    missing = conn.execute("SELECT COUNT(*) FROM messages m LEFT JOIN message_embeddings e USING(chunk_id) WHERE e.chunk_id IS NULL").fetchone()[0]
    orphan = conn.execute("SELECT COUNT(*) FROM message_embeddings e LEFT JOIN messages m USING(chunk_id) WHERE m.chunk_id IS NULL").fetchone()[0]
    if missing or orphan:
        fail(f"Missing/orphan vectors: {missing}/{orphan}")

    trigger_names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    required_triggers = {"immutable_source_message_update", "immutable_source_message_delete"}
    if not required_triggers.issubset(trigger_names):
        fail(f"Missing immutable-source triggers: {required_triggers - trigger_names}")

    # Probe immutability against direct SQL, then rollback the failed transaction.
    probe_cid = next(iter(all_chunks))
    try:
        conn.execute("UPDATE messages SET immutable_json='{}' WHERE chunk_id=?", (probe_cid,))
        conn.rollback()
        fail("Direct SQL UPDATE was not blocked by immutable trigger")
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        if "immutable source JSON" not in str(exc):
            fail(f"Unexpected immutability-trigger error: {exc}")
    conn.close()

    if len(rankings) != sum(q["split"] == "test" for q in queries):
        fail("Feature-aware ranking rows do not match test query count")
    target_ranks = []
    for row in rankings:
        ranked_ids = [x["chunk_id"] for x in row["rankings"]]
        target_ranks.append(ranked_ids.index(row["target_chunk_id"]) + 1 if row["target_chunk_id"] in ranked_ids else 999)
    recall1 = sum(r == 1 for r in target_ranks) / len(target_ranks)
    recall5 = sum(r <= 5 for r in target_ranks) / len(target_ranks)
    mrr10 = sum((1 / r if r <= 10 else 0) for r in target_ranks) / len(target_ranks)

    recorded = results["feature_aware_retrieval"]["test"]["overall"]
    for key, observed in (("Recall@1", recall1), ("Recall@5", recall5), ("MRR@10", mrr10)):
        if abs(recorded[key] - observed) > 1e-8:
            fail(f"Metric mismatch for {key}: stored={recorded[key]}, recomputed={observed}")

    print(json.dumps({
        "status": "PASS",
        "meetings": len(all_meetings),
        "styles": dict(styles),
        "messages": len(all_chunks),
        "speakers_per_meeting_min": min(len({x['username'] for x in m['messages']}) for m in all_meetings),
        "queries": len(queries),
        "project_leakage": 0,
        "sqlite_counts": counts,
        "sha256_mismatches": hash_mismatches,
        "source_json_mismatches": json_mismatches,
        "missing_vectors": missing,
        "orphan_vectors": orphan,
        "immutable_triggers": sorted(required_triggers),
        "feature_aware_test": {"Recall@1": recall1, "Recall@5": recall5, "MRR@10": mrr10},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
