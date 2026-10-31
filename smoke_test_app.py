#!/usr/bin/env python3
"""Fast end-to-end smoke tests for the Flask app's memory backend."""
import json, shutil, sqlite3, tempfile
from pathlib import Path
from memory_engine import MeetingMemoryEngine

ROOT = Path(__file__).resolve().parent
engine = MeetingMemoryEngine(device="cpu", batch_size=32)

# Built-in synthetic corpus: 60 meetings, 3,600 messages and the existing checkpoint.
ds = next(x for x in engine.dataset_summaries() if x["slug"] == "synthetic")
assert ds["meeting_count"] == 60 and ds["message_count"] == 3600
assert ds["style_counts"]["formelle"] == 30 and ds["style_counts"]["messy"] == 30
transcribed = next((x for x in engine.dataset_summaries() if x["slug"] == "transcribed"), None)
if transcribed:
    assert transcribed["meeting_count"] == 30 and transcribed["message_count"] == 1200
    assert transcribed["style_counts"]["transcribed"] == 30

# Query must return the original source JSON, not regenerated assistant text.
meeting_id = "fr2-projet-01-reunion-1"
res = engine.search("synthetic", "Quelle est la dernière décision prise concernant la base de données ?", "feature_aware", 5, meeting_id)
assert res["results"] and set(res["results"][0]["source_json"]) == {"username", "message_id", "timestamp", "message"}

# Manager transcript data must preserve the four source fields per chunk.
manager = engine.get_meeting("synthetic", meeting_id)
assert manager["messages"] and set(manager["messages"][0]["source_json"]) == {"username", "message_id", "timestamp", "message"}

# Import a custom corpus and labelled q/evidence examples into an isolated memory.
meetings_path = ROOT / "examples" / "custom_meetings.json"
queries_path = ROOT / "examples" / "custom_queries.json"
imported = engine.import_dataset(meetings_path.name, meetings_path.read_bytes(), "Smoke test custom",
                                 queries_path.read_bytes(), queries_path.name)
slug = imported["slug"]
assert imported["meeting_count"] == 2 and imported["message_count"] == 12 and imported["query_count"] == 2
# A fresh engine instance must rediscover the custom memory after an app restart.
reloaded = MeetingMemoryEngine(device="cpu", batch_size=32)
assert slug in {d["slug"] for d in reloaded.dataset_summaries()}
assert len(reloaded.get_meeting(slug, "demo-architecture-001")["messages"]) == 6
custom_search = engine.search(slug, "Quelle est la décision de migration ?", "feature_aware", 3, "demo-architecture-001")
assert custom_search["results"]

# Benchmark backend must score BM25, dense and feature-aware on custom labels.
bench = engine.benchmark(slug, "test", ["bm25", "dense", "feature_aware"])
assert bench["status"] == "OK"
assert all(bench["methods"][m]["metrics"]["evaluated_queries"] == 2 for m in ("bm25", "dense", "feature_aware"))

# Direct SQL mutation/deletion attempts must fail due to DB-level triggers.
conn = sqlite3.connect(imported["database"])
try:
    conn.execute("UPDATE messages SET username='tampered' WHERE chunk_id=?", ("demo-architecture-001:001",))
    conn.commit()
    raise AssertionError("UPDATE was unexpectedly permitted")
except sqlite3.IntegrityError:
    conn.rollback()
try:
    conn.execute("DELETE FROM messages WHERE chunk_id=?", ("demo-architecture-001:001",))
    conn.commit()
    raise AssertionError("DELETE was unexpectedly permitted")
except sqlite3.IntegrityError:
    conn.rollback()
conn.close()

# The uploaded original source remains byte-for-byte present in its dataset folder.
source_path = Path(imported["database"]).parent / meetings_path.name
assert source_path.read_bytes() == meetings_path.read_bytes()

# Remove the ephemeral smoke-test dataset so it doesn't appear in the delivered app.
shutil.rmtree(Path(imported["database"]).parent)
print(json.dumps({"status": "PASS", "synthetic_meetings": ds["meeting_count"],
                  "synthetic_messages": ds["message_count"], "custom_import": "PASS",
                  "custom_labelled_benchmark": "PASS", "custom_persistence_after_restart": "PASS",
                  "source_immutable": "PASS", "transcript_manager": "PASS"}, ensure_ascii=False, indent=2))
