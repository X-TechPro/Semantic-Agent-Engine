"""Graph-first generator keeps hidden truth traceable to turns."""
import random

from gen.latent_event_graph import make_project


def test_generated_project_has_traceable_events_mentions_and_queries():
    p = make_project("Atlas-test", 0, "test", random.Random(7))
    assert p["template"] == "C"
    assert p["events"] and p["mentions"] and p["queries"]
    turn_ids = {t["turn_id"] for t in p["turns"]}
    for m in p["mentions"]:
        assert m["turn_id"] in turn_ids
        assert m["evidence"] == next(t["text"] for t in p["turns"]
                                     if t["turn_id"] == m["turn_id"])
    for q in p["queries"]:
        assert q["gold_event_id"] in {e["event_id"] for e in p["events"]}
        assert all(t in turn_ids for t in q["evidence_turn_ids"])


def test_generated_project_ids_are_unique():
    p = make_project("Atlas-test", 0, "test", random.Random(7))
    ids = [e["event_id"] for e in p["events"]]
    assert len(ids) == len(set(ids))
