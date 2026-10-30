"""v4 §1/§2/§5: immutability, spans, append-only writes, ingestion."""
import pytest

from engine.ingest_v4 import V4Ingestor
from engine.schemas_v4 import (ChunkRecord, LinkEvent, MessageRecord,
                               StateEvent)
from engine.segment.sentences_v4 import (is_confirmation, semantic_groups,
                                         split_sentences)
from engine.store.v4 import V4Store, event_id_for


def _msg(mid="m1", text="Le budget est proposé.", **kw):
    base = dict(message_id=mid, source_id="src", project_id="Atlas",
                speaker="Marie", timestamp="2026-10-01", text=text)
    base.update(kw)
    return MessageRecord(**base)


def test_message_is_immutable_on_every_field():
    s = V4Store()
    s.add_message(_msg())
    s.add_message(_msg())                      # identical re-add is a no-op
    with pytest.raises(ValueError):
        s.add_message(_msg(text="Autre texte."))
    with pytest.raises(ValueError):
        s.add_message(_msg(speaker="Paul"))
    with pytest.raises(ValueError):
        s.add_message(_msg(timestamp="2026-10-02"))


def test_chunk_span_is_immutable_and_reconstructs_text():
    s = V4Store()
    text = "Le budget est proposé. OK, validé."
    s.add_message(_msg(text=text))
    spans = split_sentences(text)
    assert all(text[x.start:x.end] == x.text for x in spans)
    c = ChunkRecord("m1:s0.0", "m1", spans[0].start, spans[0].end,
                    spans[0].text, 0)
    s.add_chunk(c)
    s.add_chunk(c)
    with pytest.raises(ValueError):
        s.add_chunk(ChunkRecord("m1:s0.0", "m1", 0, len(text), text, 0))


def test_links_and_state_are_append_only_but_replayable():
    s = V4Store()
    link = LinkEvent("l1", "a", "b", "supersedes", 0.9, "ev", "v4",
                     "2026-10-01T00:00:00+00:00")
    s.add_link(link)
    s.add_link(link)                            # same payload, same id: no-op
    with pytest.raises(ValueError):
        s.add_link(LinkEvent("l1", "a", "b", "supersedes", 0.8, "ev", "v4",
                             "2026-10-01T00:00:00+00:00"))
    with pytest.raises(ValueError):
        s.add_link(LinkEvent("l1", "a", "c", "supersedes", 0.9, "ev", "v4",
                             "2026-10-01T00:00:00+00:00"))

    st = StateEvent("s1", "Atlas:budget", "budget", "800", "confirmed", "a",
                    "2026-10-08T00:00:00+00:00")
    s.add_state(st)
    s.add_state(st)
    with pytest.raises(ValueError):
        s.add_state(StateEvent("s1", "Atlas:budget", "budget", "900",
                               "confirmed", "a",
                               "2026-10-08T00:00:00+00:00"))


def test_sentence_grouping_follows_topic_not_length():
    text = ("On parle du budget. Le budget est de 800 euros. "
            "Ensuite le traiteur doit confirmer.")
    groups = semantic_groups(text, topic_fn=lambda t: "budget"
                             if "budget" in t else "traiteur")
    assert len(groups) == 2
    assert len(groups[0]) == 2 and len(groups[1]) == 1
    assert groups[0][0].text.startswith("On parle")


def test_confirmation_attaches_and_asr_never_invents_a_boundary():
    assert is_confirmation("OK, validé.")
    groups = semantic_groups("Le budget est proposé. OK, validé.",
                             topic_fn=lambda t: "budget")
    assert len(groups) == 1
    asr = "euh donc le budget quoi on valide huit cents euros voilà"
    spans = split_sentences(asr)
    assert len(spans) == 1 and spans[0].end == len(asr)


def test_ingest_keeps_timestamp_and_writes_state():
    s = V4Store()
    ing = V4Ingestor(s)
    report = ing.ingest(_msg("m1", "Le budget est proposé à 1200 euros.",
                             timestamp="2026-10-01T09:00:00"))
    assert report["unit_ids"] and report["states"] == 1
    unit = s.unit(report["unit_ids"][0])
    assert unit["slot"] == "budget" and unit["value"] == "1200 euros"
    assert unit["status"] == "proposed"
    assert unit["evidence"]["start"] < unit["evidence"]["end"]
    key = event_id_for(unit, s)
    cur = s.current_state(key, "budget")
    assert cur["value"] == "1200 euros" and cur["open"] is True


def test_confirmation_message_attaches_to_the_previous_proposal():
    s = V4Store()
    ing = V4Ingestor(s)
    ing.ingest(_msg("m1", "Le budget est proposé à 1200 euros.",
                    timestamp="2026-10-01T09:00:00"))
    ing.ingest(_msg("m2", "OK, validé.", timestamp="2026-10-01T09:05:00"))
    unit = s.units_for_message("m1")[0]
    key = event_id_for(unit, s)
    cur = s.current_state(key, "budget")
    assert cur["status"] == "confirmed" and cur["value"] == "1200 euros"
    assert [r["status"] for r in s.history(key, "budget")] == [
        "proposed", "confirmed"]


def test_ingest_is_idempotent():
    s = V4Store()
    ing = V4Ingestor(s)
    m = _msg("m1", "Le budget est proposé à 1200 euros.")
    ing.ingest(m)
    first = (len(s.all_units()), len(s.links()), len(s.state_events()))
    ing.ingest(m)
    second = (len(s.all_units()), len(s.links()), len(s.state_events()))
    assert first == second


def test_origin_separates_real_and_synthetic():
    s = V4Store()
    s.add_message(_msg("m1", "réel", origin="real"))
    s.add_message(_msg("m2", "synthétique", origin="synthetic"))
    assert s.message("m1")["origin"] == "real"
    assert s.message("m2")["origin"] == "synthetic"
    with pytest.raises(ValueError):
        MessageRecord("m3", "src", "p", "s", "2026-01-01", "x", origin="other")


def test_store_is_usable_from_another_thread():
    """Streamlit caches the store in one thread and reruns queries in another."""
    import threading

    from engine.store.v4 import V4Store

    store = V4Store()
    seen = {}

    def worker():
        seen["units"] = len(store.all_units())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["units"] == 0
