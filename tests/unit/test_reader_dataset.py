"""Training rows for the reader (6.2.2 / 6.2.3) — the gold the head learns from.

`engine/read/dataset.py` sits at 34% coverage, and it is the one place where a
mistake is invisible in the loss curve: rows built from a *kind string* instead of
the Event silently lose the `conditional` refinement (that happened, and nine
events were trained as `proposed`), and rows whose `kind` is separable only by the
`[speaker] ` bracket teach the head a formatting artifact instead of the skill
12.3 tests. Both failures look like a healthy training run.

The assertions here are about provenance and learnability, not accuracy: the
quality gate for the trained reader lives in `read/train.py`'s report, and copying
an accuracy number into a unit test would just freeze whatever the last run
happened to be.
"""

from __future__ import annotations

import pytest

from engine.read.dataset import (QA_KIND_TO_INTENT, QUERY_KIND, QUERY_STATUS,
                                 QUERY_TYPE, UNTRAINABLE_INTENTS, build_split,
                                 casual_variants, event_of, label_mix,
                                 predict_rules, query_rows, stack, unit_rows,
                                 user_channel_rows)
from engine.read.model import HEADS, MASK, gold_vector
from engine.schemas import Slots
from gen.ledger import reader_gold

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ledger():
    from gen.ledger import build_ledger

    return build_ledger(n_projects=8, seed=42)


@pytest.fixture(scope="module")
def split(ledger, tmp_path_factory):
    """A real ingest: the rows must come out of the engine's own segmenter, or the
    head is trained on a text distribution the product never produces."""
    from gen.splice import write_manifest

    man = write_manifest(tmp_path_factory.getbasetemp() / "manifest.json",
                         ledger.projects, seed=42)
    projects = set(man["split"]["dev"]) or {ledger.projects[0].name}
    return build_split(ledger, projects, 42, tmp_path_factory.mktemp("reader"),
                       embed_backend="hash", tag="dev")


# ------------------------------------------------------------------- gold source

def test_event_of_maps_to_events_not_kind_strings(ledger):
    """`reader_gold` derives `conditional` from `Event.notes`; a bare kind cannot."""
    events = event_of(ledger)
    assert all(hasattr(e, "notes") for e in events.values())
    conditional = [e for e in ledger.events
                   if e.kind == "proposed" and e.notes == "conditional"]
    if not conditional:
        pytest.skip("this ledger draw has no conditional event")
    ev = conditional[0]
    assert reader_gold(ev)[2] == "conditional"
    assert reader_gold(ev.kind)[2] == "proposed"      # the bug: kind-only loses it


def test_unit_rows_label_from_the_event_not_a_rederived_table(ledger):
    """One synthetic unit pointing at a conditional event must come back
    `conditional` — if this row ever reads `proposed`, the Event was dropped."""
    from helpers import make_unit

    ev = next((e for e in ledger.events
               if e.kind == "proposed" and e.notes == "conditional"), None)
    if ev is None:
        pytest.skip("no conditional event in this draw")
    rows = unit_rows(ledger, [make_unit("x", unit_id="u1",
                                        gold_event_ids=[ev.event_id])])
    assert len(rows) == 1
    assert rows[0]["labels"]["status"] == HEADS["status"].index("conditional")


def test_units_without_gold_are_dropped_not_invented(ledger):
    """Openers and fillers carry no event id. Labelling them `other` would teach a
    class the generator never asserted."""
    from helpers import make_unit

    assert unit_rows(ledger, [make_unit("Bonjour à tous.", unit_id="no-gold")]) == []


def test_the_unit_row_carries_the_real_unit(ledger):
    """`predict_rules` must read exactly what the engine read — same date, same
    speaker — or the rules baseline is scored on a different corpus."""
    from helpers import make_unit

    ev = ledger.events[0]
    u = make_unit("Le budget est validé à 1,5 M€.", unit_id="u-real",
                  speaker="Marie", gold_event_ids=[ev.event_id])
    rows = unit_rows(ledger, [u])
    assert rows[0]["unit"] is u or rows[0]["unit"].unit_id == u.unit_id
    assert rows[0]["labels"]["intent"] == MASK         # not a query: no gold intent


# ------------------------------------------------------------------- query rows

def test_query_rows_are_labelled_by_construction(ledger):
    from gen.splice import write_manifest
    import tempfile, pathlib

    man = write_manifest(pathlib.Path(tempfile.mkdtemp()) / "m.json",
                         ledger.projects, seed=42)
    projects = set(man["split"]["dev"])
    rows = query_rows(ledger, projects, 42)
    assert rows, "the QA generator must produce dev-split questions"
    for r in rows[:10]:
        assert r["labels"]["kind"] == HEADS["kind"].index(QUERY_KIND)
        assert r["labels"]["type"] == HEADS["type"].index(QUERY_TYPE)
        assert r["labels"]["status"] == HEADS["status"].index(QUERY_STATUS)
        assert r["labels"]["intent"] != MASK
        assert r["speaker"] is None                    # the user channel is unmarked


def test_unanswerable_questions_are_intent_answer_not_clarify():
    """Whether evidence exists is the retrieval layer's decision; labelling it
    `clarify` would train the head on an outcome instead of on what was asked."""
    assert QA_KIND_TO_INTENT["unanswerable"] == "answer"
    assert QA_KIND_TO_INTENT["history"] == "answer"   # a mode, not a different task
    assert set(QA_KIND_TO_INTENT.values()) <= set(HEADS["intent"])


def test_no_gold_row_exists_for_an_untrainable_intent(split):
    """5 of the 8 intents have no gold in this corpus. They must show up as
    `untrained_classes`, never as rows invented to fill them."""
    gold_intents = {g for g in split["gold_names"]["intent"] if g}
    assert gold_intents, "some intent gold is expected"
    assert not (gold_intents & set(UNTRAINABLE_INTENTS)), gold_intents


# --------------------------------------------------------------- stacking / mix

def test_stack_masks_per_head_and_keeps_gold_names_decoded(split):
    packed = stack(split["rows"])
    n = len(split["rows"])
    assert len(packed["texts"]) == n
    for name, arr in packed["labels"].items():
        assert len(arr) == n
        assert all(-1 <= int(v) < len(HEADS[name]) for v in arr), name
    assert packed["gold_names"]["intent"].count(None) == \
        int((packed["labels"]["intent"] == MASK).sum())
    for name in HEADS:
        for v in packed["gold_names"][name]:
            assert v is None or v in HEADS[name]


def test_label_mix_counts_only_unmasked_rows(split):
    mix = label_mix(split["rows"])
    for name, counts in mix.items():
        assert sum(counts.values()) == int((split["labels"][name] != MASK).sum())
        assert all(k in HEADS[name] for k in counts)


def test_info_is_not_separable_by_the_speaker_bracket(split):
    """The learnability claim in the module docstring, checked on the class it is
    actually about.

    `query` rows are *all* unbracketed by construction (a user request has no
    speaker), so requiring both channels there would assert something the design
    denies. `info` is the class at risk: without `user_channel_rows` every `info`
    row would carry `[Marie] ` and the head could reach a high kind-F1 by reading
    a bracket instead of learning question-vs-assertion.
    """
    kinds = [g for g in split["gold_names"]["kind"] if g]
    assert set(kinds) == {"info", "query"}, kinds
    info = [r for r in split["rows"] if HEADS["kind"][r["labels"]["kind"]] == "info"]
    bracketed = [r for r in info if r["speaker"] is not None]
    assert bracketed, "meeting rows should keep their speaker"
    assert len(bracketed) < len(info), \
        f"all {len(info)} info rows are bracketed: the kind head can cheat"
    assert split["n_user_channel"] > 0


def test_user_channel_rows_copies_leave_the_input_alone(ledger):
    from helpers import make_unit

    ev = ledger.events[0]
    original = make_unit("Le budget est validé à 1,5 M€.", unit_id="u-x",
                         speaker="Marie", gold_event_ids=[ev.event_id])
    rows = unit_rows(ledger, [original])
    before = [dict(r) for r in rows]
    extra = user_channel_rows(rows)
    assert len(extra) == 1 and len(rows) == 1
    assert rows == before                              # not mutated in place
    assert extra[0]["speaker"] is None
    assert extra[0]["origin"] == "user_statement"
    assert extra[0]["ref"].endswith("#u")
    assert extra[0]["labels"] == rows[0]["labels"]      # same gold, new channel


def test_query_rows_are_not_duped_into_the_user_channel(ledger):
    """The unprefixed copy is for *statements*. Duplicating a query row would
    double-count the class the bracket test above is trying to balance."""
    rows = [{"text": "Qui s'occupe du budget ?", "speaker": None, "context": None,
             "labels": gold_vector(QUERY_KIND, QUERY_TYPE, QUERY_STATUS, "answer"),
             "origin": "query", "ref": "q1"}]
    assert user_channel_rows(rows) == []


# ------------------------------------------------------- casual-French copies

def _statement_row(ledger):
    from helpers import make_unit

    ev = ledger.events[0]
    u = make_unit("Le budget est validé à 1,5 M€.", unit_id="u-c", speaker="Marie",
                  gold_event_ids=[ev.event_id])
    return unit_rows(ledger, [u])


def test_casual_copy_keeps_every_label_and_the_speaker(ledger):
    """The augmentation's whole claim: fillers change *style*, and kind/type/status
    are properties of what the utterance does. A copy that drifted its labels or
    dropped the speaker bracket would train the heads on noise."""
    rows = _statement_row(ledger)
    extra = casual_variants(rows)
    assert len(extra) == 1
    c = extra[0]
    assert c["labels"] == rows[0]["labels"]
    assert c["speaker"] == rows[0]["speaker"]
    assert c["ref"] == rows[0]["ref"] + "#c"
    assert str(c["origin"]).endswith("+casual")
    assert c["text"] != rows[0]["text"]                 # it really restyled it
    assert rows[0]["text"] == "Le budget est validé à 1,5 M€."  # input untouched


def test_casual_copy_carries_its_own_unit_text(ledger):
    """predict_rules() reads a row's `unit` when there is one — if the copy kept
    the ORIGINAL unit, the rules baseline would be scored on clean text while
    the trained head saw the casual text, and the 6.2.6 "beats the baseline"
    comparison would silently measure two different corpora."""
    rows = _statement_row(ledger)
    c = casual_variants(rows)[0]
    assert c["unit"].text == c["text"]
    assert c["unit"].unit_id == rows[0]["unit"].unit_id   # same unit, new surface


def test_casual_variants_are_deterministic(ledger):
    """Sha1 of the ref, not `random`: the corpus must rebuild identically, or
    two training runs are not the same experiment."""
    rows = _statement_row(ledger)
    a = casual_variants(rows)
    b = casual_variants(rows)
    assert [x["text"] for x in a] == [x["text"] for x in b]


def test_copies_are_never_augmented_again(split):
    """The factor is fixed at 1 copy per original: a copy of a copy would let the
    filler style outweigh the real phrasing without anyone deciding that."""
    rows = split["rows"]
    doubled = casual_variants(rows)
    assert all(not c["ref"].endswith("#c#c") and "#c#" not in c["ref"]
               for c in doubled)
    srcs = {r["ref"] for r in rows if not r["ref"].endswith(("#c", "#u"))}
    assert all(c["ref"][:-2] in srcs for c in doubled)


def test_build_split_carries_the_augmentation_with_matching_labels(split):
    """The integration the trainer actually consumes: casual rows must be present
    in every split (train AND dev/test — the gates are then measured on the
    harder corpus at full strength) and each must point at a real source row
    with identical labels."""
    casual = [r for r in split["rows"] if str(r.get("origin", "")).endswith("+casual")]
    assert casual, "build_split must ship the casual copies into dev too"
    byref = {r["ref"]: r for r in split["rows"]
             if not r["ref"].endswith(("#c", "#u"))}
    for c in casual:
        src = byref[c["ref"][:-2]]
        assert c["labels"] == src["labels"], c["ref"]


# ------------------------------------------------------------- rules baseline

def test_predict_rules_scores_the_same_rows_not_a_rebuilt_corpus(split):
    from engine.read.baseline_rules import RuleReader

    preds = predict_rules(RuleReader(), split["rows"])
    assert set(preds) == set(HEADS)
    for name, arr in preds.items():
        assert len(arr) == len(split["rows"])
        assert all(int(v) == MASK or 0 <= int(v) < len(HEADS[name]) for v in arr)
    # the rules reader is the shipped baseline, so it must at least reproduce the
    # gold `kind` on the meeting rows it was written for (not a quality gate:
    # a near-zero number here means the row order diverged from the labels)
    gold = split["labels"]["kind"]
    hit = sum(int(p) == int(g) for p, g in zip(preds["kind"], gold)
              if int(g) != MASK)
    n = int((gold != MASK).sum())
    assert n and hit / n > 0.5, f"rules reader reproduces kind on {hit}/{n} rows"
