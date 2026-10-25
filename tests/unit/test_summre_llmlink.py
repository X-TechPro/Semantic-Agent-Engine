"""CPU-tier guards for the whole-session LLM linker (D-045 round).

The GPU half is a single zero-shot pass scored against pre-declared bars, so
the parse guards must be proven here: a malformed or out-of-discipline
proposal (bad json, unknown relation, old >= new, out-of-range index) is
RECORDED as rejected, never silently scored; duplicates collapse; the
supersedes scorer is exact-pair like D-039's.
"""
from __future__ import annotations

import json

import pytest

from engine.tools.summre_llmlink import parse_relations, prf, transcript

pytestmark = pytest.mark.unit


def _raw(relations):
    return json.dumps({"relations": relations})


def test_valid_rows_kept_and_deduped():
    kept, rej = parse_relations(_raw([
        {"old": 3, "new": 9, "relation": "supersedes", "why": "venue change"},
        {"old": 3, "new": 9, "relation": "supersedes", "why": "repeat"},
        {"old": 1, "new": 2, "relation": "SUPPORTS"},
    ]), 10)
    assert kept == [{"old": 3, "new": 9, "relation": "supersedes",
                     "why": "venue change"},
                    {"old": 1, "new": 2, "relation": "supports", "why": ""}]
    assert rej == []


def test_out_of_discipline_rows_rejected_not_scored():
    kept, rej = parse_relations(_raw([
        {"old": 5, "new": 5, "relation": "supersedes"},      # not ordered
        {"old": 8, "new": 4, "relation": "supersedes"},      # reversed
        {"old": -1, "new": 2, "relation": "supports"},       # negative
        {"old": 0, "new": 10, "relation": "supports"},       # >= n_segments
        {"old": 0, "new": 1, "relation": "explains"},        # unknown relation
        {"new": 1, "relation": "supports"},                  # missing old
        "not even an object",
    ]), 10)
    assert kept == []
    assert len(rej) == 7
    assert all("reject" in r for r in rej)


def test_bad_json_is_a_recorded_rejection():
    kept, rej = parse_relations("{oops", 10)
    assert kept == [] and len(rej) == 1 and "bad json" in rej[0]["reject"]


def test_bare_list_answer_accepted():
    kept, rej = parse_relations(
        json.dumps([{"old": 0, "new": 1, "relation": "contradicts"}]), 5)
    assert kept and not rej


def test_prf_is_exact_pair_and_zero_safe():
    m = prf({(1, 2), (3, 4)}, {(1, 2), (5, 6)})
    assert m == {"n_pred": 2, "n_gold": 2, "tp": 1,
                 "precision": 0.5, "recall": 0.5}
    empty = prf(set(), set())
    assert empty["precision"] == 0.0 and empty["recall"] is None


def test_transcript_numbers_every_segment():
    segs = [{"speaker": "A", "text": " x "}, {"text": "y"}]
    lines = transcript(segs).splitlines()
    assert lines == ["[0] (A) x", "[1] (?) y"]
