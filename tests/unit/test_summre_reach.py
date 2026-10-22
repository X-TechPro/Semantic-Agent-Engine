"""Pure-logic tests for the B-13 reach instrument (no models, no GPU).

Two semantics pinned: per-meeting pairing in `_load_pair_lists` (a global pool
would compare meeting A's segments against meeting B's vectors) and the spent-
test refusal (one read stays spent). The extractor-gate helpers that used to
live here (assign_keys/consistency_table) were removed with the extractors
themselves (D-037); their verdict artefacts are static files now.
"""
from __future__ import annotations

import json

import pytest

from engine.tools.summre_reach import _load_pair_lists, _refuse_spent_test


def test_load_pair_lists_is_per_meeting(tmp_path):
    gold = {"meetings": {
        "M1": {"threads": [{"subject": "s", "relations": [
            {"old": 1, "new": 3, "relation": "supersedes"},
            {"old": 3, "new": 5, "relation": "supports"}]},
            {"subject": "t", "relations": [{"old": 10, "new": 12,
                                            "relation": "contradicts"}]}]},
        "M2": {"threads": []}}}
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold), encoding="utf-8")
    _g, per, sup = _load_pair_lists(p, meetings={})
    same, cross = per["M1"]
    assert sorted(same) == [(1, 3), (1, 5), (3, 5), (10, 12)]
    assert len(cross) == 6                # every thread-s index vs thread-t
    assert per["M2"] == ([], [])
    assert sup == [("M1", 1, 3, "supersedes"), ("M1", 3, 5, "supports"),
                   ("M1", 10, 12, "contradicts")]


def test_spent_test_refuses_once_an_artefact_exists(tmp_path):
    (tmp_path / "runs" / "summre_reach").mkdir(parents=True)
    (tmp_path / "runs" / "summre_reach" / "reach-test-2026-10-21.json").write_text(
        "{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="spent and stays spent"):
        _refuse_spent_test(tmp_path, "test")
    _refuse_spent_test(tmp_path, "dev")   # dev is always allowed


def test_removed_extractor_arms_are_refused_by_name():
    from engine.tools.summre_reach import ARM_BUILDERS

    assert ARM_BUILDERS == ("rules", "v2")
