"""Data-path tests for the smoothness tune (engine/tools/finetune_reranker.py).

The GPU half of this experiment is one shot against declared bars, so the
CPU half — what rows the model is allowed to see — is pinned here: positives
really are same-drawer, negatives really are cross-drawer within the meeting,
the row sets are deterministic across processes, and the sealed holdout plus
the spent test set cannot be reached through split_rows at all.
"""
from __future__ import annotations

import random

import pytest

from engine.config import repo_root
from engine.tools.finetune_reranker import build_rows, split_rows
from engine.tools.summre_threads import SPLITS, _drawer_keys, _load

pytestmark = pytest.mark.unit

ROOT = repo_root()


def _key_by_text(meeting: str) -> dict[str, str]:
    segs = _load(ROOT)[meeting]
    keys = _drawer_keys(ROOT, meeting)
    return {f"{segs[i].get('speaker', '?')}: {segs[i]['text'].strip()}": keys[i]
            for i in keys if i < len(segs)}


def test_positives_same_drawer_negatives_cross():
    mid = "024c_EBPH"
    pos, neg = build_rows(ROOT, mid, random.Random(1))
    assert pos and neg
    kt = _key_by_text(mid)
    for a, b in pos:
        assert kt[a] == kt[b], f"positive crosses drawers: {a[:30]!r} / {b[:30]!r}"
    for a, b in neg:
        assert kt[a] != kt[b], f"negative shares a drawer: {a[:30]!r} / {b[:30]!r}"


def test_both_orders_present():
    pos, neg = build_rows(ROOT, "047a_EDRH", random.Random(2))
    ps = set(pos)
    assert all((b, a) in ps for a, b in ps), "positives must be order-symmetric"
    ns = set(neg)
    assert all((b, a) in ns for a, b in ns)


def test_split_rows_is_deterministic_across_calls():
    p1, n1 = split_rows(ROOT, "train", 42)
    p2, n2 = split_rows(ROOT, "train", 42)
    assert p1 == p2 and n1 == n2, "row sampling must not depend on process state"


def test_holdout_and_spent_test_are_unreachable_from_train():
    assert set(SPLITS["train"]).isdisjoint(SPLITS["holdout"])
    assert set(SPLITS["sel"]).isdisjoint(SPLITS["holdout"])
    with pytest.raises(ValueError, match="unknown split"):
        split_rows(ROOT, "test", 42)          # the spent external set
    with pytest.raises(ValueError, match="unknown split"):
        split_rows(ROOT, "measure", 42)


def test_train_rows_are_plentiful_enough():
    """The whole point vs D-039: relatedness contrasts, not 25 positives."""
    pos, neg = split_rows(ROOT, "train", 42)
    assert len(pos) >= 2000, f"only {len(pos)} positive rows — the cap is off"
    assert len(neg) >= len(pos) // 2
