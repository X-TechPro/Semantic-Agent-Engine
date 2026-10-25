"""Data-path tests for the thread-conditioned assigner (D-043).

The GPU half is one shot against declared bars, so the CPU half is pinned
here: threads contain only PRECEDING same-drawer sentences (a future sentence
in a thread would silently inflate accuracy — the exact bug class that makes a
null result untrustworthy), every evaluable case has its gold drawer among the
candidates, negatives are wrong drawers, the row sets are deterministic, and
the sealed holdout / spent test splits are unreachable. The counts are pinned
exactly: a relabel that moves the fit's inputs must break loudly here.
"""
from __future__ import annotations

import random

import pytest

from engine.config import repo_root
from engine.tools.finetune_threader import (THREAD_MAX_SENTS, build_rows,
                                            split_rows)
from engine.tools.finetune_reranker import _text
from engine.tools.summre_threads import SPLITS, _drawer_keys, _load

pytestmark = pytest.mark.unit

ROOT = repo_root()
MEETING = "024c_EBPH"


def test_threads_hold_only_preceding_same_drawer_sentences():
    _, _, _, traces = build_rows(ROOT, MEETING, random.Random(7))
    keys = _drawer_keys(ROOT, MEETING)
    assert traces, "no assignment cases — the ruler would be vacuous"
    for i, gold, cands in traces:
        assert gold in cands, "gold drawer must be established to be a case"
        for d, idxs in cands.items():
            assert idxs and max(idxs) < i, f"thread {d} reaches sentence >= {i}"
            assert len(idxs) <= THREAD_MAX_SENTS
            assert all(keys[j] == d for j in idxs), "thread mixed drawers"


def test_positive_row_is_the_own_thread_text():
    pos, _, _, traces = build_rows(ROOT, MEETING, random.Random(7))
    segs = _load(ROOT)[MEETING]
    assert len(pos) == len(traces)
    for (s, t), (i, gold, cands) in zip(pos, traces):
        expect = " | ".join(_text(segs[j]) for j in cands[gold])
        assert s == _text(segs[i]) and t == expect


def test_negatives_are_wrong_drawers_by_count():
    _, neg, _, traces = build_rows(ROOT, MEETING, random.Random(7))
    assert len(neg) == sum(min(4, len(c) - 1) for _, _, c in traces)


def test_split_rows_is_deterministic():
    a = split_rows(ROOT, "train", 42)
    b = split_rows(ROOT, "train", 42)
    assert a == b


def test_declared_splits_only_and_spent_test_unreachable():
    assert set(SPLITS["train"]).isdisjoint(SPLITS["holdout"])
    with pytest.raises(ValueError, match="unknown split"):
        split_rows(ROOT, "test", 42)      # the spent external set
    with pytest.raises(ValueError, match="unknown split"):
        split_rows(ROOT, "measure", 42)


def test_counts_are_the_rows_the_fit_sees():
    """Pinned 2026-10-23: train n9 = 2060/7763 rows over 2060 cases (D-044
    expansion from n5's 878); sel 266 cases UNCHANGED — the sel pin is the
    control-reproduction invariant: if sel rows move, the base control number
    (0.4398) is no longer comparable and the round is void. A label or split
    change must break this loudly."""
    tp, tn, ti = split_rows(ROOT, "train", 42)
    sp, sn, si = split_rows(ROOT, "sel", 42)
    assert (len(tp), len(tn), len(ti)) == (2060, 7763, 2060)
    assert (len(sp), len(sn), len(si)) == (266, 1008, 266)
    cands = [len(x[4]) for x in ti]
    assert min(cands) >= 2, "assignment needs >=2 candidates to be non-trivial"
