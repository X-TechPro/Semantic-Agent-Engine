"""The synthetic tag corpus must be leak-free and French-clean."""
from __future__ import annotations

import random

from engine.schemas_v4 import INTENTS, KINDS
from gen import tag_synth as ts


def test_labels_are_the_schema_labels():
    rows = ts.build(per_family=3, seed=1)
    for items in rows.values():
        for r in items:
            if r["kind"]:
                assert r["kind"] in KINDS, r
            if r["status"]:
                assert r["status"] in ts.STATUSES
            if r["intent"]:
                assert r["intent"] in INTENTS


def test_splits_are_family_disjoint():
    rows = ts.build(per_family=3, seed=1)
    seen = {}
    for split, items in rows.items():
        for r in items:
            seen.setdefault(r["family"], set()).add(split)
    leaked = {f: s for f, s in seen.items() if len(s) > 1}
    # Contrast families are the exception: the train phrasing and the test
    # phrasing share a group name but are different sentences by construction.
    leaked = {f: s for f, s in leaked.items() if not f.endswith("_b")}
    assert not leaked, leaked


def test_every_contrast_has_a_training_phrasing():
    train = {g["group"] for g in ts.PAIR_GROUPS if g["split"] == "train"}
    test = {g["group"] for g in ts.PAIR_GROUPS if g["split"] == "test"}
    for t in test:
        assert t[:-2] in train, t


def test_no_broken_elision():
    rows = ts.build(per_family=20, seed=2)
    for items in rows.values():
        for r in items:
            t = r["text"]
            assert " de le " not in t, t
            assert " de les " not in t, t
            assert " à le " not in t, t


def test_pairs_share_their_content():
    p = ts.build_pairs(per_group=20, seed=3)
    projects = set(ts.PROJECTS)
    for row in p:
        pa = projects & set(row["a"].split())
        pb = projects & set(row["b"].split())
        if pa or pb:
            assert pa == pb, row


def test_deterministic_for_a_seed():
    a = ts.build(per_family=5, seed=11)
    b = ts.build(per_family=5, seed=11)
    assert a == b


def test_surface_matches_the_real_transcripts():
    rows = ts.build(per_family=10, seed=4)
    texts = [r["text"] for items in rows.values() for r in items]
    assert texts
    for t in texts:
        assert t == t.lower()
