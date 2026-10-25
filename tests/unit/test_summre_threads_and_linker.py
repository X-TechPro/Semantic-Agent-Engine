"""Temporal labeller + real-trained linker: the (a) round's guards (D-039).

Pins three things the round depends on:
1. the labeller refuses out-of-order pairs, unknown relations and meetings
   whose split was not declared before labelling;
2. the compiled gold counts are the ones the trainer was fitted on — a
   relabel must break this test loudly, not silently move the training set;
3. the trainer's holdout verdict stays what it was: REFUSED at the precision
   bar, no checkpoint on disk. Re-running the experiment is allowed; quietly
   wiring a refused model is not.
"""
from __future__ import annotations

import json

import pytest

from engine.config import repo_root
from engine.tools.summre_threads import SPLITS, parse_spec
from engine.tools.summre_linker import _gold_relations, select_operating_point

pytestmark = pytest.mark.unit

GOLD = repo_root() / "data/gold"


def test_parse_spec_refuses_bad_pairs():
    with pytest.raises(ValueError, match="not ordered"):
        parse_spec("5->3:supersedes", 10)
    with pytest.raises(ValueError, match="unknown relation"):
        parse_spec("3->5:replaces", 10)
    with pytest.raises(ValueError, match="not ordered/in range"):
        parse_spec("3->9:supersedes", 9)
    rows = parse_spec("3->5:supersedes note here", 10)
    assert rows == [{"old": 3, "new": 5, "relation": "supersedes",
                     "note": "note here"}]


def test_split_membership_is_declared_not_invented():
    assert "047c_EDPH" in SPLITS["sel"]
    assert "053b_EDDH" in SPLITS["holdout"]
    assert not set(SPLITS["train"]) & set(SPLITS["holdout"])
    assert not set(SPLITS["sel"]) & set(SPLITS["holdout"])


def test_gold_files_match_the_counts_the_trainer_saw():
    """train 92 rel / 32 sup, sel 22 / 8, holdout 44 / 11 — the fit's inputs."""
    expect = {"train": (92, 32), "sel": (22, 8), "holdout": (44, 11)}
    for split, (nrel, nsup) in expect.items():
        d = json.loads((GOLD / f"summre_temporal_{split}.json")
                       .read_text(encoding="utf-8"))
        rels = [r for mm in d["meetings"].values() for th in mm["threads"]
                for r in th["relations"]]
        assert len(rels) == nrel, f"{split}: {len(rels)} != {nrel}"
        assert sum(1 for r in rels if r["relation"] == "supersedes") == nsup
        assert d["split"] == split


def test_gold_loader_refuses_spent_and_mismatched():
    with pytest.raises(ValueError, match="spent external test"):
        _gold_relations(GOLD / "summre_temporal_test.json")
    bad = GOLD / "summre_temporal_train.json"
    d = json.loads(bad.read_text(encoding="utf-8"))
    d["split"] = "holdout"
    p = repo_root() / ".qwen/tmp" / "summre_temporal_train.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="declares split"):
            _gold_relations(p)
    finally:
        p.unlink(missing_ok=True)


def test_operating_point_is_precision_constrained_not_argmax_f1():
    """D-046: the refused round selected by argmax F1 on 6 positives, which put
    the chosen point at precision 0.0014 on its OWN split and made B2
    unreachable by construction. A selector that lets a high-precision point
    with lower F1 win is the fix; when none exists it must say so."""
    import numpy as np

    grid = np.linspace(-1.0, 1.0, 41)
    # separable: a threshold exists that clears the floor
    op = select_operating_point(grid, np.array([0.9, 0.8, 0.7, 0.2, 0.1]),
                                np.array([1, 1, 1, 0, 0], dtype=bool),
                                min_precision=0.70)
    assert op["mode"] == "precision_constrained"
    assert op["precision"] >= 0.70 and op["recall"] == 1.0
    # every threshold that keeps the positive also keeps a negative -> the floor
    # is unreachable, and the artefact must say that, not pick a hopeless point
    hard = select_operating_point(grid, np.array([0.6, 0.9, 0.8, 0.7]),
                                  np.array([1, 0, 0, 0], dtype=bool),
                                  min_precision=0.70)
    assert hard["mode"] == "max_f1_fallback"
    assert hard["precision_floor_met"] is False


def test_holdout_verdict_stays_refused_and_unwired():
    """The measured number (D-039): recall passed, precision did not — so no
    checkpoint exists and the shipped default cannot have silently changed."""
    hits = sorted((repo_root() / "runs" / "summre_linker")
                  .glob("result-*-refused.json"))
    assert hits, "the real-pairs training round's refusal artefact is missing"
    d = json.loads(hits[-1].read_text(encoding="utf-8"))
    assert d["cleared"] is False
    assert d["bars_declared_before_training"]["B2_precision>=0.70"] is False
    assert d["holdout"]["trained"]["recall"] >= 0.30, \
        "the recall finding is what the refusal is interesting FOR"
    assert not (repo_root() / "models" / "linker_summre.json").exists(), \
        "a refused model must not exist on disk"
    cfg_src = (repo_root() / "src/engine/config.py").read_text(encoding="utf-8")
    assert '"backend": "rules"' in cfg_src, "default linker must still be rules"


def test_paircos_auc_math():
    """The AUC used to answer the dot-product question must be the textbook
    statistic: perfect separation 1.0, reversed 0.0, ties 0.5."""
    import numpy as np

    from engine.tools.summre_paircos import auc

    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert auc(np.array([3.0, 1.0]), np.array([2.0, 2.0])) == 0.5
