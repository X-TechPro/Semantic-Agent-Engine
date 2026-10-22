"""The 6.13 gate's own decision logic — tested without running a benchmark.

`eval/concept_gate.py` is the instrument that decides whether M5 lives, so its
three judgements are worth pinning separately from the numbers it reads:

* **like-for-like is per-seed.** The first version compared `config_hash` and
  `n_questions` across *all six runs* and voided a sound experiment: the refined
  arm's config differs in `store.faiss.index_key` by construction (that is the
  whole treatment), and different seeds legitimately have different question
  counts (measured: 52, 50, 51). A guard that can never pass is not a guard.
* **the column must be the only difference**, proved by hashing the config with the
  knob erased, not by demanding the two hashes match.
* **a null on a split that cannot measure the bar is not a refutation.** The verdict
  therefore has three states, and `_verdict_word` defaults `decidable` to True -- so
  a gate run that forgot to set it would print "DROP" as though 6.13 had been
  cleared against. These tests are what keep that from being silent.
"""

from __future__ import annotations

import json

import pytest

from engine.config import load_config
from eval.concept_gate import (COLUMNS, GATE_PTS, _decidability, _headroom_note,
                               _headroom_pts, _normalised_hash,
                               _pairwise_like_for_like, _pick, _table, _verdict_word,
                               verdict_from_summary)

pytestmark = pytest.mark.unit

SEEDS = [42, 43]


def _run(column, seed, **over):
    base = {"column": column, "embed_version": "hash-256-35", "reader_backend": "rules",
            "rerank_backend": "none", "n_questions": 52 if seed == 42 else 50,
            "temporal@1": 0.50, "recall@5": 0.66, "hard_negatives": 0.86,
            "headroom": 0.0069, "gold_in_pool": 1.0}
    base.update(over)
    return base


def _runs(overrides=None, seeds=None):
    """Build the run-records a gate would hold.

    `overrides` is keyed by (column, seed) so a single field can be made to differ --
    the guard's whole job is noticing that, so the tests must be able to introduce
    one difference precisely. `seeds` is parameterised because the decidability rule
    is about the *distribution across seeds*, which cannot be expressed with a fixed
    two-seed fixture.
    """
    overrides = overrides or {}
    seeds = seeds or SEEDS
    out = {}
    for column in COLUMNS:
        for seed in seeds:
            # applied after construction, not via **kwargs: an override may name
            # `column` itself (that is one of the differences the guard must catch),
            # and a keyword collision with the positional argument is not a test of
            # anything.
            r = _run(column, seed)
            r.update(overrides.get((column, seed), {}))
            r.setdefault("normalised_hash", _normalised_hash(
                load_config(overrides={"store": {"faiss": {"index_key": column}}})))
            out[f"{column}/s{seed}"] = r
    return out


# ------------------------------------------------------------ like-for-like guard

def test_a_clean_pair_is_certified_like_for_like():
    ok, problems = _pairwise_like_for_like(_runs(), SEEDS)
    assert ok and problems == []


def test_the_column_knob_alone_does_not_void_the_comparison():
    """The bug that voided the first real run: the two arms legitimately have
    different `config_hash` values because the column under test *is* config."""
    b = _normalised_hash(load_config(overrides={"store": {"faiss": {"index_key": "base"}}}))
    r = _normalised_hash(load_config(overrides={"store": {"faiss": {"index_key": "refined"}}}))
    raw_b = load_config(overrides={"store": {"faiss": {"index_key": "base"}}}).hash()
    raw_r = load_config(overrides={"store": {"faiss": {"index_key": "refined"}}}).hash()
    assert raw_b != raw_r, "the knob really does move the raw hash"
    assert b == r, "erasing the knob must make the configs provably identical"


def test_a_real_second_difference_is_reported_not_swallowed():
    """A reader difference is a second treatment; the guard must name it."""
    runs = _runs({("refined", 42): {"reader_backend": "model"}})
    ok, problems = _pairwise_like_for_like(runs, SEEDS)
    assert not ok
    assert any("reader_backend differs" in p for p in problems), problems


def test_question_count_is_compared_within_a_seed_not_across_seeds():
    """Different seeds have different ledgers (52 vs 50). Demanding one number for
    all six runs made every real comparison fail, which is the same class of mistake
    as a CI gate that can never go green."""
    ok, problems = _pairwise_like_for_like(_runs(), SEEDS)
    assert ok, problems
    # but a mismatch inside one seed's pair is a genuine defect
    broken = _runs({("refined", 43): {"n_questions": 99}})
    ok2, problems2 = _pairwise_like_for_like(broken, SEEDS)
    assert not ok2 and any("n_questions differs" in p for p in problems2)


def test_a_missing_arm_is_a_problem_not_a_crash():
    runs = _runs()
    del runs["refined/s43"]
    ok, problems = _pairwise_like_for_like(runs, SEEDS)
    assert not ok
    assert any("missing an arm" in p for p in problems)


def test_arms_that_read_the_wrong_column_are_rejected():
    runs = _runs({("refined", 42): {"column": "base"}})
    ok, problems = _pairwise_like_for_like(runs, SEEDS)
    assert not ok and any("columns asked for" in p for p in problems)


# --------------------------------------------------------------------- verdicts

def test_verdict_word_distinguishes_refutation_from_incapability():
    """The distinction that matters for the shipped claim: 6.13's +3 pt bar against
    0.69% residual headroom is arithmetic, not evidence."""
    v = {"like_for_like": True, "passes": False, "decidable": False,
         "residual_headroom": 0.0069, "gate_pts": GATE_PTS}
    word = _verdict_word(v)
    assert "NOT DECIDABLE" in word and "DROP" not in word

    decidable = dict(v, decidable=True)
    assert _verdict_word(decidable).startswith("DROP")
    assert _verdict_word(dict(decidable, passes=True)).startswith("KEEP")
    assert _verdict_word(dict(decidable, like_for_like=False)).startswith("VOID")


def test_the_default_is_a_refutation_so_a_gate_must_set_the_flag_explicitly(tmp_path):
    """`_verdict_word` defaults a missing `decidable` to True, so an omitted flag
    reads as DROP. That default is dangerous, which is why both verdict paths set the
    key -- tested here against the recompute path rather than asserted in prose.

    Headroom is driven through the per-seed **runs**, not the summary: after the
    mean-vs-majority fix, the summary's mean is no longer the deciding value, and a
    test that set only the summary would be pinning a field nothing reads."""
    assert _verdict_word({"like_for_like": True, "passes": False,
                          "gate_pts": GATE_PTS}).startswith("DROP")

    def stored(pts_by_seed):
        seeds = list(pts_by_seed)
        runs = _runs(seeds=seeds, overrides={
            ("base", s): {"headroom": round(v / 100, 4)} for s, v in pts_by_seed.items()})
        mean = round(sum(pts_by_seed.values()) / len(pts_by_seed) / 100, 4)
        return {"split": "dev", "seeds": seeds, "n_projects": 20,
                "summary": {c: {"temporal@1": 0.5 if c == "base" else 0.49,
                                "recall@5": 0.66, "hard_negatives": 0.86,
                                "headroom": mean if c == "base" else 0.0,
                                "gold_in_pool": 1.0} for c in COLUMNS},
                "runs": runs,
                "verdict": {"gate_pts": GATE_PTS, "like_for_like": False, "passes": False}}

    for pts, want, room in (({42: 0.69, 43: 0.0}, False, 0), ({42: 4.1, 43: 5.0}, True, 2)):
        p = tmp_path / f"summary_{'_'.join(map(str, pts))}.json"
        p.write_text(json.dumps(stored(pts)))
        v = verdict_from_summary(p, list(pts))["verdict"]
        assert v["decidable"] is want, f"{pts} -> {v['decidable']}"
        assert v["seeds_with_room_for_the_bar"] == room
        assert v["headroom_pts_per_seed"] == pts
        assert v["like_for_like"] is True, "the recompute must re-certify, not trust the file"
        assert v["temporal@1_delta_pts"] == -1.0
        # the rewritten artefact is what `make report` reads, so it must be durable
        assert json.loads(p.read_text())["verdict"]["decidable"] is want


def test_a_mean_carried_by_one_seed_is_not_called_decidable():
    """The measured case that made the old rule wrong. Test-split System B headroom is
    2.04 / 10.42 / 2.24 pts: a 4.9 pt **mean** clears the +3 bar, but only one of
    three seeds can actually carry the comparison, so the bar is not measurable there.
    A mean-based check reported `decidable: true` for that corpus."""
    runs = _runs(seeds=[42, 43, 44], overrides={
        ("base", 42): {"headroom": 0.0204},
        ("base", 43): {"headroom": 0.1042},
        ("base", 44): {"headroom": 0.0224},
    })
    d = _decidability(runs, [42, 43, 44], GATE_PTS)
    assert d["headroom_pts_per_seed"] == {42: 2.04, 43: 10.42, 44: 2.24}
    assert d["seeds_with_room_for_the_bar"] == 1
    assert d["seeds_measured"] == 3
    assert d["decidable"] is False, "a mean above the bar is not a measurable comparison"
    assert "cannot decide 6.13" in d["headroom_note"]
    # the mean is still reported, so nothing is hidden -- it just cannot decide alone
    assert d["residual_headroom"] == 0.049


def test_a_seed_with_no_headroom_is_excluded_rather_than_counted_as_zero():
    """`headroom` is None when a run reported no headroom; treating that as 0.0 would
    make a missing measurement count against decidability."""
    runs = _runs(seeds=[42, 43], overrides={
        ("base", 42): {"headroom": None}, ("base", 43): {"headroom": 0.05}})
    d = _decidability(runs, [42, 43], GATE_PTS)
    assert d["seeds_measured"] == 1
    assert d["headroom_pts_per_seed"][42] is None
    assert d["decidable"] is True


def test_headroom_note_states_the_arithmetic_ceiling():
    note = _headroom_note({42: 0.69, 43: 0.0}, 3.0, 0.0034)
    assert "0.69" in note and "cannot decide 6.13" in note
    assert "can carry the comparison" not in note
    enough = _headroom_note({42: 4.1, 43: 5.0}, 3.0, 0.0455)
    assert "can carry the comparison" in enough
    # a measured zero must be distinguishable from a missing value
    assert "n/a" in _headroom_note({42: None, 43: 4.0}, 3.0, 0.04)


def test_table_renders_the_guard_problems_instead_of_hiding_them():
    out = {"split": "dev", "n_projects": 20, "seeds": SEEDS,
           "summary": {c: {"temporal@1": 0.5, "recall@5": 0.66, "hard_negatives": 0.86,
                           "headroom": 0.0069, "gold_in_pool": 1.0} for c in COLUMNS},
           "runs": _runs(),
           "verdict": {"gate_pts": GATE_PTS, "temporal@1_delta_pts": -1.01,
                       "hard_negative_delta_pts": 0.0, "like_for_like": False,
                       "like_for_like_problems": ["seed 42: reader_backend differs"],
                       "passes": False, "decidable": False, "residual_headroom": 0.0069,
                       "hard_negative_axis_measurable": False,
                       "headroom_note": _headroom_note({42: 0.69, 43: 0.0}, GATE_PTS,
                                                        0.0034)}}
    md = _table(out)
    assert "VOID" in md
    assert "reader_backend differs" in md, "a void comparison must say why it is void"
    assert "not sensitive to the column" in md, "the insensitive axis must be labelled"


def test_pick_reads_the_flat_hard_negative_key():
    """The first version read `hard["system_B"]`, which does not exist: it would have
    reported None for both arms and turned a missing key into a clean 0.0 delta."""
    res = {"comparison": {"temporal@1": {"system_B": 0.5, "baseline_A": 0.3,
                                         "delta": 0.2, "delta_ci95": [0.1, 0.3],
                                         "significant_win": True},
                          "recall@5": {"system_B": 0.66},
                          "headroom_pct": {"system_B": 0.0069}},
           "meta": {"vector_column": "refined", "n_questions": 52,
                    "embed_version": "hash-256-35", "reader_backend": "rules",
                    "rerank_backend": "none", "config_hash": "abc"},
           "system_B": {"gold_in_pool": 1.0},
           "hard_negatives": {"anchor_pairwise_accuracy": 0.8666, "n_scoreable": 30}}
    p = _pick(res)
    assert p["hard_negatives"] == 0.8666
    assert p["hard_negatives_sensitive_to_column"] is False
    assert p["column"] == "refined" and p["gold_in_pool"] == 1.0


def test_pick_survives_a_run_with_no_hard_negative_pairs():
    res = {"comparison": {"temporal@1": {"system_B": None, "baseline_A": None,
                                         "delta": None, "delta_ci95": None,
                                         "significant_win": False},
                          "recall@5": {"system_B": None}, "headroom_pct": {}},
           "meta": {"n_questions": 0}, "system_B": {}, "hard_negatives": {}}
    p = _pick(res)
    assert p["hard_negatives"] is None
    assert p["column"] == "base", "a missing vector_column is the shipped default"
