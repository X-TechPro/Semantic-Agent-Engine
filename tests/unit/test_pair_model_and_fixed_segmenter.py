"""The trained pair classifier (6.8 baseline (c)) and the fixed-window baseline (6.1).

Two modules that were effectively untested: `link/model.py` at 35% and
`segment/baseline_fixed.py` at 21%. Both are load-bearing in a specific way:

* `PairClassifier` stores coefficients as JSON so `models/linker_lr.json` is
  inspectable, and it **refuses** a checkpoint whose feature order no longer
  matches `FEATURES` rather than silently mis-indexing weights against features.
  A silent mismatch there produces a confidently wrong linker, which is the worst
  possible failure for a component whose whole job is proposing `supersedes`.
* The fixed-window segmenter is a *baseline row*, and a baseline whose numbers are
  wrong inflates the thesis. Its loop has two termination guards that are easy to
  break from the outside (stride ≥ window, stride 0), and each break shows up as an
  infinite loop or a silently missing window rather than an error.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from engine.link.model import (FEATURES, PairClassifier, build_features, sigmoid,
                               to_row)
from engine.schemas import Slots, Source, Unit, utcnow
from engine.segment.baseline_fixed import chunk_fixed_window, segment_fixed

pytestmark = pytest.mark.unit


def u(text, *, project="atlas", subject=None, amount=None, date_value=None,
      owner=None, persons=(), type="fact", status="confirmed",
      on="2026-10-12", uid="u-new", current=True, valid_to=None) -> Unit:
    return Unit(
        unit_id=uid, source_id="meeting-01", source_type="meeting",
        source_date=date.fromisoformat(on), ingested_at=utcnow(), text=text,
        sentences=[], project=project, speaker="Marie", type=type, status=status,
        valid_to=date.fromisoformat(valid_to) if valid_to else None,
        slots=Slots(subject=subject, amount=amount, date_value=date_value,
                    owner=owner, persons=list(persons)))


# --------------------------------------------------------------- build_features

def test_features_cover_the_declared_order_and_are_all_finite():
    new = u("Le budget est validé à 2,4 M€.", subject="atlas.budget",
            amount="2400000")
    old = u("Le budget est de 1,8 M€.", subject="atlas.budget", amount="1800000",
            on="2026-10-01", uid="u-old", current=False, valid_to="2026-10-12")
    f = build_features(new, old)
    assert set(f) == set(FEATURES), "a feature added/renamed must update FEATURES"
    assert all(isinstance(v, float) and v == v and abs(v) != float("inf")
               for v in f.values()), f        # no NaN / inf reaching the LR
    assert to_row(f) == [f[k] for k in FEATURES]


def test_same_subject_pair_scores_the_identity_features_high():
    new = u("Le budget passe à 2,4 M€.", subject="atlas.budget")
    old = u("Le budget est de 1,8 M€.", subject="atlas.budget",
            on="2026-10-01", uid="u-old")
    other = u("Le recrutement est ouvert.", subject="iris.hiring",
              project="iris", on="2026-10-01", uid="u-other")
    a, b = build_features(new, old), build_features(new, other)
    assert a["same_project"] == 1.0 and a["same_subject"] == 1.0
    assert b["same_project"] == 0.0 and b["same_subject"] == 0.0
    # date gap is clamped, not raw: a 10-year gap must not dominate the vector
    far = build_features(new, u("x", subject="atlas.budget", on="2000-01-01",
                                uid="u-far"))
    assert abs(far["date_signed_gap_days"]) <= 10.0
    assert far["abs_gap_days"] <= 10.0


def test_numeric_sets_equal_distinguishes_15_millions_from_1_5():
    """12.3's minimal pair: `1,5 M€` vs `15 M€` must not read as the same value."""
    a = u("Le budget est de 1,5 M€.", subject="atlas.budget")
    b = u("Le budget est de 15 M€.", subject="atlas.budget", on="2026-10-01",
          uid="u-b")
    assert build_features(a, b)["numeric_sets_equal"] == 0.0
    same = u("Le budget est de 1,5 M€ environ.", subject="atlas.budget",
             on="2026-10-01", uid="u-c")
    assert build_features(a, same)["numeric_sets_equal"] == 1.0


# ------------------------------------------------------------- sigmoid / load

def test_sigmoid_is_stable_at_the_extremes_and_monotonic():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1e6) == pytest.approx(1.0)          # no overflow
    assert sigmoid(-1e6) == 0.0                        # underflows to zero, not NaN
    assert sigmoid(2.0) > sigmoid(1.0)


def test_classifier_rejects_a_wrong_weight_count_at_construction():
    with pytest.raises(ValueError, match=f"expected {len(FEATURES)} weights"):
        PairClassifier([0.1] * (len(FEATURES) - 1), 0.0)


def test_save_load_round_trip_and_the_stale_feature_refusal(tmp_path):
    old = u("Le budget est de 1,8 M€.", subject="atlas.budget", on="2026-10-01",
            uid="u-old")
    new = u("Le budget est validé à 2,4 M€.", subject="atlas.budget",
            amount="2400000")
    clf = PairClassifier([0.5 if f == "same_subject" else 0.0 for f in FEATURES],
                         bias=-0.2, threshold=0.4, meta={"trained_on": "test"})
    path = clf.save(tmp_path / "linker_lr.json")
    stored = json.loads(path.read_text())
    assert stored["features"] == list(FEATURES)         # the guard's own evidence
    again = PairClassifier.load(path)
    assert again is not None
    assert again.score(new, old) == pytest.approx(clf.score(new, old))
    assert again.threshold == 0.4 and again.meta["trained_on"] == "test"
    assert again.proposes_supersede(new, old) is clf.proposes_supersede(new, old)

    # a checkpoint trained on a different feature order must be refused, not
    # applied column-by-column against the wrong features
    skewed = dict(stored)
    skewed["features"] = list(reversed(FEATURES))
    bad = tmp_path / "stale.json"
    bad.write_text(json.dumps(skewed))
    assert PairClassifier.load(bad) is None
    assert PairClassifier.load(tmp_path / "does-not-exist.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert PairClassifier.load(corrupt) is None


def test_explain_reports_the_largest_contributions_first():
    clf = PairClassifier([10.0 if f == "same_subject" else 0.0 for f in FEATURES], 0.0)
    new = u("x", subject="atlas.budget")
    old = u("y", subject="atlas.budget", on="2026-10-01", uid="u-old")
    top = clf.explain(new, old, top=1)
    assert top[0][0] == "same_subject" and top[0][2] > 0      # value, contribution


# ------------------------------------------------------ fixed-window segmenter

def test_chunk_fixed_window_empty_and_single():
    assert chunk_fixed_window("") == []
    assert chunk_fixed_window("   ") == []
    only = chunk_fixed_window("un seul mot", window_words=4, stride_words=2)
    assert len(only) == 1 and only[0][2] == "un seul mot"


def test_chunk_fixed_window_covers_every_word_and_respects_stride():
    text = " ".join(f"w{i}" for i in range(10))
    wins = chunk_fixed_window(text, window_words=4, stride_words=4)     # no overlap
    assert [w[2] for w in wins] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    overlapped = chunk_fixed_window(text, window_words=4, stride_words=2)
    assert overlapped[0][2].startswith("w0") and overlapped[1][2].startswith("w2")
    # surfaces must be exact slices of the original text
    for s, e, surface in overlapped:
        assert text[s:e] == surface


def test_chunk_fixed_window_terminates_on_pathological_strides():
    """A `stride_words: 0` in the config used to grow a list until the kernel
    killed the process (measured: `timeout -s KILL 6` → exit 137 on a 6-word
    string), and `window_words: 0` emitted an inverted empty span `(2, 1, '')`.
    Both are clamped to 1 now, so assert the resulting windows rather than merely
    that the call came back — "it returned" was the assertion that hung the suite."""
    text = " ".join(f"w{i}" for i in range(6))
    assert [w[2] for w in chunk_fixed_window(text, window_words=2, stride_words=0)] \
        == ["w0 w1", "w1 w2", "w2 w3", "w3 w4", "w4 w5"]        # stride -> 1
    zero_window = chunk_fixed_window(text, window_words=0, stride_words=1)
    assert [w[2] for w in zero_window] == [f"w{i}" for i in range(6)]
    assert all(s < e for s, e, _ in zero_window), "no inverted spans"
    assert len(chunk_fixed_window(text, window_words=2, stride_words=99)) == 1
    assert chunk_fixed_window(text, window_words=99, stride_words=100) \
        == [(0, len(text), text)]


def test_segment_fixed_builds_units_that_reconstruct_the_body():
    src = Source(source_id="doc-1", source_type="document",
                 source_date=date(2026, 10, 1), project="atlas",
                 text=" ".join(f"tok{i}" for i in range(9)))
    units = segment_fixed(src, window_words=4, stride_words=4)
    assert len(units) == 3
    assert len({x.unit_id for x in units}) == 3          # stable, unique ids
    assert all(x.project == "atlas" and x.source_type == "document" for x in units)
    assert " ".join(x.text for x in units) == src.body()  # nothing dropped
    for x in units:
        assert x.sentences and x.sentences[0].slice(x.text) == x.text


def test_segment_fixed_reads_its_window_from_config_not_a_hardcoded_number():
    """6.1 puts the window in config; if the function ignored it, the baseline row
    would silently measure a different system than the one recorded."""
    from engine.config import load_config

    cfg = load_config(overrides={"segment": {"fixed_window_words": 3,
                                             "fixed_stride_words": 3}})
    src = Source(source_id="doc-2", source_type="document",
                 source_date=date(2026, 10, 1),
                 text=" ".join(f"t{i}" for i in range(9)))
    assert len(segment_fixed(src, cfg)) == 3
