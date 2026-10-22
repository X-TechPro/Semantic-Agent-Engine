"""Pure helpers of `eval/frbench.py` — pinned offline, no network, no torch.

The frbench harness decides the first real-data verdict this project will publish,
so its arithmetic gets hand-computed assertions rather than smoke tests. Two traps
this file exists to close:

* **mapping drift** — the MASSIVE->intent mapping is a rule over external label
  names. If the corpus adds a label or the rule typo's, the in-scope denominator
  silently changes and the published accuracy means a different thing. The test
  enumerates every label actually in the downloaded file and pins the split.
* **the truthiness trap on a measured zero** (D-018 class): an intent accuracy of
  exactly 0.0 is the *bad news*, and `None` means "nothing was measurable". Both
  render falsy; the report must never collapse them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.frbench import (LABEL2REL, auc, build_pair_units, embed_tag,
                          map_massive_intent, massive_report, miracl_metrics,
                          parse_qrels, parse_topics, run_dir_for, xnli_report)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------- AUC

def test_auc_known_constructions_by_hand():
    assert auc([1.0, 2.0], [0.0, 0.5]) == 1.0          # perfect separation
    assert auc([0.0, 0.5], [1.0, 2.0]) == 0.0          # perfect reversal
    assert auc([1.0], [1.0]) == 0.5                     # all ties
    # one tie, one win: (1 win + 0.5 tie) / 2 = 0.75
    assert auc([2.0, 1.0], [1.0, 0.0]) == pytest.approx(0.875)


def test_auc_rejects_missing_class():
    with pytest.raises(ValueError):
        auc([], [1.0])


# --------------------------------------------------------- pair construction

def test_build_pair_units_holds_metadata_constant():
    """The AUC only measures *text* if every metadata feature is identical; if a
    date or type ever varied across pairs, the classifier could cheat on metadata."""
    from engine.link.model import FEATURES, build_features

    a1, b1 = build_pair_units("Le budget est voté.", "Le budget n'est pas voté.")
    a2, b2 = build_pair_units("X y.", "X n'y pas.")
    f1, f2 = build_features(b1, a1), build_features(b2, a2)
    # metadata features identical across pairs; only the text-derived four may vary
    textish = {"topical_overlap", "jaccard", "numeric_sets_equal", "len_ratio"}
    for k in FEATURES:
        if k in textish:
            continue
        assert f1[k] == f2[k], f"metadata feature {k} varies between pairs"
    assert f1["same_project"] == 1.0 and f1["date_signed_gap_days"] == 1 / 365
    assert b1.source_date > a1.source_date               # hypothesis is newer


def test_label_map_matches_the_verified_xnli_convention():
    # verified against data/external/facebook__xnli/README.md, not memory
    assert LABEL2REL == {0: "supports", 1: "unrelated", 2: "contradicts"}


# ----------------------------------------------------------- xnli aggregation

def test_xnli_report_requires_all_three_classes():
    with pytest.raises(ValueError, match="3 label classes"):
        xnli_report([(0.5, 0), (0.4, 0)])
    rep = xnli_report([(1.0, 0), (0.0, 0), (1.0, 1), (0.0, 1), (1.0, 2), (0.0, 2)])
    assert rep["auc_contradiction_vs_neutral"] == 0.5   # (1>0, 0<1) -> tie rate .5
    assert rep["class_counts"]["contradicts"] == 2


# ------------------------------------------------------- MASSIVE intent mapping

MASSIVE_ALL = ['alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down',
               'audio_volume_mute', 'audio_volume_other', 'audio_volume_up',
               'calendar_query', 'calendar_remove', 'calendar_set', 'cooking_query',
               'cooking_recipe', 'datetime_convert', 'datetime_query',
               'email_addcontact', 'email_query', 'email_querycontact',
               'email_sendemail', 'general_greet', 'general_joke', 'general_quirky',
               'iot_cleaning', 'iot_coffee', 'iot_hue_lightchange',
               'iot_hue_lightdim', 'iot_hue_lightoff', 'iot_hue_lighton',
               'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 'lists_createoradd',
               'lists_query', 'lists_remove', 'music_dislikeness', 'music_likeness',
               'music_query', 'music_settings', 'news_query', 'play_audiobook',
               'play_game', 'play_music', 'play_podcasts', 'play_radio',
               'qa_currency', 'qa_definition', 'qa_factoid', 'qa_maths', 'qa_stock',
               'recommendation_events', 'recommendation_locations',
               'recommendation_movies', 'social_post', 'social_query',
               'takeaway_order', 'takeaway_query', 'transport_query',
               'transport_taxi', 'transport_ticket', 'transport_traffic',
               'weather_query']


def test_every_massive_label_is_deliberately_in_or_out_of_scope():
    ours = {"answer", "compare", "build_dossier", "draft_email", "create_task",
            "schedule_event", "clarify", "none"}
    for lab in MASSIVE_ALL:
        m = map_massive_intent(lab)
        assert m is None or m in ours, lab
    # the mapped side, pinned by hand-computed rule application
    assert map_massive_intent("qa_factoid") == "answer"
    assert map_massive_intent("weather_query") == "answer"
    assert map_massive_intent("transport_traffic") == "answer"
    assert map_massive_intent("calendar_set") == "schedule_event"
    assert map_massive_intent("alarm_set") == "schedule_event"
    assert map_massive_intent("lists_createoradd") == "create_task"
    assert map_massive_intent("takeaway_order") == "create_task"
    assert map_massive_intent("email_sendemail") == "draft_email"
    # out of scope: devices, media, chit-chat, removals, conversions
    for lab in ("iot_coffee", "play_music", "general_greet", "calendar_remove",
                "datetime_convert", "social_post", "audio_volume_mute",
                "music_settings", "cooking_recipe"):
        assert map_massive_intent(lab) is None, lab


def test_mapping_still_covers_the_file_on_disk():
    """If the corpus ever gains a label outside the enumerated rule, the in-scope
    denominator silently shifts -- fail loudly instead."""
    p = Path("data/external/mteb__amazon_massive_intent/test/fr.json.gz")
    if not p.exists():
        pytest.skip("MASSIVE fr not fetched (make frdata)")
    import gzip
    labels = {json.loads(l)["label"] for l in
              gzip.open(p, "rt") if l.strip()}
    assert labels <= set(MASSIVE_ALL), sorted(labels - set(MASSIVE_ALL))


# ---------------------------------------------------------- report aggregation

def test_measured_zero_is_never_read_as_absent():
    """A perfect miss score (0.0) is bad news; None means nothing measurable. The
    D-018 truthiness trap made those two indistinguishable three times already."""
    rows = [("answer", "create_task", False), ("answer", "draft_email", False)]
    rep = massive_report(rows)
    assert rep["intent_accuracy_in_scope"] == 0.0        # not None!
    assert rep["macro_f1_in_scope"] == 0.0
    assert rep["n_out_of_scope"] == 0 and rep["oov_rejection_rate"] is None
    # no in-scope rows at all -> genuinely unmeasurable
    rep2 = massive_report([("answer", "answer", False),
                           (None, "none", False)])
    assert rep2["n_out_of_scope"] == 1 and rep2["oov_rejection_rate"] == 1.0
    assert rep2["intent_accuracy_in_scope"] == 1.0


def test_oov_leakage_names_the_leaking_predictions():
    rows = [(None, "answer", False), (None, "none", False), (None, "clarify", False)]
    rep = massive_report(rows)
    assert rep["oov_rejection_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep["oov_leakage"] == [("answer", 1)]


def test_macro_f1_hand_computed():
    # 2 correct of 3 gold 'answer'; 1 false-positive 'answer' from gold 'schedule'
    rows = [("answer", "answer", False), ("answer", "answer", False),
            ("answer", "schedule_event", False), ("schedule_event", "answer", False)]
    rep = massive_report(rows)
    f1 = rep["per_class_f1"]
    # answer: p=2/3, r=2/3 -> f1=2/3 ; schedule_event: p=1, r=0 -> 0
    assert f1["answer"] == pytest.approx(2 / 3, abs=1e-4)
    assert f1["schedule_event"] == 0.0
    assert rep["macro_f1_in_scope"] == pytest.approx(1 / 3, abs=1e-4)


# ------------------------------------------------------------ MIRACL parsing

def test_parse_qrels_excludes_labelled_negatives():
    lines = ["q1\tQ0\td1\t1", "q1\tQ0\td2\t0", "q2\tQ0\td7\t2"]
    gold = parse_qrels(lines)
    assert gold == {"q1": {"d1"}, "q2": {"d7"}}       # zeros are distractors, not gold
    assert parse_qrels(["", "  "]) == {}


def test_parse_topics_splits_on_first_tab_only():
    got = parse_topics(["q1\tsome\tquestion", "", "q2\tx"])
    assert got == {"q1": "some\tquestion", "q2": "x"}


def test_miracl_metrics_hand_computed_rankings():
    gold = {"a": {"d1"}, "b": {"d9"}}
    ranked = {"a": ["d1", "x", "y"], "b": ["x", "y", "d9"]}
    m = miracl_metrics(gold, ranked, ks=(1, 3))
    assert m["recall@1"] == pytest.approx(0.5)
    assert m["recall@3"] == 1.0
    assert m["mrr@10"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-4)
    # ndcg: perfect for 'a'; for 'b' gold at rank3 -> gain 1/log2(4)=0.5
    assert m["ndcg@10"] == pytest.approx((1.0 + 0.5) / 2, abs=1e-4)


def test_miracl_metrics_never_divides_by_empty_ranking():
    m = miracl_metrics({"a": {"d1"}}, {"a": []})
    assert m["recall@1"] == 0.0 and m["mrr@10"] == 0.0 and m["ndcg@10"] == 0.0


# ------------------------------------------------------------- artefact identity

def test_run_dirs_cannot_collide_across_arms_or_slice_sizes():
    """The tripwire-overwrote-the-headline failure class, applied to frbench."""
    a = run_dir_for("miracl", "intfloat-multilingual-e5-base-768d", 5000, "dev")
    b = run_dir_for("miracl", "hash-256-35", 5000, "dev")
    c = run_dir_for("miracl", "intfloat-multilingual-e5-base-768d", 1000, "dev")
    d = run_dir_for("xnli", "pairclassifier", None, "test")
    assert len({a, b, c, d}) == 4
    assert c.name.endswith("_n1000") and d.name.endswith("_full")
    assert embed_tag("a/b c:d") == "a-b-c-d"
    assert embed_tag("intfloat/multilingual-e5-base") == "intfloat-multilingual-e5-base"


# --------------------------------------------- NLI head metrics (pure functions)

def test_macro_f1_and_confusion_hand_computed():
    """The head's own ruler, before any checkpoint exists. One supports->unrelated
    error on four rows: supports P=1 R=.5 F1=.667, unrelated P=.5 R=1 F1=.667,
    contradicts perfect, macro=(.667+.667+1)/3, accuracy 3/4. If the aggregation
    were wrong, every number the chain prints afterwards is wrong with it -- and
    the first draft of this test proved the point on itself: the hand-expected
    0.5/1.0 was the arithmetic error, the function's 2/3/2/3 was correct."""
    from engine.link.nli_finetune import LABELS, macro_f1_and_confusion

    rep = macro_f1_and_confusion([0, 0, 1, 2], [0, 1, 1, 2])
    assert rep["per_class"]["supports"]["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep["per_class"]["supports"]["precision"] == 1.0
    assert rep["per_class"]["supports"]["recall"] == 0.5
    assert rep["per_class"]["unrelated"]["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep["per_class"]["contradicts"]["f1"] == 1.0
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3, abs=1e-3)
    assert rep["accuracy"] == pytest.approx(0.75)
    assert rep["confusion"]["supports"] == {"supports": 1, "unrelated": 1}
    assert rep["confusion"]["unrelated"] == {"unrelated": 1}
    assert list(LABELS) == ["supports", "unrelated", "contradicts"]


def test_xnli_label_map_matches_the_heads_class_order():
    """The training labels and the eval labels must be the same ordering, or the
    confusion matrix reports one class as another (the reader-gold triplication
    failure class)."""
    from engine.link.nli_finetune import LABELS, XNLI2OUR

    assert [LABELS[XNLI2OUR[i]] for i in (0, 1, 2)] == ["supports", "unrelated",
                                                         "contradicts"]
