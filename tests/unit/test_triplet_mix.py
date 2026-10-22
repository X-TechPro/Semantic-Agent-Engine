"""Training triples for the embedder fine-tune (6.4) — the split that makes the
anti-forgetting probe mean anything.

`embed/hard_negatives.py` was 157 statements at 0% coverage, and two of its
properties are load-bearing for a number I have already reported:

* **Bucket disjointness by sentence.** `_bucket()` hashes the *text*, so
  `general_rows(BUCKET_TRAIN)` and `general_rows(BUCKET_PROBE)` cannot overlap.
  The fine-tune's retention score is drawn from the probe bucket, so if the split
  were by seed instead — shuffling one corpus twice — the probe would re-see the
  same sentences and measure memorisation as retention.
* **The general budget is derived, not fixed.** 6.4 asks for 20-30% general
  language *and* ≥25% constructed negatives. A fixed 1200-row cap makes those
  unsatisfiable at once (~180 constructed rows would sit near 12%), so
  `build()` computes the limit from the synthetic count and `mix_report` refuses
  the mix when either constraint breaks.

No network here: `general_rows` reads `_FR_CACHE`, which these tests fill with
synthetic rows — the real MASSIVE mirror is apache-2.0 and is exercised by the
fine-tune run itself, not by the unit suite.
"""

from __future__ import annotations

import pytest

from engine.embed.hard_negatives import (BUCKET_PROBE, BUCKET_TRAIN, CONSTRUCTED_MIN,
                                         GENERAL_SHARE, TripletData, _bucket,
                                         build, constructed_rows, general_rows,
                                         mix_report, qa_rows, sub_seed,
                                         unit_text_by_event)
from gen.ledger import build_ledger

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ledger():
    return build_ledger(n_projects=6, meetings_per_project=3, seed=42)


@pytest.fixture(scope="module")
def train_projects(ledger):
    import tempfile, pathlib
    from gen.splice import write_manifest

    man = write_manifest(pathlib.Path(tempfile.mkdtemp()) / "m.json",
                         ledger.projects, seed=42)
    return man["split"]["train"], man["split"]["dev"]


@pytest.fixture(autouse=True)
def fake_massive(monkeypatch):
    """Synthetic same-intent French pairs, split across both buckets.

    Two intents per row set so the pairing loop has something to do, and texts
    long enough to survive the >8-character filter — the filter is why the short
    junk rows are dropped, so the corpus has to include some.
    """
    rows = []
    for intent in ("alarm_set", "takeaway_order", "calendar_set"):
        for i in range(160):
            rows.append((intent, f"{intent} numéro {i} : ouvre l'application demain matin"))
    rows.append(("alarm_set", "court"))                  # below the length filter
    monkeypatch.setattr("engine.embed.hard_negatives._FR_CACHE", {"rows": rows})
    return rows


# ------------------------------------------------------------------- buckets

def test_bucket_is_deterministic_and_split_by_sentence():
    texts = [f"phrase numéro {i} pour tester le partage du corpus" for i in range(200)]
    assert {_bucket(t) for t in texts} == {BUCKET_TRAIN, BUCKET_PROBE}
    assert all(_bucket(t) == _bucket(t) for t in texts)
    both = [t for t in texts if _bucket(t) == BUCKET_TRAIN]
    probe = [t for t in texts if _bucket(t) == BUCKET_PROBE]
    assert both and probe, "a 4-way digest parity split on 200 texts is degenerate"
    assert not (set(both) & set(probe))


def test_general_rows_buckets_are_disjoint_by_pair(ledger):
    """The retention probe must not contain a sentence the fine-tune trained on."""
    tr_a, tr_p = general_rows(limit=40, seed=42, bucket=BUCKET_TRAIN)
    pr_a, pr_p = general_rows(limit=40, seed=42, bucket=BUCKET_PROBE)
    assert tr_a and pr_a
    assert not (set(tr_a) | set(tr_p)) & (set(pr_a) | set(pr_p)), \
        "train and probe buckets overlap: retention would measure memorisation"


def test_general_rows_pairs_same_intent_and_respects_limit():
    anchors, positives = general_rows(limit=12, seed=42, bucket=BUCKET_TRAIN)
    assert len(anchors) == len(positives) == 12
    for a, p in zip(anchors, positives):
        assert a != p
        assert a.split(" numéro")[0] == p.split(" numéro")[0], "paired across intents"


def test_general_rows_is_seed_deterministic():
    a1 = general_rows(limit=8, seed=42, bucket=BUCKET_TRAIN)
    a2 = general_rows(limit=8, seed=42, bucket=BUCKET_TRAIN)
    assert a1 == a2


# ------------------------------------------------------------- constructed / qa

def test_constructed_rows_are_cue_bearing_and_project_scoped(ledger, train_projects):
    train, dev = train_projects
    a, p, n = constructed_rows(ledger, set(train), 42, per_project=6)
    assert a and len(a) == len(p) == len(n)
    # a minimal pair must differ from its own positive, and the anchor must not be
    # the negative either: a row like (x, x, x) trains the encoder to score
    # everything equally, which is the failure mode 12.3 exists to catch
    for anchor, pos, neg in zip(a, p, n):
        assert pos != neg, "positive identical to its negative"
        assert anchor != neg, "anchor identical to its negative"
    # project scoping: rows from a different project set are a different draw
    a_dev, _pd, _nd = constructed_rows(ledger, set(dev) or set(train[:1]), 42,
                                       per_project=6)
    assert set(a_dev) != set(a) or not a_dev
    # more pairs per project cannot produce fewer rows
    assert len(constructed_rows(ledger, set(train), 42, per_project=12)[0]) >= len(a)


def test_qa_rows_use_the_indexed_text_not_a_regenerated_one(ledger, train_projects,
                                                            tmp_path):
    """`texts` comes from a real ingest. Feeding it a re-render would train the
    encoder on strings retrieval never indexed."""
    train, _dev = train_projects
    texts = unit_text_by_event(ledger, set(train), 42, tmp_path)
    assert texts, "the ingest must map at least one event to a stored unit"
    anchors, positives = qa_rows(ledger, set(train), texts, 42, per_project=6)
    assert anchors
    assert set(positives) <= set(texts.values())


def test_qa_rows_drop_questions_whose_gold_was_filtered_as_chatter(ledger, train_projects):
    train, _ = train_projects
    anchors, positives = qa_rows(ledger, set(train), {}, 42, per_project=6)
    assert anchors == [] and positives == []      # no indexed text -> no row


# --------------------------------------------------------------------- mix

def _data(constructed, general, qa=0):
    d = TripletData()
    d.anchors = ["a"] * (constructed + general + qa)
    d.positives = ["p"] * len(d.anchors)
    d.negatives = ["n"] * constructed + [""] * (general + qa)
    d.source = (["constructed"] * constructed + ["general_fr"] * general
                + ["qa"] * qa)
    return d


def test_mix_report_enforces_both_of_6_4s_constraints():
    ok = mix_report(_data(40, 20, 40))            # 40% constructed, 20% general
    assert ok["ok"] is True and ok["constructed_share"] >= CONSTRUCTED_MIN
    assert GENERAL_SHARE[0] <= ok["general_share"] <= GENERAL_SHARE[1]
    assert mix_report(_data(40, 80))["ok"] is False, "general share above the band"
    assert mix_report(_data(5, 25))["ok"] is False, "constructed below 25%"
    assert mix_report(_data(0, 0))["ok"] is False, "an empty mix must not pass"


def test_mix_report_counts_rows_with_a_real_negative():
    rep = mix_report(_data(30, 10))
    assert rep["constructed_negatives"] == 30
    assert rep["n_rows"] == 40


def test_triplet_rows_keep_the_four_columns_aligned():
    d = _data(2, 1)
    rows = d.rows()
    assert len(rows) == len(d) == 3
    assert [r["source"] for r in rows] == ["constructed", "constructed", "general_fr"]
    assert rows[0]["negative"] and not rows[2]["negative"]


# -------------------------------------------------------------------- build

def test_build_refuses_a_general_share_outside_the_guide_band(tmp_path):
    with pytest.raises(SystemExit, match="outside 6.4"):
        build(projects=6, seed=42, general_share=0.5, run_dir=tmp_path)


def test_build_refuses_a_mix_that_cannot_satisfy_both_rules(tmp_path, monkeypatch):
    """With the general rows removed the constructed share is ~100% and general is
    0% — outside the band, so the builder must fail loudly instead of training a
    mix the guide forbids."""
    with pytest.raises(SystemExit, match="violates 6.4"):
        build(projects=6, seed=42, with_general=False, run_dir=tmp_path)


def test_build_produces_a_mix_that_satisfies_both_constraints(tmp_path, train_projects):
    data = build(projects=6, seed=42, general_share=0.25, per_project=4,
                 run_dir=tmp_path)
    mix = data.meta["mix"]
    assert mix["ok"] is True, mix
    assert len(data.anchors) == len(data.positives) == len(data.negatives) \
        == len(data.source)
    # 9.5: neither dev nor test projects may appear in the training mix
    train, dev = train_projects
    assert set(data.meta["train_projects"]) == set(train)
    assert not set(data.meta["train_projects"]) & set(dev)


def test_sub_seed_is_tag_specific_and_stable():
    assert sub_seed(42, "massive") == sub_seed(42, "massive")
    assert sub_seed(42, "massive") != sub_seed(42, "shuffle")
    assert sub_seed(42, "massive") != sub_seed(43, "massive")
