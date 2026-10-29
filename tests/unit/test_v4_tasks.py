"""v4 benchmark data and metrics: what the data can answer, and the baselines."""
from engine.eval.v4_data import (card, citation_set, load_change_labels,
                                 load_edges, load_meetings)
from engine.eval.v4_tasks import auroc, macro_f1, prf
from engine.link.relation_data import build_gold_rows


def test_edges_are_the_gold_annotation_and_cover_three_types():
    edges = load_edges()
    flat = [e for v in edges.values() for e in v]
    assert len(flat) == 114
    kinds = {e.relation for e in flat}
    assert kinds == {"supports", "supersedes", "contradicts"}
    assert sum(e.relation == "supersedes" for e in flat) == 40


def test_change_labels_are_turn_level_and_imbalanced():
    labels = load_change_labels()
    assert sum(len(v) for v in labels.values()) == 1475
    assert sum(sum(v) for v in labels.values()) == 187
    assert len(labels) == 6


def test_every_supersede_target_is_labelled_a_change():
    edges = load_edges()
    labels = load_change_labels()
    for mid, es in edges.items():
        for e in es:
            if e.relation == "supersedes":
                assert labels[mid][e.new] == 1


def test_citation_set_accepts_paraphrases_of_the_replaced_value():
    meetings = load_meetings()
    edges = load_edges()
    e = next(e for es in edges.values() for e in es
             if e.relation == "supersedes")
    valid = citation_set(meetings, e.meeting, e.old, e.new)
    assert e.old in valid
    # every member states the same point and value, and is earlier
    from engine.read.v4_tags import read_tags
    texts = [str(t.get("text") or "") for t in meetings[e.meeting]]
    ref = read_tags(texts[e.old])
    for j in valid:
        assert j < e.new
        tag = read_tags(texts[j])
        assert tag.slot == ref.slot
        assert tag.value.strip().lower() == ref.value.strip().lower()


def test_gold_relation_rows_never_use_machine_labels_and_add_unrelated():
    meetings = load_meetings()
    edges = load_edges()
    rows = build_gold_rows(meetings, edges)
    assert sum(r.relation == "unrelated" for r in rows) > 0
    assert all(r.source in ("gold", "negative") for r in rows)
    assert sum(r.relation == "supersedes" for r in rows) == 40


def test_card_reports_what_a_reader_needs():
    c = card(load_meetings(), load_edges(), load_change_labels())
    assert c["turns"] == 1475 and c["changes"] == 187 and c["edges"] == 114
    assert c["edge_types"]["supersedes"] == 40
    assert len(c["per_meeting"]) == 6


def test_precision_recall_f1_on_a_known_case():
    m = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                 "tp": 1, "fp": 1, "fn": 1}


def test_auroc_is_half_for_a_constant_score_and_one_for_a_perfect_one():
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_macro_f1_punishes_ignoring_a_class():
    # gold [0,0,1], pred [0,0,0]: class 0 gets tp=2 fp=1 -> F1 0.8,
    # class 1 gets F1 0, class 2 gets F1 0 -> macro 0.8/3.
    assert macro_f1([0, 0, 1], [0, 0, 0], ["a", "b", "c"]) == 0.2667
    assert macro_f1([0, 0, 1], [0, 0, 0], ["a", "b", "c"]) < 0.5
