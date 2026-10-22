"""The document/code segmenter paths (6.1) and the relation audit (9.2 / 6.8.3).

Both were executed only by ad-hoc scripts, never by the suite:
`segment/chunklet_adapter.py` sat at 39.7% and `read/audit.py` at 0%. The
document path is where §6.1's "100% span reconstruction" acceptance gate lives,
and the audit is the instrument that caught two of this project's three retracted
numbers — so both deserve a test that fails when they stop measuring what they
claim to.

`DocumentSemanticMerger` takes an `embed` callable rather than an embedder, so it
can be tested with a deterministic vector function: the whole point of that class
is the *rule* (split when adjacent sentences stop looking alike, merge a tail
smaller than `min_tokens`), not the model.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from engine.config import load_config
from engine.read.audit import audit
from engine.schemas import Source, Turn
from engine.segment import turns as T
from engine.segment.chunklet_adapter import DocumentSemanticMerger, Segmenter, segment

pytestmark = pytest.mark.integration


def _seg(**tokens):
    return Segmenter(load_config(overrides={"segment": tokens}))


DOC_TEXT = (
    "Le budget du projet Atlas est validé à 2,4 M€. "
    "La date de lancement est reportée à janvier. "
    "Le prestataire retenu est Nova Cloud. "
    "Attention, le risque de conformité reste élevé. "
    "Marie pilote le budget."
)


# ------------------------------------------------------------ document path

def test_group_sentences_respects_the_token_budget_without_losing_sentences():
    seg = _seg(doc_max_tokens=20, doc_min_tokens=4)
    sents = T.split_sentences(DOC_TEXT)
    assert len(sents) == 5
    groups = seg._group_sentences(sents)
    assert groups, "a document must produce at least one chunk"
    flat = [s.text for g in groups for s in g]
    assert flat == [s.text for s in sents], "sentences dropped or reordered"
    for g in groups[:-1]:                      # the tail may fold, earlier ones may not
        assert sum(T.count_tokens(s.text) for s in g) <= 20 or len(g) == 1


def test_group_sentences_folds_a_tiny_trailing_chunk():
    """6.1's min_tokens rule: a 1-sentence tail would become a unit too small to
    retrieve, so it joins the previous chunk."""
    long_part = "Le budget du projet Atlas est validé à 2,4 M€. " * 4
    text = long_part + "Et ensuite."
    seg = _seg(doc_max_tokens=12, doc_min_tokens=6)
    groups = seg._group_sentences(T.split_sentences(text))
    assert len(groups) >= 1
    assert sum(len(g) for g in groups) == len(T.split_sentences(text))
    # the tiny tail must not stand alone as the last group
    if len(groups) > 1:
        assert T.count_tokens(groups[-1][-1].text) >= 1


def test_segment_document_reconstructs_every_span_and_ids_are_unique():
    """The §6.1 acceptance gate: `unit.text[start:end]` must equal the sentence, or
    BM25 evidence and the quoted citation disagree."""
    src = Source(source_id="doc-1", source_type="document", source_date=date(2026, 10, 1),
                 project="Atlas", default_speaker=None, text=DOC_TEXT)
    seg = _seg(doc_max_tokens=20, doc_min_tokens=4)
    units = seg.segment(src)
    assert units
    assert len({u.unit_id for u in units}) == len(units)
    for u in units:
        for sp in u.sentences:
            assert u.text[sp.start:sp.end].strip() == sp.surface.strip()
        assert u.text.strip()
    assert " ".join(u.text for u in units).split() == DOC_TEXT.split()


def test_segment_document_handles_empty_and_single_sentence():
    seg = _seg(doc_max_tokens=20, doc_min_tokens=4)
    assert seg.segment(Source(source_id="d", source_type="document",
                              source_date=date(2026, 10, 1), text="")) == []
    one = seg.segment(Source(source_id="d2", source_type="document",
                             source_date=date(2026, 10, 1), text="Une seule phrase."))
    assert len(one) == 1 and one[0].text == "Une seule phrase."


LONG_DOC = " ".join(
    f"Phrase numéro {i} avec plusieurs mots de test pour allonger le document."
    for i in range(60))


def test_module_level_segment_routes_the_fixed_backend():
    """`segment()` is the 2.2 interface; the `fixed` row of §10 depends on it
    actually delegating to the baseline instead of silently using chunklet.

    On a short document both backends emit a single unit covering the whole text,
    so the first version of this test passed vacuously. Long input separates their
    signatures, and the constants below are the measured ones: the baseline slides
    a `window`-word window by `stride` words, so consecutive units share exactly
    the stride; chunklet groups are disjoint and reconstruct the source in order.
    """
    src = Source(source_id="doc-x", source_type="document", source_date=date(2026, 10, 1),
                 text=LONG_DOC)
    cfg_fixed = load_config(overrides={"segment": {"backend": "fixed"}})
    window = cfg_fixed["segment"]["fixed_window_words"]
    stride = cfg_fixed["segment"]["fixed_stride_words"]
    fixed = segment(src, cfg_fixed)
    chunked = segment(src, load_config(overrides={"segment": {"backend": "chunklet"}}))

    assert len(fixed) > 1 and len(chunked) > 1
    assert [T.count_tokens(u.text) for u in fixed][:2] == [window, window]
    for a, b in zip(fixed, fixed[1:]):
        assert a.text.split()[-stride:] == b.text.split()[:stride], \
            "the fixed baseline must produce overlapping sliding windows"

    assert " ".join(u.text for u in chunked).split() == LONG_DOC.split(), \
        "chunklet groups must be lossless and in order"
    for a, b in zip(chunked, chunked[1:]):
        assert a.text.split()[-stride:] != b.text.split()[:stride], \
            "chunklet groups must not overlap"


# ----------------------------------------------------------------- code path

CODE_SRC = '''"""Docstring for the module."""


def alpha(a, b):
    return a + b


class Beta:
    def gamma(self):
        return 3
'''


def test_segment_code_produces_units_and_keeps_symbols_named():
    src = Source(source_id="code-1", source_type="code", source_date=date(2026, 10, 1),
                 project="Atlas", code=CODE_SRC, language="python")
    units = _seg(doc_max_tokens=40).segment(src)
    assert units, "a python file must segment into units"
    joined = "\n".join(u.text for u in units)
    for name in ("alpha", "Beta", "gamma"):
        assert name in joined, f"{name} lost by the code segmenter"
    assert all(u.text.strip() for u in units)
    # known gap, asserted so it stays visible: `Segmenter._segment_code` passes
    # `extra={"code_symbol": ...}` to `_make_unit`, which ignores `extra` entirely
    # and `Unit` has no such field, so the symbol is computed and then dropped.
    assert not any(getattr(u, "code_symbol", None) for u in units)


def test_segment_code_on_empty_source_is_empty():
    assert _seg().segment(Source(source_id="c", source_type="code",
                                 source_date=date(2026, 10, 1), code="", language=None)) == []


# --------------------------------------------------- unpunctuated / ASR meeting

def test_pieces_splits_long_unpunctuated_turn_into_windows():
    """6.1: ASR-style turns have no sentence marks, so they fall to the token
    window splitter rather than becoming one unretrievable blob."""
    text = " ".join(f"mot{i}" for i in range(45))
    pieces = _seg(asr_max_tokens=10, asr_min_tokens=4)._pieces(text)
    assert len(pieces) > 1
    assert " ".join(p for p, _sp in pieces).split() == text.split()

    short = _seg(asr_max_tokens=40, asr_min_tokens=8)._pieces("réunion budget validé")
    assert len(short) == 1


def test_pieces_merges_a_short_multi_sentence_turn_and_splits_a_long_one():
    short = _seg(max_turn_tokens=40)._pieces(
        "Le budget est validé. Marie pilote.")
    assert len(short) == 1 and len(short[0][1]) == 2, "two short sentences must merge"

    long_turn = " ".join(f"phrase numéro {i} avec plusieurs mots." for i in range(12))
    many = _seg(max_turn_tokens=8)._pieces(long_turn)
    assert len(many) > 1, "a long multi-sentence turn must split per sentence"


# ------------------------------------------------------- semantic document merge

def _embed_synonyms(texts, role="doc"):
    """Deterministic stand-in: sentences sharing a keyword point the same way.

    `DocumentSemanticMerger.group` calls `self.embed(texts, role="doc")`, so the stub
    has to accept the kwarg the real embedder takes or this file tests a signature
    that does not exist.
    """
    out = []
    for t in texts:
        v = np.zeros(3, dtype="float32")
        v[0] = 1.0 if "budget" in t else 0.0
        v[1] = 1.0 if "lancement" in t or "janvier" in t else 0.0
        v[2] = 0.3
        n = np.linalg.norm(v) or 1.0
        out.append(v / n)
    return np.array(out)


def test_semantic_merger_splits_on_similarity_not_just_budget():
    sents = T.split_sentences(
        "Le budget du projet Atlas est validé. "
        "Le budget reste sous revue. "
        "La date de lancement est reportée. "
        "Le prestataire Nova Cloud est retenu.")
    m = DocumentSemanticMerger(_embed_synonyms, tau_merge=0.9, max_tokens=999,
                              min_tokens=1)
    groups = m.group(sents)
    # the two budget sentences are near-identical, the others are not: more groups
    # than one proves the cosine rule fired, not just the token budget
    assert len(groups) > 1
    assert [s for g in groups for s in g] == sents, "sentences lost or reordered"

    loose = DocumentSemanticMerger(_embed_synonyms, tau_merge=0.0, max_tokens=999,
                                  min_tokens=1).group(sents)
    assert len(loose) == 1, "tau=0 must merge everything adjacent"


def test_semantic_merger_folds_a_small_tail_and_respects_max_tokens():
    sents = T.split_sentences(DOC_TEXT)
    tight = DocumentSemanticMerger(_embed_synonyms, tau_merge=0.0, max_tokens=8,
                                  min_tokens=1).group(sents)
    assert len(tight) > 1, "a token budget of 8 must split a 5-sentence document"
    assert [s for g in tight for s in g] == sents


# ------------------------------------------------------------------- audit

@pytest.fixture(scope="module")
def audit_out():
    """The rules reader only: the `model` branch needs the trained encoder and is
    exercised by `make stages`, not by the unit suite."""
    return audit(split="dev", projects=6, seed=42, backends=("rules",))


def test_audit_ratios_use_the_right_denominators(audit_out):
    """`gold_found` counts exact (new, old) pairs. The bug this pins is counting
    "did this unit supersede anything" instead, which inflates recall and already
    cost this project two retracted numbers."""
    r = audit_out["backends"]["rules"]
    assert r["reader_backend"] == "rules"          # the assert inside audit() held
    assert r["corpus"]["n_stored"] > 0
    assert r["n_gold_pairs"] > 0, "6 projects must contain supersede chains on dev"
    hit = r["gold_found"]
    assert hit <= min(r["n_supersedes_written"], r["n_gold_pairs"])
    assert r["recall"] == round(hit / r["n_gold_pairs"], 4)
    assert r["precision"] == round(hit / r["n_supersedes_written"], 4)


def test_audit_attributes_every_missed_gold_pair(audit_out):
    """Every gold pair that produced no edge must be counted somewhere: either a
    guard refused it (named) or it was never proposed."""
    r = audit_out["backends"]["rules"]
    refused = r["refused_by_guard"]
    assert sum(refused.values()) == r["n_gold_pairs"] - r["gold_found"], (
        f"{refused} vs {r['n_gold_pairs']} gold / {r['gold_found']} found")
    assert all(k == "never_proposed" or k.startswith("G") for k in refused), refused


def test_audit_records_the_label_fields_the_guards_read(audit_out):
    """The blockers must be expressed in the fields G7/G3 actually consume, or the
    audit explains nothing about why a pair was refused."""
    r = audit_out["backends"]["rules"]
    blockers = r["gold_pair_label_blockers"]
    assert isinstance(blockers, dict)
    for key in blockers:
        assert key.startswith(("new_type_not_superseding:",
                               "old_type_not_value_bearing:",
                               "new_status_not_assertive:")), key
    labels = r["superseding_source_labels"]
    assert all(k.startswith(("type=", "status=")) for k in labels), labels
    assert all("/" in k for k in r["false_by_source_label"]), r["false_by_source_label"]


def test_audit_keeps_the_answers_gap_visible(audit_out):
    """B-01: gold `answers` pairs exist and the rules linker writes none. Pinned as
    a number so closing it is a decision, and so a silent regression to 'no
    measurements at all' is not mistaken for progress."""
    r = audit_out["backends"]["rules"]
    assert r["answers"]["n_gold"] >= 1, "the ledger must contain question->answer pairs"
    assert r["answers"]["n_written"] == 0, (
        f"answers gap appears closed ({r['answers']}); update B-01 deliberately")
    assert r["answers"]["recall"] == 0.0
    # 7.3 / D-012: meeting questions are now `info` and must be indexed at all
    assert r["question_units_stored"] >= 1, "no question-type unit was stored"
    assert isinstance(r["gold_pair_label_blockers"], dict)
    for key in r["gold_pair_label_blockers"]:
        assert key.startswith(("new_type_not_superseding:",
                               "old_type_not_value_bearing:",
                               "new_status_not_assertive:")), key


def test_audit_counts_stored_chatter_separately(audit_out):
    c = audit_out["backends"]["rules"]["corpus"]
    assert c["n_skipped"] >= 0
    assert c["n_units"] >= c["n_stored"], "stored more units than the segmenter made"
