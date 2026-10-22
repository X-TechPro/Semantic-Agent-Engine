"""§6e (report.py subjects_section) — pure render tests, no files, no GPU.

The rules this pins, one per project trap that already burned this repo:
generated docs must compute claims from artefacts (no hardcoded state), a
measured 0.0 must render as 0 and never become "not measured" (falsy-zero),
the honest empty state must say WHO decides (dev filter refused -> gate not
run), and the gate thresholds print from the artefact, not from a copy in
prose that can drift from the registered bar.
"""
from __future__ import annotations

import pytest

from eval.report import subjects_section

pytestmark = pytest.mark.unit


def test_no_corpus_renders_honest_instruction_not_silence():
    lines = subjects_section({})
    text = "\n".join(lines)
    assert "## 6e" in text
    assert "summre_label" in text          # tells the reader how to fix it
    assert "Corpus:" not in text           # nothing invented below the header


def test_zero_renders_as_zero_and_missing_as_not_measured():
    c = {
        "subject_corpus": [
            {"meeting": "024c_EBPH", "split": "train", "n_segments": 251,
             "drawers": 23},
            {"meeting": "039b_EDDH", "split": "measure", "n_segments": 341,
             "drawers": 9},
        ],
        "drawer_attempts": [{
            "file": "result-x-full-5e.json", "mode": "full", "seed": 42,
            "vocab_n": 57, "n_train": 1120, "saved": None,
            "failed": ["dev same-thread consistency 0.4403 < 0.6"],
            "dev": {"consistency": {
                "same_thread": {"rate": 0.0},        # a measured zero, kept
                "cross_thread": {"rate": None}},      # empty denominator
                "null_accuracy": 0.6458},
        }],
        # span head HAS an external gate artefact; the drawer classifier does
        # not (no checkpoint existed to judge) — the two states differ:
        "span_gate": {"consistency_supervised": {"gate": {
            "pass": False, "worst_same_thread": 0.0, "worst_cross_thread": None,
            "pair_coverage": 0.15,
            "thresholds": {"same_thread": 0.8, "cross_thread": 0.9,
                           "pair_coverage": 0.9}}}},
        "drawer_gate": {},
        "reach_run": {"reach": {
            "v2": {"n_pairs": 10, "marked_superseded": 0, "subject_resolved": 17,
                   "store_superseded_total": 0,
                   "ranking": {"new_above_old": 0.1}},
            "v2s": {"n_pairs": 10, "marked_superseded": 0,
                    "subject_resolved": 155, "store_superseded_total": 1,
                    "ranking": {"new_above_old": 0.0}},
        }},
    }
    text = "\n".join(subjects_section(c))
    # corpus rendered from the (declared) splits, never hardcoded prose
    assert "train 251 segments / 1 meeting(s)" in text
    assert "measure 341 segments / 1 meeting(s)" in text
    # falsy-zero: the 0.0 rate renders as 0, NOT "not measured"
    assert "| 0 | not measured | 0.6458 |" in text
    # the gate row prints the artefact's own thresholds
    assert "gate **FAIL**" in text and "0.8" in text and "0.9" in text
    assert "not run" in text               # drawer classifier: refusal stated
    # end-to-end table keeps real zeros visible (0/10, 0 edges, newer-on-top 0)
    assert "0/10" in text
    assert "| `v2s` | 155 | 0/10 | 1 | 0 |" in text


def test_saved_checkpoint_wording_switches_outcome_column():
    c = {"subject_corpus": [{"meeting": "m", "split": "train",
                             "n_segments": 10, "drawers": 2}],
         "drawer_attempts": [{"file": "f.json", "seed": 42, "vocab_n": 2,
                              "n_train": 10, "saved": "models/drawer",
                              "failed": [], "dev": {}}]}
    text = "\n".join(subjects_section(c))
    assert "saved — see the gate row" in text
    assert "external gate **not run**" in text   # saved but never gated: honest


def test_demo_real_section_renders_verdicts_and_empty_state():
    from eval.report import demo_real_section
    text = "\n".join(demo_real_section({}))
    assert "make demo-real" in text                 # honest empty state
    art = {"meetings": ["024c_EBPH"], "reader": "models/reader_v2",
           "embed": "e5-base", "engine_backends": {"subject_resolver": "gold"},
           "corpus_stats": {"n_units": 707}, "supersede_edges": 0,
           "llm_answer": "ollama qwen3.5:4b (verify-gated rewrite)",
           "llm_notes": {"answer": ["llm:nothing composed — template stands"],
                          "breakdown": ["llm:composed+verified"]},
           "checks": [{"name": "3. supersede réel", "pass": False,
                       "detail": "edges=0"},
                      {"name": "1. décision prouvée", "pass": True, "detail": ""}]}
    text = "\n".join(demo_real_section(art))
    assert "**FAIL**" in text and "edges=0" in text  # verdicts printed as recorded
    assert "PASS" in text and "1/2 checks passed" in text
    assert "hand-labelled drawer gold" in text       # annotation stated, not softened
    # a measured zero renders as 0, never as "not measured" (falsy-zero rule)
    assert "supersede edges on real speech 0" in text
    assert "resolver read back `gold`" in text
    # the LLM box: configured state AND both recorded outcomes (composed vs
    # bounced) — an outcome the file holds must appear in the file's rendering
    assert "ollama qwen3.5:4b" in text
    assert "llm:composed+verified" in text
    assert "nothing composed" in text
