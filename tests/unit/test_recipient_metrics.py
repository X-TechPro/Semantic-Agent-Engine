"""The recipient-resolution metrics (9.1) — pinned where a zero means "good".

`_recipient_metrics` computes the brief's three recipient numbers, and two of them
have a pass condition of exactly `0.0` / a denominator that can be empty. That is the
truthiness trap that produced three separate bugs in this project (D-018): `x or None`,
`if committed else 0.0`, or `f"{v:.0%}" if v else "n/a"` all look correct and all
either invent a zero where nothing was measured or hide a measured zero where the
result was good. These tests pin the distinction in both directions.

They also pin the *definition* choices, because a metric with no stated denominator
is unauditable: `email_missing` counts as committing to a person (6.12 separates
naming someone from inventing their address), and a case with no declared gold is
excluded from every denominator rather than counted as a silent pass.
"""

from __future__ import annotations

import pytest

from eval.e2e import (ASSERTED_STATUSES, RECIPIENT_GOLD, _pct, _recipient_metrics)

pytestmark = pytest.mark.unit


def _case(cid, name=None, status=None):
    return {"id": cid, "kind": "clarify", "q": "x", "ok": True, "detail": "",
            "recipient_name": name, "recipient_status": status}


def test_no_committed_recipient_yields_none_not_zero():
    """Empty numerator *and* empty denominator must be "nothing measured"."""
    m = _recipient_metrics([_case("clarify-1"), _case("clarify-2")])
    assert m["recipients_committed"] == 0
    assert m["recipient_precision_at_1"] is None, "a zero denominator must not become 0.0"
    assert m["false_recipient_rate"] is None
    # but the ambiguous denominator is real: two cases existed and both abstained
    assert m["ambiguous_recipient_cases"] == 2
    assert m["justified_abstention_rate"] == 1.0


def test_a_measured_zero_false_rate_survives_the_formatter():
    """The pass condition is 0.0. A truthiness formatter would print it as absent."""
    m = _recipient_metrics([_case("recipient-marie", "Marie", "resolved")])
    assert m["false_recipient_rate"] == 0.0
    assert m["recipient_precision_at_1"] == 1.0
    assert _pct(m["false_recipient_rate"]) == "0%"
    assert _pct(None) == "n/a"
    assert _pct(0.0) != _pct(None), "0.0 and 'absent' must not collapse"


def test_a_forced_guess_on_an_ambiguous_owner_is_counted_as_false():
    m = _recipient_metrics([_case("clarify-1", "Antoine", "resolved")])
    assert m["recipients_committed"] == 1
    assert m["recipients_correct"] == 0
    assert m["false_recipient_rate"] == 1.0
    assert m["justified_abstention_rate"] == 0.0
    # the offending case is named, so one guess cannot hide inside a rounded rate
    assert m["guessed_recipient_when_ambiguous"] == ["clarify-1"]


def test_email_missing_still_counts_as_committing_to_a_person():
    """6.12: naming someone and inventing their address are different errors."""
    assert "email_missing" in ASSERTED_STATUSES
    m = _recipient_metrics([_case("recipient-marie", "Marie", "email_missing")])
    assert m["recipients_committed"] == 1
    assert m["recipients_correct"] == 1
    wrong = _recipient_metrics([_case("recipient-marie", "Nadia", "email_missing")])
    assert wrong["false_recipient_rate"] == 1.0


def test_a_nameless_or_unresolved_recipient_is_not_a_commitment():
    for name, status in ((None, "resolved"), ("", "resolved"), ("Antoine", "not_found"),
                         ("Antoine", "ambiguous")):
        m = _recipient_metrics([_case("clarify-1", name, status)])
        assert m["recipients_committed"] == 0, (name, status)
        assert m["justified_abstention_rate"] == 1.0, (name, status)


def test_cases_without_declared_gold_are_excluded_from_every_denominator():
    """Silently counting them would let the metric grade itself."""
    m = _recipient_metrics([_case("stale-budget", "Marie", "resolved")])
    assert m["recipient_cases_labelled"] == 0
    assert m["recipients_committed"] == 0
    assert m["recipient_precision_at_1"] is None


def test_name_matching_is_case_and_whitespace_insensitive():
    m = _recipient_metrics([_case("recipient-marie", "  MARIE  ", "resolved")])
    assert m["recipients_correct"] == 1


def test_every_gold_case_id_exists_in_the_suite():
    """A typo in RECIPIENT_GOLD would silently shrink the denominator to nothing."""
    from eval.e2e import CASES

    ids = {c["id"] for c in CASES}
    missing = sorted(set(RECIPIENT_GOLD) - ids)
    assert not missing, f"gold declared for cases that do not exist: {missing}"


def test_precision_and_false_rate_share_one_denominator():
    """They must sum to 1 whenever the denominator is non-empty, or one of them is
    being computed over a different set than the other."""
    results = [_case("recipient-marie", "Marie", "resolved"),
               _case("draft-stays-draft", "Nadia", "resolved"),
               _case("clarify-1", "Antoine", "resolved"),
               _case("clarify-2")]
    m = _recipient_metrics(results)
    assert m["recipients_committed"] == 3
    assert m["recipient_precision_at_1"] + m["false_recipient_rate"] == pytest.approx(1.0)
    assert m["justified_abstention_rate"] == 0.5      # clarify-2 abstained, clarify-1 did not
