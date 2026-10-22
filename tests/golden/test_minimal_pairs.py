"""Run the 12.3 golden minimal-pair suites against the shipped configuration.

The suites in `pairs_*.jsonl` are GENERATED (see `engine.tools.gen_golden_pairs`)
and their gold comes from IDEA.md 6.8.3 / 12.3 / line 403 — never from what the
code currently does. So a failure here is one of two things, and both are worth
knowing: an engine defect, or a fixture that over-reached what the guide actually
commits to. This test does not decide which in advance.

Rows are aggregated per category rather than parametrised one-by-one: 9,640
params would flood the report and hide the pattern. The message carries the
divergence count, the guards involved and the first examples, which is what tells
you whether one rule is wrong or the vocabulary is.

`informative` gold (12.3's apple test) is asserted as NOTHING — the guide names it
informative, "not a gate" — and only reported, so a future run can see the trend
without a red suite.

12.3's last two categories, unanswerable-question and ambiguous-recipient, are
scenario-level and are exercised in `tests/e2e/` instead; they cannot be expressed
as one utterance or one pair.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from engine.config import load_config
from engine.link.supervised import build_linker
from engine.read.baseline_rules import RuleReader
from engine.schemas import Slots
from helpers import make_unit

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).resolve().parent
MIN_PAIRS = 30
#: 12.3's apple category: report the relation, never assert it.
NOT_A_GATE = "informative"

#: the keys `Slots` actually has; the fixtures carry `project` alongside slots
#: for readability, and `Slots` is extra="forbid".
SLOT_KEYS = tuple(Slots.model_fields)


def _load() -> dict[str, list[dict]]:
    suites: dict[str, list[dict]] = {}
    for path in sorted(GOLDEN.glob("pairs_*.jsonl")):
        cat = path.name[len("pairs_"):-len(".jsonl")]
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows, f"{path.name} is empty"
        suites[cat] = rows
    assert suites, f"no pairs_*.jsonl in {GOLDEN}: run `make golden`"
    return suites


SUITES = _load()


def _unit(row: dict, *, uid: str, source_type: str = "note"):
    # The fixtures flatten the slot fields into the row (see `_pair`), so the
    # slot keys are picked out of the row itself. Reading a nested `row["slots"]`
    # instead silently built every pair with an empty Slots, which made
    # `same_slot` unreachable and turned 692 harness bugs into fake engine bugs.
    slots = {k: v for k, v in row.items() if k in SLOT_KEYS}
    return make_unit(row["text"], unit_id=uid, source_date=row["date"],
                     kind=row.get("kind", "info"), type=row.get("type", "fact"),
                     status=row.get("status", "confirmed"),
                     project=row["project"], source_type=source_type,
                     slots=Slots(**slots))


def _linker():
    """The shipped configuration, not test-local knobs.

    Reading the defaults through `build_linker` is the point: an inert-knob bug
    once made two benchmark arms byte-identical, so the suite must exercise the
    same construction path the engine uses.
    """
    cfg = load_config()
    linker, backend = build_linker(cfg)
    assert backend == "rules", f"suite assumes the rules linker, got {backend}"
    return linker


def _fmt(rows: list[tuple], n: int = 6) -> str:
    return "\n".join(f"    {r}" for r in rows[:n])


class TestSuiteIntegrity:
    def test_every_category_meets_the_12_3_floor(self):
        short = {c: len(r) for c, r in SUITES.items() if len(r) < MIN_PAIRS}
        assert not short, f"12.3 requires >= {MIN_PAIRS} per category: {short}"

    def test_categories_cover_the_12_3_list(self):
        """12.3's pair/label categories, by name, so none quietly disappears."""
        expected = {"status", "person", "temporal", "out_of_order", "value_flips",
                    "ratification", "repeated", "hypothesis_after_decision",
                    "same_topic_other_attribute", "cross_project",
                    "cancel_reinstate", "apple"}
        assert set(SUITES) == expected

    def test_committed_jsonl_still_matches_its_generator(self):
        """A fixture that drifts from its generator is a silent second source."""
        from engine.tools.gen_golden_pairs import generate_all

        fresh = generate_all()
        assert fresh.keys() == SUITES.keys(), (
            f"new or missing suite: {set(fresh) ^ set(SUITES)} — run `make golden`")
        stale = [c for c, rows in fresh.items() if rows != SUITES[c]]
        assert not stale, f"{stale} differ from the generator; run `make golden`"

    def test_no_row_claims_a_label_the_schema_forbids(self):
        from engine.schemas import RELATIONS, STATUSES, TYPES

        for cat, rows in SUITES.items():
            for r in rows:
                if r["kind"] == "pair":
                    assert r["gold"] in set(RELATIONS) | {NOT_A_GATE,
                                                          "not_supersedes"}, \
                        f"{cat}: unknown gold relation {r['gold']!r}"
                    for shape in ("new", "old"):
                        assert r[shape]["type"] in TYPES, f"{cat}: bad type"
                        assert r[shape]["status"] in STATUSES, f"{cat}: bad status"
                else:
                    # `label` rows carry a dict of field -> value. `not_status` is
                    # also allowed: 12.3 sometimes promises only the negative
                    # ("negation -> not confirmed"), and inventing a specific
                    # label there is how I manufactured 522 fake failures.
                    assert set(r["gold"]) <= {"status", "not_status", "owner"}, \
                        f"{cat}: unexpected label fields {set(r['gold'])}"
                    for k, v in r["gold"].items():
                        if k in ("status", "not_status"):
                            assert v in STATUSES, f"{cat}: bad status {v!r}"
                        elif v is not None:
                            assert isinstance(v, str), f"{cat}: owner must be a name"


class TestReaderLabels:
    def test_status_labels_follow_the_guide(self):
        reader = RuleReader()
        bad: list[tuple] = []
        by_status = Counter()
        total = 0
        for cat, rows in SUITES.items():
            for r in rows:
                if r["kind"] != "label" or r.get("stage") != "reader":
                    continue
                total += 1
                u = _unit({"text": r["text"], "date": "2026-10-01",
                           "project": "Atlas", "type": "fact",
                           "status": "mention", "kind": "info"}, uid="u-label")
                got = reader.read(u).status
                gold = r["gold"]
                if "not_status" in gold:
                    ok = got != gold["not_status"]
                    want = f"not_{gold['not_status']}"
                else:
                    want = gold["status"]
                    ok = got == want
                if not ok:
                    bad.append((want, got, r["text"], r["why"]))
                    by_status[f"{want}->{got}"] += 1
        assert total >= 30, "the status category must have label rows"
        assert not bad, (
            f"{len(bad)}/{total} status rows disagree with 12.3 "
            f"({dict(by_status)}):\n{_fmt(bad)}")


class TestSlotOwners:
    def test_owner_slot_follows_the_guide(self):
        """12.3 person: an assignment/commitment binds, a mention does not."""
        from engine.slots import rules as slot_rules

        reader = RuleReader()
        bad: list[tuple] = []
        total = 0
        for cat, rows in SUITES.items():
            for r in rows:
                if r["kind"] != "label" or r.get("stage") != "slots":
                    continue
                total += 1
                roster = set(r.get("roster") or ())
                u = _unit({"text": r["text"], "date": "2026-10-01",
                           "project": r.get("project", "Atlas"), "type": "fact",
                           "status": "confirmed", "kind": "info"}, uid="u-person")
                reading = reader.read(u)
                decorated = u.model_copy(update={"kind": reading.kind,
                                                 "type": reading.type,
                                                 "status": reading.status})
                got = slot_rules.extract(decorated, roster).owner
                want = r["gold"]["owner"]
                if (got or None) != (want or None):
                    bad.append((want, got, r["text"], r["why"]))
        assert total >= 30, "the person category must have label rows"
        assert not bad, (
            f"{len(bad)}/{total} owner rows disagree with 6.3/12.3:\n{_fmt(bad)}")


class TestRelationPairs:
    def test_pairs_follow_the_guide_relation(self):
        linker = _linker()
        bad: list[tuple] = []
        by_kind = Counter()
        informative = Counter()
        guards = Counter()
        total = 0
        for cat, rows in SUITES.items():
            for i, r in enumerate(rows):
                if r["kind"] != "pair":
                    continue
                total += 1
                new = _unit(r["new"], uid=f"n-{cat}-{i}")
                old = _unit(r["old"], uid=f"o-{cat}-{i}")
                edges = [e for e in linker.link(new, [old]) if e.to_id == old.unit_id]
                got = edges[0].relation if edges else "unrelated"
                want = r["gold"]
                if want == NOT_A_GATE:
                    informative[f"{got}"] += 1
                    continue
                if want == "not_supersedes":
                    ok = got != "supersedes"
                else:
                    ok = got == want
                if not ok:
                    guard = edges[0].guard if edges else "no-edge"
                    bad.append((cat, want, got, guard, new.text, old.text, r["why"]))
                    by_kind[f"{want}->{got}"] += 1
                    guards[guard] += 1
        assert total >= 30, "the pair categories must have rows"
        assert not bad, (
            f"{len(bad)}/{total} pairs disagree with 6.8.3/12.3 "
            f"(by kind {dict(by_kind)}; guards {dict(guards)}):\n"
            + _fmt([(f"want={b[1]} got={b[2]} guard={b[3]} | {b[4]!r} vs {b[5]!r}"
                     f" | {b[6]}") for b in bad]))

    def test_apple_category_is_reported_not_gated(self):
        """12.3 names the apple test informative; the suite must not gate on it."""
        linker = _linker()
        seen = Counter()
        for i, r in enumerate(SUITES["apple"]):
            if r["kind"] != "pair":
                continue
            assert r["gold"] == NOT_A_GATE, "apple is not a gate in 12.3"
            new = _unit(r["new"], uid=f"an-{i}")
            old = _unit(r["old"], uid=f"ao-{i}")
            edges = [e for e in linker.link(new, [old]) if e.to_id == old.unit_id]
            seen[edges[0].relation if edges else "unrelated"] += 1
        # one claim does survive the "not a gate" wording: a false SUPERSEDE
        # between the brand sentence and the fruit sentence would delete evidence,
        # and 6.8 calls that the worst error in the project.
        assert "supersedes" not in seen, f"apple pairs produced supersedes: {dict(seen)}"
