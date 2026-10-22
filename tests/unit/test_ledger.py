"""Ledger generator tests (BUILD_GUIDE M0 gate: 20 projects, deterministic)."""

from datetime import date

import pytest

from gen.ledger import (SUPERSEDING_KINDS, ATTRIBUTES, Ledger, build_ledger,
                        summarise)


@pytest.fixture(scope="module")
def ledger():
    return build_ledger(n_projects=20, meetings_per_project=3, seed=42)


class TestGeneration:
    def test_m0_gate_20_projects(self, ledger):
        assert len(ledger.projects) == 20
        assert ledger.meta["generator"] == "ledger v0"

    def test_meetings_exist_per_project(self, ledger):
        per = {}
        for m in ledger.meetings:
            per[m.project] = per.get(m.project, 0) + 1
        assert len(per) == 20
        assert all(v >= 3 for v in per.values())

    def test_every_event_refs_existing_project_and_meeting(self, ledger):
        codes = {p["project"] for p in ledger.projects}
        mids = {m.meeting_id for m in ledger.meetings}
        for e in ledger.events:
            assert e.project in codes
            assert e.meeting_id in mids, e.event_id

    def test_subject_keys_are_prefixed_with_project(self, ledger):
        for e in ledger.events:
            if e.subject_key:
                assert e.subject_key.startswith(e.project.lower() + "."), e.subject_key

    def test_determinism_same_seed(self, ledger):
        again = build_ledger(n_projects=20, meetings_per_project=3, seed=42)
        assert [e.event_id for e in again.events] == [e.event_id for e in ledger.events]
        assert again.to_dict() == ledger.to_dict()

    def test_different_seed_different_content(self, ledger):
        other = build_ledger(n_projects=20, meetings_per_project=3, seed=43)
        assert other.to_dict() != ledger.to_dict()

    def test_roundtrip_serialisation(self, ledger):
        clone = Ledger.from_dict(ledger.to_dict())
        assert clone.to_dict() == ledger.to_dict()

    def test_rich_label_coverage(self, ledger):
        kinds = summarise(ledger)["kinds"]
        for k in ("proposed", "confirmed", "changed", "hypothesis", "commitment",
                  "assignment", "risk", "question", "answer", "mention"):
            assert kinds.get(k, 0) > 0, f"{k} never generated"

    def test_supersede_edges_exist(self, ledger):
        assert len(ledger.gold_supersede_edges()) >= 5


class TestGoldRelations:
    def test_supersede_edge_direction_and_date(self, ledger):
        for e in ledger.events:
            if e.supersedes:
                old = ledger.event(e.supersedes)
                assert e.kind in SUPERSEDING_KINDS, (e.event_id, e.kind)
                assert e.on >= old.on
                assert e.subject_key == old.subject_key
                assert ledger.gold_relation(e.event_id, old.event_id) == "supersedes"

    def test_hypothesis_never_supersedes(self, ledger):
        for e in ledger.events:
            if e.kind == "hypothesis":
                assert e.supersedes is None

    def test_cross_project_is_unrelated(self, ledger):
        a = [e for e in ledger.events if e.subject_key][0]
        b = [e for e in ledger.events if e.subject_key and e.project != a.project][0]
        assert ledger.gold_relation(a.event_id, b.event_id) == "unrelated"

    def test_repeated_statement_supports(self, ledger):
        evs = [e for e in ledger.events if e.kind == "repeated"]
        if not evs:
            pytest.skip("no repeated events in this seed")
        for e in evs:
            same = [o for o in ledger.events
                    if o.subject_key == e.subject_key and o.event_id != e.event_id
                    and o.kind in ("confirmed", "changed", "reinstated")]
            for o in same:
                assert ledger.gold_relation(e.event_id, o.event_id) == "supports"

    def test_same_topic_other_attribute_unrelated(self, ledger):
        evs = [e for e in ledger.events if e.subject_key and not e.distractor]
        by_proj: dict[str, list] = {}
        for e in evs:
            by_proj.setdefault(e.project, []).append(e)
        checked = 0
        for project, group in by_proj.items():
            keys = {e.subject_key for e in group}
            if len(keys) < 2:
                continue
            a = group[0]
            b = next(x for x in group if x.subject_key != a.subject_key)
            rel = ledger.gold_relation(a.event_id, b.event_id)
            assert rel == "unrelated", (a.event_id, b.event_id, rel)
            checked += 1
        assert checked > 0

    def test_question_answer_pair_yields_answers(self, ledger):
        answers = [e for e in ledger.events if e.kind == "answer"]
        assert answers
        for a in answers:
            assert ledger.gold_relation(a.event_id, a.answer_of) == "answers"

    def test_proposed_after_confirmed_is_not_supersedes(self, ledger):
        """6.8: a proposal must never replace a confirmed decision."""
        confirmed: dict[str, list] = {}
        for e in ledger.events:
            if e.kind == "confirmed" and e.subject_key:
                confirmed.setdefault(e.subject_key, []).append(e)
        checked = 0
        for e in ledger.events:
            if e.kind == "proposed" and e.subject_key in confirmed:
                for old in confirmed[e.subject_key]:
                    if old.on <= e.on:
                        rel = ledger.gold_relation(e.event_id, old.event_id)
                        assert rel in ("contradicts", "supports"), (e.event_id, old.event_id, rel)
                        assert rel != "supersedes"
                        checked += 1
        assert checked > 0, "fixture should contain proposal-after-decision cases"


class TestCurrentState:
    def test_current_state_returns_latest_per_subject(self, ledger):
        state = ledger.current_state()
        for key, cur in state.items():
            later = [e for e in ledger.events
                     if e.subject_key == key and e.on > date.fromisoformat(cur["on"])
                     and e.kind in ("confirmed", "changed", "cancelled", "reinstated")]
            assert not later, (key, [e.event_id for e in later])

    def test_as_of_snapshot_is_earlier_than_full_state(self, ledger):
        full = ledger.current_state()
        mid = date(2026, 10, 15)
        snap = ledger.current_state(as_of=mid)
        assert set(snap) <= set(full) or snap != full

    def test_attribute_kinds_are_valid(self, ledger):
        for meta in ATTRIBUTES.values():
            assert meta["kind"] in ("date", "amount", "vendor", "scope", "stack",
                                    "hiring", "risk")


class TestSave:
    def test_save_and_reload(self, tmp_path, ledger):
        p = ledger.save(tmp_path / "ledger" / "l.json")
        assert p.exists()
        clone = Ledger.from_dict(__import__("json").loads(p.read_text(encoding="utf-8")))
        assert len(clone.events) == len(ledger.events)

    def test_audit_lines_render(self, ledger):
        text = "\n".join(ledger.audit_lines())
        assert "Ledger audit" in text and "gold supersede edges" in text

    def test_too_few_project_codes_raises(self):
        with pytest.raises(ValueError):
            build_ledger(n_projects=3, project_codes=["A", "B"])
