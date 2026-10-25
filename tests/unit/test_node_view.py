"""v3: the node view selects, never rewrites."""
from datetime import date, datetime, timezone

from engine.schemas import Slots, Unit
from engine.threads.view import NodeView


def _u(i: int, month: int, day: int, text: str, *, project: str = "atlas") -> Unit:
    return Unit(unit_id=f"u-{i}", source_id="meeting-1", source_type="meeting",
                source_date=date(2026, month, day),
                ingested_at=datetime(2026, month, day, tzinfo=timezone.utc),
                text=text, sentences=[], project=project, slots=Slots(),
                turn_index=i)


UNITS = [
    _u(0, 1, 1, "on part sur une billetterie en ligne"),
    _u(1, 2, 2, "finalement on ajoute une billetterie sur place"),
    _u(2, 3, 3, "le traiteur apporte les pizzas"),
]


def test_current_is_the_latest_child_and_as_of_reads_the_past():
    view = NodeView(UNITS)
    billetterie = next(n for n in view.nodes if n.topic == "billetterie")
    assert billetterie.latest.unit_id == "u-1"
    assert billetterie.as_of(date(2026, 1, 1)).unit_id == "u-0"
    assert billetterie.as_of(date(2026, 1, 15)).unit_id == "u-0"
    assert billetterie.at_turn(0).unit_id == "u-0"
    assert billetterie.at_turn(1).unit_id == "u-1"


def test_find_returns_the_current_child_without_mutating_any_unit():
    before = [(u.unit_id, u.text, u.superseded_by) for u in UNITS]
    match = NodeView(UNITS).find("où en est la billetterie ?")[0]
    assert match.current.unit_id == "u-1"
    assert match.node.topic == "billetterie"
    after = [(u.unit_id, u.text, u.superseded_by) for u in UNITS]
    assert before == after


def test_as_of_before_the_node_exists_returns_no_current():
    view = NodeView(UNITS)
    match = view.find("billetterie", as_of=date(2025, 12, 31))
    assert match == []


def test_project_filter_and_topic_boost():
    view = NodeView(UNITS + [_u(3, 4, 4, "billetterie du projet boreal", project="boreal")])
    matches = view.find("billetterie", project="boreal")
    assert [m.node.project for m in matches] == ["boreal"]


def test_agent_prepends_node_hits_and_deduplicates():
    from engine.agent.router import Agent
    from engine.schemas import Hit

    a = Hit(unit=UNITS[0], score=1.0, evidence="", path=[])
    b = Hit(unit=UNITS[1], score=2.0, evidence="", path=[])
    out = Agent._with_nodes([b], [a, b])
    assert [h.unit.unit_id for h in out] == ["u-1", "u-0"]
