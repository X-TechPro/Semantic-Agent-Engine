"""Decision nodes: a view over immutable meeting messages.

The mentor's rule (2026-10-25): a meeting is a source of truth about what was
said at a precise date. It must never be rewritten by later information. So a
node is NOT a mutable unit: it is a grouping of messages, ordered in time, and
its "current" value is simply the latest child. Earlier children remain exactly
what was said then, with their own date and text.

One node per (project, topic) in a finished session. Children are the messages
grouped as the same thread/continuation; cross-session children can be appended
in a later session, and `as_of(date)` returns the state at that date without
touching any message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from engine.link.thread import ThreadIndex
from engine.schemas import Unit
from engine.threads.state import DecisionState
from engine.threads.state_reader import StateReader


@dataclass
class DecisionNode:
    node_id: str
    project: str | None
    topic: str
    children: list[Unit] = field(default_factory=list)
    #: last turn that confirmed/cancelled this state point (state nodes only)
    current_turn: int | None = None
    current_value: str | None = None

    @property
    def latest(self) -> Unit | None:
        return self.children[-1] if self.children else None

    def as_of(self, day: date) -> Unit | None:
        """What the node said on `day`; never mutates a child."""
        seen = [u for u in self.children if u.source_date <= day]
        return seen[-1] if seen else None

    def at_turn(self, turn_index: int) -> Unit | None:
        """What the node said at a turn of the same meeting (same-day order)."""
        seen = [u for u in self.children
                if (u.turn_index if u.turn_index is not None else -1) <= turn_index]
        return seen[-1] if seen else None

    def add(self, *units: Unit) -> "DecisionNode":
        """Append later messages; older children stay as they were."""
        for u in units:
            if all(c.unit_id != u.unit_id for c in self.children):
                self.children.append(u)
        self.children.sort(key=lambda u: (u.source_date, u.turn_index or 0))
        return self


def _state_nodes(units: list[Unit], reader: StateReader) -> list[DecisionNode]:
    """Second node set keyed by explicit state point, not by topic.

    A turn can be a continuation of a state point even when it never repeats the
    topic word. These nodes are additive: the topic nodes still exist, and the
    state nodes only claim a turn when the updater can name the point.
    """
    from engine.threads.state_reader import fold

    state = DecisionState()
    by_key: dict[tuple[str | None, str], DecisionNode] = {}
    for u in sorted(units, key=lambda x: (x.source_date, x.turn_index or 0)):
        turn = int(u.turn_index or 0)
        updates = reader.update(u.text, turn, state)
        points = set(reader.values(fold(u.text)))
        if not points and state.active and reader.continuation(u.text):
            points = {state.active}
        state.apply_all(updates)
        for point in points:
            key = (u.project, point)
            node = by_key.get(key)
            if node is None:
                node = by_key[key] = DecisionNode(
                    node_id=f"{u.project or 'global'}:state:{point}",
                    project=u.project, topic=point)
            node.add(u)
            confirmed = [x for x in updates
                         if x.point == point and x.status in
                         ("confirmed", "cancelled")]
            if confirmed:
                node.current_turn = turn
                node.current_value = confirmed[-1].value
    return sorted(by_key.values(), key=lambda n: (-len(n.children), n.node_id))


def build_session_nodes(units: list[Unit], *, merge_topic: bool = True,
                        state_reader: StateReader | None = None
                        ) -> list[DecisionNode]:
    """Group one finished session's messages into (project, topic) nodes.

    Grouping is deterministic: the persistent topic tracker names a topic when
    it is said, carries it across the turns that never repeat it, and a unit
    with no topic yet is left out rather than guessed. Every message keeps its
    text, date and turn index.

    `state_reader` adds a second node set keyed by explicit decision point; it
    is off by default so existing callers and tests keep the topic-only view.
    """
    index = ThreadIndex(window=64, topic_window=10**9)
    nodes: list[DecisionNode] = []
    if not merge_topic:
        # A thread is a CONTIGUOUS run: when the conversation moves to another
        # topic and later returns, that is a new node, not the same decision.
        run: list[Unit] = []
        run_topic: str | None = None

        def flush() -> None:
            if run and run_topic:
                nodes.append(DecisionNode(
                    node_id=f"{(run[0].project or 'global')}:{run_topic}:"
                            f"{run[0].turn_index}",
                    project=run[0].project, topic=run_topic,
                    children=list(run)))
            run.clear()

        for u in sorted(units, key=lambda x: (x.source_date, x.turn_index or 0)):
            index.add(u)
            topic = index.topic_of(u)
            if topic != run_topic:
                flush()
                run_topic = topic
            if topic:
                run.append(u)
        flush()
        if state_reader is not None:
            nodes += _state_nodes(units, state_reader)
        return sorted(nodes, key=lambda n: (-len(n.children), n.node_id))

    by_topic: dict[tuple[str | None, str], DecisionNode] = {}
    for u in sorted(units, key=lambda x: (x.source_date, x.turn_index or 0)):
        index.add(u)
        topic = index.topic_of(u)
        if not topic:
            continue
        key = (u.project, topic)
        node = by_topic.get(key)
        if node is None:
            node = by_topic[key] = DecisionNode(
                node_id=f"{(u.project or 'global')}:{topic}",
                project=u.project, topic=topic)
        node.add(u)
    out = list(by_topic.values())
    if state_reader is not None:
        out += _state_nodes(units, state_reader)
    return sorted(out, key=lambda n: (-len(n.children), n.node_id))


__all__ = ["DecisionNode", "build_session_nodes"]
