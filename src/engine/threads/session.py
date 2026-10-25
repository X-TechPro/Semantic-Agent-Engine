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


@dataclass
class DecisionNode:
    node_id: str
    project: str | None
    topic: str
    children: list[Unit] = field(default_factory=list)

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


def build_session_nodes(units: list[Unit], *, merge_topic: bool = True) -> list[DecisionNode]:
    """Group one finished session's messages into (project, topic) nodes.

    Grouping is deterministic: the persistent topic tracker names a topic when
    it is said, carries it across the turns that never repeat it, and a unit
    with no topic yet is left out rather than guessed. Every message keeps its
    text, date and turn index.
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
    return sorted(by_topic.values(), key=lambda n: (-len(n.children), n.node_id))


__all__ = ["DecisionNode", "build_session_nodes"]
