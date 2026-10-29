"""Typed thread graph: nodes are turns, edges are learned relations.

A node can carry several edges, so it is not forced into one event cluster. A
component is a convenience for "what is the current turn in this thread"; the
typed edges remain the source of truth and preserve evidence/confidence.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from engine.link.relation_scorer import PairMeta, RelationScorer, features


@dataclass
class GraphNode:
    node_id: str
    project: str
    turn: int
    text: str
    speaker: str = ""
    timestamp: str = ""
    embedding: np.ndarray | None = None


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: float
    evidence: str


@dataclass
class ThreadGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    _adj: dict[str, list[GraphEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        self._adj[edge.source].append(edge)
        self._adj[edge.target].append(edge)

    def neighbors(self, node_id: str) -> list[GraphEdge]:
        return list(self._adj.get(node_id, []))

    def component(self, node_id: str) -> list[GraphNode]:
        seen = {node_id}
        dq = deque([node_id])
        while dq:
            cur = dq.popleft()
            for e in self._adj.get(cur, []):
                other = e.target if e.source == cur else e.source
                if other not in seen:
                    seen.add(other); dq.append(other)
        return [self.nodes[i] for i in seen if i in self.nodes]

    def current(self, node_id: str) -> GraphNode | None:
        comp = self.component(node_id)
        if not comp:
            return None
        return max(comp, key=lambda n: (n.timestamp, n.turn))

    def edges_of(self, node_id: str, relation: str | None = None) -> list[GraphEdge]:
        return [e for e in self._adj.get(node_id, [])
                if relation is None or e.relation == relation]


def build_thread_graph(project: str, turns: list[dict], embeddings: np.ndarray,
                       scorer: RelationScorer, *, top_k: int = 16,
                       threshold: float = 0.5) -> ThreadGraph:
    """Retrieve prior candidates by cosine, then keep typed learned edges.

    A message can have edges to more than one prior event; no single cluster
    assignment is made here.
    """
    g = ThreadGraph()
    E = np.asarray(embeddings, dtype=np.float32)
    for i, turn in enumerate(turns):
        g.add_node(GraphNode(
            node_id=f"{project}:{i}", project=project, turn=i,
            text=str(turn.get("text") or ""),
            speaker=str(turn.get("speaker") or ""),
            timestamp=str(turn.get("timestamp") or turn.get("date") or ""),
            embedding=E[i]))
    for i in range(len(turns)):
        if i == 0:
            continue
        prior = np.arange(i)
        sims = E[prior] @ E[i]
        order = prior[np.argsort(-sims)[:max(1, top_k)]]
        xs, metas = [], []
        for j in order:
            xs.append(features(
                str(turns[i].get("text") or ""), str(turns[j].get("text") or ""),
                E[i], E[j],
                PairMeta(project, str(turns[i].get("speaker") or ""), i,
                         str(turns[i].get("timestamp") or "")),
                PairMeta(project, str(turns[j].get("speaker") or ""), int(j),
                         str(turns[j].get("timestamp") or ""))))
            metas.append(int(j))
        scores = scorer.score_many(xs)
        for j, s in zip(metas, scores):
            if s.relation == "unrelated" or s.p_same < threshold:
                continue
            g.add_edge(GraphEdge(
                source=f"{project}:{j}", target=f"{project}:{i}",
                relation=s.relation, confidence=s.p_same,
                evidence=str(turns[j].get("text") or "")[:240]))
    return g


__all__ = ["GraphNode", "GraphEdge", "ThreadGraph", "build_thread_graph"]
