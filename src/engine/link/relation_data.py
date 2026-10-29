"""§4 supervision: dated pairs with hard negatives, derived from real meetings.

The architecture asks for hard negatives of three shapes:

* same project, different value → `contradicts`;
* same value, different project or date → `contradicts`;
* proposal vs confirmed vs cancelled → status supervision.

Only real turns are used; nothing is templated. Every row carries the status of
both sides so the status head can be trained from the same pass. Rows are
grouped by meeting so leave-one-meeting-out can be honoured by the trainer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from engine.link.thread import detect_topic
from engine.read.v4_tags import Tags, read_tags

SELF_LABELS = ("supersedes", "answers", "supports", "creates_action",
               "contradicts", "unrelated")


@dataclass(frozen=True)
class Row:
    meeting: str
    query: str
    candidate: str
    relation: str
    query_status: str
    candidate_status: str
    query_slot: str
    candidate_slot: str
    source: str

    def key(self) -> tuple:
        return (self.meeting, self.query, self.candidate)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _tags(texts: list[str]) -> list[Tags]:
    return [read_tags(t) for t in texts]


def build_rows(
    meetings: dict[str, list[dict]],
    pairs: dict[str, list[tuple[str, int, int]]],
    *,
    mids: list[str] | None = None,
    hard_negatives: int = 3,
    answers_per_meeting: int = 12,
    actions_per_meeting: int = 8,
    supports_per_meeting: int = 8,
    seed: int = 42,
) -> list[Row]:
    import random

    rng = random.Random(seed)
    rows: list[Row] = []
    # only labelled meetings: an unlabelled or spent meeting must not donate
    # supervision just because it is present in the corpus file.
    for mid in (sorted(mids) if mids else sorted(pairs)):
        if mid not in meetings:
            continue
        texts = [str(t.get("text") or "") for t in meetings[mid]]
        if not texts:
            continue
        tags = _tags(texts)
        topics = [detect_topic(t) for t in texts]
        gold = {(old, new) for _topic, old, new in pairs.get(mid, [])}
        gold_by_new: dict[int, set[int]] = {}
        for old, new in gold:
            gold_by_new.setdefault(new, set()).add(old)

        # 1. gold antecedents: the only real positives we have
        for old, new in sorted(gold):
            if old >= len(texts) or new >= len(texts):
                continue
            rows.append(Row(mid, texts[new], texts[old], "supersedes",
                            tags[new].status, tags[old].status,
                            tags[new].slot, tags[old].slot, "gold"))

        # 2. hard negatives: same slot, different value, *lexically closest*
        #    first. Random same-slot turns made the head look good in training
        #    (17/40@1) and fail on the retrieval pool (2/40@1); the pool is
        #    lexically confusable by construction, so the negatives must be too.
        from engine.store.fts import tokenize

        def jac(a: str, b: str) -> float:
            A = set(tokenize(a)); B = set(tokenize(b))
            return len(A & B) / max(1, len(A | B))

        for old, new in sorted(gold):
            if new >= len(texts):
                continue
            slot, value = tags[new].slot, _norm(tags[new].value)
            candidates = []
            for j in range(len(texts)):
                if j in (old, new) or j in gold_by_new.get(new, set()):
                    continue
                if tags[j].slot != slot or _norm(tags[j].value) == value:
                    continue
                candidates.append((jac(texts[new], texts[j]), j))
            candidates.sort(reverse=True)
            for _score, j in candidates[:hard_negatives]:
                rows.append(Row(mid, texts[new], texts[j], "contradicts",
                                tags[new].status, tags[j].status, slot,
                                tags[j].slot, "hard_negative"))
            # a same-topic, different-slot turn is the "unrelated" hard case
            for j in range(len(texts)):
                if j in (old, new) or j in gold_by_new.get(new, set()):
                    continue
                if topics[j] and topics[j] == topics[new] and tags[j].slot != slot:
                    rows.append(Row(mid, texts[new], texts[j], "unrelated",
                                    tags[new].status, tags[j].status, slot,
                                    tags[j].slot, "hard_negative"))
                    break

        # 3. answers: a question and the next substantive turn
        taken = 0
        for i, t in enumerate(tags):
            if taken >= answers_per_meeting:
                break
            if t.kind != "question":
                continue
            for j in range(i + 1, min(len(texts), i + 4)):
                if tags[j].kind != "question":
                    rows.append(Row(mid, texts[j], texts[i], "answers",
                                    tags[j].status, tags[i].status,
                                    tags[j].slot, tags[i].slot, "derived"))
                    taken += 1
                    break

        # 4. creates_action: a commitment/action turn and the decision it acts on
        taken = 0
        for i, t in enumerate(tags):
            if taken >= actions_per_meeting:
                break
            if t.kind not in ("commitment", "action"):
                continue
            for j in range(max(0, i - 12), i):
                if tags[j].kind == "decision" and tags[j].slot == t.slot:
                    rows.append(Row(mid, texts[i], texts[j], "creates_action",
                                    tags[i].status, tags[j].status,
                                    t.slot, tags[j].slot, "derived"))
                    taken += 1
                    break

        # 5. supports: same slot, same value, both live statements
        taken = 0
        for i in range(len(texts)):
            if taken >= supports_per_meeting:
                break
            for j in range(max(0, i - 12), i):
                if (tags[i].slot == tags[j].slot
                        and _norm(tags[i].value) and _norm(tags[i].value) == _norm(tags[j].value)
                        and tags[i].status == tags[j].status == "confirmed"):
                    rows.append(Row(mid, texts[i], texts[j], "supports",
                                    tags[i].status, tags[j].status,
                                    tags[i].slot, tags[j].slot, "derived"))
                    taken += 1
                    break

    seen: set[tuple] = set()
    out: list[Row] = []
    for r in rows:
        k = r.key()
        if k in seen or r.query == r.candidate:
            continue
        seen.add(k)
        out.append(r)
    return out


def build_gold_rows(meetings: dict[str, list[dict]],
                    edges: dict[str, list], *,
                    unrelated_per_meeting: int = 20,
                    seed: int = 42) -> list[Row]:
    """Relation supervision from the gold annotation only.

    All 114 typed edges are used (not just `supersedes`), and `unrelated` is
    sampled explicitly because the annotation only marks related pairs. No row
    here is machine-labelled: this is the honest set for a pair classifier.
    """
    from engine.eval.v4_data import unrelated_rows

    rows: list[Row] = []
    for mid, es in edges.items():
        texts = [str(t.get("text") or "") for t in meetings.get(mid, [])]
        for e in es:
            if e.old >= len(texts) or e.new >= len(texts):
                continue
            rows.append(Row(mid, texts[e.new], texts[e.old], e.relation,
                            read_tags(texts[e.new]).status,
                            read_tags(texts[e.old]).status,
                            read_tags(texts[e.new]).slot,
                            read_tags(texts[e.old]).slot, "gold"))
    for r in unrelated_rows(meetings, edges, per_meeting=unrelated_per_meeting,
                            seed=seed):
        rows.append(Row(r["meeting"], r["query"], r["candidate"], "unrelated",
                        read_tags(r["query"]).status,
                        read_tags(r["candidate"]).status,
                        read_tags(r["query"]).slot,
                        read_tags(r["candidate"]).slot, "negative"))
    return rows


def status_rows(meetings: dict[str, list[dict]], mids: list[str] | None = None
                ) -> list[tuple[str, str, str]]:
    """(meeting, text, status) for the four head labels; `unknown` is dropped."""
    out = []
    keep = {"proposed", "confirmed", "cancelled", "completed"}
    for mid in (sorted(mids) if mids is not None else sorted(meetings)):
        if mid not in meetings:
            continue
        for t in meetings[mid]:
            text = str(t.get("text") or "")
            if not text.strip():
                continue
            tag = read_tags(text)
            if tag.status in keep:
                out.append((mid, text, tag.status))
    return out


__all__ = ["Row", "build_rows", "status_rows", "SELF_LABELS"]
