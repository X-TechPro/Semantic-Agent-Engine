"""Training rows for the reader: ledger gold, never the rules reader's output.

Two row sources, and the split between them is the whole point:

* **units** — text segmenter-produced from a rendered meeting, labelled by
  looking up the ledger event the turn came to (`gen.ledger.reader_gold`). Units
  with no gold pointer (openers, fillers) are *dropped*, not labelled `other`:
  the generator never asserted what they are, and inventing a label here would
  teach the head a class that only exists in my convenience.
* **queries** — the QA generator's questions, whose intent is fixed when the
  question is written, so `kind=query` and `intent=...` are gold by construction.

The `kind` head is only learnable because of the channel cue. A meeting question
("Qui s'occupe du budget ?") and a user question with the same words have
opposite gold labels (7.3), so text alone cannot decide it — the `[speaker]`
prefix from 6.2.3 *is* the signal, which is why meeting rows carry a speaker and
query rows deliberately do not. Without that, the measured kind F1 would be
noise, not skill.

Intent coverage is partial and stated as such: the generators can only gold
`answer`, `compare` and `build_dossier`. `draft_email` / `create_task` /
`schedule_event` / `clarify` / `none` have no gold rows in the ledger corpus, so
the intent macro-F1 is computed over the classes that exist and the rest is
reported as `untrained_classes`, not silently averaged in as a free win.

Casual French is trained for, not assumed away (`casual_variants`): the heads
must read "bon, du coup on reporte le truc tu vois" as the same speech act as
the rendered sentence it came from, because real ASR meetings (B-13's measured
status mix: 994/1019 `mention`) speak the sloppy register only. The augmented
copy keeps every label and speaker prefix — the filler changes style, and
kind/type/status are properties of what the utterance *does* — and it flows
through dev and test too, so the §6.2.6 gates are measured on the harder corpus
at full strength rather than on the generator's tidy phrasing.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config
from engine.read.model import HEADS, MASK, gold_vector
from engine.schemas import Unit

#: the QA generator's question kinds -> the reader's intent label. `history` is a
#: retrieval *mode*, not a different task: the user still wants an answer.
#: `unanswerable` is also `answer` — whether evidence exists is the retrieval
#: layer's decision (6.9 abstention), not a different intention, and labelling it
#: `clarify` would train the head on an outcome instead of on what was asked.
QA_KIND_TO_INTENT = {
    "current_decision": "answer", "history": "answer",
    "proposal_vs_decision": "answer", "owner": "answer", "date": "answer",
    "unanswerable": "answer", "compare": "compare", "build_dossier": "build_dossier",
}
#: labels of a query row other than intent, by construction (a user request is a
#: question with no asserted state)
QUERY_KIND, QUERY_TYPE, QUERY_STATUS = "query", "question", "mention"
UNTRAINABLE_INTENTS = ["draft_email", "create_task", "schedule_event", "clarify",
                       "none"]


def event_of(ledger) -> dict[str, Any]:
    """event_id -> the Event itself.

    The Event, not its kind string: `reader_gold` derives the `conditional`
    status from `Event.notes`, so handing it a bare kind silently trained 9
    conditional events as `proposed` (the same drift the literal tables had).
    """
    return {e.event_id: e for e in ledger.events}


def unit_rows(ledger, units: list[Unit]) -> list[dict[str, Any]]:
    """One row per stored unit that has a ledger gold pointer."""
    from gen.ledger import reader_gold

    events = event_of(ledger)
    by_source: dict[tuple[str, int], Unit] = {}
    for u in units:
        if u.turn_index is not None:
            by_source[(u.source_id, u.turn_index)] = u
    rows: list[dict[str, Any]] = []
    for u in units:
        ev = events.get(u.gold_event_ids[0]) if u.gold_event_ids else None
        if ev is None:
            continue
        k, type_, status = reader_gold(ev)
        prev = by_source.get((u.source_id, (u.turn_index or 0) - 1))
        rows.append({"text": u.text, "speaker": u.speaker,
                     "context": prev.text if prev else None,
                     "labels": gold_vector(k, type_, status, None),
                     "origin": "unit", "ref": u.unit_id, "project": u.project,
                     # the real unit, so the rules baseline reads exactly what
                     # the engine read (same source_date, same speaker prefix)
                     "unit": u})
    return rows


def query_rows(ledger, projects: set[str], seed: int,
               per_project: int = 8) -> list[dict[str, Any]]:
    """One row per generated question, labelled by construction."""
    from gen.qa_from_ledger import build_questions

    qa = build_questions(ledger, seed=seed, per_project=per_project)
    rows: list[dict[str, Any]] = []
    for item in qa.items:
        if item["project"] not in projects:
            continue
        intent = QA_KIND_TO_INTENT.get(item["kind"])
        if intent is None:
            continue
        rows.append({"text": item["question"], "speaker": None, "context": None,
                     "labels": gold_vector(QUERY_KIND, QUERY_TYPE, QUERY_STATUS,
                                           intent),
                     "origin": "query", "ref": item["qid"],
                     "project": item["project"]})
    return rows


def stack(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Columns the model consumes, plus the per-head gold arrays."""
    labels = {name: np.array([r["labels"].get(name, MASK) for r in rows])
              for name in HEADS}
    return {
        "texts": [r["text"] for r in rows],
        "speakers": [r.get("speaker") for r in rows],
        "contexts": [r.get("context") for r in rows],
        "labels": labels,
        # gold as label *names*, so a report can show confusion without decoding
        "gold_names": {name: [HEADS[name][int(v)] if int(v) >= 0 else None
                              for v in labels[name]] for name in HEADS},
    }


def label_mix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Class distribution per head over the unmasked rows — the imbalance the
    class weights exist to counter, reported so it can be checked."""
    out: dict[str, dict[str, int]] = {}
    for name in HEADS:
        vals = [r["labels"].get(name, MASK) for r in rows]
        keep = [v for v in vals if v >= 0]
        out[name] = {HEADS[name][v]: int(c) for v, c in Counter(keep).items()}
    return out


def user_channel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same statements presented as *user* text, to make `kind` learnable.

    Why this exists: meeting rows carry a `[speaker] ` prefix and query rows do
    not, so on the raw corpus "starts with a bracket" predicts `info` perfectly.
    A kind head trained like that measures a formatting artifact, not the skill
    12.3 actually tests — deciding between a question and an assertion in plain
    user text. Adding an unprefixed copy of every statement gives the head a
    real, balanced decision on the channel where the reader has no other cue.

    The label is not invented: kind means "does this utterance request or
    assert", which is a property of the sentence, not of where it was said.
    """
    extra: list[dict[str, Any]] = []
    for r in rows:
        if r.get("speaker") is None or r.get("origin") != "unit":
            continue
        copy = dict(r)
        copy["speaker"] = None
        copy["context"] = None
        copy["origin"] = "user_statement"
        copy["ref"] = r["ref"] + "#u"
        extra.append(copy)
    return extra


_FILL_PREFIXES = ("bon", "euh", "du coup", "ben", "bah", "alors", "écoute", "en fait")
_FILL_TAILS = ("tu vois", "en fait", "je crois", "quoi", "bref")
_ET_CUE = re.compile(r"\bet\b(?! puis)", re.IGNORECASE)


def _pick(salt: str, options: tuple[str, ...]) -> str:
    """Deterministic choice — `random` would make the corpus depend on call
    order, and the augmentation must be reproducible from the ledger alone."""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def casual_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A casual-French copy of every original row, labels untouched.

    Rules, in order of application, and every one of them *preserves* what the
    sentence does: a leading filler ("bon,", "du coup"), a spoken "et puis"
    where the rendered text says "et", a hedge tail ("tu vois", "en fait"),
    lower-casing of the first letter the way ASR writes it. Nothing is deleted
    from the utterance — removing a "ne pas" or a date would change the label
    while keeping it, which would teach the head to ignore exactly the cues the
    guards later rely on.

    Copies-of-copies are skipped (`#c`/`#u` refs), so the factor is fixed at 2×
    and the augmented rows stay auditable back to their source row.
    """
    extra: list[dict[str, Any]] = []
    for r in rows:
        ref = str(r.get("ref", ""))
        if ref.endswith("#c") or ref.endswith("#u"):
            continue
        if not r.get("text"):
            continue
        salt = f"casual|{ref}"
        h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest(), 16)
        text = r["text"]
        pre = _pick(salt + "|p", _FILL_PREFIXES)
        text = f"{pre}, {text}" if h % 4 else f"{pre} {text}"
        text = text[0].lower() + text[1:]              # ASR-style sentence start
        if h % 3 == 0:
            text = _ET_CUE.sub("et puis", text, count=1)
        if h % 2:
            text = f"{text} {_pick(salt + '|t', _FILL_TAILS)}"
        final = text.rstrip(" .?!") + (text[-1] if text[-1] in "?!" else "")
        copy = dict(r)
        copy["text"] = final
        if r.get("unit") is not None:
            # predict_rules() reads the row's own `unit` when present — a copy
            # still holding the ORIGINAL unit would score the rules baseline on
            # clean text while the trained head saw the casual one, and the
            # 6.2.6 "beats the baseline" comparison would be measuring two
            # different corpora. Clone the unit with the casual text.
            copy["unit"] = r["unit"].model_copy(update={"text": final})
        copy["origin"] = f"{r.get('origin')}+casual"
        copy["ref"] = ref + "#c"
        extra.append(copy)
    return extra


def build_split(ledger, projects: set[str], seed: int, run_dir: Path,
                embed_backend: str = "hash", tag: str = "dev",
                with_queries: bool = True) -> dict[str, Any]:
    """Render + segment this split's meetings, then attach ledger gold.

    The units come out of a real `Engine` ingest rather than by re-implementing
    segmentation here: if the training rows were built by a second, hand-rolled
    splitter, the head would be trained on a text distribution the engine never
    produces.
    """
    from engine.engine import Engine

    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / f"reader_{tag}.db"
    for stale in run_dir.glob(f"reader_{tag}.db*"):
        stale.unlink()
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": embed_backend},
                                 "reader": {"backend": "rules"}})
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    eng = Engine(cfg, path=db, roster=roster, linker_enabled=False)
    from gen.render import render_meeting

    for m in ledger.meetings:
        if m.project in projects:
            eng.ingest(render_meeting(ledger, m.meeting_id, seed=seed))
    units = eng.store.all_units()
    eng.close()
    for stale in run_dir.glob(f"reader_{tag}.db*"):
        stale.unlink()
    rows = unit_rows(ledger, units)
    if with_queries:
        rows = rows + query_rows(ledger, projects, seed)
    # unprefixed copies of the statements, so `kind` cannot be decided by the
    # presence of a `[speaker] ` bracket (see user_channel_rows)
    originals = list(rows)
    rows = rows + user_channel_rows(rows)
    # casual-French copies of the ORIGINAL rows (not of the channel copies):
    # fixed 1× augmentation factor, every variant auditable to its source
    casual = casual_variants(originals)
    rows = rows + casual
    packed = stack(rows)
    packed["rows"] = rows
    packed["mix"] = label_mix(rows)
    packed["n_units"] = len(units)
    packed["n_user_channel"] = sum(1 for r in rows if r.get("speaker") is None)
    return packed


def predict_rules(reader, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """What the shipped rules reader answers on the *same* rows.

    This is the baseline the trained head has to beat (0.4 / 6.2.6). It runs on
    gold-labelled rows rather than on units re-read through the store, so the
    comparison cannot be accused of scoring a different corpus.
    """
    out: dict[str, list[int]] = {name: [] for name in HEADS}
    for r in rows:
        unit = r.get("unit") or _unit_from_row(r)
        reading = reader.read(unit)
        for name in HEADS:
            value = getattr(reading, name)
            index = HEADS[name].index(value) if value in HEADS[name] else MASK
            out[name].append(index)
    return {name: np.array(vals) for name, vals in out.items()}


def _unit_from_row(row: dict[str, Any]) -> Unit:
    """A query row as a Unit, so the rules baseline can read it.

    Query rows have no unit of their own — the QA generator emits a question and
    an intent, and the engine reads bare text through `read_text`. This builds
    the same shape that path builds, so `predict_rules` compares readers on
    identical inputs.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    return Unit(unit_id=row.get("ref", "row"), source_id="dataset",
                source_type="note", source_date=now.date(), ingested_at=now,
                text=row["text"], speaker=row.get("speaker"),
                project=row.get("project"), gold_event_ids=[])


__all__ = ["event_of", "unit_rows", "query_rows", "stack", "label_mix",
           "build_split", "predict_rules", "QA_KIND_TO_INTENT",
           "UNTRAINABLE_INTENTS", "QUERY_KIND", "QUERY_TYPE", "QUERY_STATUS"]
