"""MASSIVE-fr: the one place real French requests become reader rows.

The mapping table and the loader lived in `eval/frbench.py` until the engine
needed them too (B-12's warm-up): an eval module owning vocabulary the engine
consumes is the import direction reversed, and copying the table into the engine
would repeat the reader-gold failure -- one decision table, owned by the layer
that applies it, imported by the layer that scores it.

Labels follow `read/dataset.py`'s query-row convention verbatim (QUERY_KIND /
QUERY_TYPE / QUERY_STATUS): a MASSIVE row is a user request, so kind=query,
type=question, status=mention are gold by construction. The intent label is
MASSIVE's own name mapped through `MAP_RULES`, or `none` when the request falls
outside our seven-operation vocabulary -- which is exactly what
`frbench.massive_report` counts as a correct rejection, so training and scoring
read the same table rather than two that can drift.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import repo_root
from engine.read.dataset import QUERY_KIND, QUERY_STATUS, QUERY_TYPE
from engine.read.model import HEADS

#: Rule over MASSIVE's own label names, verified against its 59 labels, not
#: recalled. `None` = out of scope: the reader must not claim an in-scope
#: intent. A substring rule beats hand-picking label names because the rule is
#: auditable; exclusions are listed explicitly.
MAP_RULES: tuple[tuple[str, str], ...] = (
    ("qa_", "answer"),                    # qa_factoid, qa_definition, qa_maths...
    ("_query", "answer"),                 # weather_query, news_query, email_query...
    ("transport_traffic", "answer"),
    ("calendar_set", "schedule_event"),
    ("alarm_set", "schedule_event"),
    ("lists_createoradd", "create_task"),
    ("takeaway_order", "create_task"),
    ("email_sendemail", "draft_email"),
)
EXCLUDED_FROM_ANSWER = {"general_quirky"}  # never matched anyway; explicit for readers

#: the mteb mirror's per-language jsonl.gz layout (pinned revision in
#: PROVENANCE.json; the canonical AmazonScience/massive repo is parquet and is
#: fetched separately for DATASETS identity -- frbench scores this one)
MASSIVE_DIR_NAME = "mteb__amazon_massive_intent"


def map_massive_intent(label: str) -> str | None:
    for needle, ours in MAP_RULES:
        if needle in label and label not in EXCLUDED_FROM_ANSWER:
            return ours
    return None


def load_massive(split: str = "test") -> list[dict]:
    p = (repo_root() / "data" / "external" / MASSIVE_DIR_NAME / split / "fr.json.gz")
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def massive_rows(split: str = "train", *, limit: int | None = None,
                 seed: int = 42) -> dict[str, Any]:
    """Deterministic row selection with its composition measured, not assumed.

    `limit` takes a shuffled subset (the "warm up on a bit of data" step): the
    seed makes it reproducible, and the returned counts tell the truth about
    what the subset contains -- including a 0.0 share, which is a number, not
    an absence (the falsy-zero trap this project has paid for three times).
    """
    raw = load_massive(split)
    needed = {"text", "label"}
    missing = needed - set(raw[0])
    if missing:
        raise ValueError(f"{split} rows lack fields {sorted(missing)} "
                         f"(got {sorted(raw[0])[:6]}...) -- wrong layout?")
    if limit is not None and limit < len(raw):
        raw = random.Random(seed).sample(raw, limit)
    rows = [{"text": r["text"], "speaker": None, "context": None,
             "intent": map_massive_intent(r["label"]) or "none",
             "massive_label": r["label"], "ref": f"massive:{split}:{r.get('id')}",
             "origin": "massive"} for r in raw]
    mix = Counter(r["intent"] for r in rows)
    return {
        "rows": rows,
        "n": len(rows),
        "n_mapped": sum(1 for r in rows if r["intent"] != "none"),
        "intent_mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
        "none_share": round(mix.get("none", 0) / max(1, len(rows)), 4),
    }


def row_label_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Head-index label columns for `ReaderModel.fit`.

    Every MASSIVE row is a user request: kind/type/status are constant by
    construction (the dataset.py query convention), and those three heads train
    on that constant only as part of the REPLAY mix -- warm() passes type/status
    fully masked for these rows, because those labels feed the guards and the
    guards' inputs must not move for a kind/intent tune-up.
    """
    def idx(head: str, value: str) -> int:
        return HEADS[head].index(value)
    return {
        "kind": np.array([idx("kind", QUERY_KIND)] * len(rows)),
        "type": np.array([idx("type", QUERY_TYPE)] * len(rows)),
        "status": np.array([idx("status", QUERY_STATUS)] * len(rows)),
        "intent": np.array([idx("intent", r["intent"]) for r in rows]),
    }


__all__ = ["MAP_RULES", "EXCLUDED_FROM_ANSWER", "map_massive_intent",
           "load_massive", "massive_rows", "row_label_arrays", "MASSIVE_DIR_NAME"]
