"""Export the Pydantic contracts to `schemas/*.json` (BUILD_GUIDE 2.3 / 3).

These files are what an external agent validates against; regenerating them is
part of `make schemas`, and a test asserts every contract has a schema on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.schemas import (ActionResult, AnswerWithEvidence, Claim,
                            Clarification, Comparison, ComparisonItem, Dossier,
                            DossierSection, DraftEmail, Edge, EventProposal,
                            Filters, Hit, IngestReport, Query, ReadContext,
                            Reading, Recipient, Source, Slots, Span, TaskProposal,
                            Turn, Unit)

CONTRACTS: dict[str, type] = {
    "unit": Unit,
    "slots": Slots,
    "span": Span,
    "source": Source,
    "turn": Turn,
    "reading": Reading,
    "read_context": ReadContext,
    "edge": Edge,
    "query": Query,
    "filters": Filters,
    "hit": Hit,
    "recipient": Recipient,
    "draft_email": DraftEmail,
    "task_proposal": TaskProposal,
    "event_proposal": EventProposal,
    "clarification": Clarification,
    "claim": Claim,
    "dossier_section": DossierSection,
    "dossier": Dossier,
    "answer_with_evidence": AnswerWithEvidence,
    "comparison_item": ComparisonItem,
    "comparison": Comparison,
    "action_result": ActionResult,
    "ingest_report": IngestReport,
}


def export_all(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, model in CONTRACTS.items():
        schema = model.model_json_schema()
        (out_dir / f"{name}.schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index[name] = {
            "title": schema.get("title"),
            "required": schema.get("required", []),
            "properties": sorted(schema.get("properties", {})),
        }
    (out_dir / "index.json").write_text(
        json.dumps({"contracts": index, "label_sets": label_sets()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(CONTRACTS)


def label_sets() -> dict[str, list[str]]:
    """The closed vocabularies a contract's string fields must come from.

    These are the values an external agent cannot recover from JSON Schema alone:
    `Unit.type` is a plain `str` there, so without this block the consumer has to
    guess the allowed labels. It used to emit `{"source_type": []}` because the
    expression was `model_fields["source_type"].metadata and []`, which is `[]`
    for any metadata value — an empty list that *looked* like a finished field.
    """
    from engine.schemas import (INTENTS, KINDS, RELATIONS, SOURCE_TYPES, STATUSES,
                                TYPES)

    return {"source_type": list(SOURCE_TYPES), "kind": list(KINDS),
            "type": list(TYPES), "status": list(STATUSES),
            "intent": list(INTENTS), "relation": list(RELATIONS)}


def main(argv=None) -> int:
    from engine.config import repo_root

    out = Path(repo_root() / "schemas")
    n = export_all(out)
    print(f"wrote {n} schemas -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
