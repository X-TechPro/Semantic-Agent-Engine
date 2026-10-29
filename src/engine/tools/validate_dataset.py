"""Validate the two things a dataset author writes: meetings and questions.

An input meeting carries four fields per message and nothing else. Tags, units,
links and state are the architecture's output (`schemas/memory_record.schema.json`)
and are rejected if they appear in an input file.

Usage:
    python -m engine.tools.validate_dataset data/mine/*.meeting.json
    python -m engine.tools.validate_dataset data/mine/*.questions.json \\
        --meetings data/mine/*.meeting.json

Exit code 1 on any error; warnings do not fail the run.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"
STYLES = {"messy", "formal", "tricky", "chat"}


class Report:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, m: str) -> None:
        self.errors.append(m)

    def warn(self, m: str) -> None:
        self.warnings.append(m)


def _schema_check(doc: dict, name: str, rep: Report) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        rep.warn("jsonschema not installed: structural checks only")
        return
    schema = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    checker = Draft202012Validator.FORMAT_CHECKER
    for e in sorted(Draft202012Validator(schema,
                                         format_checker=checker).iter_errors(doc),
                    key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "<root>"
        rep.err(f"schema {where}: {e.message}")


def _ts(value: str) -> bool:
    """A date is not a time, and a naive time is not a timestamp."""
    s = str(value)
    if "T" not in s:
        return False
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt.tzinfo is not None


def validate_meeting(doc: dict, rep: Report) -> set[str]:
    _schema_check(doc, "meeting.schema.json", rep)
    ids: set[str] = set()
    for i, m in enumerate(doc.get("messages") or []):
        mid = m.get("message_id", f"<#{i}>")
        if mid in ids:
            rep.err(f"duplicate message_id {mid}")
        ids.add(mid)
        if not _ts(m.get("timestamp", "")):
            rep.err(f"{mid}: timestamp is not ISO date-time: {m.get('timestamp')!r}")
        if not str(m.get("text", "")).strip():
            rep.err(f"{mid}: empty text")
        if not str(m.get("speaker", "")).strip():
            rep.err(f"{mid}: empty speaker")
    return ids


def validate_questions(doc: dict, rep: Report, known: set[str] | None) -> None:
    _schema_check(doc, "questions.schema.json", rep)
    seen: set[str] = set()
    for q in doc.get("questions") or []:
        qid = q.get("query_id", "<no id>")
        if qid in seen:
            rep.err(f"duplicate query_id {qid}")
        seen.add(qid)
        if q.get("style") not in STYLES:
            rep.err(f"{qid}: unknown style {q.get('style')!r}")
        exp = q.get("expected_message_ids") or []
        if q.get("must_abstain") and exp:
            rep.err(f"{qid}: must_abstain but names {len(exp)} expected messages")
        if not q.get("must_abstain") and not exp:
            rep.err(f"{qid}: no expected_message_ids and must_abstain is false")
        if known is not None:
            for mid in exp:
                if mid not in known:
                    rep.err(f"{qid}: expected message {mid} does not exist")
    styles = {q.get("style") for q in doc.get("questions") or []}
    if styles and styles <= {"formal"}:
        rep.warn("every question is formal: add messy and tricky ones")


def detect(doc: dict) -> str:
    if "messages" in doc:
        return "meeting"
    if "questions" in doc:
        return "questions"
    return "unknown"


def validate_file(path: Path, known: set[str] | None = None) -> Report:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rep = Report(path)
    kind = detect(doc)
    if kind == "meeting":
        validate_meeting(doc, rep)
    elif kind == "questions":
        validate_questions(doc, rep, known)
    else:
        rep.err("not a meeting (no `messages`) and not a questions file (no `questions`)")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="meeting or questions JSON files")
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="meeting files, to check that cited ids exist")
    args = ap.parse_args(argv)

    known: set[str] | None = None
    if args.meetings:
        known = set()
        for raw in args.meetings:
            rep = validate_file(Path(raw))
            known |= set()
            doc = json.loads(Path(raw).read_text(encoding="utf-8"))
            for m in doc.get("messages") or []:
                known.add(m.get("message_id", ""))
            print(f"{'OK' if not rep.errors else 'FAIL'} {raw} (meeting)"
                  f"  errors={len(rep.errors)} warnings={len(rep.warnings)}")
            for e in rep.errors[:20]:
                print("  error:", e)

    bad = 0
    for raw in args.paths:
        rep = validate_file(Path(raw), known)
        kind = detect(json.loads(Path(raw).read_text(encoding="utf-8")))
        print(f"{'OK' if not rep.errors else 'FAIL'} {raw} ({kind})"
              f"  errors={len(rep.errors)} warnings={len(rep.warnings)}")
        for e in rep.errors[:20]:
            print("  error:", e)
        for w in rep.warnings[:10]:
            print("  warn :", w)
        bad += bool(rep.errors)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
