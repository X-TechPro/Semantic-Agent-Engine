"""Input files carry four fields; questions cite message ids. Both are enforced."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from engine.tools.validate_dataset import detect, validate_file

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "data" / "examples"


def _meeting() -> dict:
    return json.loads((EXAMPLES / "atlas.meeting.json").read_text(encoding="utf-8"))


def _questions() -> dict:
    return json.loads((EXAMPLES / "atlas.questions.json").read_text(encoding="utf-8"))


def _run(doc: dict, tmp_path: Path, known: set[str] | None = None):
    p = tmp_path / "input.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return validate_file(p, known)


def _ids(doc: dict) -> set[str]:
    return {m["message_id"] for m in doc["messages"]}


def test_the_example_meeting_is_valid(tmp_path):
    assert _run(_meeting(), tmp_path).errors == []


def test_the_example_questions_are_valid_against_its_messages(tmp_path):
    assert _run(_questions(), tmp_path, _ids(_meeting())).errors == []


def test_a_generated_field_in_the_input_is_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][0]["units"] = []
    assert any("schema" in e for e in _run(doc, tmp_path).errors)


def test_a_label_in_the_input_is_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][1]["is_change"] = True
    assert any("schema" in e for e in _run(doc, tmp_path).errors)


def test_a_fifth_field_is_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][0]["channel"] = "audio"
    assert any("schema" in e for e in _run(doc, tmp_path).errors)


def test_duplicate_message_ids_are_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][1]["message_id"] = doc["messages"][0]["message_id"]
    assert any("duplicate message_id" in e for e in _run(doc, tmp_path).errors)


def test_a_bad_timestamp_is_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][0]["timestamp"] = "2021-10-05"  # a date is not a time
    assert any("ISO date-time" in e for e in _run(doc, tmp_path).errors)


def test_an_empty_message_is_rejected(tmp_path):
    doc = _meeting()
    doc["messages"][0]["text"] = " "
    assert any("empty text" in e for e in _run(doc, tmp_path).errors)


def test_a_question_without_an_expected_id_or_abstain_is_rejected(tmp_path):
    doc = _questions()
    q = doc["questions"][0]
    q["expected_message_ids"] = []
    q["must_abstain"] = False
    assert any("no expected_message_ids" in e for e in _run(doc, tmp_path).errors)


def test_abstaining_while_naming_a_message_is_rejected(tmp_path):
    doc = _questions()
    q = doc["questions"][-1]
    q["must_abstain"] = True
    q["expected_message_ids"] = ["m-001:1"]
    assert any("must_abstain" in e for e in _run(doc, tmp_path).errors)


def test_an_unknown_style_is_rejected(tmp_path):
    doc = _questions()
    doc["questions"][0]["style"] = "angry"
    assert any("schema" in e for e in _run(doc, tmp_path).errors)


def test_a_cited_message_that_does_not_exist_is_rejected(tmp_path):
    doc = _questions()
    doc["questions"][0]["expected_message_ids"] = ["m-001:99"]
    assert any("does not exist" in e
               for e in _run(doc, tmp_path, _ids(_meeting())).errors)


def test_all_formal_questions_only_warn(tmp_path):
    doc = _questions()
    for q in doc["questions"]:
        q["style"] = "formal"
        if not q["must_abstain"] and not q["expected_message_ids"]:
            q["expected_message_ids"] = ["m-001:1"]
    rep = _run(doc, tmp_path, _ids(_meeting()))
    assert not rep.errors
    assert any("formal" in w for w in rep.warnings)


def test_an_unrecognised_file_is_rejected(tmp_path):
    assert detect({"meetings": []}) == "unknown"
    rep = _run({"meetings": []}, tmp_path)
    assert rep.errors and "not a meeting" in rep.errors[0]
