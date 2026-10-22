"""--merge compile: coarse families from fine specs, loud on unmapped keys.

The merge is a partition change, never a relabelling — these tests pin the
three properties that make that auditable: fine labels survive beside the
merged ones, nulls stay null, and a fine key missing from the map is a hard
error (a silent identity would let a half-written map quietly under-merge).
"""
from __future__ import annotations

import json

import pytest

from engine.tools.summre_label import compile_gold

pytestmark = pytest.mark.unit

MEETINGS = {"M1": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}


def _spec(tmp_path, body):
    p = tmp_path / "M1.train.txt"
    p.write_text(body, encoding="utf-8")
    return p


def _gold(tmp_path):
    return json.loads((tmp_path / "out" / "summre_subjects_M1.json")
                      .read_text(encoding="utf-8"))


def test_merge_maps_keys_and_keeps_fine_labels(tmp_path):
    compile_gold("M1", _spec(tmp_path, "0-1:budget\n2:null\n"), MEETINGS,
                 "train", tmp_path / "out", merge={"budget": "financement"})
    g = _gold(tmp_path)
    assert g["keys"] == {"0": "financement", "1": "financement", "2": None}
    assert g["fine_keys"] == {"0": "budget", "1": "budget", "2": None}
    assert g["counts"]["financement"] == 2 and "budget" not in g["counts"]
    assert g["merge"]["families"] == ["financement"]


def test_unmapped_fine_key_is_a_loud_error(tmp_path):
    with pytest.raises(ValueError, match="absent from the merge"):
        compile_gold("M1", _spec(tmp_path, "0:budget\n1-2:null\n"), MEETINGS,
                     "train", tmp_path / "out", merge={"other": "x"})


def test_null_stays_null_under_merge(tmp_path):
    compile_gold("M1", _spec(tmp_path, "0-2:null\n"), MEETINGS, "train",
                 tmp_path / "out", merge={"a": "b"})
    assert list(_gold(tmp_path)["keys"].values()) == [None, None, None]
