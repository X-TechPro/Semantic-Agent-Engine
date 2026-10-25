"""Token-level concept attention (D-048): the score plumbing, not the model.

The model numbers live in `runs/token_attention/`; these tests pin the two
places a silent mistake would change them: subword merging (the owner's example
splits "pizzas" into "pizza"+"##s", and a naive score lands on the suffix) and
the AUC statistic used to compare signals.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.eval.token_attention import _words, auc

pytestmark = pytest.mark.unit


class FakeTok:
    all_special_tokens = ["[CLS]", "[SEP]"]
    _tok = {1: "pizza", 2: "##s", 3: "[SEP]", 4: "le", 5: "budget"}

    def convert_ids_to_tokens(self, i):
        return self._tok[i]


def test_wordpiece_continuations_merge_into_one_word():
    assert _words(FakeTok(), [1, 2]) == [(0, 2, "pizzas")]


def test_specials_are_dropped_and_spans_stay_indexed():
    spans = _words(FakeTok(), [4, 1, 2, 3, 5])
    assert spans == [(0, 1, "le"), (1, 3, "pizzas"), (4, 5, "budget")]


def test_auc_is_the_textbook_statistic():
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
