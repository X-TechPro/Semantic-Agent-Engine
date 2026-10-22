"""The shared drawer validator — `normalize_attr` and the project-echo guard.

Moved here from the (deleted) LLM extractor when the extractor line was
removed (D-037). The functions stay live because the labelling tool runs every
hand-written drawer through `normalize_attr`: gold labels and any future
model output therefore share one format rule by construction, not by
convention. `echoes_project` is the D-028 smoke catch — a key made only of
the project's own tokens would file every unit of the meeting into one slot
and merge all threads.
"""
from __future__ import annotations

import pytest

from engine.slots.subjects import echoes_project, normalize_attr

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw,want", [
    ("budget", "budget"),                      # exact vocabulary
    ("Budget", "budget"),                      # case-folded
    ("coût", "budget"),                        # surface form, accents folded
    ("date de lancement", "launch_date"),      # longest surface form wins
    ("menu_traiteur", "menu_traiteur"),        # well-shaped coined key
    ("Le buffet des mariés", "buffet_maries"),  # coined, stop-words dropped
    ("none", None), (None, None), (True, None), ("", None), ("null", None),
    ("je vais au marché puis je reviens demain", None),   # >3 tokens
    ("x" * 50, None),                          # shape guard
])
def test_normalize_attr_validates_every_drawer_name(raw, want):
    assert normalize_attr(raw) == want


def test_echo_guard_refuses_project_named_keys():
    assert echoes_project("019c eapd", "019c_EAPD")
    assert not echoes_project("buffet", "019c_EAPD")
    assert not echoes_project("buffet", None)
