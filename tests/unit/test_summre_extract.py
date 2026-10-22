"""summre_extract's pure parts — the filter IS the quality gate of the label pass.

The hand-labelled temporal gold is built from exactly what `is_clean` keeps and
`meeting_segments` orders, so these tests pin what the annotator will actually
see: ASR placeholders out, sub-15-char fragments out, `sil` mid-sentence kept
(it's anchored, not substring -- real French words start with those letters),
and meetings reconstructed in time order with speaker provenance intact.
"""

from __future__ import annotations

import pytest

from engine.tools.summre_extract import is_clean, meeting_segments

pytestmark = pytest.mark.unit


def test_garbage_and_short_segments_never_reach_the_label_pass():
    for junk in ("sil", "sil w_1 w_2 w_3", "w_12", "*", "", "   ", "court"):
        assert not is_clean(junk), junk
    assert is_clean("on pourrait faire une proposition de pub")
    # anchored on purpose: a substring filter would delete real speech
    assert is_clean("c'est une silhouette devant le budget annuel du projet")


def test_meeting_segments_rebuilds_time_order_with_speakers():
    rows = [
        {"meeting_id": "m1", "speaker_id": "02", "segments": [
            {"start": 120.0, "transcript": "finalement on part sur nova cloud pour le cloud"},
            {"start": 30.0, "transcript": "je propose openstack plutôt pour le cloud"},
            {"start": 40.0, "transcript": "sil"},
            {"start": 45.0, "transcript": "petit"},
        ]},
        {"meeting_id": "m1", "speaker_id": "01", "segments": [
            {"start": 0.0, "transcript": "bonjour on parle du cloud du projet"},
        ]},
        {"meeting_id": "m2", "speaker_id": "01", "segments": [
            {"start": 5.0, "transcript": "zut alors est-ce que ça marche vraiment"},
        ]},
    ]
    got = meeting_segments(rows)
    assert list(got) == ["m1", "m2"], "meetings sorted by id, deterministic"
    m1 = got["m1"]
    assert [s["start_min"] for s in m1] == [0.0, 0.5, 2.0]
    assert [s["speaker"] for s in m1] == ["01", "02", "02"]
    assert all("sil" != s["text"] for s in m1)
    assert m1[1]["text"].startswith("je propose")
