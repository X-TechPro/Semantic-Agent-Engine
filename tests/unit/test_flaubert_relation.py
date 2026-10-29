"""§4 heads: shapes, untrained refusal, checkpoint contract.

No network and no checkpoint download: the encoder is not instantiated here.
A wrong signature or a silently-random head is exactly what went unnoticed
before, so the contract is asserted directly.
"""
import json

import pytest
import torch

from engine.link.flaubert_relation import (RELATION_LABELS, STATUS_LABELS,
                                           Heads)


def test_head_shapes_match_the_pair_interaction():
    h = Heads(hidden=32, n_relation=len(RELATION_LABELS),
              n_status=len(STATUS_LABELS))
    feat = torch.randn(5, 4 * 32)
    logits = h.forward_pair(feat)
    assert logits.shape == (5, len(RELATION_LABELS))
    st = h.forward_unit(torch.randn(5, 32))
    assert st.shape == (5, len(STATUS_LABELS))
    ch = h.forward_change(torch.randn(5, 32))
    assert ch.shape == (5, 2)                   # the primary target


def test_a_checkpoint_without_the_encoder_cannot_pass_for_finetuned():
    """Saving only heads discarded encoder fine-tuning: reload scored 0.62 AUC
    against 0.73 for the trained model. The contract is asserted here."""
    from engine.link.flaubert_relation import FlaubertRelationModel
    import inspect

    src = inspect.getsource(FlaubertRelationModel.save)
    assert "encoder.pt" in src
    assert "change_threshold" in src


def test_label_sets_are_the_schema_ones():
    from engine.schemas_v4 import RELATIONS
    assert tuple(RELATION_LABELS) == RELATIONS
    assert set(STATUS_LABELS) == {"proposed", "confirmed", "cancelled", "completed"}


def test_checkpoint_must_carry_matching_labels(tmp_path):
    from engine.link.flaubert_relation import FlaubertRelationModel

    p = tmp_path / "ckpt"
    p.mkdir()
    torch.save(Heads(hidden=8).state_dict(), p / "heads.pt")
    (p / "config.json").write_text(json.dumps({
        "model_id": "x", "hidden": 8,
        "relation_labels": ["supersedes"],       # wrong on purpose
        "status_labels": list(STATUS_LABELS),
        "trained": True}), encoding="utf-8")
    from engine.link.flaubert_relation import RelationPrediction  # noqa: F401
    with pytest.raises(ValueError):
        FlaubertRelationModel.from_checkpoint(p, device="cpu")


def test_untrained_model_refuses_to_predict():
    from engine.link.flaubert_relation import FlaubertRelationModel

    # bypass __init__ (which would load the encoder): build the object state
    m = object.__new__(FlaubertRelationModel)
    m.trained = False
    with pytest.raises(RuntimeError):
        m.relation_probs(["a"], ["b"])
