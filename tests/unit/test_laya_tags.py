"""Laya tagger wiring: encoding, option counts, and the fine-tuned model.

The model tests skip unless the checkpoint and the fine-tuned weights exist, so
the suite stays importable on a machine without torch or without the download.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]


def _have(*parts: str) -> bool:
    return all((ROOT / p).exists() for p in parts)


needs_model = pytest.mark.skipif(
    not _have("runs/laya_tags/multilingual_v2/model.safetensors"),
    reason="fine-tuned tagger not present")


def test_encoding_matches_the_option_count():
    from engine.read import laya_tags

    pytest.importorskip("transformers")
    try:
        _, tokz, _, _ = laya_tags.load_laya("multilingual")
    except (FileNotFoundError, OSError):
        pytest.skip("Laya checkpoint not available")
    row = {"text": "on n'a pas encore validé le budget", "status": "proposed"}
    item = laya_tags._item(row, "status", tokz, 192, 128)
    assert len(item["target"]) == len(laya_tags.TASKS["status"]["labels"])
    assert len(item["markers"]) == len(laya_tags.TASKS["status"]["labels"])


@needs_model
def test_negation_is_proposed_not_confirmed():
    from engine.read import laya_tags

    model, tok, _, _ = laya_tags.load_laya("multilingual")
    from safetensors.torch import load_file
    model.load_state_dict(load_file(
        str(ROOT / "runs/laya_tags/multilingual_v2/model.safetensors")),
        strict=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    texts = ["on n'a pas encore validé le budget",
             "ce n'est pas encore acté pour atlas",
             "rien n'est acté sur le devis"]
    labels = laya_tags.TASKS["status"]["labels"]
    probs = laya_tags.predict(model, tok, texts, "status", device=device)
    pred = [labels[max(range(len(p)), key=p.__getitem__)] for p in probs]
    assert pred == ["proposed"] * 3, pred


@needs_model
def test_reader_returns_tags_with_model_fields():
    from engine.read import laya_reader
    from engine.read.v4_tags import Tags

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags = laya_reader.tag_texts(
        ["on annule le devis pour atlas", "j'envoie le contrat vendredi"],
        ROOT / "runs/laya_tags/multilingual_v2", device=device)
    assert len(tags) == 2 and all(isinstance(t, Tags) for t in tags)
    assert tags[0].status == "cancelled"
    assert tags[1].kind == "action"
