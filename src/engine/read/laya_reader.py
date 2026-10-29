"""Model-backed §3 tags: the fine-tuned Laya heads replace the keyword rules.

The model predicts the three tag fields; it does not predict spans. Entities,
slot, value and the evidence clause still come from the rule reader, which is
the part that already worked and is span-addressable.

`read_tags` stays the rules-only path (no torch). Callers that want the model
use `tag_texts`, which batches every turn of a meeting through three forward
passes instead of one per sentence.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from engine.read import laya_tags
from engine.read.v4_tags import Tags, read_tags

_MODEL: dict[tuple[str, str], tuple] = {}


def load(model_dir: str | Path, device: str = "cuda"):
    key = (str(model_dir), device)
    if key not in _MODEL:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from laya.common import build_model

        meta = json.loads((Path(model_dir) / "meta.json").read_text())
        root = Path(meta["source_checkpoint"])
        cfg = json.loads((root / "rl_agent_config.json").read_text())
        model = build_model(cfg, encoder_dir=str(root / "encoder"))
        model.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")),
                              strict=True)
        tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        model.to(device).eval()
        _MODEL[key] = (model, tok)
    return _MODEL[key]


def tag_texts(texts: list[str], model_dir: str | Path, *,
              device: str = "cuda") -> list[Tags]:
    model, tok = load(model_dir, device)
    probs = {t: laya_tags.predict(model, tok, texts, t, device=device)
             for t in ("kind", "status", "intent")}
    out = []
    for i, text in enumerate(texts):
        base = read_tags(text)
        pick = {}
        conf = 1.0
        for task in ("kind", "status", "intent"):
            labels = laya_tags.TASKS[task]["labels"]
            p = probs[task][i]
            j = max(range(len(p)), key=p.__getitem__)
            pick[task] = labels[j]
            conf = min(conf, p[j])
        out.append(replace(base, kind=pick["kind"], status=pick["status"],
                           intent=pick["intent"], confidence=round(conf, 4)))
    return out
