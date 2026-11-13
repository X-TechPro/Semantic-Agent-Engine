#!/usr/bin/env python3
"""pplx-context variants on the QMSum test split (meeting-scoped, same metrics).

Backbone: perplexity-ai/pplx-embed-context-v1-0.6b, the stock model Rexu v3 loads
(no LoRA adapter is loaded anywhere in the Rexu v3 path). Variants:
- rexu_v3    : stock backbone + trained Dense(1024->128) late head + MaxSim
- raw_maxsim : stock backbone, no head: normalized 1024-d contextual tokens + MaxSim
- mean_cos   : stock backbone, no head: per-turn mean of contextual tokens, cosine

Usage: python evaluate_qmsum_pplx_variants.py --max-tokens 15000 --variants rexu_v3 raw_maxsim mean_cos
Writes artifacts/qmsum_pplx_variants_w<max_tokens>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import memory_engine as me
from evaluate_qmsum_baselines import metrics
from pplx_rexu import RexuV3Retriever

ROOT = Path(__file__).resolve().parent


PLAIN_MODEL = "perplexity-ai/pplx-embed-v1-0.6B"


@torch.inference_mode()
def plain_embed(texts: list[str], model, tok, device: str, bs: int = 32) -> torch.Tensor:
    """Mean-pooled, L2-normalised embeddings from the non-context pplx-embed-v1 model."""
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out)


def load_plain(device: str):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PLAIN_MODEL, trust_remote_code=True, cache_dir=str(ART_HF))
    model = AutoModel.from_pretrained(PLAIN_MODEL, trust_remote_code=True,
                                      cache_dir=str(ART_HF)).to(device).eval()
    return model, tok


ART_HF = ROOT / "artifacts" / "hf_cache"


def rank(r: RexuV3Retriever, ds: dict, q: dict, variant: str, plain=None) -> list[str]:
    cands = ds["by_meeting"][q["meeting_id"]]
    if variant == "pplx_embed_v1":
        model, tok, docs = plain["model"], plain["tok"], plain["docs"]
        mid = q["meeting_id"]
        if mid not in docs:
            docs[mid] = plain_embed([r.records[c]["encoder_text"] for c in cands], model, tok, r.device)
        qv = plain_embed([q["query"]], model, tok, r.device)[0]
        scores = (docs[mid] @ qv).numpy()
        return [cands[i] for i in np.argsort(-scores)]
    cands = ds["by_meeting"][q["meeting_id"]]
    if variant == "rexu_v3":
        out = r.search(q["query"], top_k=len(cands), meeting_id=q["meeting_id"], method="rexu_v3")
        return [x["chunk_id"] for x in out["results"]]
    if variant == "raw_maxsim":
        _, qt = r._encode_one(q["query"])
        qt = qt.to(r.device)
        scores = np.array([float((qt @ r._docs_tokens[c].to(r.device).T).max(dim=1).values.sum())
                           for c in cands])
    elif variant == "mean_cos":
        qm, _ = r._encode_one(q["query"])
        scores = np.array([float(qm @ r._docs_mean[c]) for c in cands])
    else:
        raise ValueError(variant)
    return [cands[i] for i in np.argsort(-scores)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=15000)
    ap.add_argument("--variants", nargs="+", default=["rexu_v3", "raw_maxsim", "mean_cos"])
    args = ap.parse_args()

    e = me.MeetingMemoryEngine(device="cuda")
    ds = e._get_dataset("qmsum")
    qs = [q for q in ds["queries"] if q.get("split") == "test"]
    r = RexuV3Retriever(e, "qmsum", max_tokens=args.max_tokens)
    needs_context = any(v != "pplx_embed_v1" for v in args.variants)
    if needs_context:
        r._ensure()
    plain = None
    if "pplx_embed_v1" in args.variants:
        model, tok = load_plain("cuda")
        plain = {"model": model, "tok": tok, "docs": {}}

    results = {}
    for variant in args.variants:
        ranked = [(rank(r, ds, q, variant, plain), set(q["target_chunk_ids"])) for q in qs]
        results[variant] = metrics(ranked)
        print(variant, json.dumps(results[variant]), flush=True)

    payload = {"dataset": "qmsum-test", "scope": "meeting", "backbone": "pplx-embed-context-v1-0.6b (stock)",
               "max_tokens_per_window": args.max_tokens, "n_queries": len(qs), "variants": results}
    (ROOT / "artifacts" / f"qmsum_pplx_variants_w{args.max_tokens}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
