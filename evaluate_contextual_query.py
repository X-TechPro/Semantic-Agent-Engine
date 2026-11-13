#!/usr/bin/env python3
"""Late interaction with a meeting-contextualised query (QMSum test, meeting-scoped).

Both arms keep the documents as before (meeting windows, late chunking) but encode the
query inside each window of its meeting: the input is window_text + query, and only the
query's token states are kept. The query therefore attends to the meeting, the same way
the document tokens do. Each chunk is scored with the query states from the window that
contains it, using MaxSim.

Arms (reported only):
- pplx_ctx_query      : pplx-embed-context-v1-0.6b stock tokens, no trained head, 15k windows
- bge_colbert_ctx_query: BGE-M3 late-chunked tokens with the official colbert_linear
                         projection (applied to the contextual states, CLS dropped), 8.1k windows

Writes artifacts/qmsum_contextual_query_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import memory_engine as me
from evaluate_qmsum_baselines import metrics
from pplx_rexu import RexuV3Retriever
from colbert_v2 import LateChunkedColBERTv2

ROOT = Path(__file__).resolve().parent
HF = ROOT / "artifacts" / "hf_cache"
PPLX_WINDOW = 15000
BGE_BUDGET = 8100


@torch.inference_mode()
def pplx_query_tokens(r: RexuV3Retriever, window_text: str, query: str) -> torch.Tensor:
    """Contextual token states of the query, read from the forward pass over window + query."""
    tok = r._tok
    text = window_text + tok.sep_token + query
    enc = tok([text], padding=True, truncation=True, max_length=r.max_tokens + 1024,
              return_tensors="pt").to(r.device)
    hidden = r._model(**enc).last_hidden_state.float()[0]
    ids = enc["input_ids"][0]
    mask = enc["attention_mask"][0].bool()
    seps = ((ids == tok.sep_token_id) & mask).nonzero(as_tuple=True)[0]
    start, end = int(seps[-1]) + 1, int(mask.sum())
    toks = hidden[start:end][mask[start:end]]
    return F.normalize(toks, dim=-1)


def run_pplx(e, ds, qs_by_meeting) -> dict[str, list]:
    r = RexuV3Retriever(e, "qmsum", max_tokens=PPLX_WINDOW)
    r._ensure()  # loads model and the cached per-chunk document tokens (head is not used)
    ranked = {}
    window_counts = []
    for mid, qs in qs_by_meeting.items():
        cands = ds["by_meeting"][mid]
        windows = r._windows(cands)
        window_counts.append(len(windows))
        for q in qs:
            scores: dict[str, float] = {}
            for window in windows:
                text = r._tok.sep_token.join(r.records[c]["encoder_text"] for c in window)
                qt = pplx_query_tokens(r, text, q["query"])
                for c in window:
                    dt = r._docs_tokens[c].to(r.device)
                    scores[c] = float((qt @ dt.T).max(dim=1).values.sum())
            order = sorted(cands, key=lambda c: -scores[c])
            ranked[q["query_id"]] = (order, set(q["target_chunk_ids"]))
            print(f"  pplx {len(ranked)} queries done", flush=True) if len(ranked) % 20 == 0 else None
    print("pplx windows per meeting:", sorted(window_counts), flush=True)
    return ranked


def load_bge_projection(device: str) -> nn.Linear:
    path = hf_hub_download("BAAI/bge-m3", "colbert_linear.pt", cache_dir=str(HF))
    linear = nn.Linear(1024, 1024).to(device)
    linear.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return linear.eval()


@torch.inference_mode()
def bge_projected(lc: LateChunkedColBERTv2, linear: nn.Linear, text: str):
    enc = lc._tokenizer([text], return_offsets_mapping=True, truncation=True,
                        max_length=lc.max_len, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(lc.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    hidden = lc._model(**enc).last_hidden_state[0].float()
    return F.normalize(linear(hidden), dim=-1), offsets


def run_bge(e, ds, qs_by_meeting) -> dict[str, list]:
    lc = LateChunkedColBERTv2(e._get_dataset("qmsum")["records"], ds["meetings"], "qmsum",
                              device=e.device)
    lc._ensure_model()
    linear = load_bge_projection(lc.device)
    ranked = {}
    window_counts = []
    missing = 0
    for mid, qs in qs_by_meeting.items():
        cands = ds["by_meeting"][mid]
        windows = lc._meeting_windows(mid, BGE_BUDGET)
        window_counts.append(len(windows))
        # Document side: no query in the input, so these states are query-independent.
        doc_windows = []
        for document, spans, window_ids in windows:
            proj, offsets = bge_projected(lc, linear, document)
            chunk_tokens = {}
            for cid in window_ids:
                start, end = spans[cid]
                idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
                if idx:
                    chunk_tokens[cid] = proj[idx]
                else:
                    missing += 1
            doc_windows.append((document, window_ids, chunk_tokens))
        for q in qs:
            scores: dict[str, float] = {}
            for document, window_ids, chunk_tokens in doc_windows:
                # Query side: the query is appended to this window; keep only its tokens.
                proj, offsets = bge_projected(lc, linear, document + q["query"])
                q_idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= len(document)]
                qt = proj[q_idx]
                for cid, dt in chunk_tokens.items():
                    scores[cid] = float((qt @ dt.T).max(dim=1).values.sum())
            for cid in cands:
                scores.setdefault(cid, float("-inf"))
            order = sorted(cands, key=lambda c: -scores[c])
            ranked[q["query_id"]] = (order, set(q["target_chunk_ids"]))
            print(f"  bge {len(ranked)} queries done", flush=True) if len(ranked) % 20 == 0 else None
    print("bge windows per meeting:", sorted(window_counts), "chunks without tokens:", missing, flush=True)
    return ranked


def main() -> None:
    e = me.MeetingMemoryEngine(device="cuda")
    ds = e._get_dataset("qmsum")
    qs = [q for q in ds["queries"] if q.get("split") == "test"]
    by_meeting: dict[str, list] = {}
    for q in qs:
        by_meeting.setdefault(q["meeting_id"], []).append(q)

    results = {}
    for name, runner in (("pplx_ctx_query", run_pplx), ("bge_colbert_ctx_query", run_bge)):
        ranked = runner(e, ds, by_meeting)
        results[name] = metrics([ranked[q["query_id"]] for q in qs])
        print(name, json.dumps(results[name]), flush=True)
        torch.cuda.empty_cache()

    payload = {"dataset": "qmsum-test", "scope": "meeting", "n_queries": len(qs), "arms": results}
    (ROOT / "artifacts" / "qmsum_contextual_query_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
