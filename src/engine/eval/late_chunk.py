"""Late chunking with jina v5, evaluated on the repository's own units.

`chonkie.LateChunker` takes a SentenceTransformer and re-chunks the document by
its own rules; it cannot take v5's custom model, and its chunks are not our
turns. So this implements the same idea directly:

    one forward over the whole meeting
    -> token-level hidden states
    -> mean-pool each turn's token span
    -> that vector is the turn's embedding, carrying its meeting context

Then the two things an embedding is for in this architecture:
step 6 (does the true antecedent rank in the top k) and step 4 (gold vs decoy).

Decoder caveat: v5 is a Qwen3 decoder, so a token's state sees only its left
context, not the future. Encoder-based late chunking sees both directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from engine.config import repo_root
from engine.eval.jina_ab import _norm, pair_set
from engine.eval.mlm_link import auc
from engine.eval.real_pairs import load_meetings
from engine.eval.v4_data import citation_set, load_edges, texts_of

V5 = "jinaai/jina-embeddings-v5-text-small"


class LateV5:
    def __init__(self, device: str = "cuda", max_len: int = 32768):
        from transformers import AutoModel

        self.device = device
        self.max_len = max_len
        self.m = AutoModel.from_pretrained(
            V5, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
        self.base = self.m.base_model.model
        self.tok = self.m.tokenizer
        self.tok.padding_side = "right"

    @torch.no_grad()
    def meeting(self, texts: list[str]) -> np.ndarray:
        """One embedding per turn, each pooled inside the meeting's forward pass."""
        keep = [i for i, t in enumerate(texts) if t.strip()]
        doc, spans, pos = "", {}, 0
        for i in keep:
            if doc:
                doc += "\n"
                pos += 1
            spans[i] = (pos, pos + len(texts[i]))
            doc += texts[i]
            pos += len(texts[i])
        batch = self.tok(doc, return_offsets_mapping=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        offsets = batch.pop("offset_mapping")[0].tolist()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        h = self.base(**batch).last_hidden_state[0].float()
        out = np.zeros((len(texts), h.size(-1)), dtype=np.float32)
        for i in keep:
            s, e = spans[i]
            idx = [j for j, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e]
            if not idx:
                out[i] = h.mean(0).cpu().numpy()
            else:
                out[i] = h[idx].mean(0).cpu().numpy()
        return _norm(out)


def evaluate(late: LateV5, root: Path) -> dict:
    meetings = load_meetings(root)
    edges = load_edges(root)
    emb = {mid: late.meeting(texts_of(meetings, mid)) for mid in sorted(edges)}

    hits = {5: 0, 10: 0, 20: 0}
    total = 0
    for mid in sorted(edges):
        texts = texts_of(meetings, mid)
        for e in edges[mid]:
            pool = [i for i in range(min(e.new, len(texts))) if texts[i].strip()]
            if not pool:
                continue
            sims = emb[mid][pool] @ emb[mid][e.new]
            order = [pool[j] for j in np.argsort(-sims)]
            valid = citation_set(meetings, mid, e.old, e.new)
            total += 1
            for cut in hits:
                if any(i in valid for i in order[:cut]):
                    hits[cut] += 1

    _, rows = pair_set(root)
    out = {"retrieval": {"n": total,
                         **{f"recall@{c}": f"{h}/{total}" for c, h in hits.items()}}}
    for decoy in ("slot", "topic"):
        sub = [r for r in rows if decoy in r]
        if not sub:
            continue
        gold = [float(emb[r["meeting"]][r["new"]] @ emb[r["meeting"]][r["old"]])
                for r in sub]
        neg = [float(emb[r["meeting"]][r["new"]] @ emb[r["meeting"]][r[decoy]])
               for r in sub]
        out[decoy] = {"n": len(sub), "gold": round(float(np.mean(gold)), 4),
                      "decoy": round(float(np.mean(neg)), 4),
                      "auc": round(auc(gold, neg), 4)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v4/late_chunk.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = repo_root()
    late = LateV5(args.device)
    res = evaluate(late, root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
