#!/usr/bin/env python3
"""Fine-tune the Rexu v3 late head with set-level hard-negative loss.

The pplx-context backbone stays frozen. Evaluation uses only each dataset's
held-out test queries.
"""
from __future__ import annotations

import json, random, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import memory_engine as me

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'archive' / 'rexu_v3_1'))
from evaluate_pplx_variants import PplxContextTokens
from pplx_rexu import LateHead

ART = ROOT / "artifacts"
DEVICE = "cuda"


def maxsim(q, d):
    return (q @ d.T).max(dim=1).values.sum()


@dataclass
class Pack:
    name: str
    records: dict
    by_meeting: dict
    queries: list
    docs_mean: dict
    docs_tokens: dict
    qmean: dict
    qtokens: dict

    def candidates(self, q):
        mid = q.get("meeting_id")
        return self.by_meeting.get(mid, []) if mid else list(self.records)

    def targets(self, q):
        x = q.get("target_chunk_ids") or q.get("evidence_chunk_ids") or q.get("evidence") or []
        if isinstance(x, str): x = [x]
        if not x and q.get("target_chunk_id"): x = [q["target_chunk_id"]]
        return x


def load_packs():
    # v7 train split, built directly from the JSON corpus (no engine/sqlite
    # dependency: the artifact cache is not required to rebuild the head).
    v7m = json.loads((ROOT / "data/french_synthetic_meetings.json").read_text())
    v7q = json.loads((ROOT / "data/french_retrieval_queries.json").read_text())
    r7, b7 = {}, {}
    for split, rows in v7m.items():
        for m in rows:
            mid = m["meeting_id"]; b7[mid] = []
            for x in m["messages"]:
                cid = f"{mid}:{int(x['message_id']):03d}"; b7[mid].append(cid)
                text = x.get("message", x.get("text", ""))
                speaker = x.get("username", x.get("speaker", ""))
                r7[cid] = {"chunk_id": cid, "meeting_id": mid, "encoder_text":
                           f"Réunion : {m['title']}. Intervenant : {speaker}. "
                           f"Intervention : {text}", "message_text": text}
    # v8
    d8 = json.loads((ROOT / "data/v8_meetings.json").read_text())
    q8 = json.loads((ROOT / "data/v8_queries.json").read_text())
    if isinstance(q8, dict): q8 = q8["queries"]
    d8m = d8["meetings"] if isinstance(d8, dict) else d8
    r8, b8 = {}, {}
    for m in d8m:
        mid = m["meeting_id"]; b8[mid] = []
        for x in m["messages"]:
            cid = x.get("id") or f"{mid}:{x['message_id']:04d}"; b8[mid].append(cid)
            speaker = x.get("speaker", x.get("username", ""))
            text = x.get("text", x.get("message", ""))
            r8[cid] = {"chunk_id": cid, "meeting_id": mid, "encoder_text":
                       f"Réunion : {m['title']}. Intervenant : {speaker}. Intervention : {text}",
                       "message_text": text}
    # v9
    d9 = json.loads((ROOT / "data/v9_meetings.json").read_text())
    q9 = json.loads((ROOT / "data/v9_queries.json").read_text())
    if isinstance(q9, dict): q9 = q9["queries"]
    d9m = d9["meetings"] if isinstance(d9, dict) else d9
    r9, b9 = {}, {}
    for m in d9m:
        mid = m["meeting_id"]; b9[mid] = []
        for x in m["messages"]:
            cid = x["id"]; b9[mid].append(cid)
            r9[cid] = {"chunk_id": cid, "meeting_id": mid, "encoder_text":
                       f"Réunion : {m['title']}. Intervenant : {x['speaker']}. Intervention : {x['text']}",
                       "message_text": x["text"]}
    packs = [Pack("v7", r7, b7, v7q, {}, {}, {}, {}),
             Pack("v8", r8, b8, q8, {}, {}, {}, {}),
             Pack("v9", r9, b9, q9, {}, {}, {}, {})]
    enc = PplxContextTokens()
    for p in packs:
        print("encode", p.name, len(p.records), flush=True)
        for mid, chunks in p.by_meeting.items():
            mean, toks = enc.encode_meeting([p.records[c]["encoder_text"] for c in chunks])
            for c, mv, tv in zip(chunks, mean, toks):
                p.docs_mean[c] = mv; p.docs_tokens[c] = tv
        qid = lambda q, i: q.get("query_id") or q.get("id") or f"{p.name}-{i}"
        for i, q in enumerate(p.queries):
            m, t = enc.encode_one(q["query"])
            p.qmean[qid(q, i)] = m; p.qtokens[qid(q, i)] = t
    return packs


def main():
    torch.manual_seed(17); random.seed(17)
    packs = load_packs()
    late = LateHead().to(DEVICE)
    head_path = ART / "pplx_v3_late_head_multi.pt"
    if head_path.exists():
        late.load_state_dict(torch.load(head_path, map_location="cpu",
                                        weights_only=False)["late"])
    else:
        print("no existing late head; training from scratch", flush=True)
    late.train()
    params = list(late.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-3)
    train_by = {p.name: [q for q in p.queries if q.get("split") == "train"] for p in packs}
    print("train counts", {k: len(v) for k, v in train_by.items()}, flush=True)

    @torch.inference_mode()
    def hard_negative(pack, q, positives, k=5):
        cand = pack.candidates(q)
        qid = q.get("query_id") or q.get("id") or f"{pack.name}-{pack.queries.index(q)}"
        ql = late(pack.qtokens[qid].to(DEVICE))
        # Score all candidates under the current head and choose the hardest
        # non-evidence chunks. Same meeting/candidate pool only.
        scores = np.asarray([float(maxsim(ql, late(pack.docs_tokens[c].to(DEVICE))))
                             for c in cand])
        neg = [cand[i] for i in np.argsort(-scores) if cand[i] not in positives][:k]
        return neg

    def sample(pack, n):
        pool = train_by[pack.name]
        out = []
        for _ in range(n):
            q = random.choice(pool); cand = pack.candidates(q)
            positives = [t for t in dict.fromkeys(pack.targets(q)) if t in cand]
            if not positives: continue
            neg = hard_negative(pack, q, positives, k=5)
            if not neg:
                neg = [c for c in cand if c not in positives][:5]
            out.append((pack, q, positives, positives + neg))
        return out

    for step in range(1, 251):
        batch = sample(packs[0], 4) + sample(packs[1], 2) + sample(packs[2], 2)
        if not batch: break
        losses = []
        for pack, q, positives, cand in batch:
            qid = q.get("query_id") or q.get("id") or f"{pack.name}-{pack.queries.index(q)}"
            ql = late(pack.qtokens[qid].to(DEVICE))
            sl = torch.stack([maxsim(ql, late(pack.docs_tokens[c].to(DEVICE))) for c in cand])
            p = len(positives)
            # Set-level objective: all evidence chunks outrank all hard negatives.
            listwise = -torch.logsumexp(sl[:p], dim=0) + torch.logsumexp(sl, dim=0)
            losses.append(listwise)
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0: print("step", step, "loss", round(float(loss), 4), flush=True)
    late.eval()
    torch.save({"late": late.state_dict()}, ART / "pplx_v3_late_head_multi.pt")

    @torch.inference_mode()
    def eval_pack(pack, mode):
        qs = [q for q in pack.queries if q.get("split") == "test"]
        ranks = []; ev = []; ah = []; details = []
        for qi, q in enumerate(qs):
            qid = q.get("query_id") or q.get("id") or f"{pack.name}-{pack.queries.index(q)}"
            cand = pack.candidates(q)
            ql = late(pack.qtokens[qid].to(DEVICE))
            sc = np.asarray([float(maxsim(ql, late(pack.docs_tokens[c].to(DEVICE)))) for c in cand])
            order = [cand[i] for i in np.argsort(-sc)]
            gold = set(pack.targets(q))
            p = [order.index(c) + 1 for c in gold if c in order]
            rank = min(p) if p else len(order) + 1
            found = set(order[:5]) & gold
            ranks.append(rank); ev.append(len(found) / len(gold)); ah.append(float(found == gold))
            details.append({"query": q["query"], "rank": rank, "evidence@5": ev[-1],
                            "all_hops@5": ah[-1], "top5": order[:5]})
        a = np.asarray(ranks)
        return {"R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
                "MRR": float(np.where(a <= 10, 1 / a, 0).mean()),
                "EvidenceRecall@5": float(np.mean(ev)), "AllHops@5": float(np.mean(ah)),
                "n": len(a), "details": details}
    out = {p.name: eval_pack(p, "late") for p in packs}
    for name, v in out.items():
        print(name, {k: v[k] for k in ("R@1","R@5","MRR","EvidenceRecall@5","AllHops@5")}, flush=True)
    (ART / "pplx_multidataset_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
