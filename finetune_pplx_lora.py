#!/usr/bin/env python3
"""LoRA fine-tune pplx-context backbone for Rexu v3 late retrieval.

Frozen-head baseline: v7 .784/.858, v8 .900/.950.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import memory_engine as me
from pplx_rexu import LateHead, RexuV3Retriever

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"; HF = ART / "hf_cache"; DEVICE = "cuda"
MODEL = "perplexity-ai/pplx-embed-context-v1-0.6b"


def maxsim(q, d):
    return (q @ d.T).max(dim=1).values.sum()


class Trainer:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model
        self.tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(HF), trust_remote_code=True)
        base = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                         cache_dir=str(HF)).to(DEVICE)
        base.gradient_checkpointing_enable()
        cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         lora_dropout=0.05, bias="none")
        self.model = get_peft_model(base, cfg).to(DEVICE)
        self.model.train()
        self.late = LateHead().to(DEVICE)
        self.late.load_state_dict(torch.load(ART / "pplx_v3_late_head.pt",
                                             map_location="cpu", weights_only=False)["late"])
        self.late.train()
        self.sep = self.tok.sep_token

    def _parts(self, hidden, ids, mask):
        seps = ((ids == self.tok.sep_token_id) & mask).nonzero(as_tuple=True)[0].tolist()
        segments, start = [], 0
        for p in seps:
            segments.append((start, p)); start = p + 1
        segments.append((start, int(mask.sum())))
        out = []
        for a, b in segments:
            t = hidden[a:b][mask[a:b]]
            if t.size(0): out.append(F.normalize(t, dim=-1))
        return out

    def encode_meeting(self, chunks):
        text = self.sep.join(chunks)
        enc = self.tok([text], padding=True, truncation=True, max_length=4096,
                       return_tensors="pt").to(DEVICE)
        h = self.model(**enc).last_hidden_state[0]
        return self._parts(h, enc["input_ids"][0], enc["attention_mask"][0].bool())

    def encode_query(self, text):
        enc = self.tok([text + self.sep], padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to(DEVICE)
        h = self.model(**enc).last_hidden_state[0]
        return self._parts(h, enc["input_ids"][0], enc["attention_mask"][0].bool())[0]

    def eval_scores(self, query, chunks):
        with torch.inference_mode():
            self.model.eval()
            qt = self.late(self.encode_query(query))
            dt = [self.late(t) for t in self.encode_meeting(chunks)]
            self.model.train()
        return np.asarray([float(maxsim(qt, d)) for d in dt])

    def cache_docs(self, texts_by_meeting):
        with torch.inference_mode():
            self.model.eval()
            out = {mid: [self.late(t) for t in self.encode_meeting(chunks)]
                   for mid, chunks in texts_by_meeting.items()}
            self.model.train()
        return out


def main():
    torch.manual_seed(3); random.seed(3)
    eng = me.MeetingMemoryEngine(device=DEVICE); ds = eng._get_dataset("synthetic")
    queries = json.loads((ROOT / "data/french_retrieval_queries.json").read_text())
    train = [q for q in queries if q["split"] == "train"]
    tr = Trainer()
    params = [p for p in tr.model.parameters() if p.requires_grad] + list(tr.late.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4)
    print("trainable", sum(p.numel() for p in params), flush=True)
    for step in range(1, 201):
        q = random.choice(train); mid = q["meeting_id"]; chunks = ds["by_meeting"][mid]
        target = q["target_chunk_id"]
        negs = [c for c in chunks if c != target]
        negs = random.sample(negs, min(4, len(negs)))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            docs = tr.encode_meeting([ds["records"][c]["encoder_text"] for c in chunks])
            qt = tr.late(tr.encode_query(q["query"]))
            # target + negatives token vectors; listwise over 5 candidates
            idx = {c: i for i, c in enumerate(chunks)}
            cand = [target] + negs
            scores = torch.stack([maxsim(qt, tr.late(docs[idx[c]])) for c in cand])
            loss = F.cross_entropy(scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=DEVICE))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0:
            print("step", step, "loss", round(float(loss), 4), flush=True)
    tr.model.eval(); tr.late.eval()
    torch.save({"adapter": {k: v.detach().cpu() for k, v in tr.model.state_dict().items()
                            if "lora" in k}, "late": tr.late.state_dict()},
               ART / "pplx_v3_lora.pt")

    def eval_v7():
        qs = [q for q in queries if q["split"] == "test"]
        texts = {mid: [ds["records"][c]["encoder_text"] for c in chunks]
                 for mid, chunks in ds["by_meeting"].items()}
        cache = tr.cache_docs(texts)
        ranks = []
        for q in qs:
            chunks = ds["by_meeting"][q["meeting_id"]]
            with torch.inference_mode():
                tr.model.eval(); qt = tr.late(tr.encode_query(q["query"]))
                sc = np.asarray([float(maxsim(qt, d)) for d in cache[q["meeting_id"]]])
            order = [chunks[i] for i in np.argsort(-sc)]
            gold = set(q["target_chunk_ids"]); p = [order.index(c) + 1 for c in gold if c in order]
            ranks.append(min(p) if p else len(order) + 1)
        a = np.asarray(ranks)
        return {"R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
                "MRR": float(np.where(a <= 10, 1/a, 0).mean()), "n": len(a)}
    v7 = eval_v7(); print("v7", v7, flush=True)
    # v8
    d8 = json.loads((ROOT / "data/v8_meetings.json").read_text())
    q8 = json.loads((ROOT / "data/v8_queries.json").read_text())
    v8_texts = {m["meeting_id"]: [f"Réunion : {m['title']}. Intervenant : {x['username']}. Intervention : {x['message']}" for x in m["messages"]] for m in d8["meetings"]}
    v8_cache = tr.cache_docs(v8_texts)
    ranks = []; ev = []; ah = []
    for q in q8:
        scored = []
        with torch.inference_mode():
            tr.model.eval(); qt = tr.late(tr.encode_query(q["query"]))
            for m in d8["meetings"]:
                sc = [float(maxsim(qt, d)) for d in v8_cache[m["meeting_id"]]]
                for i, s in enumerate(sc):
                    scored.append((f"{m['meeting_id']}:{m['messages'][i]['message_id']:04d}", s))
        scored.sort(key=lambda x: -x[1]); order = [c for c, _ in scored]
        gold = set(q["evidence_chunk_ids"]); p = [order.index(c) + 1 for c in gold if c in order]
        ranks.append(min(p) if p else len(order) + 1)
        found = set(order[:5]) & gold; ev.append(len(found)/len(gold)); ah.append(float(found == gold))
    a = np.asarray(ranks)
    v8 = {"R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
          "MRR": float(np.where(a <= 10, 1/a, 0).mean()),
          "EvidenceRecall@5": float(np.mean(ev)), "AllHops@5": float(np.mean(ah)), "n": len(a)}
    print("v8", v8, flush=True)
    out = {"v7": v7, "v8": v8, "meta": {"fine_tune": "LoRA r=8 on q/k/v/o/gate/up/down, 300 steps",
                                        "train_queries": len(train), "frozen": "pplx-context base otherwise frozen"}}
    (ART / "pplx_lora_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
