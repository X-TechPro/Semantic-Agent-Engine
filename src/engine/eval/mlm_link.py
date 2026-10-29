"""Does a token-level likelihood ratio separate "continues" from "same topic"?

The cheap pre-test for the state/SSM idea, using the checkpoint already on disk.

For a target sentence B and two contexts A (the gold antecedent) and A' (same
entity, different thread), score the masked pseudo-likelihood of B under each:

    PLL(B | A) = sum_i log p(b_i | A, B without b_i)

`p(b_i | ...)` is read from the tied input/output embeddings, because this
checkpoint ships no LM head (`pred_layer` is absent; `share_inout_emb` is true,
so the missing bias is the only approximation).

The null context (`PLL(B)`) is computed as well, but note: when both contexts
are compared on the *same* B, the null cancels exactly. It matters only for a
global threshold across different sentences, not for the pairwise AUC.

Also measured: whether the highest-surprise tokens land on the value that
changed ("une ou deux semaines" -> "trois semaines"). That is the signal the
state layer would need, and no pooled embedding exposes it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from engine.eval.v4_data import load_edges, texts_of
from engine.link.thread import detect_topic
from engine.read.v4_tags import extract_entities, read_tags

MODEL = "inokufu/flaubert-base-uncased-xnli-sts"


class MaskedScorer:
    def __init__(self, model_id: str = MODEL, device: str = "cuda",
                 max_ctx: int = 96, max_tgt: int = 64):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device if torch.cuda.is_available()
                                   or device == "cpu" else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.E = self.model.get_input_embeddings().weight.detach()
        self.mask_id = int(self.tok.mask_token_id)          # 5
        self.bos, self.sep = 0, 1
        self.max_ctx, self.max_tgt = max_ctx, max_tgt

    def encode(self, text: str) -> list[int]:
        return self.tok(text or "", add_special_tokens=False).input_ids

    def _sequences(self, ctx_ids: list[int], tgt_ids: list[int],
                   causal: bool = True):
        """One sequence per scored token.

        `causal=True` masks the token *and every token after it*, so the model
        predicts b_i from A and b_<i only. That is the quantity a sequence model
        (or an SSM) would compute. Masking only the token itself leaves the rest
        of B visible, which lets the sentence predict itself and makes the
        context almost irrelevant — a measured dead end.
        """
        prefix = [self.bos] + ctx_ids[-self.max_ctx:] + [self.sep]
        seqs = []
        for i, tok_id in enumerate(tgt_ids):
            seq = prefix + list(tgt_ids) + [self.sep]
            pos = len(prefix) + i
            if causal:
                for j in range(pos, len(prefix) + len(tgt_ids)):
                    seq[j] = self.mask_id
            else:
                seq[pos] = self.mask_id
            seqs.append((seq, pos, tok_id))
        return seqs

    def pll(self, contexts: list[str], targets: list[str], batch: int = 32,
            causal: bool = True) -> list[dict]:
        """Return per-target sum log p, token count and per-token surprises."""
        jobs = []
        order = []
        for k, (ctx, tgt) in enumerate(zip(contexts, targets)):
            tgt_ids = self.encode(tgt)[:self.max_tgt]
            if not tgt_ids:
                order.append((k, [], 0.0))
                continue
            for item in self._sequences(self.encode(ctx), tgt_ids, causal):
                jobs.append((k, item[0], item[1], item[2]))
            order.append((k, tgt_ids, None))

        sums: dict[int, list[float]] = {}
        with torch.no_grad():
            for s in range(0, len(jobs), batch):
                chunk = jobs[s:s + batch]
                width = max(len(x[1]) for x in chunk)
                ids = torch.full((len(chunk), width), self.tok.pad_token_id,
                                 dtype=torch.long)
                att = torch.zeros((len(chunk), width), dtype=torch.long)
                for r, (_k, seq, _pos, _t) in enumerate(chunk):
                    ids[r, :len(seq)] = torch.tensor(seq)
                    att[r, :len(seq)] = 1
                ids = ids.to(self.device)
                att = att.to(self.device)
                h = self.model(input_ids=ids, attention_mask=att
                               ).last_hidden_state
                pos = torch.tensor([x[2] for x in chunk], device=self.device)
                tgt = torch.tensor([x[3] for x in chunk], device=self.device)
                h_pos = h[torch.arange(len(chunk), device=self.device), pos]
                logits = h_pos @ self.E.T                       # tied output
                logp = torch.log_softmax(logits.float(), dim=-1)
                vals = logp.gather(1, tgt[:, None]).squeeze(1).cpu().tolist()
                for (k, _seq, _pos, _t), v in zip(chunk, vals):
                    sums.setdefault(k, []).append(float(v))

        out = []
        for k, tgt_ids, _ in order:
            vals = sums.get(k, [])
            out.append({"pll": sum(vals) / max(1, len(vals)) if vals else 0.0,
                        "sum": sum(vals), "n": len(vals),
                        "per_token": vals})
        return out


def decoys(texts: list[str], old: int, new: int,
           gold_olds: set[int] | None = None) -> dict[str, int]:
    """Two negatives with different meanings.

    `slot`   — same point, different value, highest word overlap. Hard, but
               ambiguous: in a negotiation chain an earlier value can also be a
               valid citation, so failing this is not necessarily an error.
    `topic`  — a different thread, highest word overlap. Unambiguously not the
               antecedent: this is the "same words, other subject" case.
    """
    from engine.store.fts import tokenize

    def jac(a: str, b: str) -> float:
        A = set(tokenize(a)); B = set(tokenize(b))
        return len(A & B) / max(1, len(A | B))

    exclude = {old, new} | set(gold_olds or ())
    tag_new = read_tags(texts[new])
    topic_new = detect_topic(texts[new])
    out: dict[str, int] = {}
    slot_pool, topic_pool = [], []
    for j, t in enumerate(texts):
        if j in exclude or not t.strip():
            continue
        tag = read_tags(t)
        if tag_new.slot and tag.slot == tag_new.slot \
                and tag.value.strip().lower() != tag_new.value.strip().lower():
            slot_pool.append((jac(texts[new], t), j))
        topic_j = detect_topic(t)
        if topic_new and topic_j and topic_j != topic_new:
            topic_pool.append((jac(texts[new], t), j))
    if slot_pool:
        out["slot"] = max(slot_pool)[1]
    if topic_pool:
        out["topic"] = max(topic_pool)[1]
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score_positive > score_negative) over all pairs."""
    if not pos or not neg:
        return 0.0
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/v4/mlm_link.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    edges = load_edges(root)
    scorer = MaskedScorer(device=args.device)

    rows = []
    for mid in sorted(edges):
        texts = texts_of(meetings, mid)
        gold_olds: dict[int, set[int]] = {}
        for e in edges[mid]:
            gold_olds.setdefault(e.new, set()).add(e.old)
        for e in edges[mid]:
            if e.new >= len(texts) or e.old >= len(texts):
                continue
            pick = decoys(texts, e.old, e.new, gold_olds.get(e.new))
            if not pick:
                continue
            rows.append({"meeting": mid, "relation": e.relation,
                         "old": e.old, "new": e.new,
                         "neg_slot": pick.get("slot"),
                         "neg_topic": pick.get("topic"),
                         "new_text": texts[e.new], "old_text": texts[e.old],
                         "slot_text": texts[pick["slot"]]
                         if "slot" in pick else None,
                         "topic_text": texts[pick["topic"]]
                         if "topic" in pick else None})
    if args.limit:
        rows = rows[:args.limit]
    print("pairs:", len(rows),
          "| with slot decoy:", sum(r["neg_slot"] is not None for r in rows),
          "| with topic decoy:", sum(r["neg_topic"] is not None for r in rows),
          flush=True)

    targets = [r["new_text"] for r in rows]
    null = scorer.pll([""] * len(rows), targets)
    gold = scorer.pll([r["old_text"] for r in rows], targets)
    for r, n, g in zip(rows, null, gold):
        r["pll_null"] = round(n["pll"], 4)
        r["delta_gold"] = round(g["pll"] - n["pll"], 4)
        r["surprise_gold"] = [round(-v, 3) for v in g["per_token"]]

    for kind, key in (("slot", "neg_slot"), ("topic", "neg_topic")):
        idx = [i for i, r in enumerate(rows) if r[key] is not None]
        if not idx:
            continue
        out = scorer.pll([rows[i][f"{kind}_text"] for i in idx],
                         [rows[i]["new_text"] for i in idx])
        for i, o in zip(idx, out):
            rows[i][f"delta_{kind}"] = round(o["pll"] - rows[i]["pll_null"], 4)

    # cosine on exactly the same triples, for reference
    def embed(texts, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = scorer.tok(texts[i:i + bs], padding=True,
                                 truncation=True, max_length=256,
                                 return_tensors="pt").to(scorer.device)
                h = scorer.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 1)

    tg = embed(targets)
    go = embed([r["old_text"] for r in rows])
    cos_gold = (go * tg).sum(1).tolist()
    for r, c in zip(rows, cos_gold):
        r["cos_gold"] = round(c, 4)
    for kind in ("slot", "topic"):
        idx = [i for i, r in enumerate(rows)
               if r.get(f"delta_{kind}") is not None]
        if not idx:
            continue
        cn = embed([rows[i][f"{kind}_text"] for i in idx])
        cos_neg = (cn * tg[idx]).sum(1).tolist()
        for i, c in zip(idx, cos_neg):
            rows[i][f"cos_{kind}"] = round(c, 4)

    by_rel: dict[str, list] = {}
    for r in rows:
        by_rel.setdefault(r["relation"], []).append(r)

    def block(rs: list[dict], kind: str) -> dict:
        pairs = [(r["delta_gold"], r[f"delta_{kind}"], r["cos_gold"],
                  r[f"cos_{kind}"]) for r in rs
                 if r.get(f"delta_{kind}") is not None]
        if not pairs:
            return {"n": 0}
        pos = [p[0] for p in pairs]
        negv = [p[1] for p in pairs]
        cpos = [p[2] for p in pairs]
        cneg = [p[3] for p in pairs]
        prefer = sum(p > n for p, n in zip(pos, negv))
        return {"n": len(pairs),
                "mean_delta_gold": round(statistics.mean(pos), 3),
                "mean_delta_neg": round(statistics.mean(negv), 3),
                "gold_preferred": prefer,
                "gold_preferred_rate": round(prefer / len(pairs), 4),
                "auc_pll": auc(pos, negv),
                "auc_cosine": auc(cpos, cneg),
                "mean_cos_gold": round(statistics.mean(cpos), 3),
                "mean_cos_neg": round(statistics.mean(cneg), 3)}

    result = {"model": MODEL, "pairs": len(rows),
              "gold_vs_slot": block(rows, "slot"),
              "gold_vs_topic": block(rows, "topic"),
              "by_relation": {k: {"slot": block(v, "slot"),
                                  "topic": block(v, "topic")}
                              for k, v in sorted(by_rel.items())}}

    # where does the surprise land? Only for supersede pairs with a value.
    loc_hits = loc_total = 0
    examples = []
    for r in rows:
        if r["relation"] != "supersedes":
            continue
        new_vals = [e for e in extract_entities(r["new_text"])
                    if e["type"] in ("amount", "date")]
        old_vals = {str(e["value"]).lower() for e in
                    extract_entities(r["old_text"]) if e["type"] in
                    ("amount", "date")}
        if not new_vals:
            continue
        ent = next((e for e in new_vals
                    if str(e["value"]).lower() not in old_vals), new_vals[0])
        start, end = int(ent["start"]), int(ent["end"])
        if end <= start:
            continue
        before = scorer.encode(r["new_text"][:start])
        inside = scorer.encode(r["new_text"][start:end])
        if not inside:
            continue
        lo, hi = len(before), len(before) + len(inside)
        sur = r["surprise_gold"]
        if hi > len(sur):
            continue
        loc_total += 1
        peak = max(range(len(sur)), key=lambda i: sur[i])
        hit = lo <= peak < hi
        loc_hits += hit
        if len(examples) < 5:
            examples.append({"new": r["new_text"][:90], "value": ent["value"],
                             "peak_token": int(peak), "value_span": [lo, hi],
                             "hit": bool(hit)})
    result["value_localization"] = {
        "n": loc_total, "peak_in_value": loc_hits,
        "rate": round(loc_hits / loc_total, 4) if loc_total else None,
        "examples": examples}

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**result, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
