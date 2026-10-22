"""Fine-tune the embedder on synthetic ledger data (BUILD_GUIDE 6.4 / 9.1).

What it should teach and what it must not break:

* teach: keep apart the minimal pairs the ledger can prove — status, owner, date,
  number, attribute. These are `hard_negatives`' constructed rows, 43% of the mix
  where 6.4 demands >= 25%, and they are deliberately NOT filtered by the
  NV-Retriever similarity rule (being near-identical is the point);
* must not break: general French. A held-out slice of MASSIVE-fr pairs that the
  trainer never sees is scored before and after. That slice is disjoint BY
  SENTENCE (`BUCKET_PROBE`), not by seed — two seeds over one corpus shuffle the
  same sentences, which would have measured memorisation as retention.

Two deliberate choices about who does the maths:

* the loss and tokenization come from sentence-transformers
  (`MultipleNegativesRankingLoss` + `smart_batching_collate`) rather than a
  hand-written InfoNCE, because pooling is defined by the ST model card and a
  hand-rolled twin can silently disagree with it;
* the BEFORE/AFTER measurement is taken through `SentenceTransformerEmbedder`,
  the same class the engine uses at runtime, including its prefix handling — so
  the number reported here is the number the retrieval stack would see.

LoRA r=16 (6.4's "LoRA first"): the 0.6B base needs ~11 GB of
weights+grads+Adam-state in fp32 and this device has 8 GB. The adapter is merged
before saving so the engine loads a plain model with no peft dependency, and the
query/doc prefixes are written beside it as `sae_prefixes.json` — without that
sidecar, `KNOWN_PREFIXES` (keyed by hub id) would not match a local path, the
`Instruct:` prompt would vanish, and the benchmark would grade a different model.

Seed discipline: rows from TRAIN only, DEV is the selection/reporting split here,
TEST is untouched (9.5).

Run:  PYTHONPATH=src python -m engine.embed.finetune --epochs 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import repo_root
from engine.embed.hard_negatives import BUCKET_PROBE, build as build_triplets
from engine.embed.hard_negatives import general_rows
from engine.embed.hf_embedder import KNOWN_PREFIXES, SentenceTransformerEmbedder
from gen.hard_negs import build_hard_negatives
from gen.ledger import build_ledger
from gen.splice import write_manifest

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LORA_R = 16
SCALE = 20.0                 # 6.4: temperature 0.02-0.05, i.e. scale 20-50
LR = 1e-4
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROBE_PAIRS = 300


def splits(projects: int, seed: int) -> dict[str, set[str]]:
    root = repo_root()
    ledger = build_ledger(n_projects=projects, seed=seed)
    return write_manifest(root / "data/splits/ledger.json", ledger.projects,
                          seed=seed)["split"]


def _rows(triplets) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Triples with a hard negative, and plain pairs without one.

    Kept as two batches because `MultipleNegativesRankingLoss` treats every
    column after the positive as an extra negative: padding the pair rows with a
    duplicate of their own positive would train against a false negative.
    """
    triples, pairs = [], []
    for r in triplets.rows():
        (triples if r["negative"] else pairs).append(r)
    return triples, pairs


def discriminate(model_id: str, items: list[tuple[str, str, str]],
                 device: str = "cuda") -> dict[str, Any]:
    """Share of items where the true text outscores its trap, via the engine class.

    `items` are (anchor, positive, negative). Anchors go through the query role,
    the two candidates through the doc role — the same prefixes retrieval uses.
    """
    if not items:
        return {"n": 0}
    emb = SentenceTransformerEmbedder(model_id, device=device, batch_size=32,
                                      max_seq_length=256)
    q = emb.embed([a for a, _p, _n in items], role="query")
    p = emb.embed([p for _a, p, _n in items], role="doc")
    n = emb.embed([n for _a, _p, n in items], role="doc")
    sp = (q * p).sum(1)
    sn = (q * n).sum(1)
    return {"n": len(items), "model": model_id, "backend": emb.version,
            "pairwise_accuracy": round(float((sp > sn).mean()), 4),
            "mean_margin": round(float((sp - sn).mean()), 4),
            "median_pos_sim": round(float(np.median(sp)), 4)}


def weights_moved(base_id: str, tuned_id: str, device: str,
                  n: int = 24) -> dict[str, Any]:
    """Do the saved weights actually differ from the frozen base?

    Encodes the same sentences through BOTH and compares the vectors. A fine-tune
    whose adapter never reached the forward path still prints a falling loss
    curve and still writes a model directory, so "it trained" cannot be taken on
    trust — the loss could have moved nothing but the LoRA copy in memory.
    """
    probe = ["Le budget est validé.", "La date de lancement est prévue en décembre.",
             "C'est Marie qui gère le prestataire.", "Le périmètre est annulé."]
    probe = (probe * 7)[:n]
    a = SentenceTransformerEmbedder(base_id, device=device, batch_size=n,
                                    max_seq_length=256).embed(probe, role="doc")
    b = SentenceTransformerEmbedder(tuned_id, device=device, batch_size=n,
                                    max_seq_length=256).embed(probe, role="doc")
    cos = float((a * b).sum(1).mean())
    return {"changed": bool(cos < 0.9999), "mean_cosine": round(cos, 6),
            "n_probed": n}


def dev_items(ledger, dev: set[str], seed: int) -> list[tuple[str, str, str]]:
    """DEV hard-negative items — cue-bearing only, `repeat` pairs excluded."""
    return [(h.anchor, h.positive, h.negative)
            for h in build_hard_negatives(ledger, seed=seed, per_project=8)
            if h.project in dev and h.cue
            and h.meta.get("scoreable", True) is not False]


def probe_items(seed: int, n: int = PROBE_PAIRS) -> list[tuple[str, str, str]]:
    """Held-out general French: same-intent positive, different-intent trap."""
    a, p = general_rows(limit=n + 8, seed=seed, bucket=BUCKET_PROBE)
    out = []
    for i in range(min(n, len(a))):
        trap = p[(i + 7) % len(p)]
        if trap != p[i]:
            out.append((a[i], p[i], trap))
    return out


def train(epochs: int = 6, batch: int = 16, lr: float = LR, seed: int = 42,
          projects: int = 20, out: Path | None = None, device: str | None = None,
          verbose: bool = True) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer
    # ST 5.5 deprecates the old `sentence_transformers.losses` path; importing
    # from the module it points at keeps the run warning-free.
    from sentence_transformers.sentence_transformer.losses import \
        MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    root = repo_root()
    out_dir = Path(out) if out else (root / "models" / "embed"
                                     / f"qwen3-ledger-s{seed}")
    run_dir = root / "runs" / "embed_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

    sp = splits(projects, seed)
    ledger = build_ledger(n_projects=projects, seed=seed)
    data = build_triplets(projects=projects, seed=seed, general_share=0.25)
    mix = data.meta["mix"]
    if not mix["ok"]:
        raise SystemExit(f"training mix violates 6.4: {mix}")
    say(f"rows {mix['n_rows']} | constructed {mix['constructed_share']:.1%} "
        f"| general {mix['general_share']:.1%}")
    triples, pairs = _rows(data)
    say(f"triples {len(triples)} | pairs {len(pairs)}")

    st = SentenceTransformer(BASE_MODEL, device=device,
                             model_kwargs={"trust_remote_code": False})
    st.max_seq_length = 256
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05,
                     bias="none", task_type="FEATURE_EXTRACTION",
                     target_modules=TARGET_MODULES)
    #: Hold our OWN reference to the wrapped model. Reading `st[0].auto_model` back
    #: after the assignment handed back the inner `Qwen3Model`, so the merge step
    #: raised `'Qwen3Model' object has no attribute 'merge_and_unload'` — and the
    #: naive fix (merge the attribute) would have merged nothing at all.
    peft_model = get_peft_model(st[0].auto_model, cfg)
    st[0].auto_model = peft_model
    trainable = [p for p in st.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("LoRA attached but nothing is trainable: the adapter is "
                         "not in the forward path and the run would change nothing")
    say(f"LoRA r={LORA_R} {TARGET_MODULES}: "
        f"{sum(p.numel() for p in trainable):,} trainable params")

    def collate_for(cols: list[str]):
        """Per-column feature dicts, the way ST 5.5 collates internally.

        `smart_batching_collate` was tried first and raised `'dict' object has no
        attribute 'texts'`: in 5.5 it expects `InputExample` objects, not the
        dict rows an HF `Dataset` yields. Reading its source showed the real step
        is `[self.preprocess(sentence) for sentence in zip(*texts)]`, so doing
        that directly here is the same work without the undocumented class.
        """

        def collate(batch: list[dict[str, str]]) -> tuple[list[dict], Any]:
            import torch

            # `preprocess` is typed `dict[str, Tensor | Any]`, so it legitimately
            # carries non-tensor entries (the raw texts for chat-style models).
            # Moving EVERY value to the GPU raised 'str' object has no attribute
            # 'to'; only tensors may be moved.
            def move(feat: dict[str, Any]) -> dict[str, Any]:
                return {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in feat.items()}

            feats = [move(st.preprocess([row[c] for row in batch])) for c in cols]
            labels = torch.zeros(len(batch), dtype=torch.long, device=device)
            return feats, labels

        return collate

    def loader(rows: list[dict[str, str]], cols: list[str]) -> DataLoader:
        ds = Dataset.from_list([{c: r[c] for c in cols} for r in rows])
        return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                          collate_fn=collate_for(cols))

    dl_t = loader(triples, ["anchor", "positive", "negative"])
    dl_p = loader(pairs, ["anchor", "positive"])
    loss = MultipleNegativesRankingLoss(st, scale=SCALE)
    opt = torch.optim.AdamW(trainable, lr=lr)
    steps = max(len(dl_t), len(dl_p))
    history: list[float] = []
    for ep in range(1, epochs + 1):
        tot, seen = 0.0, 0
        for (bt, bl), (bp, pl) in zip(_cycle(dl_t, steps), _cycle(dl_p, steps)):
            opt.zero_grad()
            l = loss(bt, bl) + loss(bp, pl)
            l.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += float(l.detach())
            seen += 1
        history.append(round(tot / max(1, seen), 4))
        say(f"epoch {ep}/{epochs} loss {history[-1]:.4f}")

    # Merge through OUR reference, not `st[0].auto_model` (which yields the inner
    # base model and has no merge_and_unload).
    merged = peft_model.merge_and_unload()
    st[0].auto_model = merged
    out_dir.mkdir(parents=True, exist_ok=True)
    # ST 5.5's save_pretrained has no `save_to_hub` kwarg (verified against the
    # live signature); passing it would raise TypeError only after a full run.
    st.save_pretrained(str(out_dir))
    prefixes = KNOWN_PREFIXES.get(BASE_MODEL, {"query": "", "doc": ""})
    (out_dir / "sae_prefixes.json").write_text(json.dumps(prefixes, indent=1),
                                               encoding="utf-8")
    #: Did the saved weights actually move? A fine-tune that writes an unchanged
    #: copy of the base would still print a loss curve, so this is checked rather
    #: than assumed (same failure family as the inert config knobs).
    moved = weights_moved(BASE_MODEL, str(out_dir), device)
    if not moved["changed"]:
        raise SystemExit(
            f"saved model is numerically identical to {BASE_MODEL} "
            f"(mean cosine {moved['mean_cosine']}) — the adapter never reached the "
            f"forward path, so this run trained nothing")
    say(f"weights moved: mean cosine to frozen base "
        f"{moved['mean_cosine']:.5f} (< 0.9999 required)")

    dev = dev_items(ledger, set(sp["dev"]), seed)
    probe = probe_items(seed)
    before = discriminate(BASE_MODEL, dev, device)
    after = discriminate(str(out_dir), dev, device)
    gen_before = discriminate(BASE_MODEL, probe, device)
    gen_after = discriminate(str(out_dir), probe, device)
    d_hard = round(after["pairwise_accuracy"] - before["pairwise_accuracy"], 4)
    d_gen = round(gen_after["pairwise_accuracy"] - gen_before["pairwise_accuracy"], 4)
    result = {
        "base": BASE_MODEL, "out_dir": str(out_dir), "seed": seed,
        "epochs": epochs, "batch": batch, "lr": lr, "scale": SCALE,
        "lora_r": LORA_R, "mix": mix, "loss_history": history,
        "dev_items": len(dev), "probe_items": len(probe),
        # the guard that the weights really reached the saved artefact — recorded,
        # not merely printed, so the JSON is the evidence and not the terminal
        "weights_moved": moved,
        "hard_negative_dev": {"frozen": before, "finetuned": after,
                              "delta_accuracy": d_hard},
        "general_french_heldout": {"frozen": gen_before, "finetuned": gen_after,
                                   "delta_accuracy": d_gen,
                                   "note": "PROXY for the 6.4 MTEB-French "
                                           "regression gate; mteb is not "
                                           "installed so that gate is UNMEASURED"},
        "gate": {"hard_negative_improved": d_hard > 0.0,
                 "general_drop_within_1pt": d_gen >= -0.01,
                 "acceptance_6_4": "Recall@5 on dev must be >= frozen + 5 pts; "
                                   "run `make bench-st` with --model <out_dir>"},
    }
    (run_dir / f"result_{seed}.json").write_text(json.dumps(result, indent=1,
                                                           default=str),
                                                 encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({
        "base": BASE_MODEL, "seed": seed, "epochs": epochs, "batch": batch,
        "lr": lr, "scale": SCALE, "lora_r": LORA_R, "mix": mix,
        "train_projects": sorted(sp["train"]), "prefixes": prefixes,
    }, indent=1), encoding="utf-8")
    say(json.dumps({"hard_neg_delta": d_hard, "general_delta": d_gen,
                    "gate": result["gate"]}, indent=1))
    return result


def _cycle(loader, n: int):
    it = iter(loader)
    for _ in range(n):
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)
            yield next(it)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = train(args.epochs, args.batch, args.lr, args.seed, args.projects,
                Path(args.out) if args.out else None)
    print(f"\nsaved {res['out_dir']}")
    print("hard-negative dev Δ accuracy:",
          res["hard_negative_dev"]["delta_accuracy"])
    print("general-French held-out Δ accuracy:",
          res["general_french_heldout"]["delta_accuracy"])
    ok = res["gate"]["hard_negative_improved"] and res[
        "gate"]["general_drop_within_1pt"]
    print("gates:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
