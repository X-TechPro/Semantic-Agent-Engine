#!/usr/bin/env python3
"""Compare French pretrained sentence encoders with the custom tiny model.

Runs on exactly the synthetic French corpus/splits produced by run_french_lab.py.
The validation set chooses prefix configuration and whether fine-tuning is needed;
the test split is only evaluated once after all choices are frozen.

Requires network access on first run to download Hugging Face weights.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SEED = 20261031
MODEL_SPECS = [
    {
        "name": "antoinelouis/french-me5-small",
        "prefix_modes": {
            # The model card demonstrates raw sentence encoding. Test E5-style
            # prefixes as a validation-only ablation rather than assuming them.
            "plain": ("", ""),
            "e5_prefix": ("query: ", "passage: "),
        },
    },
    {
        "name": "intfloat/multilingual-e5-small",
        # The official model card explicitly requires query:/passage: prefixes
        # for asymmetric retrieval, including non-English queries.
        "prefix_modes": {"e5_prefix": ("query: ", "passage: ")},
    },
]


def load_data():
    meetings_by_split = json.loads((DATA / "french_synthetic_meetings.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA / "french_retrieval_queries.json").read_text(encoding="utf-8"))
    messages: dict[str, dict[str, Any]] = {}
    meetings: dict[str, dict[str, Any]] = {}
    for split, meetings_rows in meetings_by_split.items():
        for meeting in meetings_rows:
            meetings[meeting["meeting_id"]] = meeting
            for msg in meeting["messages"]:
                cid = f"{meeting['meeting_id']}:{msg['message_id']:03d}"
                messages[cid] = {
                    "chunk_id": cid,
                    "meeting_id": meeting["meeting_id"],
                    "split": split,
                    "style": meeting["style"],
                    "text": f"Réunion : {meeting['title']}. Intervenant : {msg['username']}. Intervention : {msg['message']}",
                    "raw": msg,
                }
    return meetings, messages, queries


def metrics_for_queries(queries, messages, passage_vectors, query_vectors, k_values=(1, 3, 5, 10)):
    by_meeting: dict[str, list[str]] = {}
    for cid, row in messages.items():
        by_meeting.setdefault(row["meeting_id"], []).append(cid)
    hits = {k: [] for k in k_values}
    reciprocal, ndcg = [], []
    rankings = []
    for q in queries:
        candidates = by_meeting[q["meeting_id"]]
        scores = np.asarray([float(np.dot(query_vectors[q["query_id"]], passage_vectors[c])) for c in candidates])
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        target = q["target_chunk_id"]
        rank = ranked.index(target) + 1 if target in ranked else len(ranked) + 1
        reciprocal.append(1.0 / rank if rank <= 10 else 0.0)
        for k in k_values:
            hits[k].append(float(rank <= k))
        ndcg.append(1.0 / np.log2(rank + 1) if rank <= 5 else 0.0)
        if q.get("split") == "test":
            rankings.append({
                "query_id": q["query_id"], "query": q["query"], "meeting_id": q["meeting_id"],
                "target_chunk_id": target,
                "top10": [{"chunk_id": candidates[i], "source": messages[candidates[i]]["raw"], "score": float(scores[i])}
                          for i in order[:10]],
            })
    out = {f"Recall@{k}": float(np.mean(v)) for k, v in hits.items()}
    out["MRR@10"] = float(np.mean(reciprocal))
    out["nDCG@5"] = float(np.mean(ndcg))
    return out, rankings


def encode_all(model, queries, messages, query_prefix, passage_prefix, batch_size, device):
    ids = list(messages)
    # Embed exactly the meeting-scoped chunks; timestamps remain structured metadata.
    passage_texts = [passage_prefix + messages[c]["text"] for c in ids]
    query_rows = list(queries)
    query_texts = [query_prefix + q["query"] for q in query_rows]
    pvec = model.encode(
        passage_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True, device=device,
    )
    qvec = model.encode(
        query_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True, device=device,
    )
    pv = {cid: pvec[i].astype(np.float32, copy=False) for i, cid in enumerate(ids)}
    qv = {q["query_id"]: qvec[i].astype(np.float32, copy=False) for i, q in enumerate(query_rows)}
    return pv, qv


def differentiable_encode(model, texts, device):
    features = getattr(model, "preprocess", model.tokenize)(texts)
    features = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in features.items()}
    output = model(features)
    return F.normalize(output["sentence_embedding"], p=2, dim=-1)


def finetune_on_train(model, train_queries, messages, query_prefix, passage_prefix,
                      epochs, batch_queries, learning_rate, device):
    """Small MNRL/InfoNCE-style adaptation using train queries + five hard negatives."""
    rows = [q for q in train_queries if q["split"] == "train" and len(q.get("hard_negative_chunk_ids", [])) >= 5]
    if not rows:
        raise RuntimeError("No train queries with five hard negatives were found")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        random.Random(SEED + epoch).shuffle(rows)
        model.train()
        losses = []
        for start in range(0, len(rows), batch_queries):
            batch = rows[start:start + batch_queries]
            texts = []
            for q in batch:
                pos_id = q["target_chunk_id"]
                neg_ids = q["hard_negative_chunk_ids"][:5]
                texts.extend([query_prefix + q["query"], passage_prefix + messages[pos_id]["text"]])
                texts.extend(passage_prefix + messages[nid]["text"] for nid in neg_ids)
            emb = differentiable_encode(model, texts, device)
            b = len(batch)
            emb = emb.view(b, 7, -1)
            qv = emb[:, 0]
            candidates = emb[:, 1:]
            logits = torch.einsum("bd,bkd->bk", qv, candidates) / 0.05
            target = torch.zeros(b, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        history.append({"epoch": epoch, "train_infonce_loss": float(np.mean(losses))})
        print(f"  fine-tune epoch {epoch}/{epochs}: InfoNCE={np.mean(losses):.4f}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fine-tune-epochs", type=int, default=2)
    ap.add_argument("--fine-tune-gap", type=float, default=0.05,
                    help="Fine-tune if base validation MRR is this much below the custom tiny baseline")
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    ART.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ART / "hf_cache"))
    status_path = ART / "pretrained_comparison.json"
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        status = {
            "status": "NOT_RUN_MISSING_DEPENDENCY",
            "reason": f"sentence-transformers import failed: {type(exc).__name__}: {exc}",
            "required_install": "pip install -r requirements.txt",
            "models": [x["name"] for x in MODEL_SPECS],
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    if args.check_only:
        status = {
            "status": "READY_DEPENDENCY_PRESENT",
            "sentence_transformers_available": True,
            "models": [x["name"] for x in MODEL_SPECS],
            "note": "This checks the Python dependency only; model weights still need to be downloaded on first full run.",
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    meetings, messages, all_queries = load_data()
    tiny = json.loads((ART / "french_results.json").read_text(encoding="utf-8"))
    tiny_val_mrr = tiny["retrieval_metrics"]["val"]["dense"]["MRR@10"]
    tiny_test = tiny["retrieval_metrics"]["test"]["dense"]
    train_q = [q for q in all_queries if q["split"] == "train"]
    val_q = [q for q in all_queries if q["split"] == "val"]
    test_q = [q for q in all_queries if q["split"] == "test"]
    device = args.device
    torch.manual_seed(SEED)
    run_start = time.time()
    result = {
        "protocol": {
            "dataset": "fully synthetic French meetings only",
            "split": "project-grouped train/validation/test; identical queries and meeting-scoped candidate sets",
            "model_selection": "validation MRR@10 selects prefix and whether fine-tuning is triggered; test evaluated once afterwards",
            "fine_tuning": "train split only; InfoNCE with one positive and five same-meeting hard negatives",
            "tiny_baseline_val_MRR@10": tiny_val_mrr,
            "tiny_baseline_test": tiny_test,
            "device": device,
        },
        "models": {},
    }

    for spec in MODEL_SPECS:
        name = spec["name"]
        print(f"\nLoading {name}")
        start = time.time()
        try:
            model = SentenceTransformer(name, device=device, cache_folder=str(ART / "hf_cache"))
        except Exception as exc:
            result["status"] = "PARTIAL_MODEL_LOAD_FAILURE"
            result["failed_model"] = name
            result["failure"] = f"{type(exc).__name__}: {exc}"
            result["elapsed_seconds"] = time.time() - run_start
            status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"status": result["status"], "failed_model": name, "failure": result["failure"]}, ensure_ascii=False, indent=2))
            return 2
        model.max_seq_length = 128
        prefix_results = {}
        candidates = {}
        for mode, (qp, pp) in spec["prefix_modes"].items():
            print(f"  validation prefix mode: {mode}")
            pv, qv = encode_all(model, val_q, {c:r for c,r in messages.items() if r["split"] == "val"}, qp, pp, args.batch_size, device)
            m, _ = metrics_for_queries(val_q, {c:r for c,r in messages.items() if r["split"] == "val"}, pv, qv)
            prefix_results[mode] = m
            candidates[mode] = (qp, pp)
        best_mode = max(prefix_results, key=lambda k: prefix_results[k]["MRR@10"])
        qp, pp = candidates[best_mode]
        base_val = prefix_results[best_mode]
        best_state = copy.deepcopy(model.state_dict())
        finetune = {"triggered": False, "reason": "base validation is within 0.05 MRR of custom tiny model"}

        if not args.no_finetune and base_val["MRR@10"] < tiny_val_mrr - args.fine_tune_gap:
            finetune = {"triggered": True, "reason": f"base validation MRR {base_val['MRR@10']:.4f} is more than {args.fine_tune_gap:.2f} below tiny baseline {tiny_val_mrr:.4f}"}
            base_best = base_val["MRR@10"]
            val_history = []
            train_history = finetune_on_train(
                model, train_q, messages, qp, pp, args.fine_tune_epochs,
                batch_queries=8, learning_rate=2e-5, device=device,
            )
            # Keep the post-tune model only if validation improved over the base checkpoint.
            pv, qv = encode_all(model, val_q, {c:r for c,r in messages.items() if r["split"] == "val"}, qp, pp, args.batch_size, device)
            tuned_val, _ = metrics_for_queries(val_q, {c:r for c,r in messages.items() if r["split"] == "val"}, pv, qv)
            if tuned_val["MRR@10"] > base_best:
                base_val = tuned_val
                best_state = copy.deepcopy(model.state_dict())
                finetune["accepted"] = True
            else:
                model.load_state_dict(best_state)
                finetune["accepted"] = False
            finetune["training_history"] = train_history
            finetune["validation_after"] = tuned_val

        model.load_state_dict(best_state)
        model.eval()
        # Freeze all decisions before the one-and-only test evaluation.
        test_messages = {c:r for c,r in messages.items() if r["split"] == "test"}
        pv, qv = encode_all(model, test_q, test_messages, qp, pp, args.batch_size, device)
        test_metrics, rankings = metrics_for_queries(test_q, test_messages, pv, qv)
        model_key = name.replace("/", "__")
        if finetune.get("accepted"):
            save_dir = ART / (model_key + "_finetuned")
            model.save(str(save_dir))
            finetune["saved_model"] = str(save_dir.relative_to(ROOT))
        result["models"][name] = {
            "selected_prefix_mode": best_mode,
            "validation_by_prefix": prefix_results,
            "selected_validation_metrics": base_val,
            "fine_tuning": finetune,
            "test_metrics": test_metrics,
            "test_rankings": rankings,
            "elapsed_seconds": time.time() - start,
        }
        print("  selected:", best_mode, "val MRR:", base_val["MRR@10"], "test:", test_metrics)

    result["status"] = "COMPLETED"
    result["elapsed_seconds"] = time.time() - run_start
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
