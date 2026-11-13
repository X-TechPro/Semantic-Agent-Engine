> Current state (2026-11-06): Rexu v3 is active; v3.1 is archived.
> v10 is the active hand-written corpus (5×55 messages, 19 queries).
> v7 remains the legacy 60-meeting experiment. Generated v8/v9 are archived.

# v7 — unified unseen-discussion experiment

## Why v7 exists

v6 had two datasets with different assumptions: synthetic (templates, custom
encoder in-distribution) and transcribed (short turns with explicit
`décision:`/ISO-date/closing markers). The two tables were not comparable and
the adaptive `graphrag_best` was dataset-aware. v7 replaces both with **one
corpus, one split protocol, and one evaluation for all methods**.

v6 data and artifacts are archived in `data/archive_v6/` and
`artifacts/archive_v6/`; the v6 README is `docs/README_V6.md`.

## Corpus and split protocol

- 15 projects × 4 meetings = 60 meetings; 60 messages each; 3,600 messages.
- Project-disjoint split: **train 9 projects / 36 meetings**, **val 3 / 12**,
  **test 3 / 12**. No project appears in two splits.
- Speaker-disjoint pools: train uses 12 names, val 6, test 6. No speaker is
  shared across splits.
- Test projects: **Delta**, **Équinoxe**, **Mistral**. Train projects include
  Atlas, Cobalt, Falcon, Horizon, Iris, Junon, Kestrel, Nova, Orion.
- Test query phrasings are **held out**: train/val use one template family;
  test uses a different paraphrase family. Exact train↔test query overlap: **0**.
- Test adds a **multi-gold count query** per meeting (6 decisions per meeting).
- Test set: 264 queries — 72 topic-latest-decision, 72 commitment-owner, 72
  objection, 24 global-latest-decision, 12 latest-message, 12 decision-count.

## Training protocol

| Component | Trained on | Selected on | Evaluated on |
|---|---|---|---|
| Custom encoder + topic/act heads | v7 train | v7 val MRR | v7 test |
| Feature-aware retriever | v7 train (encoder) | v7 val policy | v7 test |
| Relation classifier | v7 train pairs | — | v7 test pairs |
| Intent classifier | generic examples | — | v6 archived |
| GraphRAG fusion / category weights | v7 train | v7 val | v7 test |
| French ME5, multilingual E5 | v7 train (InfoNCE fine-tune) | v7 val | v7 test |
| BGE-M3 ColBERT projection | v7 train (1 pos + 5 hard neg) | v7 val | v7 test |
| FT-ColBERT router | deterministic rule | v7 val | v7 test |
| BGE-M3 base / ColBERT-XM / v2 | frozen | — | v7 test |

No test query, test project, test speaker or test pair is used for training or
selection.

## v7 test results (264 queries, held-out phrasings, unseen projects/speakers)

`Recall@K` credits the fraction of gold targets on the 12 multi-gold count
queries; `AnyR@K` credits at least one gold target.

| Method | R@1 | AnyR@1 | R@5 | AnyR@5 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.119 | 0.125 | 0.344 | 0.352 | 0.210 | 0.236 |
| BGE-M3 · dense only | 0.321 | 0.352 | 0.597 | 0.617 | 0.467 | 0.493 |
| ColBERT · BGE-M3 only | 0.275 | 0.307 | 0.601 | 0.617 | 0.436 | 0.471 |
| **Rexu v2.1 · BGE-M3 FT + ColBERT v1** | **0.614** | **0.652** | **0.895** | **0.905** | **0.756** | **0.787** |
| **Rexu v2.2 · BGE-M3 FT + ColBERTv2** | **0.625** | **0.663** | **0.907** | **0.917** | **0.763** | **0.797** |
| **GraphRAG · FT-ColBERT router** | **0.633** | **0.670** | **0.926** | **0.936** | **0.776** | **0.812** |

The benchmark keeps these seven methods. Rexu v2.1 is BGE-M3 FT + ColBERT v1;
v2.2 adds ColBERTv2 compression; v2.3 is archived. Custom dense and the other
GraphRAG variants are removed from the benchmark and remain in the full
component ablation: `artifacts/v7_ablation.json`.

## Fine-tuned BGE-M3 ColBERT projection

The frozen BGE-M3 ColBERT signal was the weak part of GraphRAG. Fine-tuning a
1024→1024 projection on top of frozen BGE-M3 token states is cheap and highly
effective:

- One positive + five v7 hard negatives per query.
- Listwise MaxSim loss, 2 epochs, projection selected on v7 val.
- Base BGE-M3 remains frozen; the checkpoint is ~4 MB.

| Method | Val R@1 | Val MRR | Test R@1 | Test MRR |
|---|---:|---:|---:|---:|
| BGE-M3 ColBERT, frozen | 0.337 | 0.474 | 0.275 | 0.436 |
| **BGE-M3 ColBERT, fine-tuned projection** | **0.837** | **0.885** | **0.617** | **0.743** |
| E5 | 0.778 | 0.847 | 0.475 | 0.617 |
| GraphRAG hybrid | 0.885 | 0.925 | 0.521 | 0.666 |
| **FT-ColBERT router** | **0.873** | **0.913** | **0.636** | **0.763** |

Test R@1 .617 is fractional on the multi-gold count queries; AnyR@1 is .652.

The router keeps FT ColBERT for semantic categories and switches to GraphRAG
hybrid only for `latest_message` and global latest-decision queries, where
FT ColBERT loses to temporal/graph signals. Graph fusion on every query does
**not** help:

| Fusion | Val MRR | Test R@1 | Test MRR |
|---|---:|---:|---:|
| Fine-tuned ColBERT alone | 0.885 | 0.617 | 0.743 |
| Replace graph `colbert` signal with fine-tuned model | 0.786 | 0.492 | 0.622 |
| Val-tuned E5 + graph + fine-tuned ColBERT | 0.928 | 0.576 | 0.700 |
| **Router: FT ColBERT + graph only for temporal** | **0.913** | **0.636** | **0.763** |

The graph re-ranker overfits the val template family; routing it only to the
categories where temporal structure matters is better. `graphrag_ft_router` is
the default; GraphRAG hybrid remains for comparison only.

## Earlier GraphRAG hybrid improvement

Before projection fine-tuning, the best GraphRAG result was a two-signal rank
fusion:

```text
score = 0.7 * rank_pct(fine-tuned E5) + 0.3 * rank_pct(learned GraphRAG)
```

The weight is selected on v7 val only. Test effect:

| Method | R@1 | MRR@10 |
|---|---:|---:|
| E5 alone | 0.485 | 0.617 |
| GraphRAG learned | 0.470 | 0.571 |
| GraphRAG hybrid | 0.534 | 0.666 |

It is superseded by the fine-tuned ColBERT projection. Token-level
cross-message edges are still a separate lever: v7 test is single-target, and
semantic token edges would mainly help cross-meeting/multi-hop queries that v7
does not currently test.

## Seen-template vs held-out phrasing

In-distribution = v7 val queries (same template family as train, unseen
projects/speakers). OOD = v7 test (held-out phrasings, unseen projects/speakers).

| Method | Val R@1 | Test R@1 | Val MRR | Test MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.175 | 0.125 | 0.288 | 0.210 |
| BGE-M3 · dense only | 0.381 | 0.352 | 0.493 | 0.467 |
| ColBERT · BGE-M3 only | 0.337 | 0.307 | 0.474 | 0.436 |
| ColBERT mixed + V2 compression | 0.806 | 0.663 | 0.867 | 0.763 |
| **FT-ColBERT router** | **0.849** | **0.670** | **0.899** | **0.776** |

The drop from val to test is the actual generalization cost. Feature-aware and
the custom dense encoder lose the most; the fine-tuned pretrained encoders are
stable.

## French ColBERTv2 checkpoint ablation

`antoinelouis/colbertv2-camembert-L4-mmarcoFR` is a 54M-parameter French
ColBERTv2 model: CamemBERT-L4, 32-d token projection, query max 32, doc max 160,
trained on French mMARCO.

| Variant | Val R@1 | Val MRR | Test R@1 | Test MRR |
|---|---:|---:|---:|---:|
| Frozen checkpoint | 0.290 | 0.410 | 0.265 | 0.372 |
| Fine-tuned 2 epochs on v7 train | 0.758 | 0.831 | 0.398 | 0.528 |
| BGE-M3 ColBERT (current) | 0.337 | 0.474 | 0.275 | 0.436 |
| Multilingual E5 | 0.778 | 0.847 | 0.475 | 0.617 |
| **GraphRAG hybrid** | **0.885** | **0.925** | **0.521** | **0.666** |

Fine-tuning was 1 positive + 5 v7 hard negatives, MaxSim listwise loss, 2 epochs,
checkpoint selected on v7 val. It improves the checkpoint a lot, and beats the
frozen BGE-M3 ColBERT, but still trails E5 and the hybrid. A three-way blend of
E5 + graph + French ColBERT selected **weight 0** for French ColBERT on v7 val;
it adds no value. It is not adopted. Artifacts:
`artifacts/french_colbert_eval.json`,
`artifacts/french_colbert_finetuned_eval.json`,
`artifacts/french_colbert_v7.pt`; scripts: `evaluate_french_colbert.py`,
`finetune_french_colbert.py`.

### Full-size French ColBERTv2

`AdrienB134/ColBERTv2.0-mmarcoFR` is the full checkpoint: CamemBERT-base,
12 layers, 110M params, 128-d, ColBERTv2 training on mMARCO-fr.

| Variant | Val R@1 | Val MRR | Test R@1 | Test MRR |
|---|---:|---:|---:|---:|
| Full checkpoint, frozen | 0.393 | 0.502 | 0.303 | 0.406 |
| Full checkpoint, projection fine-tuned | 0.742 | 0.821 | 0.458 | 0.581 |
| ColBERT-XM, `fr_XX` adapter, frozen | 0.337 | 0.428 | 0.277 | 0.362 |
| ColBERT-XM, 128-d projection, 4 epochs | **0.829** | **0.883** | **0.519** | **0.629** |
| ColBERT-XM, 512-d projection | 0.718 | 0.806 | 0.413 | 0.534 |
| CamemBERT-L4 checkpoint, fine-tuned | 0.758 | 0.831 | 0.398 | 0.528 |
| BGE-M3 projection, fine-tuned | 0.837 | 0.885 | 0.617 | 0.743 |
| Multilingual E5 | 0.778 | 0.847 | 0.475 | 0.617 |

The real full model beats the L4 checkpoint when frozen, and fine-tuning its
768→128 projection closes much of the gap. It still loses to the fine-tuned
BGE-M3 projection by 0.159 R@1 and 0.162 MRR, and to E5 by 0.017 R@1 / 0.036
MRR. The architecture is not the bottleneck; the missing operation/field
constraints are. Not adopted. Artifacts:
`artifacts/real_colbertv2_fr_eval.json`,
`artifacts/real_colbertv2_fr_finetuned_eval.json`,
`artifacts/real_colbertv2_fr_projection.pt`; scripts:
`evaluate_real_colbertv2_fr.py`, `finetune_real_colbertv2_fr.py`.

The multilingual `antoinelouis/colbert-xm` (XMOD, 128-d, 81 languages) scores
test R@1 .303 / R@5 .473 / MRR .379 with the library's default path. With the
`fr_XX` adapter and a 128-d projection trained for 4 epochs, it reaches test
R@1 .519 / R@5 .803 / MRR .629. That beats the full French checkpoint's
fine-tuned projection (.458/.581) but is still below BGE ColBERT
(.617/.743) and E5 (.475/.617). A 512-d projection is worse (.413/.534):
capacity is not the only bottleneck. Artifacts: `artifacts/colbert_xm_v7_eval.json`,
`artifacts/colbert_xm_fr_finetuned_eval_128.json`,
`artifacts/colbert_xm_fr_projection_128.pt`; script:
`finetune_colbert_xm_fr.py`.

### Where ColBERT-XM breaks

Test set, XM 4-epoch vs BGE projection:

| Category | XM R@1 | BGE R@1 | XM R@5 | BGE R@5 |
|---|---:|---:|---:|---:|
| topic_latest_decision | 0.333 | 0.500 | 0.778 | 0.917 |
| find_commitment | 0.986 | 1.000 | 1.000 | 1.000 |
| find_objection | 0.417 | 0.667 | 0.833 | 0.931 |
| global_latest_decision | 0.125 | 0.208 | 0.708 | 0.542 |
| latest_message | 0.000 | 0.000 | 0.000 | 0.167 |
| count_decisions | 0.750 | 0.917 | 0.750 | 1.000 |
| **overall** | **0.519** | **0.652** | **0.811** | **0.879** |

XM's top-1 confusions: 69 same-topic/wrong-act, 22 same-act/wrong-topic,
36 neither. BGE: 52, 24, 16. XM has 23 queries outside top-10 vs BGE's 20.
The loss is semantic precision on topic/objection, not a time-field issue.

Efficiency:

| | XM 128-d | BGE 1024-d |
|---|---:|---:|
| Token dimension | 128 | 1024 |
| Index tokens | 148,449 | 170,432 |
| Projected index | 38 MB | 349 MB |
| Parameters (with adapters) | 852M | 568M |
| Query encode + score | 59.0 ms | 6.5 ms |

XM is ~9× smaller in token storage, but **not faster** here: the XMOD adapter
stack is 852M parameters and runs FP32. It is a storage win, not a latency win.

### BGE + ColBERTv2 compression

Applying the ColBERTv2 codec to the fine-tuned BGE token index is the one
combination that gives both accuracy and compression:

| Centroids | R@1 | R@5 | MRR | Compressed vs fp16 index |
|---:|---:|---:|---:|---:|
| 1024 | 0.633 | 0.879 | 0.730 | 12.7× |
| 2048 | **0.659** | **0.879** | **0.747** | **11.1×** |
| 4096 | 0.659 | 0.879 | 0.748 | 8.7× |

The 2048-centroid variant is shipping as `colbert_ft_v2`; with the mixed-style
projection it scores .625 R@1 / .763 MRR with a 31 MB index versus 349 MB for
fp16. This is BGE accuracy with ColBERTv2 storage, and it is slightly better
than the uncompressed mixed BGE. Artifacts:
`artifacts/bge_colbert_v2_eval.json`, `artifacts/bge_colbert_v2_stats.json`,
`artifacts/bge_colbert_v2_latency.json`; implementation: `bge_colbert_v2.py`.

Latency, measured on 40 test queries after warmup:

| | BGE fp16 | BGE + V2 |
|---|---:|---:|
| Warm query | 7.8 ms | **6.7 ms** |
| Cold index load | 0.06 s | 0.56 s |
| Disk/compressed index | 349 MB | **31 MB** |

Warm query latency is the same within noise. The compressed index is
decompressed once at load, so cold start is ~0.5 s slower and resident GPU
memory is unchanged. This is a **storage** optimization, not a query-speed
optimization. A true memory/speed win would need scoring against centroids and
codes directly (PLAID-style); I tested that next and it was worse.

### Transfer to messy transcriptions

The archived v6 corpus is split by project into messy train/val/test. The
held-out messy test has 30 queries:

| Model on held-out messy test | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Raw BGE-M3 MaxSim | 0.400 | 0.667 | 0.900 | 0.524 |
| BGE v7-only projection | 0.400 | 0.600 | 0.867 | 0.494 |
| **BGE mixed-style projection** | **0.733** | **1.000** | **1.000** | **0.867** |

The v7-only projection does not transfer. The mixed projection was trained on
v7 train plus six messy projects and evaluated on two unseen messy projects;
it fixes the transfer failure without giving up v7. The 30-query test is small,
so treat the exact number as provisional. Artifact:
`artifacts/bge_mixed_style_eval.json`; script: `finetune_bge_mixed.py`.

### PLAID-style pruning (negative result)

PLAID is the ColBERTv2 retrieval driver: centroid interaction, pruning, then
exact rerank. At v7 scale it is counterproductive:

| Method | R@1 | R@5 | R@10 | MRR | Query |
|---|---:|---:|---:|---:|---:|
| Exact compressed scan | **0.659** | **0.879** | **0.924** | **0.747** | **6.7 ms** |
| PLAID 8 probes / 128 candidates | 0.663 | 0.777 | 0.803 | 0.713 | 8.5 ms |
| PLAID 8 probes / 256 candidates | 0.667 | 0.848 | 0.864 | 0.736 | 10.0 ms |
| PLAID 4 probes / 64 candidates | 0.606 | 0.652 | 0.693 | 0.629 | 9.7 ms |

The exact GPU scan over ~170k tokens is already cheap at 3,600 messages.
Centroid scoring and candidate gathering cost more than the scan they replace,
and pruning loses R@5/R@10. PLAID becomes worthwhile at much larger corpora
with a real inverted index, not here. Artifact: `artifacts/plaid_bge_eval.json`;
script: `plaid_bge_colbert.py`.

### Temporal reranking

I found no established time-token variant of ColBERT; the practical approach
is a separate temporal field/score. Query plan determines `latest`/`earliest`;
candidate scores are `rank(ColBERT) + λ × rank(recency)`, with λ selected on
val.

| Model | λ | Test R@1 | Test R@5 | Test MRR |
|---|---:|---:|---:|---:|
| XM, no time | 0 | 0.519 | 0.811 | 0.631 |
| XM, time rerank | 0.1 | **0.530** | **0.833** | **0.651** |
| BGE, no time | 0 | 0.652 | 0.879 | 0.743 |
| BGE, time rerank | 0.2 | **0.670** | **0.905** | **0.769** |

It helps: global-latest XM R@1 .125→.250, BGE .208→.458; latest-message BGE
R@1 0→.083. Topic/objection barely move. Time is a real but narrow signal,
not a replacement for act/topic constraints. Artifact:
`artifacts/colbert_time_rerank_eval.json`.

## Token-level distant-message edges (negative result)

I tested non-adjacent message edges built from fine-tuned BGE ColBERT token
similarities: mean-pool candidate selection, MaxSim rerank, symmetric edges
added to the chunk PPR graph at weights 0.05 / 0.10 / 0.20. 9,774 edges were
added.

| Graph | Hybrid R@1 | Hybrid R@5 | Hybrid MRR | Router R@1 | Router MRR |
|---|---:|---:|---:|---:|---:|
| Baseline chunk graph | 0.534 | 0.852 | 0.666 | 0.670 | 0.763 |
| + distant edges, w=0.05 | 0.542 | 0.841 | 0.666 | 0.670 | 0.763 |
| + distant edges, w=0.10 | 0.542 | 0.845 | 0.666 | 0.670 | 0.763 |
| + distant edges, w=0.20 | 0.534 | 0.845 | 0.662 | 0.670 | 0.763 |

The edges do not help the router at all, and the hybrid gain is a +0.008 R@1
with a worse R@5. v7 test has no cross-meeting or multi-hop query type, so this
is expected. Not adopted. Artifact: `artifacts/distant_edges_eval.json`;
script: `evaluate_distant_edges.py`.

## Hard-negative discrimination (v7 test, 264 queries)

Component ablation; the five-method benchmark is above. Negatives are
same-meeting, same-topic/different-act first, then same-act. The router is not
in this table; its component is the fine-tuned ColBERT row.

| Method | R@5 | MRR@10 | gold>hard | intrusion@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.352 | 0.210 | 0.386 | 0.120 |
| Dense · custom | 0.428 | 0.295 | 0.458 | 0.109 |
| Feature-aware | 0.553 | 0.396 | 0.500 | 0.158 |
| ColBERT | 0.617 | 0.436 | 0.523 | 0.175 |
| **ColBERT fine-tuned** | **0.879** | **0.743** | **0.792** | **0.198** |
| GraphRAG fixed | 0.689 | 0.556 | 0.712 | 0.158 |
| GraphRAG learned | 0.716 | 0.571 | 0.720 | 0.157 |
| GraphRAG hybrid | 0.852 | 0.666 | 0.773 | 0.204 |
| GraphRAG best | 0.705 | 0.573 | 0.720 | 0.159 |
| GraphRAG v2 | 0.576 | 0.425 | 0.564 | 0.149 |
| ColBERTv2 | 0.231 | 0.118 | 0.265 | 0.142 |

## Per-unseen-project results

Recall@1 / MRR@10 on each test project:

| Method | Delta | Équinoxe | Mistral |
|---|---|---|---|
| French ME5 | 0.477 / 0.621 | 0.443 / 0.614 | 0.466 / 0.580 |
| Multilingual E5 | 0.443 / 0.588 | 0.545 / 0.676 | 0.466 / 0.587 |
| ColBERT fine-tuned | 0.625 / 0.733 | 0.670 / 0.767 | 0.659 / 0.730 |
| GraphRAG hybrid | 0.511 / 0.655 | 0.602 / 0.719 | 0.489 / 0.625 |
| **FT-ColBERT router** | **0.670 / 0.773** | **0.682 / 0.785** | **0.659 / 0.730** |

No single project drives the result; the ranking is stable across the three
unseen discussions.

## Components

- Topic head macro-F1 **0.647**, accuracy 0.639.
- Act head macro-F1 **0.960**, accuracy 0.958.
- Relation classifier macro-F1 **0.975**, accuracy 0.973 on test pairs
  (`supports`, `contradicts`, `supersedes`, `creates_action`, `answers`,
  `unrelated`). The pairs are annotation-derived; this is not open-world NLI.
- Memory: 60 meetings, 3,600 messages, 3,600 units, 2,413 relations; 0 hash
  mismatches, 0 missing/orphan vectors.

## Findings

1. **Fine-tuning the BGE ColBERT projection is the main win** (test R@1 .617,
   MRR .743). Frozen BGE ColBERT was .275/.436; E5 is .475/.617.
2. **Graph fusion helps only on temporal categories.** Replacing the graph's
   ColBERT signal everywhere drops test R@1 to .492. Routing GraphRAG only to
   `latest_message` and global latest-decision queries lifts the router to
   .636/.763, above both parents.
3. **Feature-aware was a template specialist.** Its v6 92.1% R@1 drops to 28.0%
   on held-out phrasings.
4. **Late chunking is the wrong tool here.** ColBERTv2+late chunking collapses
   (R@1 0.022); GraphRAG v2's fusion masks some damage but remains below v1.
5. **`graphrag_ft_router` is the default.** The agent path uses `colbert_ft`.
   The benchmark keeps the five methods above; other variants are in
   `v7_ablation.json`.
6. **Token-level distant edges do not pay off on v7.** 9,774 non-adjacent
   MaxSim edges leave the router unchanged and move hybrid R@1 only .534→.542
   while reducing R@5. v7 has no cross-meeting/multi-hop query type; add those
   queries before revisiting this.
7. **The full French ColBERTv2 is better than the L4 one, but not competitive.**
   Frozen .303/.406; projection fine-tuned .458/.581; still below BGE ColBERT
   .617/.743. Bigger ColBERT does not solve operation/field constraints.
8. **BGE + ColBERTv2 compression works.** 2048 centroids + 1-bit residual keeps
   BGE token accuracy (.659/.747) while shrinking the index from 349 MB to
   31 MB. It is available as `colbert_ft_v2`, not the default router.

## Limitations

- The corpus is still synthetic. It removes the v6 explicit markers and uses
  project/speaker-disjoint splits, but it is not AMI/ICSI. A real public corpus
  with decision annotations is the next required step.
- The relation test pairs derive from annotations, so macro-F1 0.975 measures
  template-defined relations, not open-world contradiction detection.
- BGE-M3's base is frozen; only its 1024→1024 ColBERT projection, ME5/E5 and
  the custom encoder are trained.
- The v7 agent evaluation is not rebuilt yet; v6 agent cases are archived.

## Reproduce

```bash
source .venv/bin/activate
python run_french_lab.py --force-data --train-only --epochs 6
python run_french_lab.py --eval-only
python train_retrieval.py --device cuda
python run_pretrained_comparison.py --batch-size 64
python finetune_bge_colbert.py --device cuda --epochs 2
python evaluate_distant_edges.py --device cuda   # negative ablation
python evaluate_v7.py --device cuda
python evaluate_ood.py --device cuda          # val -> held-out test
python app.py
```

Artifacts: `artifacts/v7_benchmark.json`, `artifacts/v7_ood_results.json`,
`artifacts/pretrained_comparison.json`, `artifacts/french_results.json`,
`artifacts/bge_colbert_projection.pt`, `artifacts/bge_colbert_finetuned_eval.json`,
`artifacts/bge_colbert_hybrid_eval.json`, `artifacts/distant_edges_eval.json`.
