# Semantic Meeting Memory — v7

One corpus, one split protocol, one evaluation for every model. v7 replaces the
separate synthetic/transcribed v6 benchmarks with a project- and
speaker-disjoint corpus and held-out query phrasings.

v6 is archived:

- Data: `data/archive_v6/`
- Results and finetuned v6 models: `artifacts/archive_v6/`
- v6 README: `docs/README_V6.md`

## What v7 is

- 15 projects × 4 meetings, 60 messages per meeting: 60 meetings, 3,600
  messages.
- **Project-disjoint split:** train 9 projects / 36 meetings, val 3 / 12, test
  3 / 12.
- **Speaker-disjoint pools:** train 12 names, val 6, test 6; no overlap.
- **Held-out test phrasings:** train/val use one template family, test uses a
  different paraphrase family. Exact train↔test query overlap is **0**.
- Test projects: **Delta**, **Équinoxe**, **Mistral**. Test speakers: aya,
  ibrahim, lucas, manon, nadia, théo.
- Test queries: 264 — topic-latest-decision, commitment-owner, objection,
  global-latest-decision, latest-message, and a 6-gold decision-count query.
- The corpus is still synthetic, but it no longer uses `décision:` prefixes,
  ISO dates in text, or scripted closings.

Full protocol and tables: [docs/EXPERIMENT_V7.md](docs/EXPERIMENT_V7.md).

## Training protocol

| Component | Trained on | Selected on | Tested on |
|---|---|---|---|
| Custom encoder + topic/act heads | v7 train | v7 val MRR | v7 test |
| Feature-aware retriever | v7 train (encoder) | v7 val policy | v7 test |
| Relation classifier | v7 train pairs | — | v7 test pairs |
| GraphRAG fusion weights | v7 train | v7 val | v7 test |
| French ME5 / multilingual E5 | v7 train (InfoNCE) | v7 val | v7 test |
| BGE-M3 / ColBERT / ColBERTv2 / late chunking | frozen | — | v7 test |

No test query, project, speaker, or relation pair is used for training or
selection.

## v7 test results (264 held-out queries, unseen projects/speakers)

These are the v7 protocol's baselines. The table predates Rexu v3: v2.1/v2.2
and GraphRAG were the entries measured on v7, and v2.2 remains the BGE
comparison baseline. Rexu v3 is the shipped default and is measured on the
pyramid corpora in **Cross-architecture results** at the bottom of this README.

`Recall@K` credits the fraction of gold targets on multi-gold queries;
`AnyR@K` credits at least one gold target.

| Method | R@1 | AnyR@1 | R@5 | AnyR@5 | MRR@10 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| BM25 | 0.119 | 0.125 | 0.344 | 0.352 | 0.210 | 0.236 |
| BGE-M3 · dense only | 0.321 | 0.352 | 0.597 | 0.617 | 0.467 | 0.493 |
| ColBERT · BGE-M3 only | 0.275 | 0.307 | 0.601 | 0.617 | 0.436 | 0.471 |
| **Rexu v2.1 · BGE-M3 FT + ColBERT v1** | **0.614** | **0.652** | **0.895** | **0.905** | **0.756** | **0.787** |
| **Rexu v2.2 · BGE-M3 FT + ColBERTv2** | **0.625** | **0.663** | **0.907** | **0.917** | **0.763** | **0.797** |
| **GraphRAG · FT-ColBERT router** | **0.633** | **0.670** | **0.926** | **0.936** | **0.776** | **0.812** |

**Rexu naming.** v2.1 is BGE-M3 FT + ColBERT v1; v2.2 adds ColBERTv2
compression; v2.3 is archived. The older parser-based temporal path and its
.920 varied-suite number were removed; Rexu v3 is the current default; v2.2 remains the BGE comparison baseline.

## Rexu v3 architecture

Rexu v3 keeps the whole meeting in context instead of pooling each message
alone. Every message is a span separated by the encoder's separator token, so
a message vector already carries what was said before it. Query and document
tokens go through the same 1024→128 late head, and MaxSim scores a query token
against its best match anywhere in the message. No BGE-M3, no graph.

```mermaid
flowchart LR
    M["Meeting<br/>all messages in order"] --> J["Join with separator token"]
    J --> E["pplx-embed-context-v1-0.6b<br/>contextual encoder (16k tokens)"]
    E --> S["Split on separator:<br/>one token span per message"]
    S --> H["Late head 1024 → 128<br/>per token, L2-normalised"]
    Q["Question"] --> QE["Same encoder"] --> QH["Same late head"]
    H --> MS["MaxSim<br/>for each query token: max dot over message tokens,<br/>then sum over query tokens"]
    QH --> MS
    MS --> R["Rank messages<br/>(max turn score per session)"]
```

Training uses the v7 train split only: one positive message per query and five
hard negatives (same topic, wrong status/person/date), with the encoder frozen
and the 128-d projection trained. Test queries, projects, speakers and relation
pairs never enter training or model selection.


**Fine-tuning the BGE ColBERT projection is the main win.** The projection is
trained on a 50/50 mixture of v7 train and project-disjoint messy transcriptions,
with one positive + five hard negatives; the base BGE-M3 is frozen. The router
keeps FT ColBERT for semantic queries and uses GraphRAG only for
`latest_message` and global latest-decision queries. Test R@1 .633 / MRR .776,
vs BGE-only .321/.467 and ColBERT-only .275/.436. Graph fusion on every query
**hurts**; routing it only to temporal cases helps. Removed variants remain in
`artifacts/v7_ablation.json`.

Token-level distant-message edges were tested and rejected: 9,774 non-adjacent
MaxSim edges leave the router unchanged, move hybrid R@1 only .534→.542, and
lower R@5. v7 has no cross-meeting/multi-hop queries to reward them.

Hard-negative discrimination on the same split: fine-tuned ColBERT R@5 0.879,
MRR 0.743, gold-beats-hard 0.792, intrusion@5 0.198. Full table:
[docs/EXPERIMENT_V7.md](docs/EXPERIMENT_V7.md).

French ColBERTv2 ablation: frozen `antoinelouis/colbertv2-camembert-L4-mmarcoFR`
scores test R@1 0.265 / MRR 0.372; fine-tuning on v7 train lifts it to
0.398 / 0.528. The full `AdrienB134/ColBERTv2.0-mmarcoFR` (12 layers, 110M)
scores 0.303 / 0.406 frozen and 0.458 / 0.581 with a fine-tuned projection.
ColBERT-XM (`fr_XX`, 128-d, 4 epochs) reaches 0.519 / 0.629; its 38 MB index is
9× smaller than BGE's, but query latency is 59 ms vs 6.5 ms and total adapters
are 852M params. Temporal reranking lifts XM to 0.530 / 0.651 and BGE to
0.670 / 0.769, but does not fix topic/objection. BGE remains the best ColBERT
backbone; not adopted.

## Seen templates vs held-out phrasings

In-distribution = v7 val (same template family as train, unseen projects and
speakers). OOD = v7 test (held-out phrasings, unseen projects and speakers).

| Method | Val R@1 | Test R@1 | Val MRR | Test MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.175 | 0.125 | 0.288 | 0.210 |
| BGE-M3 · dense only | 0.381 | 0.352 | 0.493 | 0.467 |
| ColBERT · BGE-M3 only | 0.337 | 0.307 | 0.474 | 0.436 |
| Rexu v2.2 | 0.806 | 0.663 | 0.867 | 0.763 |

## Per unseen project (Recall@1 / MRR@10)

| Method | Delta | Équinoxe | Mistral |
|---|---|---|---|
| BGE-M3 · dense only | 0.375 / 0.494 | 0.330 / 0.437 | 0.352 / 0.469 |
| ColBERT · BGE-M3 only | 0.318 / 0.454 | 0.284 / 0.407 | 0.318 / 0.446 |
| Rexu v2.2 | 0.648 / 0.759 | 0.682 / 0.772 | 0.659 / 0.759 |

## Components

- Topic head macro-F1 0.647; act head macro-F1 0.960.
- Relation classifier macro-F1 0.975 on test pairs (annotation-derived, not
  open-world NLI).
- Memory: 60 meetings, 3,600 messages, 3,600 units, 2,413 relations; 0 hash
  mismatches, 0 missing/orphan vectors.

## Run

```bash
source .venv/bin/activate
pip install -r requirements.txt

# v10 is the hand-written multi-hop corpus
python evaluate_v10.py

# v7 legacy experiment
python evaluate_v7.py --device cuda

# UI
python app.py                               # http://127.0.0.1:5000
# or ./run_app.sh, which uses .venv/bin/python directly
```

The active built-in dataset is **v10 · Les Rivières de Sel**: 5 hand-written
meetings × 55 messages, 19 labelled multi-hop queries, split 9 train / 6 val /
4 test. v7 remains available as `synthetic` (legacy). The generated v8/v9
corpora are archived under `archive/generated_v8_v9/`.

## Defaults

- The UI default is **`v10`** with **`rexu_v3`**. v3.1 is archived.
- `rexu_v3` = pplx-context(full meeting) → contextual 1024-d tokens →
  Dense(1024→128) → MaxSim. No BGE-M3.
- **Rexu v2.1/v2.2**, BM25, BGE-M3 dense, raw BGE-M3 ColBERT, and GraphRAG
  remain available for comparison.
- v10 zero-shot with the existing v3 head: R@1 .500, R@5 1.000, MRR .688,
  Evidence@5 .475, All-Hops@5 .000 on 4 test queries.
- Graph view: projects → meetings in a spaced grid; meeting view: message
  chunks → speakers/acts/topics, with `precedes` and relation edges. Large
  graphs use short meeting labels and retain full titles in the inspector.

## Limits

- Still synthetic. It is harder and leakage-free, but not a real meeting
  corpus. A public corpus with decision annotations (AMI/ICSI/MeetingBank/
  QMSum) is the next required step.
- The relation classifier learns annotation-defined pair types; macro-F1 0.975
  is not evidence of open-world contradiction detection.
- BGE-M3/ColBERT are frozen; only the custom encoder, GraphRAG fusion, and
  ME5/E5 are trained.
- The v7 agent evaluation is not rebuilt; v6 agent cases are archived.
- GraphRAG v2/late chunking is worse on every v7 test; it is not a default.

## Docs

- [docs/EXPERIMENT_V7.md](docs/EXPERIMENT_V7.md) — protocol, full tables,
  per-project results, findings.
- [docs/README_V6.md](docs/README_V6.md) — archived v6 README.
- [docs/DATA_REALISM_AND_TRANSFER.md](docs/DATA_REALISM_AND_TRANSFER.md) — why
  the old cross-dataset comparison was invalid.
- [docs/GRAPH_RAG_V2_ABLATION.md](docs/GRAPH_RAG_V2_ABLATION.md),
  [docs/V1_ERROR_ANALYSIS.md](docs/V1_ERROR_ANALYSIS.md) — v6 diagnostics.

## Cross-architecture results (pyramid corpora, turn level)

Rexu v3 is the shipped default, so it needs numbers of its own, measured on the
same gold as the other engines rather than on the older v7 protocol. The corpus
is `data/pyramid/`: two hand-authored 206-turn meetings, 50 queries each (20
1-hop, 10 2-hop, 10 3-hop, 10 4-hop), never used for training.

`R@1` = the main answer turn is ranked first; `R@5/10/20` = at least one gold
evidence turn in the top k; `MRR` = rank of the main turn; `ev@5` = mean
fraction of all gold evidence in the top 5.

| dataset | method | R@1 | R@5 | R@10 | R@20 | MRR | ev@5 |
|---|---|---|---|---|---|---|---|
| EN | BM25 | 0.020 | 0.260 | 0.560 | 0.980 | 0.153 | 0.240 |
| EN | BGE-M3 dense | 0.040 | 0.540 | 0.800 | 1.000 | 0.272 | 0.430 |
| EN | Pyramid v3.1 | 0.260 | **0.960** | **1.000** | **1.000** | 0.464 | **0.753** |
| EN | **Rexu v3** | **0.400** | 0.920 | 0.960 | 0.960 | **0.564** | 0.653 |
| EN | SAE v4 + reranker | 0.060 | 0.580 | 0.720 | 0.780 | 0.244 | 0.427 |
| EN | SAE v4, no reranker | 0.000 | 0.040 | 0.380 | 0.780 | 0.078 | 0.040 |
| FR | BM25 | 0.060 | 0.440 | 0.600 | 0.980 | 0.185 | 0.335 |
| FR | BGE-M3 dense | 0.020 | 0.600 | 0.720 | 0.960 | 0.258 | 0.480 |
| FR | Pyramid v3.1 | 0.180 | **0.920** | **0.980** | **1.000** | 0.357 | **0.677** |
| FR | **Rexu v3** | **0.460** | 0.860 | 0.960 | 0.980 | **0.610** | 0.665 |
| FR | SAE v4 + reranker | 0.160 | 0.600 | 0.720 | 0.760 | 0.319 | 0.465 |
| FR | SAE v4, no reranker | 0.040 | 0.260 | 0.520 | 0.760 | 0.151 | 0.188 |

Resources (RTX 5060 8 GB; build/index and model load included where noted):

| method | build / index | VRAM peak | RAM peak | query latency |
|---|---|---|---|---|
| BM25 | 0.02 s | — | — | < 1 ms |
| BGE-M3 dense | 0.9 s encode | 2.3–2.9 GB | 3.6–5.2 GB | one 206×50 matmul |
| Pyramid v3.1 | 11.8–13.7 s | 2.9 GB | 4.7–5.3 GB | included in build |
| **Rexu v3** | 5.4–11.6 s (cached docs) | 2.4 GB | 3.5–4.4 GB | 43–45 ms (p95 52–55) |
| SAE v4 + reranker | 1.0–1.3 s | 2.6 GB | 2.8–4.2 GB | 200–224 ms |
| SAE v4, no reranker | 1.0–1.3 s | 1.1 GB | 2.6–3.7 GB | 42–83 ms |

What it says:

- **Rexu v3 is the best top-1 retriever** on both languages: R@1 0.40/0.46 and
  MRR 0.56/0.61, ahead of pyramid (0.26/0.18 and 0.46/0.36) and far ahead of
  dense/BM25.
- **Pyramid wins evidence coverage** (R@5 0.96/0.92, ev@5 0.75/0.68): it finds
  the chain of supporting turns even when the main turn is not ranked first.
  That is the multi-hop property, not a top-1 win.
- **SAE is reranker-dominated here**: without it R@5 falls to 0.04/0.26; with
  it, 0.58/0.60. Its unit model and rule readers target French decision/state
  questions, not these chain questions.
- **Cost**: Rexu is ~5× faster per query than SAE with the reranker, uses
  2.4 GB VRAM, and reuses a cached document encode. Pyramid pays a ~12 s
  one-off build and then answers in milliseconds.

Previous numbers stay as measured and are not replaced:

- v7 protocol (264 held-out queries, unseen projects/speakers): BGE-M3 dense
  R@1 .321 / MRR .467; FT-ColBERT v2.2 .625/.763; GraphRAG router .633/.776
  (table above).
- v10 zero-shot with the v3 head: R@1 .500, R@5 1.000, MRR .688,
  Evidence@5 .475, All-Hops@5 .000 on 4 test queries (Defaults section).
- QMSum and the doc-level caches are local artefacts: `artifacts/` is ignored
  by git, so these numbers come from the raw result JSONs in
  `data/pyramid/results/`.

Caveats: n = 50 per language, one meeting per language. SAE is truncated to
its top 30, so its MRR and full-recall are floors. Rexu builds its document
tokens once per meeting and caches them; that build is reported separately
from per-query latency. Reproduction scripts live in the pyramid repository
(`eval_baselines.py`, `eval_rexu.py`, `eval_sae.py`) because they import its
metric and data code.
