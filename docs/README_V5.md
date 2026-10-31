# French Synthetic Meeting Memory — Piscine Prototype

A runnable research prototype for the Intelligence Lab / Makers Embedding Lab piscine. **All meeting data in this repository is synthetic. No real meeting corpus is used.**

## What is implemented

### Five-stage memory pipeline

1. **Meeting ingestion and immutable JSON chunking**
   - 60 synthetic meetings: 30 formal and 30 messy/transcription-style.
   - 60 messages per meeting, six distinct speakers per meeting (3,600 messages total).
   - Every source message has exactly `{ "username", "message_id", "timestamp", "message" }`.
   - `chunk_id` is external to the source JSON and links that immutable record to all derived data.

2. **Meeting-level semantic feature extraction**
   - A trainable neural head predicts high-level topic and discourse-act activations.
   - Whole-meeting aggregation computes recurrence, importance and co-activation edges.
   - Per-meeting MiniBatchKMeans clusters provide additional latent groups.
   - Meeting summaries are generated from model-predicted activations, not synthetic gold labels.

3. **Message and meeting embeddings**
   - Custom one-layer Transformer encoder, 64-dimensional normalized vectors, topic and act heads.
   - Message encoder input includes meeting title, speaker and message; timestamps remain structured metadata.
   - Each meeting also has a global vector derived from its message vectors and a textual summary.

4. **Persistent memory**
   - SQLite stores source JSON separately from derived vectors and feature data.
   - SHA-256 hashes verify source records.
   - SQLite triggers reject direct UPDATE/DELETE of source messages.
   - `message_embeddings.chunk_id` is a foreign key to the immutable source message.
   - Feature nodes, co-activation edges, semantic units and predicted relations are persisted separately.

5. **Retrieval and evidence return**
   - BM25, custom dense retrieval and validation-tuned rank fusion.
   - Feature-aware retrieval uses query/message topic and act predictions, dense/BM25 ranks and explicit chronology for temporal questions.
   - Query returns the original source JSON objects, never regenerated JSON.
   - Retrieval is meeting-scoped by default in evaluation. The CLI can optionally search all meetings.

## Dataset and evaluation split

Fifteen fictional projects have four meetings each: two formal and two messy. Projects are assigned wholly to train, validation or test to prevent cross-meeting/project leakage:

| Split | Projects | Meetings | Formal | Messy | Messages |
|---|---:|---:|---:|---:|---:|
| Train | 9 | 36 | 18 | 18 | 2,160 |
| Validation | 3 | 12 | 6 | 6 | 720 |
| Test | 3 | 12 | 6 | 6 | 720 |
| **Total** | **15** | **60** | **30** | **30** | **3,600** |

There are 1,260 synthetic query-to-exact-message pairs (756 train, 252 validation, 252 test). Each query has one gold source chunk. The test set is kept out of training and policy tuning.

Messy meetings are generated in French with oral paraphrases, fillers, false starts/repairs, repeated words, missing punctuation, informal syntax and light ASR-like transcription artifacts. They remain synthetic; this is not a claim of realistic ASR coverage.

## Custom model training

The custom encoder has approximately 94k parameters, one Transformer encoder layer and a 64-dimensional embedding projection. It is trained from scratch on this French synthetic corpus; the older English prototype weights are not reused because the vocabulary and language distribution differ.

Objectives:
- Query-positive InfoNCE with in-batch negatives.
- Five same-meeting hard negatives per training query.
- Supervised topic and discourse-act feature losses, sampled across messages in each meeting.
- Checkpoint selection based on validation MRR@10.

## Run from scratch

Python 3.10+ is recommended. PyTorch uses CUDA when available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate the synthetic corpus and train the custom encoder.
python run_french_lab.py --force-data --epochs 6 --batch-size 32 --train-only

# Build/update SQLite memory, train the auxiliary heads/classifiers and evaluate.
python run_french_lab.py --eval-only

# Render the full human-readable report from the saved metrics.
python write_report.py

# Verify corpus, hashes, FK relations, immutable triggers and metric reproducibility.
python smoke_test_french.py
```

`--train-only` writes `artifacts/french_tiny_encoder.pt`; `--eval-only` loads that checkpoint, builds derived memory and runs the evaluation suite. It also writes `artifacts/FRENCH_PIPELINE_SUMMARY.md`; `write_report.py` produces the complete `artifacts/FRENCH_REPORT.md`.

## Query the memory

```bash
python run_french_lab.py \
  --eval-only \
  --query "Quelle décision a finalement été retenue concernant la base de données ?" \
  --meeting-id fr2-projet-01-reunion-1 \
  --top-k 5
```

Each result contains `rank`, `chunk_id`, `score` and the original four-field `source_json`.

## Compare pretrained encoders

The benchmark runner compares both requested models on the exact same meeting-scoped validation/test queries:

- `antoinelouis/french-me5-small`
- `intfloat/multilingual-e5-small`

```bash
# Requires internet access on first run to download model weights.
python run_pretrained_comparison.py --batch-size 64

# Skip optional fine-tuning and compare frozen pretrained encoders first.
python run_pretrained_comparison.py --no-finetune
```

Prefix handling follows the model cards: the French ME5 model is tested with raw text and an E5-prefix ablation; multilingual-E5 uses `query: ` and `passage: ` prefixes for asymmetric retrieval. The runner selects prefix mode on validation only. If a pretrained model is more than 0.05 MRR below the custom model on validation, it fine-tunes on train queries only using one positive plus five same-meeting hard negatives. Fine-tuning is kept only if validation MRR improves. Test is evaluated after those choices are frozen.

Run it from the project virtual environment (`source .venv/bin/activate`). A system-wide Python install can have a `torchvision` wheel built for a different `torch` version; importing it then raises `RuntimeError: operator torchvision::nms does not exist`, which surfaces as `Could not import module 'PreTrainedModel'` from `transformers`.

Held-out test results from `artifacts/pretrained_comparison.json` (prefix mode and fine-tuning chosen on validation only; both models were fine-tuned because their frozen validation MRR was more than 0.05 below the custom dense baseline):

| Model | Prefix | Val MRR@10 | Test Recall@1 | Test Recall@5 | Test MRR@10 | Test nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| `antoinelouis/french-me5-small` (fine-tuned) | `e5_prefix` | 0.865 | 83.3% | 94.8% | 0.882 | 0.899 |
| `intfloat/multilingual-e5-small` (fine-tuned) | `e5_prefix` | 0.875 | 82.1% | 95.2% | 0.876 | 0.895 |

Both fine-tuned checkpoints are saved under `artifacts/*_finetuned/` and are reused by the Flask model loader and `compare_retrieval.py`.

## Hybrid ColBERT + GraphRAG memory

`hybrid_retrieval.py` adds an end-to-end retrieval path next to the custom encoder. It is available in the UI method selector as **GraphRAG · hybride** and **ColBERT · BGE-M3**, and via `memory_engine.search(..., method="graphrag" | "colbert")`.

**Late interaction.** Chunks are encoded as token sequences with `BAAI/bge-m3` and scored with ColBERT's MaxSim: each query token takes its maximum similarity over the chunk's tokens, and those maxima are summed. ColBERTv2's published checkpoint is English MS-MARCO; BGE-M3 is the multilingual token-level encoder used here so French meeting text is not pushed through an English vocabulary. This is the ColBERT late-interaction scoring function, not the full ColBERTv2 stack: there is no residual compression, no PLAID index and no late chunking. Chunks are turns/messages, and the token index is cached at `artifacts/hybrid/`.

**Tagged graph.** Nodes are typed (`chunk`, `speaker`, `topic`, `act`, `meeting`, `project`) and carry tags; edges are typed (`said`, `act`, `topic`, `precedes`, `contains`, `theme`, and the persisted `supports` / `contradicts` / `supersedes` / `creates_action` / `answers` / `unrelated` relations). The graph view renders the whole corpus as a project constellation and each meeting as a relation graph.

**Differentiable propagation.** GraphRAG ranking is personalized PageRank over a torch sparse adjacency. Edge weights are `torch` parameters, so propagation is differentiable with respect to relation confidence. Fusion over `{ColBERT rank, BM25 rank, graph PPR, temporal subspace, act match, custom dense}` is a torch parameter vector trained with listwise cross-entropy by `HybridRetriever.train_fusion()`. The checked-in default weights were selected on `data/paraphrase_val.json`, a 12-query paraphrase dev set built from validation meetings; the OOD set was never used for selection.

**Temporal subspace.** Following TMRL (arXiv 2601.05549), the temporal signal is a compact subspace rather than an embedded timestamp: Fourier time features for hour/day/week/month/year, a recency or interval score, and an explicit query plan (`dernier/dernière/last`, author, act keywords, ISO dates, `HH:MM`). TMRL learns this subspace with LoRA on temporal contrastive data; this prototype uses the deterministic subspace and trained fusion instead of claiming that fine-tuning.

**Operator plan instead of a router.** The old act-argmax router silently sent risk questions to the commitment policy. The hybrid path parses the query lexically into constraints, applies speaker and act filters as hard/soft constraints, and keeps lexical evidence in the fusion, so an out-of-distribution paraphrase cannot discard BM25.

```bash
# Evaluates the existing custom methods, ColBERT, and GraphRAG on the 252
# templated test queries and the 18 hand-written OOD queries.
python evaluate_ood.py --device cuda --batch-size 64

# Optional: train the fusion weights (train split only) instead of defaults.
python evaluate_ood.py --train-fusion
```

Held-out results written to `artifacts/ood_results.json`:

| Split | Method | Recall@1 | Recall@5 | MRR@10 |
|---|---|---:|---:|---:|
| 252 templated test (meeting-scoped) | Feature-aware | 92.1% | 100.0% | 0.956 |
| 252 templated test (meeting-scoped) | GraphRAG | 52.0% | 81.7% | 0.646 |
| 18 hand-written OOD | BM25 | 44.4% | 55.6% | 0.513 |
| 18 hand-written OOD | Dense · custom | 33.3% | 33.3% | 0.373 |
| 18 hand-written OOD | Feature-aware | 22.2% | 44.4% | 0.315 |
| 18 hand-written OOD | ColBERT (MaxSim) | 72.2% | 88.9% | 0.787 |
| 18 hand-written OOD | **GraphRAG hybrid** | **94.4%** | **100.0%** | **0.972** |

The in-distribution benchmark rewards the encoder trained on that exact synthetic template distribution; the OOD set is the honest test of paraphrases such as "Qu'est-ce qui pourrait mal tourner si le trafic augmente ?" and multi-constraint questions such as "Quelle décision Inès a-t-elle prise en dernier ?". The two numbers should be read together.

## Transcribed evaluation corpus

The original synthetic French corpus is untouched. A second, harder corpus ships alongside it:

- `data/transcribed_meetings.json`: 30 meetings across 10 unrelated projects (mobile refactor, data migration, security, hiring, pricing, support, accessibility, infrastructure, partnerships, compliance). 1,200 messages, written as messy French transcription (fillers, false starts, repeated words, missing final punctuation, lowercase). Every meeting has exactly three decisions, a commitment, an objection and a risk.
- Cross-session references: later meetings cite the previous session by ISO date, and each project's first meeting cites the previous project's first session. `GraphMemory` turns those date references into `references` edges between meeting nodes, so the global and per-meeting graphs show crossover.
- `data/transcribed_queries.json`: 179 queries, including 60 easy and 119 hard. Categories: `easy_temporal` (last message), `easy_decision` (first decision), `hard_count` (how many decisions, multi-gold), `hard_multipart` (last message + author + time), `hard_commitment` (who owns an action), `hard_cross_meeting` (what was decided in a referenced session, global scope).

Regenerate deterministically with:

```bash
python generate_transcribed_corpus.py
```

The corpus is loaded automatically as the built-in dataset **Transcribed · 30 réunions**; select it in the sidebar, then use Recherche, Graphe or Benchmarks exactly as for the synthetic corpus.

Multi-answer queries (count, cross-meeting) carry a `target_chunk_ids` list. Benchmark Recall@K is the mean fraction of gold chunks found in the top K; MRR@10 uses the best matching rank.

Benchmark results on the 179 transcribed test queries:

| Method | Recall@1 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 16.4% | 29.8% | 0.245 |
| Dense · custom | 1.5% | 12.1% | 0.090 |
| Feature-aware | 27.7% | 34.5% | 0.321 |
| French ME5 | 19.2% | 32.6% | 0.312 |
| Multilingual E5 | 18.2% | 33.1% | 0.277 |
| ColBERT · BGE-M3 | 30.0% | 64.1% | 0.511 |
| **GraphRAG · hybride** | **54.0%** | **96.1%** | **0.809** |

GraphRAG by category (Recall@5 / MRR): last message 1.00 / 0.87, first decision 1.00 / 0.56, decision count 0.91 / 0.92, last message + author + time 1.00 / 1.00, commitment owner 0.97 / 0.82, cross-meeting 0.89 / 0.69. Recall@1 is structurally low for multi-gold queries (three decisions cannot all fit in one slot), which is why Recall@5 and MRR are the primary numbers here.

## Main results (current saved run)

Held-out test retrieval:

| Retrieval mode | Recall@1 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 17.9% | 48.4% | 0.295 |
| Custom dense | 62.7% | 87.3% | 0.733 |
| Feature-aware | **91.7%** | **99.6%** | **0.954** |

The feature-aware ranker uses learned topic/act activations and explicit chronological ordering where the question requests “latest”. Its operation routing reaches 100% on the 252 templated synthetic test questions; this is an intentionally controlled synthetic task and should not be interpreted as real-world language-understanding accuracy.

Feature-aware test performance by meeting style:

| Style | Recall@1 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| Formal | 96.8% | 100.0% | 0.983 |
| Messy | 86.5% | 99.2% | 0.925 |

See `artifacts/FRENCH_REPORT.md`, `artifacts/french_results.json` and `artifacts/french_feature_aware_test_rankings.json` for complete results.

## Piscine coverage and boundaries

Implemented in this prototype:
- BM25, dense and hybrid retrieval baselines.
- Contrastive retrieval training and hard-negative mining.
- Topic/act feature heads, meeting aggregation, feature recurrence/importance, co-activation graph and latent clustering.
- Immutable source memory, message/meeting vectors and semantic-unit/relation persistence.
- Synthetic classification of `supports`, `contradicts`, `supersedes`, `creates_action`, `answers`, `unrelated`.
- Small intent classifier for `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event`, `clarify`.
- Sourced evidence answers, cross-meeting timeline comparison, dossier output, simulated email/task/event proposals and clarification.

Not production-complete:
- No LLM natural-language synthesis; the demo agent is deterministic.
- No real email/calendar/task connector is called and no external action is executed.
- Cross-meeting contradiction resolution is not production-grade.
- Messy data includes synthetic ASR-like artifacts, not a validated speech recognizer or real transcript corpus.
- Relation and intent metrics come from controlled synthetic annotations and small held-out synthetic tests.

The scores demonstrate that the pipeline executes and learns useful patterns on the defined synthetic benchmark. They do not establish robustness on real meetings or unrestricted French speech.

---

# Local Flask dashboard (search, chat manager, custom datasets)

The project also includes a local Flask application with a dark UI. It does not replace or mutate the built-in synthetic dataset. Instead, imported corpora are stored as isolated datasets with separate SQLite memories, so the reference experiment remains reproducible while you can switch the active memory in the UI.

## Start the app

Recommended on your Arch Linux + RTX 5060:

```bash
chmod +x setup_rtx5060_arch.sh run_app.sh
./setup_rtx5060_arch.sh
source .venv/bin/activate
python app.py
```

Open `http://127.0.0.1:5000`. The app binds to localhost only. To force CPU or CUDA, set `MEETING_DEVICE=cpu` or `MEETING_DEVICE=cuda` before `python app.py`.

For Windows PowerShell, run `setup_rtx5060_windows.ps1`, then `python app.py` (or `run_app.bat`).

### RTX 5060 tuning

- Setup pins PyTorch 2.14.1 + CUDA 13.2 wheels, which are published for Linux and Windows and support Blackwell-class compute capability 12.0+. See the official PyTorch install matrix and release wheel listing: https://pytorch.org/get-started/previous-versions/ and https://download.pytorch.org/whl/cu132/torch/.
- TF32 is enabled for CUDA matmul/cuDNN where supported.
- Custom encoder inference uses CUDA autocast FP16; pretrained encoders are moved to CUDA and cast to FP16.
- Pretrained passage encoding defaults to batch size 64, max sequence length 128; this is a conservative starting point for an 8 GB GPU. Adjust the CLI `--batch-size` if you see OOM or have headroom.
- Query vectors are batch-encoded during benchmark runs. Passage vectors are cached per dataset/model, so repeated benchmarks do not re-encode the entire corpus.
- Use `python check_gpu.py` to verify the actual Torch CUDA build, detected GPU, compute capability, VRAM and FP16 matrix multiply.

Install order matters: the setup scripts install the CUDA-enabled Torch wheel first, then install the rest of `requirements.txt`. On a CPU-only system, install the CPU Torch build instead and run with `--device cpu`.

## UI features

### Two interfaces, one switch

The top bar has a `FIXED / CONSOLE` segmented switch. `CONSOLE` is the default and implements the "terminal-technical dark console" language in `design.md`: pure-black canvas, 1px hairlines, corner registration marks, Anton wordmark, Inter text, JetBrains Mono micro-labels, Lucide outline icons, and larger type and controls. `FIXED` restores the original dark dashboard. The choice is stored in `localStorage`, so it survives reloads. Both interface modes share the same backend and behaviour; only `static/console.css`, the icon markup, and the theme switch differ.

- **Recherche:** query the active memory using GraphRAG, ColBERT (BGE-M3 MaxSim), BM25, custom dense embeddings, feature-aware retrieval, French ME5, multilingual E5, or compare all five original methods for a single live query. Results include rank, score, source speaker/time, meeting, and the exact source JSON chunk.
- **Réunions:** browse meeting metadata and render messages as a transcript grouped by speaker and timestamp. Every message can reveal its underlying JSON object.
- **Datasets:** upload your own JSON or JSONL corpus and optionally a labelled query/evidence file. A custom import builds model embeddings, predicted feature activations, meeting summaries, feature nodes/edges and an isolated SQLite database. Select the dataset from the sidebar to search it.
- **Benchmarks:** launch a background evaluation against labelled queries in the active dataset. It reports Recall@1/3/5/10, MRR@10 and nDCG@5 for BM25, custom dense, feature-aware, French ME5, multilingual E5, ColBERT (BGE-M3 MaxSim) and GraphRAG. Missing pretrained weights/dependencies are reported as unavailable, never as fabricated scores.
- **Graphe:** global project constellation (projects, meetings, dominant theme per project) and a per-meeting graph with messages, speakers, topics, acts, temporal edges and predicted relations. Nodes are tagged and clickable; the inspector shows the speaker, timestamp, act, topic, text and tags. The graph is served by `/api/graph` and `/api/graph/<meeting_id>`.

## Custom data formats

Supported meeting JSON layouts:

1. A list of meeting objects, each containing `meeting_id`, optional metadata and `messages: [...]`.
2. `{ "meetings": [ ... ] }`.
3. A single meeting object containing `messages: [...]`.
4. The existing split format `{ "train": [...], "val": [...], "test": [...] }`.
5. A JSON array of message chunks, grouped into one meeting.
6. JSONL where each line is a message chunk; `meeting_id` is used for grouping where present.

Recommended message fields are the original four: `username`, `message_id`, `timestamp`, `message`. The importer also understands `speaker`/`author`, `id`, `time`, and `text`/`content` aliases for encoding and display. **Each uploaded message object is stored unchanged as its source JSON**; aliases are not written back into the original chunk. The source file bytes are also retained in the custom dataset folder.

The optional query/evidence JSON can be a list or `{ "queries": [...] }`. Each row should contain:

```json
{
  "query_id": "q-001",
  "split": "test",
  "meeting_id": "meeting-001",
  "query": "Quelle décision a été prise ?",
  "target_chunk_id": "meeting-001:004"
}
```

`target_message_id` can be used instead of `target_chunk_id`, and `target_json` can be used to identify the exact source object. Use one gold chunk per query in this prototype. Without labelled queries, live retrieval still works, but Recall/MRR cannot be calculated.

Examples are in `examples/custom_meetings.json` and `examples/custom_queries.json`.

## CLI comparison (all five methods)

Run on the built-in synthetic test split:

```bash
python compare_retrieval.py --dataset synthetic --split test --device cuda --batch-size 64
```

Run on your own corpus and labelled questions:

```bash
python compare_retrieval.py \
  --meetings-json /path/to/meetings.json \
  --queries-json /path/to/queries.json \
  --name "My French meetings" \
  --split test --device cuda --batch-size 64
```

The script writes a timestamped JSON report under `artifacts/`. You can select methods explicitly with `--methods bm25 dense feature_aware french_me5 multilingual_e5`.

## Important interpretation caveat

Custom uploaded meeting text is embedded using the **small custom encoder trained on the synthetic French benchmark**. This is a distribution-shift test, not automatic adaptation. The predicted topic/act heads have a fixed synthetic label space, so feature-aware retrieval may not generalize to new domains or specialized vocabulary. The pretrained ME5/E5 methods provide an independent comparison. If they perform better on your custom corpus, use that as evidence that the synthetic-trained feature heads need real domain adaptation—not as a reason to tune on your test queries.

The web app and backend were syntax-checked and the engine's end-to-end smoke tests were run in the artifact-building environment. The two pretrained checkpoints have since been downloaded and fine-tuned locally under `artifacts/hf_cache/` and `artifacts/*_finetuned/`, so the five-way comparison no longer depends on first-run internet access. Install via the included scripts on your laptop, then run `python check_gpu.py` before comparing all methods.

### Optional pretrained adaptation before the five-way comparison

On the built-in synthetic train/validation/test split, run:

```bash
python run_pretrained_comparison.py --device cuda --batch-size 64
```

This existing training script evaluates raw-vs-E5 prefixes for French ME5 on validation, uses the trained custom model as its validation reference, and only fine-tunes a pretrained encoder on the **training queries** if its validation MRR falls behind by more than the configured gap. It retains the fine-tuned checkpoint only when validation MRR improves. The Flask model loader and `compare_retrieval.py` automatically reuse an accepted local checkpoint and the validation-selected prefix mode. The final held-out comparison is then run with:

```bash
python compare_retrieval.py --dataset synthetic --split test --device cuda --batch-size 64
```

Do not tune on test queries. This optional adaptation script is currently defined for the built-in synthetic benchmark; it does not automatically fine-tune on arbitrary uploaded datasets.
