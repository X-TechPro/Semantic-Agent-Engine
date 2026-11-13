> Current state (2026-11-06): Rexu v3 is the active default; v3.1 is archived.
> v10 is the active hand-written dataset: 5 meetings × 55 messages, 19 queries.
> v7 remains legacy; generated v8/v9 are archived under `archive/generated_v8_v9/`.
> v10 zero-shot: R@1 .500, R@5 1.000, MRR .688, Evidence@5 .475.
> The benchmark page no longer shows architecture formulas.

# Flask dashboard implementation report

> Historical build report (v5). Current v7 results are in [README.md](README.md)
> and [docs/EXPERIMENT_V7.md](docs/EXPERIMENT_V7.md). v6 material is archived
> under `data/archive_v6/` and `artifacts/archive_v6/`.
> The "NOT RUN" and "no Flask/CUDA" statements below describe the original
> artifact host, not the current checkout.

## Delivered application

A local Flask app was added around the existing French meeting-memory experiment.

- `app.py`: local-only Flask server and JSON APIs.
- `memory_engine.py`: custom encoder loading, all five retrieval modes, synthetic/custom dataset handling, SQLite persistence and labelled evaluation.
- `templates/index.html`, `static/style.css`, `static/app.js`: responsive dark dashboard with search, meeting manager, dataset import and benchmark views.
- `compare_retrieval.py`: all-five-method CLI evaluation.
- `setup_rtx5060_arch.sh`, `setup_rtx5060_windows.ps1`, `run_app.sh`, `run_app.bat`, `check_gpu.py`: installation, startup and hardware diagnostics.

## Built-in corpus

The existing synthetic French corpus was preserved as the reference dataset:

- 60 meetings (30 formal and 30 messy)
- 3,600 message chunks
- 6 speakers per meeting
- 1,260 query-to-target examples, split by project
- existing tiny French Transformer checkpoint and SQLite memory

## Persisted synthetic test metrics

These are the already-trained held-out synthetic results, reproduced with the new CLI backend:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1786 | 0.4841 | 0.6032 | 0.2952 | 0.3296 |
| Custom dense | 0.6270 | 0.8730 | 0.9206 | 0.7333 | 0.7637 |
| Feature-aware | 0.9167 | 0.9960 | 1.0000 | 0.9537 | 0.9642 |
| French ME5 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Multilingual E5 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

The last two models are loaded lazily by the application and benchmark script. They were not scored in this build environment because `sentence-transformers` is absent and outbound DNS/model downloads are unavailable. Their metrics are intentionally left unavailable rather than fabricated.

## Custom dataset lifecycle

- The built-in synthetic source and reference SQLite database are never overwritten by an upload.
- Each uploaded dataset gets a random isolated folder under `data/user_datasets/`, its original uploaded source file, optional query file, `manifest.json` and a separate `memory.sqlite3`.
- The SQLite source-message table has UPDATE and DELETE triggers. Message vectors, feature activation arrays, meeting summaries and feature nodes are derived rows linked by `chunk_id`.
- On app restart, custom datasets are rehydrated from their manifest and SQLite memory.
- Supported input: meeting JSON arrays, `{ "meetings": [...] }`, one meeting object, train/val/test split JSON, JSON arrays of message chunks, and JSONL message chunks.
- Labelled queries support `target_chunk_id`, `target_message_id` or `target_json`. One gold chunk per query is currently evaluated.
- Without labels, the user can still query and inspect the top-k results; Recall/MRR/nDCG are not computed.

## Verification performed in the build environment

`smoke_test_app.py` passed:

- 60/3,600 built-in meeting/message counts.
- Built-in message JSON schema preserved and feature-aware query returns source JSON.
- Meeting manager transcript extraction.
- Custom dataset import and embedding.
- Custom labelled BM25/dense/feature-aware benchmark.
- Custom dataset persistence after constructing a fresh engine instance.
- SQLite rejects direct source-message UPDATE and DELETE operations.

`node --check static/app.js` and Python bytecode compilation passed. The five-way backend comparator reproduced BM25, dense and feature-aware metrics and reported both pretrained models as `UNAVAILABLE` due to the missing dependency.

## Environment limitation

The artifact-building environment has CPU-only PyTorch 2.10, no Flask, no `sentence-transformers`, no CUDA runtime and no Hugging Face DNS access. Consequently, the Flask server itself and the two downloaded pretrained weights could not be launched here. The app, templates, CSS/JS, dependencies, GPU setup scripts and tests are included for execution on the user's RTX 5060 laptop. Run `python check_gpu.py` after installation to verify CUDA before the five-way benchmark.

## Interpretation

The custom model was trained on the synthetic French corpus. Applying it to uploaded data is intentionally a domain-shift test, not automatic adaptation. Its topic and discourse-act heads have a fixed synthetic label space. The French ME5 and multilingual E5 paths provide independent pretrained dense retrieval baselines. A high feature-aware synthetic score is not evidence of general robustness on arbitrary transcripts.

## Fresh verification snapshot

- `python -m py_compile` passed for the app, engine, training, comparison, GPU-check and smoke-test modules.
- `node --check static/app.js` passed.
- `smoke_test_app.py`: PASS (60 meetings, 3,600 messages; custom import; labelled benchmark; persistence after restart; immutable source rows; chat transcript rendering).
- `compare_retrieval.py --dataset synthetic --split test --device cpu --batch-size 64` reproduced all BM25/custom-dense/feature-aware metrics over 252 held-out questions. Full values are in `artifacts/latest_synthetic_test_comparison.json`.
- French ME5 and multilingual E5 remain `UNAVAILABLE` in this build environment because `sentence-transformers` is not installed and this environment has no model-download access. They are included in the CLI and UI, but their scores are not fabricated.
- This build host has CPU-only PyTorch, no CUDA device and no Flask package. Therefore code/static syntax and engine smoke tests were run, but a live HTTP/browser session and RTX 5060 inference were not possible here. The installation and GPU diagnostic scripts are included.
