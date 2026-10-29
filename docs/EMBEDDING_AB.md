# Embedding A/B — jina v3, jina v5, e5; and what chonkie can and cannot do

All numbers measured on this repo's data: 114 annotated relation pairs (63 of
them genuinely distant, median gap 3 turns), 1,475 real turns with 187 changes,
and the 4,180-row held-out synthetic tag split. Same harness, same pools.

## Step 3 — classification, apples to apples

Same rows, same splits, same 2-layer head, same epochs, same label smoothing.
The only difference is the encoder and how much of it trains.

| model | regime | kind F1 | status F1 | intent F1 | real AUC |
|---|---|---:|---:|---:|---:|
| Laya mmBERT-base | end-to-end, option-marker head | **0.755** | **0.861** | **0.948** | **0.691** |
| Laya mmBERT-base | frozen, 2-layer head | 0.655 | 0.519 | 0.686 | 0.642 |
| jina v5-small | frozen, last-token pooling | 0.424 | 0.421 | 0.457 | 0.532 |
| jina v5-small | frozen, mean pooling | 0.471 | 0.468 | 0.441 | 0.524 |
| jina v5-small | last 4 layers + head | 0.535 | 0.508 | 0.519 | 0.532 |

v5's representation is much worse for this task even before training: 0.42 vs
0.66 on kind with the identical head. Mean pooling instead of last-token pooling
lifts it to 0.47/0.47 and does not close the gap (Laya frozen, same head: 0.66 /
0.52; Laya end-to-end: 0.76 / 0.86). Fine-tuning four layers recovers ~0.11 and
drops training loss 0.63 → 0.41, still short. Laya stays.

## Step 4 — relation pairs (AUC over gold vs decoy)

| model | values, sentence | threads, sentence | values, +3 turns | threads, +3 turns |
|---|---:|---:|---:|---:|
| jina v3 | 0.492 | 0.738 | 0.790 | 0.823 |
| jina v5 text-matching | 0.457 | 0.735 | 0.793 | 0.807 |
| jina v5 retrieval | 0.470 | 0.754 | 0.791 | 0.846 |
| e5-base | 0.344 | 0.548 | 0.789 | 0.827 |

Every model converges to ~0.79 with the window and every model sits at or below
chance on values without it. The window gain is partly leakage: 58 of 114 pairs
have the gold inside the new turn's 3-turn window. On the 28 genuinely distant
pairs the clean numbers are 0.71 (jina v3), 0.73 (e5) — the window still helps,
the encoder does not separate them.

## Step 6 — retrieval, true antecedent in top-k (114 edges, citation set)

| model | recall@5 | recall@10 | recall@20 |
|---|---:|---:|---:|
| jina v5 retrieval | 52 | 67 | **79** |
| jina v3 text-matching | 52 | 65 | **79** |
| jina v5 text-matching | 52 | 64 | 78 |
| e5-base | 40 | 55 | 64 |
| jina v5 classification | 22 | 31 | 41 |

v5 ties v3. Both beat e5 by 15 points. The classification adapter is the wrong
one for retrieval.

### Late chunking (one forward over the meeting, pool each turn's span)

`chonkie.LateChunker` is bound to a SentenceTransformer and re-chunks by its own
rules, so it cannot take v5 and does not preserve turn boundaries. The algorithm
is implemented directly in `engine/eval/late_chunk.py` and evaluated on the same
pools:

| unit embedding | recall@5 | recall@10 | recall@20 | values AUC | threads AUC |
|---|---:|---:|---:|---:|---:|
| per-turn, plain | 52 | 67 | 79 | 0.470 | 0.754 |
| **late chunked (v5, 32k)** | **75** | **94** | **98** | **0.738** | 0.693 |

This is the largest single gain measured in this repository: +19 edges at
recall@20 (69% → 86%) and the value decoy goes from chance (0.47) to 0.74, with
no per-pair forward pass. Caveats: v5 is a decoder, so each token sees only its
left context; gold and decoy cosines saturate at ~0.99, so the ranking is doing
the work, not the absolute scale; and the set is six meetings.

## Chonkie — what it is

A chunking library. Its chunkers decide **where to cut text**; none of them
classify, link, or track state.

Measured on `024c_EBPH` (251 ASR turns, 18k chars, no punctuation):

| chunker | chunks | median words | note |
|---|---:|---:|---|
| SentenceChunker | 226 | 12 | ≈ one chunk per turn; no punctuation to split on |
| RecursiveChunker | 395 | 9 | cuts mid-sentence, worse |
| SemanticChunker | 89–92 | 24–26 | embedding boundaries; **insensitive to chunk_size** (same 89 at 512 / 1024 / 2048) |

So, step by step:

- **2 (chunk sentences)** — marginal *on this corpus*. Measured over 7,960
  turns: median 12 tokens, p95 43, p99 75, max 190; 42 turns (0.53%) exceed the
  tagger's 96-token window and 13 exceed 128. Those are the long monologues where
  a paragraph gets one tag and its tail is never read. The current segmenter
  keeps an unpunctuated turn whole by design, so the fix is a **length-triggered
  split**, not replacing step 2. SentenceChunker cannot do it without
  punctuation; SemanticChunker can (89 blocks at ~25 words per block).
- **3 (classify)** — no. A chunker has no labels.
- **4 (link)** — not the decision, but a legitimate **candidate narrower**: the
  semantic segmentation gives ~90 blocks for 251 turns, i.e. ~3 turns per block,
  which is the same granularity as the 3-turn window that produced the 0.79. It
  is not thread-level (a meeting has ~10 threads, not 90 blocks).
- **5 (state)** — no.
- **6 (search)** — the best fit. `LateChunker` is the one mechanism with a real
  claim: embed the document once, pool per chunk, so each chunk vector carries
  its surrounding context. The catch is that it needs a long-context encoder and
  a token-level pooling hook; with v5 (32k) that is custom code, not a chonkie
  provider, because our turns are already the retrieval unit.

**The one experiment worth running**: LateChunker over a whole meeting with v5
as the encoder. It is the index-time version of the window effect that lifted
the pair AUC to 0.79, computed once instead of per query.

## Operational notes

- Both jina models are **CC BY-NC 4.0**: fine for the piscine, not for a product.
- v3's remote code needs two shims under transformers 5.5 (a `torch_dtype`
  string check and `all_tied_weights_keys`); applied in the local HF module cache.
- v5's `AutoModel` wrapper cannot be instantiated twice per process (`config_class`
  is missing), and its `encode()` has no internal batching: one process per task,
  chunk the calls.
- The host's NVIDIA kernel module and user library are mismatched
  (`nvidia-smi` fails, `NVML_SUCCESS` assert inside the torch caching allocator).
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` works around it; a reboot fixes it.
- The Streamlit app holds a CUDA context on the same 8 GB card and must be
  stopped before any embedding run.
