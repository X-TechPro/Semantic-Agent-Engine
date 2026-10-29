# Tagger V4 — Laya fine-tuned on the piscine tags

Step 3 of the architecture ("classify each chunk") was a keyword reader: 0.06 F1
on real change turns, ranking quality 0.525. This replaces the classification
part of it with a fine-tuned model. It does **not** replace the span/entity/slot
extraction, which stays on the rule reader.

## What was trained

| | |
|---|---|
| checkpoint | `convaiinnovations/laya` → `multilingual` (mmBERT-base, 322M) |
| why not the repo root | the root checkpoint is ModernBERT-large, English-only; the Laya card says so and the corpus is French ASR |
| head | Laya's own: `[CLS] <question> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`, one logit per option, 2 transformer layers, `type_emb` slot per task |
| objective | cross-entropy over the options (not the shipped RLCD objective — this is supervised tagging, not calibrated question answering) |
| encoder lr / head lr | 2e-5 / 1e-3, cosine, 6% warmup |
| epochs | 2, batch 32 |
| regularisation | label smoothing 0.05, dropout 0.1, token embeddings frozen |
| sizes | 12,540 train / 1,980 val / 4,180 test rows, 320 contrast pairs |

Weights: `runs/laya_tags/multilingual_v2/model.safetensors` (whole model, 1.2 GB —
the embedding alone is 786 MB, which is why saving only heads is wrong).

## Tag schema

From piscine.md §4 (unit types), the semantic-unit `status`, and §5 (intents).

| task | options |
|---|---|
| kind | decision, action, commitment, question, answer, fact, risk, idea |
| status | proposed, confirmed, cancelled, completed, unknown |
| intent | answer, compare, dossier, email, task, schedule, clarify, none |

`replaced` is excluded: it is derived by the linker, not said in a sentence.
`none` is added to intent because most meeting turns are not requests.

## Data

- **46 families**, each a phrasing pattern with 2–8 surface forms, filled from
  44 names, 36 projects, 24 deliverables, 15 dates, 14 amounts.
- **Family-disjoint splits.** Whole phrasings go to one split; the 8 names and
  6 projects reserved for test never appear in train.
- **Contrast pairs.** Same content, one cue changed: `on valide X` /
  `on n'a pas encore validé X`; `on part sur X` / `on part sur X ou pas`;
  `X glisse` / `X risque de glisser`; `j'envoie X` / `je m'engage à envoyer X`.
  Every contrast has a *training* phrasing and a *different test* phrasing, so a
  test failure means the cue was not learned, not that it was never seen.
- **ASR surface.** Lowercase, final punctuation dropped, fillers only at places
  a speaker can pause, repeated words, correct elision (`du`, `au` — the first
  version produced "on gardu support", which is a bug, not noise).
- **Oversampling.** Contrast rows are emitted twice; without it they were ~15%
  of a batch and the model fit the template families instead of the cues.

## Results

Held-out phrasing (test split), 2,640 rows for kind and status, 1,540 for intent:

| task | macro-F1 | accuracy | ECE |
|---|---:|---:|---:|
| kind | 0.755 | 0.772 | 0.162 |
| status | **0.861** | **0.870** | 0.087 |
| intent | 0.948 | 0.950 | 0.050 |

Contrast pairs, both members must be right:

| field | pairs | both correct | per contrast |
|---|---:|---:|---|
| kind | 200 | **1.00** | question 1.0, idea 1.0, risk 1.0, answer 1.0, commitment 1.0 |
| status | 120 | 0.792 | completed 1.0, cancelled 0.975, proposed 0.40 |

Real corpus (1,475 turns, 187 annotated changes), zero-shot transfer:

| probe | AUC |
|---|---:|
| status "carries a state" vs annotator change | **0.691** |
| kind ∈ {decision, action, commitment} vs annotator change | 0.540 |

For reference, the keyword reader on the same turns ranks 0.525.

## Known weaknesses

1. **"c'est acté" is read as `completed`, not `confirmed` (0.65).** The synthetic
   data uses "c'est acté" for confirmed and "c'est déjà acté depuis …" for
   completed; the model generalised to the wrong one. This is the whole of the
   failing status contrast. Fix by tightening the definition, not the model:
   confirmed = validated now, completed = already done before now.
2. **Kind does not transfer to real meetings (0.54).** The synthetic kind cues
   ("et si", "risque de", "je m'engage") are rarer and differently worded in the
   ASR transcripts. Status transfers because its cues are grammatical
   (`pas`, `plus`, `déjà`, `annule`), not topical.
3. **Intent has no validation split.** Each intent has two families; one goes to
   train, one to test. The 0.95 is real but measured on one held-out phrasing
   per intent.
4. **Synthetic policy ≠ annotator policy.** The 187 real change labels are
   binary; they do not say which of "status" or "kind" the annotator meant. The
   0.691 is a transfer probe, not an in-domain score.
5. **No slot/value/span prediction.** Those fields still come from
   `read_tags`; only kind/status/intent are model output.

## Reproduce

```bash
PYTHONPATH=src python -m gen.tag_synth --per-family 220 --pairs 40 --out data/synth/tags
PYTHONPATH=src python -m engine.read.laya_tags --epochs 2 --lr-enc 2e-5 \
    --out runs/laya_tags/multilingual_v2
PYTHONPATH=src python -m pytest tests/unit/test_tag_synth.py tests/unit/test_laya_tags.py -q
```

## Using it

`engine.read.laya_reader.tag_texts(texts, model_dir)` batches whole meetings
through the three heads and returns the existing `Tags` objects with kind,
status and intent replaced. `read_tags` remains the rules-only path, so nothing
that does not want torch pays for it.

## Files

- `src/gen/tag_synth.py` — generator, families, contrasts, splits
- `src/engine/read/laya_tags.py` — load, fine-tune, predict, metrics
- `src/engine/read/laya_reader.py` — drop-in adapter to `Tags`
- `data/synth/tags/{train,val,test,pairs}.jsonl`
- `runs/laya_tags/multilingual_v2/{model.safetensors,meta.json,metrics.json}`
- `tests/unit/test_tag_synth.py`, `tests/unit/test_laya_tags.py`
