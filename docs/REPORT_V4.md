# v4 report — measured on the questions the data can answer

All numbers come from `runs/v4/tasks.json` (benchmark) and
`runs/v4/linker.json` (training). Six real meetings, one meeting held out at a
time, so no model is scored on a meeting it saw.

## Why the target changed

The previous benchmark asked "find the one earlier turn the annotator picked".
That question is not solvable: 37 of the 40 pairs have another earlier turn that
is an equally good citation by word overlap, and the trivial rule "return the
previous turn" scores 4/40 at rank 1 and 23/40 at rank 5 — the semantic
pipeline scored 4/40 and 13/40. The project's own definition
(`docs/CHANGE_DEFINITION.md`) already says exact-pair scoring is a citation
metric, never the primary target. The data (`docs/DATASET_V4.md`) is now used
for the questions it actually labels.

Correction carried forward: the earlier "zero-shot FlauBERT" rows were not
zero-shot. The checkpoint has no classifier weights, so those numbers came from
a randomly initialised head.

## 1. Change detection — the primary target

1,475 turns, 187 changes (12.7%). "Did this message change something?"

| arm | precision | recall | F1 | AUC |
|---|---:|---:|---:|---:|
| never change | 0.000 | 0.000 | 0.000 | 0.500 |
| deterministic reader | 0.218 | 0.067 | 0.063 | 0.525 |
| trained head (LOMO) | **0.321** | **0.360** | **0.282** | **0.758** |

This is the first component in the project that is above chance on real data.
The rules are at chance on changes: they look for French cue words
("validé", "annulé") and the ASR transcripts rarely contain them. The trained
head separates changes from non-changes with AUC 0.76, and its F1 is 4.5x the
rules'.

Per meeting, AUC ranges 0.60 to 0.91. F1 is modest because 87% of turns are
negatives and the positive class is a mix of new decisions, task definitions
and date changes.

## 2. State tracking — what is still broken

Ingest each meeting turn by turn; compare what the engine put in force with the
gold change labels.

| measure | result |
|---|---:|
| a change produced a live state update | **11.0%** |
| a non-change produced a live state update | **10.4%** |
| replaced rows created on non-change turns | 0 |

The two rates are the same. That means the state layer does not distinguish a
change from chatter on this corpus: it is driven by the rule reader, which
section 1 shows is at chance. This is the clearest remaining defect, and it is
a *wiring + data* defect, not a missing concept: a binary "something changed"
signal does not say *which value* changed, so the state layer cannot use it
without a slot/value prediction.

## 3. Relation classification — given two turns

114 gold pairs (60 supports, 40 supersedes, 14 contradicts), pair supplied,
name the relation.

| measure | result |
|---|---:|
| accuracy | 0.435 |
| majority-class accuracy | **0.541** |
| macro-F1 | 0.274 |

Below the majority baseline. With ~20 held-out pairs per fold, this is a data
limit, not a reason to change architecture: a 110M-parameter encoder cannot
learn three relation classes from this.

## 4. Citation retrieval — kept, with its baseline

Same 40 pairs, now scored as a set (any earlier turn stating the replaced value
counts; mean set size 1.07).

| arm | rank 1 | rank 5 |
|---|---:|---:|
| previous-turn baseline | **4/40** | **23/40** |
| pipeline, rules only (no links) | 1/40 | 11/40 |
| pipeline with typed links + head | 3/40 | 14/40 |

Typed links move the retrieval arm from 1/40 to 3/40 at rank 1, which is the
one place in this report where the §4/§5 machinery is measurably ahead of its
own rules-only ablation. Both arms remain below "return the previous turn" at
rank 5.

## What the numbers mean together

- The engine can now tell a change from a non-change (AUC 0.76) — that is new
  and it is the dimension the brief asks to measure first.
- The state layer does not yet use that signal, so "what is in force" is not
  trustworthy on real speech.
- Relation classification and antecedent retrieval need more labelled data,
  not another architecture: 114 pairs and 40 citations are demonstration
  samples.
- The largest available data gain does not need a download: 22 of the 28
  extracted meetings (~6,485 turns) are unlabelled with the same binary schema.

## Artefacts

- `runs/v4/tasks.json` — every section above, per meeting
- `runs/v4/linker.json` — per-fold training metrics
- `models/flaubert_v4/` — final heads + encoder, plus one checkpoint per held meeting
- `docs/DATASET_V4.md` — what is labelled, what is not, splits, licences

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m engine.link.train_relation --epochs 3
PYTHONPATH=src .venv/bin/python -m engine.eval.v4_tasks
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```
