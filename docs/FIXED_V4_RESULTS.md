> **Historical.** Measured before the synthetic-only pivot and before the
> point-aware state. Current numbers: `docs/STEPS_EVAL.md`.

# Fixed v4, end to end on real meetings

Four arms through the shipped `V4Ingestor → V4Store → V4Linker → V4Search →
answer`, each meeting in its own store with its own leave-one-meeting-out
relation checkpoint (`models/flaubert_v4/fold_<mid>`). Six hand-labelled
SUMM-RE meetings, 7,960 turns, 40 annotated pairs, 114 edges, 187 changes.

| arm | links | states | @1 | @5 | @20 | citation@20 | state F1 | P | R | answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (rules) | 324 | 7,977 | 4 | 13 | 22 | 55/114 (48%) | 0.177 | 0.167 | 0.187 | 37/40 |
| + Laya tags | 641 | 8,043 | 4 | 13 | 23 | 55/114 (48%) | **0.272** | 0.180 | **0.551** | 39/40 |
| + late chunking | 324 | 7,977 | 6 | 15 | 27 | 78/114 (68%) | 0.177 | 0.167 | 0.187 | 37/40 |
| **both** | 641 | 8,043 | **6** | **15** | **27** | **79/114 (69%)** | **0.272** | 0.180 | **0.551** | 39/40 |

`@k` is the rank of the annotated earlier turn for the 40 pairs. `citation@20`
lets any turn stating the same value count (the annotation is under-specified on
its own).

## What each change actually bought

- **Late chunking** (one forward per meeting, pool each turn's span, contextual
  query vector): +5 pairs at rank 20, citation recall 48% → 68%. This is the
  retrieval win, and it survives being put inside the real fusion/rerank/state
  pipeline — but only after the query vector is contextual too. With a
  context-free query and pre-computed vectors it scored 9/40, worse than
  baseline.
- **Laya tags**: state detection recall 0.19 → 0.55, F1 0.18 → 0.27, answers
  verified 37 → 39. It also doubles the typed links (324 → 641) because more
  units carry a status and a slot. Precision is flat at 0.18: on real ASR the
  tagger over-fires "this carries a state" (468 false positives).
- They compose: neither cancels the other.

## Where it still fails

- **`at_1` is 6/40 (15%).** Retrieval finds the right region; it does not pick
  the exact antecedent.
- **`cites_gold` is 1/40 in every arm.** The answer cites the current state
  owner, and its transition source almost never equals the annotated replaced
  turn. Better retrieval did not become better supersession: the link decision
  is still the bottleneck, not candidate generation.
- **Reading precision 0.18.** Recall nearly tripled, but four in five predicted
  states are false. Wiring this into step 5 without a precision fix would
  overwrite memory constantly (the earlier measurement: updates at 11% of
  changes and 10.4% of non-changes).
- 6 meetings, 40 pairs. `citation@20` is permissive.

## What was tried and failed (recorded so it is not retried)

| attempt | result |
|---|---|
| status-probability threshold, 0.4 → 0.9 | precision flat at 0.20–0.21, recall 0.50–0.54, F1 0.29. No threshold separates changes from assertions. |
| change head on late-chunked v5 vectors, LOMO over the six meetings | per-fold AUC 0.44–0.54, F1 ≈ 0. The binary change label is not a property of one turn. |
| relation head on retrieval's top-20 instead of the recency window | gold covered 25/40, picked first 4/40 — the same 4/40 as on the recency window. Candidate source is not the limit. |
| deterministic "latest same-slot wins", the architecture's own rule | 3/40. The annotated old turn is usually not the latest same-slot unit before the new one. |

All four land at 2–6 of 40. That is the signature of a label problem, not a
model problem: the annotation circles *one* earlier turn, but the audit already
showed 37 of 40 pairs have another earlier message that states the same value.
On the citation-set metric — any turn stating the same value — the pipeline
reaches 79/114 (69%), just under the 0.7 bar. `at_1` on the exact circled turn
is the wrong thing to optimise against.

## Synthetic side

The tagger holds on held-out synthetic phrasing (kind 0.755, status 0.861,
intent 0.948, contrast pairs 1.00/0.79) and both existing demos pass unchanged
(`make demo` 19/19, `make demo-real` 10/10, full test suite green). The
degradation is specific to real ASR, and it is the same two places as before:
the exact antecedent and the supersede decision.

## Reproduce

```bash
PYTHONPATH=src python -m engine.eval.v4_fixed --arm laya+late
PYTHONPATH=src python -m engine.eval.v4_fixed --arm baseline
```

Files: `runs/v4/fixed_{baseline,laya,late,laya_late}.json`,
`src/engine/eval/v4_fixed.py`.
