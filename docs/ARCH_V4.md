# Architecture v4 — real-data memory, immutable messages, explicit state

Status: v4 path; v3 remains the rollback. The v2 diagram is the visual base;
v4 adds v3’s immutability and the five-layer search.

## 1. Message receipt — immutable ledger

Every message produces one immutable row:

```
message_id, source_id, project_id, speaker, timestamp, text,
content_hash, ingested_at, source_type
```

No later process edits that row. Links, event membership and state are separate
append-only records. “Replaced” is a `LinkEvent(supersedes)` plus a current-state
view, never a rewrite of the old message.

## 2. Sentence spans, then semantic units

```
chunk_id, message_id, start, end, text, sentence_index
```

The first layer is exact sentence spans with offsets. The second layer groups
adjacent spans by topic/intent. A short confirmation (“OK, validé”) stays
attached to the preceding proposal. If ASR has no punctuation, the turn is the
fallback; a boundary is never invented.

## 3. Reading — piscine tags with confidence

```
kind      decision | action | commitment | question | answer | fact | risk | idea
status    proposed | confirmed | cancelled | completed | replaced | unknown
intent    answer | compare | dossier | email | task | schedule | clarify
entities  person | project | object | date | amount | action | event
owner, deadline, evidence_span, confidence, model_version
```

Tags are annotations, not a gate. Low confidence lowers link/answer confidence.

## 4. Linking — FlauBERT XNLI/STS is a reranker

```
candidate retrieval (BM25 + dense + entity/event/action)
  → inokufu/flaubert-base-uncased-xnli-sts
  → relation head: supersedes | answers | supports | creates_action | contradicts | unrelated
  → status head: proposed | confirmed | cancelled | completed
```

NLI is not `supersedes`. Relation/status heads are trained on dated pairs with
hard negatives: same project/different value, same value/different project or
date, proposal vs confirmed vs cancelled. The node stores children and evidence;
a summary is a rendered view.

## 5. Memory update — append-only state

```
LinkEvent   from_unit, to_unit, relation, confidence, evidence, model_version
StateEvent  event_id, slot, value, status, evidence_unit, created_at
```

Current state:

- proposal stays open;
- confirmed/cancelled/completed updates current;
- a later proposal cannot displace a confirmed value;
- old values stay in history;
- missing evidence stays `unknown`.

## 6. Search — five layers + temporal state

1. **Query interpretation and reference resolution** — intent, requested slot,
   entities; resolve “it / he / she / that / him / then”.
2. **Entity and event retrieval (highest priority)** — separate people,
   projects, objects, decisions, actions, events indexes.
3. **BM25 + dense** — exact names/numbers/dates and paraphrase, fused by RRF.
4. **Graph expansion** — `same_event`, `answers`, `creates_action`, `supports`,
   `supersedes`.
5. **Contextual reranking** — FlauBERT XNLI/STS or a trained pair scorer reads
   query + candidate + nearby turns + event history.
6. **Temporal/state-aware ranking** — proposed/confirmed/completed/cancelled;
   latest valid state and the evidence that changed it.

## 7. Answer — evidence or abstention

Return the answer, citations (`message_id`/`chunk_id`/span), the state transition
and an explicit `unknown` when evidence is missing. Template by default; optional
LLM rewrite checked by a verifier. The LLM never writes memory.

## Real-data result (2026-10-28)

Six real meetings, one held out at a time. Produced by `engine.eval.v4_tasks`
(`runs/v4/tasks.json`, `runs/v4/linker.json`); the dataset is documented in
`docs/DATASET_V4.md`.

The reported target is the one §3 and §5 exist to serve, and the one the change
definition names as primary: **did this message change something**. The earlier
"find the one annotated earlier turn" table is now section 4, because that
question is under-determined by the labels (37/40 pairs have another equally
good citation; "return the previous turn" scores 4/40 at rank 1 against the
pipeline's 4/40).

| target | baseline | result |
|---|---:|---:|
| change detection, F1 (1,475 turns, 187 changes) | 0.063 rules / 0.000 never | **0.282** |
| change detection, AUC | 0.525 rules | **0.758** |
| state update on a change turn | — | 0.110 |
| state update on a non-change turn | — | 0.104 |
| relation classification, accuracy (114 pairs) | 0.541 majority | 0.435 |
| citation, rank 1 (40 pairs, set-scored) | 4/40 previous turn | 3/40 with links (1/40 rules-only) |

What the numbers say:

- **The trained change head works.** AUC 0.76 against 0.53 for the rules and
  0.50 for chance, F1 4.5x the rules. This is the first component measured above
  chance on real French speech.
- **The state layer is still broken**: it updates on 11.0% of change turns and
  10.4% of non-change turns — the same rate, because it is driven by the rule
  reader. A binary change signal does not name the slot or value, so it cannot
  fix this alone.
- **Relation classification is below the majority baseline** on 114 gold pairs.
  That is a data limit; the pair head itself was validated at 0.76 AUC on the
  larger change set.
- **Citation retrieval is below recency.** Until the target is defined better
  (several earlier turns are valid evidence), the previous-turn baseline is the
  number to beat.

Correction to the earlier table: the rows labelled "zero-shot FlauBERT" were
**not** zero-shot. `inokufu/flaubert-base-uncased-xnli-sts` ships a bare
`FlaubertModel` (196 tensors, no `classifier`, no `sequence_summary`), so
`AutoModelForSequenceClassification` built a randomly initialised head. The
heads are now defined in `link/flaubert_relation.py` and trained in
`link/train_relation.py`; an untrained model refuses to answer.

This is the current limit, not a shipped default. v3 remains the rollback.

## Files

- `src/engine/schemas_v4.py` — immutable/append-only contracts, §1-§5 labels
- `src/engine/store/v4.py` — SQLite store, FTS5, current-state view
- `src/engine/ingest_v4.py` — message → chunks → units → links → state
- `src/engine/segment/sentences_v4.py` — spans and topic/intent grouping
- `src/engine/read/v4_tags.py` — §3 tags, evidence span, slot/value
- `src/engine/link/flaubert_relation.py` — relation + status heads
- `src/engine/link/relation_data.py` — supervision from dated real pairs
- `src/engine/link/train_relation.py` — LOMO training, per-fold checkpoints
- `src/engine/link/v4_linker.py` — guards + typed `LinkEvent`s
- `src/engine/threads/state_v4.py` — append-only `StateEvent` derivation
- `src/engine/retrieve/entity_index.py`, `event_index.py` — entity/event lookups
- `src/engine/query/interpret.py`, `query/v4_search.py` — §6 five-layer search
- `src/engine/answer/v4.py` — cited answer, transition, abstention, verifier
- `src/engine/eval/v4_data.py` — gold relations, change labels, citation sets
- `src/engine/eval/v4_tasks.py` — the four measured targets above
- `src/engine/eval/v4_real.py` — antecedent/candidate read, per layer

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m engine.link.train_relation --epochs 3
PYTHONPATH=src .venv/bin/python -m engine.eval.v4_tasks   # the table above
PYTHONPATH=src .venv/bin/python -m engine.eval.v4_real    # per-layer candidate read
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```
