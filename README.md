# Semantic Agent Engine

A memory for meeting talk that knows **what is still true**.

Someone says "the budget is proposed", later "approved", later still "actually,
cancelled". Search finds all three and picks whichever *sounds* most like your
question. This engine stores every message unchanged, groups the statements that
belong to one decision, keeps the latest one as the current state, and answers
with the sentence it came from — or refuses when the transcript does not say.

The seven steps, in one screen:

```
1  store the message unchanged (id, speaker, timestamp, text)
2  split it into sentence spans with exact character offsets
3  read each span: kind, status, intent, point, value
4  link a new statement to the earlier one it replaces
5  update memory: append-only state, current view, history
6  retrieve: BM25 + dense + typed links + cross-encoder rerank
7  answer: retrieved turns + memory state → LLM → cited answer or refusal
```

## Current measured state

The working corpus is synthetic: **five hand-written meetings** (CEO, PM, dev,
SRE, HR, finance, assistant, office manager), 103 turns, five topics, plus 36
questions — 26 answerable, 10 deliberately tricky. Gold and questions live in
`data/mine/`; the formats are `schemas/meeting.schema.json` and
`schemas/questions.schema.json`.

| step | metric | result |
|---|---|---|
| 1 store | edit refused; re-ingest no-op | refused; no-op; 0 duplicates |
| 2 chunk | offsets index their text | 0 mismatches |
| 3 label | change recall; values that are values | 1.00; 32 of 32 |
| 4 link | replaced turn linked | 4 of 4 |
| 5 memory | current value; leaked state | 12 of 12; 0 leaked |
| 6 search | hit@1 / @3 / @10 | .577 / .846 / .962 |
| 7 answer | cites expected; tricky refused | .808; 7 of 10 |

End-to-end on the 36 questions: **0.778 correct** (formal .833, messy .750,
tricky .700). Details in `docs/STEPS_EVAL.md` and `docs/STEP7_DEBUG.md`.

Step 7 is an LLM over the retrieved context, not a template: the top turns with
their ids and timestamps, plus the memory's current state and the unit that
proves it, go to a local model; the reply is kept only if it cites a supplied id
and every number it states appears in the cited material. Alone it scores .583;
when memory already holds the answer with proof, the deterministic path answers
first (hybrid, the default). The reranker is Qwen3-Reranker-0.6B fused with the
pipeline order (`docs/EMBEDDING_AB.md`).

Honest limits: five meetings and 36 questions, gold written by the author of the
meetings; the status/point/slot cues are rules fitted to them, and there is no
held-out meeting yet. Three of ten tricky questions still get an answer from the
LLM. Treat these as a working baseline, not a leaderboard result.

## Commands

```bash
make setup       # environment + torch/cuda check
make test        # full test suite
make meetings    # step 7 end to end on the five meetings (hybrid answerer)
make steps       # the per-step table above
make rerank      # Qwen3 reranker vs the pipeline order
make tag         # fine-tune Laya on the tag taxonomy (heavy, needs the GPU)
make demo        # legacy v3: the 7 mandatory scenarios, synthetic
make demo-real   # legacy v3: the same scenarios on SUM-RE meetings
make report      # rebuild the v3 report from its artefacts
```

Python 3.14 on an RTX 5060 (8 GB). The venv uses `--system-site-packages`; do
not install ML packages into it from the system torch path. The reranker and the
local LLM both want the GPU: Ollama's 7B model holds ~4 GB, so the answerer sets
`keep_alive: 30s` and the GPU tests must not run beside it.

## The UI

```bash
make ui        # http://localhost:8501
```

Four views in the sidebar:

- **Ask** — the v3 memory: the answer card is built from the same units as the
  evidence panel (status, date, speaker, unit id), so the two can never
  disagree; superseded statements are listed as history.
- **Meetings (v4)** — the current pipeline on the five hand-written meetings:
  retrieval table (rank · unit · slot · point · status), memory state, the
  answer with its proof, and the exact prompt the LLM receives. Answerer is
  selectable (hybrid / llm / rules), rerank can be switched off for a GPU-free
  run.
- **Timeline** — every statement about one subject, in order, current vs
  replaced.
- **Debug** — per-stage read and hit scores.

Shareable state: `?q=...` on Ask, `?view=v4&m=nova.meeting.json&q=...` on
Meetings. The UI keeps the embedder (and the reranker when enabled) on the GPU,
so stop it before `make test`.

## Data and schemas

```
data/mine/        five hand-written meetings, questions, gold changes
data/examples/    the minimal examples the schema doc uses
schemas/meeting.schema.json        4 fields per message, nothing else
schemas/questions.schema.json      style + expected message ids + abstain flag
schemas/memory_record.schema.json  what the architecture writes (not an input)
```

The input is deliberately dumb: `message_id`, `speaker`, `timestamp`, `text`.
Tags, spans, links and state are produced by the engine, never authored.
`engine/tools/validate_dataset.py` checks both formats, including that a cited
message exists and that a timestamp carries a time.

## What works, what does not

Working: immutable storage with SQL-level triggers, exact spans, value
extraction (`150 000`, `M€`, `1er avril`, `5 minutes`), point-aware state
(`project:slot:point`), deterministic supersede links, retrieved-context
answers with citations.

Not working yet: three of ten tricky questions are answered by the local 7B
model instead of refused; hit@1 is .577 (the right turn is in the top 3 for .846
and top 10 for .962, but not always first); headcount and anaphoric questions
("on les garde pour la formation") still miss; the rule cues are fitted to these
five meetings.

Legacy v3 (rollback) stays in `docs/ARCH_V3.md` with its own demo and report;
its real-data numbers are not comparable to the table above and are not used by
the current pipeline.

## Layout

```
src/engine/   segment · read (v4 tags, Laya tagger) · link (rules, v4 linker) ·
              store · threads (state) · query (search, rerank) · answer (v4,
              llm) · ingest_v4 · eval/* · tools/*
src/gen/      synthetic generators (tag corpus, latent graphs)
src/engine/eval/  v4_fixed · steps_eval · step7_debug · meetings_eval ·
              rerank_qwen · jina_ab · late_chunk · laya_real · change_head
api/  ui/     FastAPI and Streamlit over the v3 engine
schemas/      the JSON contracts
docs/         ARCH_V4 · STEPS_EVAL · STEP7_DEBUG · DATASET_SCHEMA ·
              EMBEDDING_AB · TAGGER_V4 · MEETINGS_EVAL · FIXED_V4_RESULTS
runs/         measured artefacts (meetings, steps, laya_tags, v4)
tests/        unit · property · integration · golden · e2e
```

## Rules this repo holds itself to

- **Nothing is sent.** Every action is a draft that needs confirmation.
- **Nothing is invented.** No address, person or value that is not in a stored
  message; missing information is refused, not guessed.
- **Nothing is deleted.** Superseded statements stay as history; state is a view.
- **A number that cannot move is not a test**, and a refusal that cites the right
  turn is not an answer (the answer metric counts only real answers).
- **Splits by meeting, never by sentence** — and a held-out meeting set is the
  next honest test.
