> **Legacy corpus.** Describes the six SUMM-RE meetings, which the current
> pipeline no longer uses. Input and question formats: `docs/DATASET_SCHEMA.md`.

# Dataset v4 — what is labelled, what is not, and what it can answer

Everything here describes data already in the repository. No new corpus was
downloaded for this change.

## Source

| | |
|---|---|
| corpus | `linagora/SUMM-RE`, French meeting transcripts (ASR, no punctuation) |
| revision | `47cb4ec3437ef5497b6092b8c6d1e0b5b1d86dc0` |
| licence | cc-by-sa-4.0 (recorded in `data/external/summre_test/PROVENANCE.json`) |
| extract | 28 meetings, 7,960 turns (`data/external/summre_test/meetings.json`) |
| used here | 6 meetings, 1,475 turns |

## Two independent annotations

**1. Change labels** — `data/gold/summre_change_labels.json`

- one binary label per turn: did this message change something;
- 1,475 turns, **187 changes** (12.7%);
- definition: `docs/CHANGE_DEFINITION.md`.

**2. Typed edges** — `data/gold/summre_temporal_train.json`,
`data/gold/summre_temporal_sel.json`

- 114 pairs of turns with a relation:
  **60 supports, 40 supersedes, 14 contradicts**;
- each edge carries a one-line human note ("pub une deux semaines -> trois
  semaines").

The two agree where they overlap: all 40 `supersedes` targets are labelled
`change`. `supports` targets are changes 16/60 of the time, `contradicts` 1/14.

## What is *not* labelled

- no slot or value per turn (the engine extracts them, the rules are in
  `read/v4_tags.py`; they are not ground truth);
- no responsible person per action, no recipient per request;
- no "unanswerable question" set;
- one annotator, one domain (organising an event: dates, budget, venue,
  communication, recruitment).

## Splits

Leave-one-meeting-out over the 6 labelled meetings. No meeting contributes to
both training and test. The corpus also declares holdout meetings
(`008a_EARH`, `049b_EBDH`, `053b_EEDH`) that are not labelled here.

## Why the primary target is change detection

`docs/CHANGE_DEFINITION.md` states the primary target is the new turn:
`change` / `no_change`, with citations scored separately. The earlier
benchmark ignored that and scored "find the one earlier turn the annotator
picked":

- 37 of 40 pairs have **another earlier turn that is an equally good citation**
  by word overlap (median: five alternatives);
- the trivial rule "always return the previous turn" scores 4/40 at rank 1 and
  23/40 at rank 5 — the semantic pipeline scores 4/40 and 13/40.

Asking a model to pick one of several correct answers is not a solvable target.
Hence:

- **citation set** — the annotated old turn plus every earlier turn stating the
  same value for the same point (`eval/v4_data.py:citation_set`). Any member
  counts as correct.
- **change detection** — 187 clear positives.
- **relation classification** — given two turns, name the relation. 114 gold
  pairs; this is a fair question because both turns are supplied.

## Hard cases from the brief: covered vs missing

| hard case (brief) | status |
|---|---|
| change of date / budget / owner | covered (dates and amounts dominate) |
| proposal vs decision | partly (40 supersedes edges) |
| restatement vs change | covered (1288 negatives) |
| conditional agreement vs final validation | missing |
| same topic across several meetings | missing (each meeting is its own project) |
| person cited vs person responsible | missing |
| information request vs instruction to act | missing |
| recipient absent or ambiguous | missing |
| questions with no answer in the sources | missing |

## Data expansion path

22 of the 28 extracted meetings (~6,485 turns) are unlabelled. Labelling them
with the same binary schema would multiply the change set by roughly ten and
is the single highest-value data investment available. It does not require new
downloads, only annotation.

## Reproduce

```bash
# heads (relation + change + status), leave-one-meeting-out
PYTHONPATH=src .venv/bin/python -m engine.link.train_relation --epochs 3
# benchmark: change detection, state tracking, relation classification, citations
PYTHONPATH=src .venv/bin/python -m engine.eval.v4_tasks
```
