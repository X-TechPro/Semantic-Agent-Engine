> **Historical.** These are the first end-to-end numbers on the five
> meetings, before the point field, the rule linker and the reranker.
> Current: `docs/STEPS_EVAL.md` (step table) and `docs/STEP7_DEBUG.md`.

# Hand-written meetings — first end-to-end result

Five meetings written by hand (not generated): 24, 20, 20, 18 and 20 messages,
five topics, realistic roles (CEO, PM, dev, SRE, HR, finance, assistant, office
manager), messy French with disfluencies and no punctuation. 36 questions:
18 formal, 8 messy, 10 tricky (answerable-looking but not answered in the
transcript). Files: `data/mine/*.meeting.json`, `data/mine/*.questions.json`;
results: `runs/meetings/eval.json`.

| metric | value |
|---|---:|
| questions correct | **0.222** |
| right turn in top 1 / 3 / 10 (answerable 26) | 0.192 / 0.500 / **0.808** |
| answer cites an expected turn | 0.231 |
| answer contains the expected value | 0.192 |
| tricky questions refused (10) | **0.000** |

By style: messy 0.625 correct, formal 0.167, tricky 0.000.

## What works

Retrieval. The turn that states the answer is in the top 10 for 81% of
answerable questions and in the top 3 for half. Step 6 is not the blocker on
this data.

## What fails, in order of damage

1. **Step 7 never refuses.** 0 of 10 tricky questions got "I don't have enough
   information". Every question found some state and printed it. A meeting with
   any date in it will answer "who is the landlord?" with that date.
2. **The answer reports the top hit's state, whatever the question asks.**
   "Quand se termine le bail actuel ?" answered "septembre (confirmed)" — a date
   from a different point, cited from a different turn.
3. **Values are whole sentences, not values.** The state for
   "55 000 c était le budget initial mais 52 000 c est acté alors" stores the
   entire sentence as its value. Nothing downstream can compare 52 000 with
   55 000 if the "value" is the sentence.
4. **Slots disagree between question and state.** `_query_slot("Quand le
   deuxième poste est-il ouvert ?")` returns `mode`, while the state is `date`.
   A "when" question can never match its own answer.
5. **State is attached to unrelated units.** `V4Search` gives every unit on a
   slot the *current* state for that slot, so `hiring:8` and `hiring:10` both
   carry `hiring:17`'s state and evidence. The answer then cites the right slot
   but the wrong turn.

## What this means

The architecture's retrieval and storage layers survive hand-written meetings.
The answer layer does not: it answers every question, with whichever state
happened to rank first, and cannot tell "not in the transcript" from "in the
transcript". That is the same wall as on SUM-RE, now visible on clean,
purpose-built data — so it is the pipeline, not the corpus.

## Next fixes, in dependency order

1. **Extract values, not sentences** (step 3/5). `52 000`, `5 avril`,
   `1er septembre`, not the clause. Everything else is downstream of this.
2. **One slot vocabulary for questions and states.** "when" → `date`,
   "how much / price / rent / salary" → `amount`, "who" → `owner`. Today
   `mode` and `date` are different words for the same question.
3. **Refuse unless the state answers the question**: slot matches, and the
   value appears in that same turn's evidence. Not in some other turn's.
4. **Attach state only to its owner unit**, not to every unit on the slot.
