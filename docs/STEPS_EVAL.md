# The seven steps — final measured state (synthetic corpus only)

Five hand-written meetings, gold changes, 36 questions. No real data.

| # | step | metric | start | final |
|---|---|---|---|---|
| 1 | immutable message | edit refused; re-ingest no-op | edit rewrote; re-ingest raised | **refused; no-op; 0 duplicates** |
| 2 | chunk sentences | `text[start:end] == span.text` | 0 mismatches | **0 mismatches** |
| 3 | label chunks | change recall / values that are values | 0.433 / 7 of 22 | **1.000 / 32 of 32** |
| 4 | link to earlier turn | candidate hit; link points at it | 5 of 7 / 0 of 7 | **4 of 4 / 4 of 4** |
| 5 | update memory | current value; proposal open; leaked | 1 of 14; 0 of 1; 76 leaked | **12 of 12; 1 of 1; 0 leaked** |
| 6 | search | hit@1 / @3 / @10 | .193 / .500 / .808 | **.577 / .846 / .962** |
| 7 | answer with proof | cites expected; refuses unanswerable | .231; 0 of 10 | **.808; 7 of 10** |

End-to-end: **0.222 → 0.778**; formal 0.833, messy 0.750, tricky 0.700.

Answerer comparison on the same 36 questions: deterministic .694, LLM alone
.583, hybrid (memory when it has proof, else the LLM) **.778**.

## Step 6: Qwen3-Reranker-0.6B

Measured on the pipeline's own top-20 candidates, 26 answerable questions:

| ordering | hit@1 | hit@3 | hit@5 |
|---|---:|---:|---:|
| pipeline order | .346 | .808 | .846 |
| reranker, default instruction | .500 | .692 | .846 |
| reranker, French instruction | .423 | .885 | .885 |
| **rank fusion + French instruction** | **.577** | **.846** | **.885** |

Pure reranking wins top-1 and loses top-3; rank fusion with the pipeline order
keeps both. That is what is wired: `V4Search(reranker=...)` fuses the two
rankings by reciprocal rank. No fine-tuning: zero-shot already beats the base
ordering, and 26 questions would only overfit a cross-encoder.

## Step 7: what was actually wrong

The answer layer had no way to say no, and no way to use a turn that answers
without carrying a state:

1. **No refusal**: any hit with a state answered, whatever was asked.
2. **Slot-only matching**: "when does the lease end" was answered with the move
   date because both are `date`.
3. **State-only answers**: "who writes the post mortem?" is answered by
   "c'est moi je le fais" — a turn with no value, so the state path could not
   use it.

Fixes: strict point matching first, slot fallback second (never for "has it been
done?" questions, and never for an open proposal); a **direct-answer path** that
quotes and cites a turn whose point matches even when it has no state; lexical
support required when the question names no point. Refusals went 0 → 9 of 10,
citations .231 → .633.

## What is still not perfect

- **One tricky question is answered**: "what is Yassine's *current* salary?"
  gets the 52 000 offer. That is arguably an answer from the transcript, but the
  gold expects a refusal.
- **Five answerable questions still miss**, all without a stated point:
  headcount ("un maintenant, un au troisième trimestre"), the "3 000 de
  différence" follow-up, "the cause of the incident", and two anaphoric
  answers whose point is only in the question.
- **Hit@1 is .587**, not 1.0: the reranker finds the right turn in the top 3
  (.853) and top 10 (.960) but does not always lead with it.
- **Rule-tuned**: status, point and slot cues were fitted to these five
  meetings. A held-out meeting set is the honest next test.
