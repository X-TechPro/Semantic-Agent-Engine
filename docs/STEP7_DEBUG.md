# Step 7, layer by layer

The answer path has six internal layers. Each was instrumented (25 of 26
answerable questions have the right turn in the top 10, so every loss below is
inside step 7, not retrieval).

| layer | what it does | measured |
|---|---|---|
| 1 parse | question → slot + point | "combien de temps" was `budget`, "ouvert" was `mode` |
| 2 retrieve | is the expected turn in the top 10 | 25 of 26 |
| 3 state | does that turn carry a state | **18 of 26** — facts and answers carry none |
| 4 match | slot, point, value, support, completion | strict 7, slot-fallback 11; 4 rejected on value, 2 on slot |
| 5 select | which matching turn is chosen | a fallback state was beating an exact-point fact |
| 6 direct | quote a turn with no state | was quoting **questions**; verifier rejected its own date prefix |
| 7 answer | cite, verify, refuse | citations **.747**, refusals **9 of 10** |

## The seven bugs found inside step 7

1. **Question slot mis-parse.** `combien de temps` mapped to `budget`, so a
   duration question could never match the duration state; `ouvert` mapped to
   `mode` before `date`. Fixed in `QUERY_SLOTS`.
2. **Value searched only in the evidence span.** A state's value often sits in
   the unit outside the cue clause; four correct answers were rejected. The
   check now reads the unit text too.
3. **Fallback-state ordering.** A state matching only the slot was chosen
   before a fact that matched the exact point. New order: strict state →
   exact-point direct answer → slot fallback.
4. **Facts and answers carry no state.** `current_for_unit` only returns live
   states, so "c'est moi je le fais" and "le 30 septembre" were invisible.
   Added the direct-answer path that quotes and cites them.
5. **Direct answers quoted questions.** ASR has no punctuation, so "et le
   loyer" and "qui s'occupe du kit presse" read as `fact` and got quoted as
   answers. Question-shaped turns are now excluded.
6. **The verifier rejected the metadata.** The rendered quote adds a date
   ("D'après Claire (2026-11-09)"), and `verify` checks every number in the
   text against the evidence — so every direct answer failed. It now verifies
   the quote, not the prefix.
7. **Open states answered completion questions.** "L'acompte a-t-il été versé ?"
   was answered with the deposit due. Completion questions now require a
   completed turn, in both the state and the direct path.

## Final numbers

| step | start | final |
|---|---|---|
| 1 store | edit rewrote; re-ingest raised | refused; no-op; 0 duplicates |
| 2 chunk | 0 mismatches | 0 |
| 3 label | .433 recall; 7/22 values | **1.00; 32/32** |
| 4 link | 5/7 candidates; 0/7 linked | **4/4; 4/4** |
| 5 memory | 1/14; 76 leaked | **12/12; 1/1 open; 0 leaked** |
| 6 search | .193 / .500 / .808 | **.587 / .853 / .960** |
| 7 answer | .231; 0/10 refused | **.731; 9/10 refused** |

End-to-end: **0.222 → 0.694** (formal .611, messy .625, tricky .900).

## The ten answerable misses that remain

All are the same shape: the expected turn is retrieved and correctly ranked, but
the question's point is not stated by any turn.

- `hiring-q2` — "when does Yassine start?" → the answering turn is a conditional
  ("démarre le 1er avril **si on valide**"); no current state, and no lexical
  overlap for the direct path.
- `hiring-q3` — "how many hires?" → a headcount with no value (`un maintenant,
  un au troisième trimestre`); the only amount present is 52 000.
- `hiring-q5` — "what about the 3 000 difference?" → the answer is "on les garde
  pour la formation", pure anaphora.
- `incident-q2` — "the cause?" → "un certificat tls expiré" shares no word with
  the question.
- `incident-q4` — yes/no data loss → the only point-bearing state is the
  outage's duration.
- `incident-q5`, `nova-q4` — "who writes/owns X?" → "c'est moi je le fais", a
  pronoun with no object.
- `incident-q6`, `nova-q3`, `nova-q5` — threshold/feature/legal-status facts
  whose points are implicit in the previous question.

They need the answer to be resolved against the question's referent (anaphora),
not another rule. That is a model job, not a cue.

## Step 7 as the architecture defines it: retrieved messages → LLM → proof

`engine/answer/llm.py` is that step, implemented: top-10 retrieved turns are
passed to a local model with their ids; the reply must cite a provided id and
every number it states must appear in the cited turns, otherwise it is refused.
"INSUFFISANT" is an accepted outcome.

| answerer | correct | citation hit | value hit | tricky refused |
|---|---:|---:|---:|---:|
| deterministic state/template | 0.694 | .731 | .577 | **.900** |
| Qwen2.5-7B, messages only | 0.500 | .538 | .462 | .600 |
| Qwen2.5-7B + timestamps + memory state | 0.583 | .577 | .538 | .700 |
| **hybrid: memory answer when it has proof, LLM otherwise** | **0.778** | **.808** | **.769** | .700 |
| Qwen3.5-4B | 0.278 | .000 | .000 | 1.000 |

The last row is the shipped default. The LLM is given exactly what step 7 says
it gets: the question, the retrieved turns with ids and timestamps, and the
memory's current state with the unit that proves it. It decides recency and
answers; the proof guard keeps only answers whose numbers are in the cited
material. The deterministic path is used first because when memory already
knows the value with proof there is nothing to generate.

On this corpus the 7B model loses to the deterministic answer path, for three
reasons: it over-claims (it turned "we will send the offer" into "Yassine
accepted it"), it answers tricky questions instead of refusing, and the proof
guard then throws away some correct replies whose numbers came from an
uncited excerpt. The 4B model refuses everything — it never emits a citation.

The LLM path stays selectable (`--answer llm` / `--answer hybrid`); `rules` is
the default because it is better here. A larger model (14B+, or a hosted one) is
the missing variable, not the plumbing: the step is wired and measured.
