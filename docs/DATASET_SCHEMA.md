# Dataset files — input and questions

Two files, two jobs. The **meeting** is dumb data: four fields per message and
nothing else. Tags, units, links and state are what the architecture produces
and stores; they must never appear in the input. The **questions** file is what
the pipeline is scored on, and it cites message ids, not text.

Validate both with:

```bash
PYTHONPATH=src python -m engine.tools.validate_dataset data/examples/atlas.meeting.json
PYTHONPATH=src python -m engine.tools.validate_dataset data/examples/atlas.questions.json \
    --meetings data/examples/atlas.meeting.json
```

## 1. Input: one meeting

`schemas/meeting.schema.json`. Every block is exactly `message_id`, `speaker`,
`timestamp`, `text`. `additionalProperties: false`, so a stray `units` or
`labels` key fails validation.

```json
{
  "meeting_id": "m-001",
  "messages": [
    {"message_id": "m-001:0", "speaker": "Marie",
     "timestamp": "2026-10-12T09:00:00+02:00",
     "text": "euh on pourrait partir sur un budget de 1,5 M€ pour atlas"},
    {"message_id": "m-001:1", "speaker": "Sophie",
     "timestamp": "2026-10-12T09:02:30+02:00",
     "text": "bon finalement on valide 2,4 M€ c'est acté"},
    {"message_id": "m-001:2", "speaker": "Karim",
     "timestamp": "2026-10-12T09:04:00+02:00",
     "text": "et le lancement on le décale à janvier ou pas"},
    {"message_id": "m-001:3", "speaker": "Marie",
     "timestamp": "2026-10-12T09:04:36+02:00",
     "text": "oui on décale à janvier"}
  ]
}
```

Rules:

1. **`message_id` is stable and unique** across the corpus. It is the only
   handle the questions and the memory have; renaming it breaks citations.
2. **`timestamp` is a real ISO date-time with a timezone.** Never a constant
   date — that silently kills every "is it still valid?" question.
3. **`text` is the message.** Keep the mess in it: disfluencies, no punctuation,
   ASR mistakes. Cleaning belongs to the pipeline, not the file.
4. **`meeting_id` at the top is optional metadata** for grouping; if absent the
   file stem is used.
5. **No generated fields.** If a file contains `units`, `change`, `relations` or
   `state`, the schema rejects it. Those are output.

## 2. Evaluation: questions

`schemas/questions.schema.json`. Each question carries a style, the text, and
the message ids any correct answer may cite. `tricky` questions cite nothing and
must be refused.

```json
{
  "corpus_id": "atlas-eval-v1",
  "questions": [
    {"query_id": "q-001", "style": "formal",
     "text": "Quel budget a été validé pour le projet Atlas ?",
     "expected_message_ids": ["m-001:1"], "expected_value": "2,4 M€",
     "must_abstain": false},
    {"query_id": "q-002", "style": "messy",
     "text": "et du coup le budget on part sur combien ?",
     "expected_message_ids": ["m-001:1"], "expected_value": "2,4 M€",
     "must_abstain": false},
    {"query_id": "q-004", "style": "tricky",
     "text": "Qui gère le budget du projet Atlas ?",
     "expected_message_ids": [], "expected_value": null, "must_abstain": true}
  ]
}
```

Rules:

1. **`expected_message_ids` is a set.** Any message stating the value is valid.
   Same semantics as the change definition: one canonical gold is not the
   target, and scoring exact identity on it made the previous evaluation
   under-specified.
2. **`tricky` + `must_abstain: true` must cite nothing.** A confident answer to
   an unanswerable question is the failure mode that matters.
3. **Styles are explicit** (`messy`, `formal`, `tricky`, `chat`) so results are
   reported per style instead of averaged. The validator warns if every question
   is formal.
4. **`expected_value` is optional** and only checks that the answer contains the
   string; the citation set is the primary score.
5. **Split by meeting, not by question**, and keep at least one style or
   generator out of training, or you measure the wording rather than the engine.

## 3. Output: what the architecture writes

Not an input file. After ingest the engine produces the memory record — units
with spans and tags, typed links, state events — described by
`schemas/memory_record.schema.json`. That is what the store keeps, and it is what
a gold file would look like *if* you later want to score steps 3–5. It is kept
separate so the input stays four fields.

## What each file feeds

| file | feeds |
|---|---|
| `meeting.messages[]` | steps 1–2 (immutable message, chunking) |
| `questions[].text` | steps 6–7 (search, answer) |
| `questions[].expected_message_ids` | step 6 recall and step 7 citation |
| `questions[].must_abstain` | step 7 refusal |
| `questions[].style` | per-style reporting (messy / formal / tricky) |
