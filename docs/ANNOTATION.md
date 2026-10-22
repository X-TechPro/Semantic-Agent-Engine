# ANNOTATION — the label schema, and how gold was produced

The brief requires a documented annotation schema and a justification of label
quality ("concevoir un schéma d'annotation, produire ou adapter des exemples et
justifier leur qualité"). This file is that document.

The short version: **labels are not hand-written and not model-predicted — they are
authored by the generator and looked up afterwards.** A unit's gold label is derived
from the ledger event that produced its sentence, so it cannot drift from the text,
and no annotator's judgement is buried inside what is presented as ground truth.

---

## 1. Label vocabularies (closed sets)

Defined once in `engine/schemas.py`; every schema, test and report reads them there.

| Field | Values |
|---|---|
| `source_type` | `meeting`, `document`, `code`, `note` |
| `kind` (speech act) | `info`, `query` |
| `type` | `decision`, `action`, `commitment`, `question`, `answer`, `fact`, `risk`, `idea`, `other` |
| `status` | `proposed`, `hypothesis`, `conditional`, `confirmed`, `cancelled`, `mention` |
| `intent` | `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event`, `clarify`, `none` |
| `relation` | `supports`, `contradicts`, `supersedes`, `creates_action`, `answers`, `unrelated` |

Two vocabularies are load-bearing in ways that are easy to get wrong:

* **`kind` is not `type`.** A question asked *inside a meeting* is `kind=info,
  type=question`; only a user query is `kind=query`. Conflating them made the
  `answers` relation unreachable in practice (7.3), because the linker keyed on
  `kind == "query"` and no meeting unit ever had it.
* **`repeated` is not a decision.** Its gold relation is `supports`
  (`Ledger.gold_relation`); labelling it `decision` would hand the linker a
  superseding speech act for something the generator says merely restates the
  in-force value.

## 2. Gold by construction, then by lookup

```
ledger event (kind, subject_key, value, on, person, notes, supersedes)
        │  gen/render.py writes a French sentence per event, and records
        │  Unit.gold_event_ids = [event_id] on the unit that carries it
        ▼
unit ──► reader_gold(event) ──► (kind, type, status)     ← the ONLY such table
```

`READER_GOLD` in `gen/ledger.py` maps all 13 event kinds to a label triple. Three
rules worth stating because each was a bug found the hard way:

| Event kind | → kind / type / status | Why it is this way |
|---|---|---|
| `proposed` + `notes="conditional"` | `info / idea / conditional` | a conditional is its own status, not a plain proposal — scoring it `proposed` hid the class the reader must learn (measured: conditional F1 0.00) |
| `changed`, `reinstated`, `assignment` | `info / decision / confirmed` | all three assert a state of the world |
| `question` | `info / question / mention` | asked, not decided |
| `repeated` | `info / fact / confirmed` | restates; never supersedes |

**`reader_gold()` is the single source of truth.** `gen.render.reader_status`,
`reader_type` and `engine.read.dataset` all delegate to it, and
`_STATUS_OF` is *derived* from it rather than re-listed. This matters: the table
existed in two copies before, they disagreed about the `conditional` refinement, and
9 dev events were trained as `proposed` while scored as `conditional` — which
inverted a reader verdict (DECISIONS, and the project memory
"reader labels are guard inputs"). Reader labels are **inputs to the supersede
guards**, so a wrong label is not a cosmetic error — it changes which edges the
system is allowed to write.

`reader_gold` accepts an `Event` or a kind string. Passing the string is allowed but
loses the `conditional` refinement, so callers holding an `Event` should pass it.

## 3. Relation labels

`Ledger.gold_relation(new_id, old_id)` labels an **ordered** pair, and the ordering
is the same one the store uses for edges (`from_id = newer, to_id = older`).

Derivation order, and the reason for each branch:

1. different project → `unrelated` (6.8.3's cross-project rule);
2. `answer` ↔ its `answer_of` question → `answers`;
3. `commitment` ↔ its `commit_ref` → `creates_action`;
4. **a `question` on either side → `unrelated`** — a question asserts nothing, so it
   can neither support nor contradict. Added after measuring 75 of 154 dev edges
   labelled `contradicts` purely because a question's `value` field (the subject
   label) differs from a real value. Without this branch the reference label is
   wrong in the direction that makes a working linker look broken (D-021);
5. a `supersedes` pointer in either direction → `supersedes` (the pointer wins, even
   when the surface value is unchanged, which is what cancel/reinstate needs);
6. `repeated`, or same slot + same value + same kind → `supports`;
7. same slot, assertive, different value → `contradicts`.

## 4. Hard negatives (6.4) — minimal pairs, cue in the positive

`gen/hard_negs.py` builds pairs that differ in exactly one thing, with the flip
drawn from `["status", "status", "person", "date", "number", "attribute"]` — status
weighted 2× because it is the distinction the thesis depends on.

| flip | cue | positive | negative |
|---|---|---|---|
| `proposed` | `proposé` | "…est proposé." | "…est validé." |
| `hypothesis` | `peut-être` | "On pourrait peut-être revoir X." | "X est acté." |
| `conditional` | `sous réserve` | "D'accord pour X sous réserve du budget." | "X est acté, sans condition." |
| `negation` | `pas encore` | "Ce n'est pas encore validé pour X." | "X est validé." |
| `person` | `Marie` | "C'est Marie qui gère X." | "C'est Karim qui gère X." |
| `date` | `décembre` | "X est prévu en décembre." | "Finalement X passe en janvier." |
| `number` | `1,5 M€` | "X est de 1,5 M€." | "X est de 15 M€." |

Two rules that were learned from a broken instrument, not designed well first:

* **The cue must live in the POSITIVE.** For the negation pair the negated sentence
  is the positive, because the anchor asks about the thing not yet validated.
* **A pair whose anchor cannot distinguish the two texts is dropped, not scored.**
  Such a pair pins pairwise accuracy at chance and makes the whole 6.4 number
  unauditable — and a positive result from a broken instrument is worse than a
  negative one. `n_scoreable`, `ties` and `min_n_per_flip` are reported beside the
  accuracy for this reason; cells with n < 12 are labelled descriptive.

## 5. Questions and their gold evidence

`gen/qa_from_ledger.py` emits `current_decision`, `history`,
`proposal_vs_decision`, `owner`, `date`, `unanswerable`, `compare`, `build_dossier`.
Gold is **never a stored string**: each question carries the evidence unit ids it was
generated from, so a change in rendering invalidates the gold rather than silently
stranding it. `unanswerable` items are built from a *negative* search over the
corpus — a question is only admitted as unanswerable when no unit could support it.

## 6. Splits and leakage

`gen/splice.py` splits **by project** (60/20/20), never by sentence, with `Atlas`
forced into `test` so no tuning decision can be made on the demo project. The
manifest at `data/splits/ledger.json` records its own `leakage` block, and
`tests/unit/test_split_leakage.py` pins both halves of the truth:

* project ids are disjoint across splits — asserted, and the manifest's
  `leakage.projects` must be empty;
* **person names are not disjoint** — 16 names appear in more than one split at 20
  projects. The test asserts that `leakage.persons` is *populated*, so if someone
  later "fixes" the generator to disjoint people the test fails loudly and the
  change becomes a decision with a re-measurement attached. The brief asks for
  unseen people only "si possible"; this generator does not deliver it, and no claim
  here says otherwise.

## 7. What is *not* annotated, and why that is a limit

Every label above is authored by the same generator that wrote the sentences. That
makes the corpus internally consistent and fully auditable, and it also means
**label quality is a property of my templates, not of independent human judgement**.
The brief anticipates this: synthetic conversations are allowed but "must be
complemented by an evaluation that avoids measuring only the model's ability to
recognise formulations generated by the same patterns". The complement is partially
present (a held-out general-French pair probe) and partially missing (the MTEB-French
gate is unmeasured, and no public meeting corpus has been labelled by hand).
`docs/LIMITS.md` §1 states what that does and does not license.
