# Blockers

> Engineering log — start with `README.md` (§5 says all of this in plain
> words); this file is the detail behind each open item.

What the build cannot currently complete, each with the measurement that shows
it. These are not TODOs — a TODO implies the work is known and available. Each
entry states what was tried, what the number was, and what would unblock it.

## B-01 — the `answers` relation is unreachable for the rules baseline

**Gate.** BUILD_GUIDE 6.8 defines six relations. `answers` is one of them.

**Measured.** On the dev split of the synthetic ledger, 19 answer events carry an
`answer_of` pointer to a question event. None is dated before its question, and
**all 19 pairs span different meetings**. System B produced **0 `answers`
edges** — under either reader backend.

**Why.** Two independent causes, both found by measurement:

1. *Channel vs speech act.* `RuleLinker.propose` gated `answers` on
   `old.kind == "query"`. But 7.3 says a question asked **inside a meeting** is
   not a user query, and `gen/ledger.py:READER_GOLD` encodes it as
   `("info", "question", "mention")`. Query units are never stored (2.1), so the
   two conditions could not both hold: no stored unit could ever be a `question`
   neighbour. **Fixed** — meeting questions are now read as `info` and indexed
   (dev corpus 63 → 74 units, 8 of them questions).
2. *The baseline cannot name an answer.* `RuleReader` has no code path returning
   `type == "answer"`; in the rules reader `answer` exists only as an **intent**
   value. So the other half of the condition (`new.type == "answer"`) is
   unsatisfiable for rules text. Cause 1 is repaired; cause 2 is the blocker.

**Rejected attempt.** A slot-matching fallback — "a value-bearing unit on a
question's own subject is that answer" — was implemented and measured: 0.8 recall
but **72 edges against 5 gold pairs (precision 0.055)**. Adding the
one-answer-per-question invariant (`IngestPipeline._one_answer_per_question`)
collapsed the over-fire to 16 edges and **lost all 4 hits (recall 0.0)**: the
first value-bearing unit on a subject is usually the decision, not the reply.
Both rule changes were reverted; the invariant itself is kept because the guide's
relation is one-to-one.

**Unblocks when.** A reader emits `answer` as a *type*. After the gold-table fix
(B-02) the trained head does produce `answers`: 14 edges, 1 of 5 gold pairs
(recall 0.20, precision 0.071) on dev, while the rules backend still writes **0**
because it has no path to `type == "answer"`. 14-for-1 is not shippable either;
the relation is produced but not identified.

## B-02 — reader gold was triplicated; fixing it reversed an earlier conclusion

**The defect.** Three literal tables mapped EventKind -> reader labels:
`ledger._STATUS_OF`, `render._TYPE_MAP/_STATUS_MAP`, and
`engine/ledger.READER_GOLD`. They agreed on type and status for all 12 common
kinds — and disagreed on the one case that mattered: `proposed` with
`notes == "conditional"`, whose refinement lived only in `render.reader_status`.
So `READER_GOLD` trained **12 conditional rows as `proposed`**, and
`eval/run_stage.py` scored a third, hand-coded `kind` gold that called a meeting
question `query` — contradicting 7.3 and scoring the correct labelling as an
error. Now one table (`READER_GOLD`, via `reader_gold(Event)`), derived views
elsewhere, delegation asserted by `tests/unit/test_reader_gold.py` (22 cases).

**Why this matters more than the fix looks.** Before it, I measured and reported
that the trained reader "wins every head gate and loses the relation layer".
That conclusion was **wrong** — an artifact of contaminated gold. Corrected:

| (test, 217 rows) | rules | trained |
|---|---|---|
| kind macro-F1 | 1.0 | 1.0 |
| type macro-F1 | 0.6609 | **0.8733** |
| status macro-F1 | 0.8745 | **0.9255** |
| intent macro-F1 | 0.8827 | 1.0 (3 gold classes) |

| dev, corpus-matched | rules | trained |
|---|---|---|
| supersedes R / P / F1 | 0.714 / 0.517 / 0.600 | 0.714 / **0.652** / **0.682** |
| questions stored | 8 | 9 |
| Δ temporal@1 (seed 42) | +0.1942 [0.127, 0.2614] | **+0.2362** [0.1339, 0.3766] |

| test, 3 seeds | rules | trained |
|---|---|---|
| A / B | 0.2498 / 0.5518 | 0.2596 / **0.5523** |
| Δ | **+0.3077** | +0.2953 |
| seeds significant | **3/3**, spread 0.1714 | 2/3, spread 0.1490 |

**The honest verdict.** The trained reader wins the mechanism metric (relation
F1 +0.082, precision +0.135 at equal recall) and dev, and **ties** the headline
on test (B differs by 0.0005; rules keeps 3/3 significance and a larger mean Δ).
So 0.4 — "keep the baseline if the trained version doesn't beat it" — does not
settle this: neither does it clearly win on the governing metric. `reader.backend`
therefore stays `rules` as the default **because the headline is undecided, not
because the trained reader was shown to hurt**. Deciding it needs more seeds and
projects, not a re-labelled gold table.

**Side effect worth stating.** With corrected labels the rules baseline itself
passes the 6.2.6 status gate on test (**0.8525**, n=80; previously reported
0.7962 and 0.6883 from the two contaminated paths). Part of the trained head's
original motivation was measuring the baseline against broken gold.

**Unblocks when.** A larger protocol (≥3 seeds × ≥40 projects, or bootstrap over
questions rather than over the single dev split) separates +0.3077 from +0.2953.

## B-03 — the reader is a template-matcher on unseen phrasing

**Measured.** On §12.3 flagship sentences, which are *not* ledger-rendered:

| sentence | rules | trained | gold |
|---|---|---|---|
| "Le lancement est prévu en décembre." | fact/mention | **answer/hypothesis** | idea/proposed |
| "Finalement, le lancement est reporté à janvier." | decision/confirmed | decision/**cancelled** | decision/confirmed |

`answer ∉ VALUE_BEARING_TYPES`, so G7 refuses the supersede for a *different*
reason than the rules reader's — the flagship update can still fail.

**Why.** Training rows come from `gen/render.py` templates ("la date de
lancement : proposition novembre"), so "est prévu en décembre" was never in the
distribution. 7.2 asks for ≥3 hand re-phrasings per event precisely for this;
those do not exist yet.

**Boundary.** Golden fixtures must **not** be trained on (9.5), so the fix
belongs in the generator's phrasing diversity, not in importing the test set.

## B-04 — intent coverage is 3 of 8 labels

`draft_email`, `create_task`, `schedule_event`, `clarify` and `none` have no gold
rows in the ledger corpus; the QA generator can only gold `answer`, `compare`,
`build_dossier`. The intent head is scored over the classes that exist and the
gate carries that caveat in `runs/reader/result.json → gates.intent`. No claim is
made about the other five.

## B-05 — the consistency audit still reports multi-head slots

`consistency_audit.ok` is `false` on dev: hypotheses are never superseded, so
several current units accumulate on one slot. G12 was added for the false-edge
consequence (test precision 0.9167 → 0.9565) and the audit now reports those
slots, but the partition itself is unresolved. It is a data-shape property of the
generator, not a store bug: no guard can decide between two unretracted
statements.

## B-06 — M5 concept attention is built, gated, and the gate could not decide it

Superseded on 2026-10-21 (it previously read "an empty placeholder ... nothing there
is measured", which the build below made false). Full record: DECISIONS D-020.

`src/engine/concept/` now holds `model.py`, `losses.py`, `data.py`, `train.py`,
`refine_job.py`, `corpus.py`, wired into config (`concept`), `run_offline`, the CLI
(`sae concept-train`, `sae concept-refine`), `bench --vectors refined`, four Make
targets, `eval/concept_gate.py` and report §6c. 46 tests cover it.

The measured result, both arms from one code path differing only in the vector column
(`like_for_like: true`, seeds 42/43/44, 20 projects, dev): temporal@1 **0.5092 →
0.4991 = −1.01 pts** against 6.13's +3 pt bar.

**What is actually blocking the decision, in priority order:**

1. **No split in this corpus can carry the bar.** The denominator that matters is
   **System B's own** residual headroom, because 6.13's bar is a gain *over System B*.
   On dev that is **0.69 pts** (per seed 2.08 / 0 / 0), so a +3 pt gain was
   arithmetically impossible on the deciding split, and `DROP` here would be an
   instrument result rather than evidence.

   An earlier version of this entry claimed the test split "has the headroom to settle
   it (22.1%)" -- that figure is the **unscoped** arm's (`a_scoped`) headroom, not
   System B's, so it was a denominator error. Measured against the right arm, test
   leaves **2.04 / 10.42 / 2.27 pts per seed**: only one of three seeds has room for
   the bar at all, and the 4.91 pt mean is carried entirely by that seed.

   So the blocker is not "9.5 forbids deciding on test" -- it is that **the corpus
   does not contain three points of temporal failure left for a geometry change to
   fix, on either split**. Closing it means more temporal churn per project (longer
   supersede chains), not a rule change. `concept.enabled` stays false and the report
   says `NOT DECIDABLE`, which is neither a pass nor a refutation.
2. **Two of M5's three outputs reach nothing.** `state` is now trained (a triplet
   objective; it previously received no gradient at all) but the linker does not read
   it, and `validity` is unused by retrieval, which weights on stored edge status
   instead. A head with no consumer is an untested claim, so 6.13 is not *fully*
   implemented even once the number exists. See also B-09 and B-10.
3. **The general-French bound (9.7: ≤1 pt regression) is unmeasured** for refined
   vectors, because the MTEB gate itself is unmeasured (B-07). Refined vectors were
   fitted on ledger French only.

What is *not* in doubt: the anchoring machinery works. The refine job **refused to
publish** an unanchored column (measured cosine-to-base 0.0105 at init, 0.5081 after
soft-penalised training), and the fitted model now ships anchored at 0.9812 with drift
p95 0.2009 inside the 0.25 bound. A model that re-embedded rather than refined could
not have reached retrieval at all.

**External check, later the same day (D-022).** The pooled MIRACL-fr run gave
concept attention the decidable test this corpus could not: real French questions,
a human-judged hard-negative pool, 343/343 scoreable, and an anchored refine on the
real store (cosine-to-base 0.9853, drift p95 0.1763, all 3,407 rows). It **lost**
-- MRR@10 0.6965 → 0.6774 dense, 0.6046 → 0.5898 hybrid (~1.5–1.9 pts). The
temporal@1 bar itself remains NOT DECIDABLE (external corpora carry no supersede
gold), but the mechanism claim 6.13 rests on -- that context-refined vectors rank
better -- is now measured against data with headroom, and it did not hold.

## B-07 — unmeasured milestones

Stated plainly so the completion percentage cannot be read as "done":

- **M3**: the embedding fine-tune **was built and run** (`embed/finetune.py` +
  `embed/hard_negatives.py`) and is a measured negative end-to-end (D-013). The
  reranker is now measured too and is the first trained component that beats the
  floor (System B 0.5518 → 0.6616, 3/3 significant on dev *and* test, D-015); it
  stays opt-in on cost. §10 rows 3 and 8/11 remain unmeasured because they need a
  real embedder, not the offline hash stand-in. **Row 9 (concept attention) is now
  measured and its answer is "not decidable on this split", not "no" — see B-06.**
- **Anti-forgetting gate (6.4): still UNMEASURED.** `eval/regression_mteb.py` exists
  and refuses to substitute a proxy for the gate: `mteb` is not installed, so it
  writes `mode: proxy / gate: unmeasured` plus the labelled proxy (held-out
  general-French pairwise accuracy 0.595 → 0.625). Installing `mteb` is the only way
  to close it.
- **Eval harnesses**: `latency`, `oracle_swap`, `e2e` (adversarial) and
  `regression_mteb` are built, run and wired into `make` + the report. Missing:
  §10 rows 1–12 as one consolidated table.
- **B-08 abstention leak**: 3 of 12 unanswerable adversarial cases are answered
  with a real-but-irrelevant quote (measured refusal rate 9/12).
- **Docs/tooling**: complete and artifact-driven — `README.md`, `docs/LIMITS.md`,
  `docs/ANNOTATION.md` and `.pre-commit-config.yaml` exist, their claims cite
  artefacts, and `ui/app.py` has been run and *looked at* (populated-store
  screenshot; the defects it exposed were fixed). **Coverage** is re-measured with
  every code change and quoted at the exact figure, never pytest's rounded one
  (`runs/cov*.log`). The persistent gap is offline-hostile modules — GPU-training or
  live-HF paths: `embed/finetune.py`, `read/train.py`, `tools/verify_models.py`,
  `tools/verify_datasets.py`, `link/train.py`, `link/nli_finetune.py` (its
  `train()` needs the GPU; head load/eval are integration-tested) — which is the
  same install-time decision as B-07's MTEB gate.
- **UI**: run and verified via screenshot (`make ui`); two defects it exposed —
  a config menu offering a file that does not exist, and metrics read from
  session state instead of the store — are fixed (D-019 trail).
- **Git**: only `IDEA.md` is committed; everything else is untracked.

## B-09 — the trained linker has never run inside the engine (2026-10-21)

`build_linker` could not construct it at all until D-019: `SupervisedLinker.__init__`
rejected two of the six keywords it was always handed, so every
`link.backend: logreg` attempt raised `TypeError` before an engine could start.

That constructor bug is fixed and pinned by a test, **but no §10 row exists for the
trained linker**, because producing one means an end-to-end A/B run (rules arm vs
logreg arm, same corpus, seeds 42/43/44) and that has never been executed. D-016's
dev F1 0.6981 / test F1 0.7328 are classifier metrics measured offline against the
gold ledger — they say the model ranks pairs better than the rules do, not that the
system answers better. Any completion percentage that counts M3/M4 as delivered must
read this line first.

## B-10 — 6.10's history half is fixed but has no surface

`graph_expand.predecessors` returned the wrong direction of a supersede chain (D-018,
now corrected and tested). It still has **no caller in `src/`, `api/` or `ui/`**, so a
user asking "what does this decision replace?" gets nothing: `expand()` surfaces the
chain head when a superseded unit is hit, and nothing walks the other way. Wiring it
into the answer payload changes what the §13 scenarios render, so it is a deliberate
piece of work with its own acceptance cases rather than a one-line call — recorded
here so the 6.10 checkbox cannot be read as complete.

## B-11 — 6.14's threshold re-calibration is not wired into the offline pass (2026-10-21)

6.14 lists six batch steps; five are implemented and reported by `run_offline`
(`audit`, `reembed`, `concept`, `rebuilt_vectors`, `clusters`). The sixth —
**"re-calibrate thresholds if model version changed"** — is not.

`ReaderModel.calibrate()` exists and works (`read/model.py:390`), but its only
caller is `read/train.py:118`, i.e. it runs at *training* time. So a store whose
reader checkpoint was replaced, or whose encoder version moved, keeps whatever
temperature and per-head thresholds the previous training run wrote. Nothing in the
nightly pass notices, and `sae offline` reports a clean job.

Why it is a blocker and not a polish item: `reader.abstain_threshold` gates
abstention, and D-014 already showed what an inert reader knob does — it silently
abstained on every question for the whole recorded suite. A stale calibration is the
same failure with a slower fuse. The fix needs a decision about where dev logits come
from at calibration time (the offline pass has no dev split loaded), so it is not a
one-liner. `README.md` §1 states this in the diagram rather than implying 6.14 is
complete.

**Resolved (D-027, same date)**: the sixth step ships as `offline.recalibrate_thresholds`,
wired into `run_offline` (reported key `recalibrate`) and reachable from
`sae offline` — a pass this consequential having no CLI surface was a second
gap found on the way. B-11's open decision — where dev logits come from — is
answered by rebuilding the reader's own dev split deterministically
(`rebuild_dev_rows`: the same ledger, seed and `build_split` the trainer used;
the store holds no gold, and a cached split hides exactly the drift the audit
hunts). What the audit measured the day it shipped, on 336 current-dev rows:
`models/reader_v2` refits to its own shipped temperatures (0.1451/0.5234/0.294
reproduced to four decimals — the instrument's positive control), while
`models/reader`, trained before casual variants existed, refits its status head
off the clamp floor it was trained at (T 0.05 → 0.3014, dev ECE 0.0398 →
0.0202) — drift visible as a number for the first time. The apply path ran live
once: `--apply-recal` wrote `models/reader_recal` (same weights, refit
temperatures — a full opt-in checkpoint, no shipped dir touched), and the
candidate audits as its own fixed point. `reader.abstain_threshold` stays a
human knob — the report ships the abstain curve (v2: 2.4% abstain at floor 0.5,
kept-accuracy 0.9665) so any future default flip has a number. Kept honest: the
curve is measured on the synthetic dev distribution; over-firing on raw French
is a subject/distribution problem temperature cannot touch (B-13 stands).

## B-12 — external data names the reader as the remaining choke, and the fixed linker head is not wired in (2026-10-21)

`make frbench` (DECISIONS D-022, artefacts `runs/frbench/`): on 2,974 real French
requests the trained reader calls **12.2%** of them queries and scores **16.2%**
intent accuracy on the mapped subset — while averaging **0.96 confidence with
0.0000 abstention**. The synthetic-dev calibration does not travel. Because reader
labels feed the guards (the reader-gold lesson), this cannot be fixed by a quiet
retrain: the guide's own MASSIVE warm-up step must ship with a synthetic
re-validation pass alongside it, or it trades an external gap for an internal one.

The mirror-image gap sat in the linker: the XNLI-fine-tuned head **does** fix text
relation discrimination externally (test macro-F1 0.7986 vs chance 0.3333, read
once) and only existed as a checkpoint plus a `frbench` arm. **The linker half of
this blocker is now closed as wired-but-gated** (D-023): `link.backend: nli`
consults the head on the text-only fall-through and nothing else; re-validation
shows supersedes P/R and guard refusals byte-identical to the rules baseline
(the head cannot move a guard input) and no ledger-decidable gain either way
(~75 of its ~79 changed edges per split sit on pairs with no gold label; macro-F1
test 0.4169 -> 0.4169). Default stays `rules` per 0.4: the gate needs a decidable
corpus. **The reader half is now closed as warmed-and-priced** (D-024): the
MASSIVE-fr warm-up (`models/reader_warm`, selectable via `make readwarm` /
`sae stage --reader warm` / `sae frbench --reader-model`) took test kind 0.1221 →
**0.9718** and in-scope intent macro-F1 0.0904 → **0.605**, with confidence finally
beside accuracy (0.96 → 0.345 against 0.291). The guard-input promise held structurally:
encoder + type/status heads frozen, drift asserted 0.0, and the stage rows are byte-identical
`model(reader)` vs `model(reader_warm)`. The recorded price: out-of-scope rejection fell
0.9448 → **0.2863** (over-claims `compare`/`build_dossier` on French it should abstain on)
and ledger intent 1.0 → 0.33. Making that safe to ship is now B-11's job -- a re-fitted
`abstain_threshold` on a checkpoint change is the mechanism the over-claims need before any
default flip.

## B-13 — the temporal layer's external test exists, was read once, and the shipped system cannot reach it (2026-10-21)

The mandate's last uncovered claim: the headline is the *relation share* of the
synthetic delta, and nothing external carried supersede gold. That gap is now
closed in the measuring sense and open in the winning sense. `make frdata` +
`python -m engine.tools.summre_extract` pull SUMM-RE **test**-split meetings by
remote parquet range-read (audio never downloaded; ~1.3 s/shard), and
`data/gold/summre_temporal_test.json` holds hand-labelled threads: **1,002
sentences reviewed, 65 relations, 28 supersedes pairs**, every index
machine-validated, labels frozen before any score existed, one read guarded in
code. The read happened (`runs/frbench/summre_intfloat-multilingual-e5-base-768d_test_full`,
A/B arms sharing one e5, `sae frbench --component summre`, deliberately outside
the `all` sweep):

- **mechanism reach 0/28** — the rules linker marked no gold-old unit superseded
  (chain-direct 0), because real ASR speech ships no slots and the rules reader
  called **994/1019 units `mention`** (confirmed: 15); guards with no input
  cannot authorise, exactly the dependency the synthetic corpus hides;
- ranking consequence: newer-over-superseded **A 0.0 → B 0.0**; MRR(new)
  0.0312 → 0.0471 and both-in-pool 7 → 10 — and with zero edges marked, that
  small B movement is the *scoping* flag, not the relation layer: row 6, not
  row 7. The query design (the superseded claim's own text pins it at rank 0)
  makes this a ceiling test, and it says the ceiling is currently the floor;
- instrument honesty: 9 turns produced no unit (segment splits/merges), costing
  one contradicts and one supports pair from the denominators (n=7/8, 28/29 —
  counted in the artefact, not silently dropped).

**What this blocks**: any sentence of the form "the relation layer generalises to
real meetings" — it is now measured-unreached, not assumed-fine. **What it does
not block**: the synthetic headline stands as internal validity (and the
mechanism itself was proven reachable there: oracle-swap gold relations +0.057),
because this null bounds the shipped default system's *signal chain*, not the
architecture's machinery. What would move it is a reader whose statuses fire on
raw speech (B-07/B-11 territory: the trained reader's statuses on real ASR, then
B-11's re-fitted thresholds) — a re-run of this arm after such a checkpoint is a
new experiment with a new artefact tag (`summre_*` dirs are named by embedder +
backend read-backs), and this set may not be re-read: one spent read stays spent
(`eval_summre_temporal` raises if its artefact exists).

**Fix-2 outcome (D-026, same date)**: the two missing inputs were measured
separately on FRESH dev gold (the spent set was not touched; `runs/summre_reach/
reach-dev-*-gate+run.json`). Statuses now fire — reader_v2 calls 510/781 dev
units assertive where rules called 9 — so the status half of this blocker is
closed (with an over-firing caveat that B-11 owns). Subjects do not resolve:
the lexicon hits 17/781, and the cosine nearest-attribute lookup failed its
amended consistency gate at every threshold tested (same-thread agreement
0.27/0.29 vs a 0.8 bar; real speech sits at cosine 0.85-0.92 to some centroid,
so no `tau` separates), and it ships wired but OFF. End-to-end on dev: 0/10
gold-old units marked superseded in both shipped arms, 0 spurious edges store-
wide. B-13 stays open, narrowed to its remaining cause: real slot extraction
(§6.3), not statuses, not guards.

**6.3(c) also measured (D-028, same date)**: the LLM fallback extractor was
built (validated proposals, startup refusal without a live backend, self-echo
guard that rejected 47/171 real answers), instrument-audited against a second
schema and a second model, and judged by the SAME pre-registered gate: pair
coverage **0.05 vs 0.90** — the model answers 97% of the time, but with meeting
titles and project ids, not keys. It ships wired and OFF. What now blocks B-13
is a *supervised* extractor (or a model beyond this 8 GB host): the dev gold
threads are exactly its training signal, and no further unsupervised
proposals are left to try.

**6.3(d/e) supervised also measured (D-029, 2026-10-22)**: the supervised path
was built and fine-tuned — v1 sentence-independent (dev 1.0 on its own task,
external-gate pair coverage 0.30) and v2 dialogue-context-aware (template
French 0.95, anaphoric accuracy 0.0/6 — the ledger corpus carries ~4 anaphoric
training rows, because the templates always name the attribute in-sentence;
gate coverage 0.15). Both ship wired and OFF: `sae subject-train` retrains,
`--gate-sup` re-measures, the v2s arm reaches end-to-end. The blocker's cause
is now evidence-bounded: no architecture or training recipe fixes a corpus
that lacks the phenomenon — closing it needs real annotated meeting speech,
and the dev gold's 40 thread units are the gate's measurement set, so training
on them would spend it.

**6.3(f) real-corpus supervision measured too (D-030, 2026-10-22)**: it was not
left hypothetical — 1,625 real segments across six SUMM-RE meetings were
hand-labelled with subject drawers (four train, one dev, one held-out measure;
`data/gold/summre_subjects_*.json`, split declared in each file, the trainer
refuses the measure tape) and a window-attention classifier (last 4 turns +
current as ONE camembert input, 57 drawers + NULL) was trained in three modes.
None passed its dev consistency filter (best same-thread 0.5597 vs the 0.60
save bar; the external PROPOSER_GATE never reached, never moved), with the
failure shaped precisely: agreement and abstention trade against epochs, and
names shared across meetings learn while meeting-specific ones starve at
~20 rows each. The corpus and the loop remain as the asset; B-13's remaining
cause is now class support for one domain, not a missing extractor.

**Fifth meeting + tape repair (D-031, same date)**: growing the extraction to
12 shards re-completed 039b_EDDH across its old shard boundary (242→341
segments); the byte-digest check caught the drift BEFORE anything trained on a
shifted tape — all 242 old labels carried to new indices by text identity, the
99 inserted segments labelled, the measure gold now 341/341 with the reindex
documented in its own spec header. A fifth TRAINING meeting (047a_EDRH, the
tape's music-festival domain) then measured the generalisation question with
the dev set, code and gates literally unchanged: reachable dev rows rose
110→129 as hoped, same-thread agreement FELL (0.4403→0.2819 at 5 epochs;
0.5597→0.4149 at 3), nothing passed the dev filter, nothing saved, and the
external PROPOSER_GATE has still never been run for the classifier. Meetings
add classes faster than examples-per-class — the blocker's cause stands,
sharpened and now with its counter-example on file. The programme's numbers
render from artefacts in `docs/REPORT.md` §6e; `make drawer` is the one-command
loop (train → early-stop retry → external gate + end-to-end only if a
checkpoint exists → report).

**Supervised line closed on its declared condition (D-032, same date)**: the
pre-flight audit found and fixed the last instrument flaw — the dev filter ran
on an off-domain meeting (2/9 tape names) — and the rebuild (dev = 047c at
6/9, train = 7 meetings, 1,879 rows, 70 drawers) measured the answer under
unchanged gates: same-thread 0.2285/0.2453 against the 0.60 filter, separation
0.9247/0.9467 (above the external bar — topics ARE distinguishable; agreement
at ~27 examples/drawer is what cannot be bought). No checkpoint, external gate
never reached, end-to-end arm stays unexercised-live by design. B-13 remains
open as a DATA-SCALE blocker with every alternative excluded by measurement:
lexicon 2%, cosine 27%, LLM 5%, span-heads 0.30/0.15, classifier agreement
0.23. Not architecture, not method, not this machine — annotated meeting
speech at a scale the repo has grown to 2,673 segments of and still needs
more of.

**7th proposal excluded (D-034, same date)**: live-graph projection (the tag
as "which existing thread does this sentence land in", centroid of each
thread's REAL sentences, online join/seed/tie) was swept on dev across
windows {0,4} × floors 0.30–0.95 and cleared no bar — it either absorbs the
whole meeting into one drawer (separation 0.0) or fragments it (agreement
0.002); windowing moves the transition, never widens it, because e5 sits at
0.9+ between domain-adjacent French (`runs/graph_proj/gate-graph-*.json`,
measure tape never touched). The geometry itself is the floor. B-13 stays
open with its single named supply: annotated real conversations at scale.

**The data supply answered once — partially (D-035, same date)**: three more
same-domain meetings labelled (12 total, 3,237 train rows / 72 drawers) and
retrained per protocol; refused again (dev same-thread 0.19/0.26 vs the 0.60
filter). The per-class audit sharpens the blocker from "more data" to
"MATCHED data": topics with 200+ examples file well (lieu 0.64), topics the
label passes happened not to discuss stay at zero (hebergement, international
have NO examples — four event meetings covered four kinds of event, not
lodging or security). Volume alone does not move the aggregate; topic mix
does. B-13's remaining supply is now specified: labelled meetings on the
under-covered topics, or a merged-drawer pass; `make drawer` re-tests either.

**The merged-drawer lever measured too (D-036, same date)**: 72 fine drawers
→ 16 families through one map applied identically to every split (specs remain
the source of truth; `fine_keys` preserved in each gold file). It closed the
tape's zero-support gaps (lodging now has 12 examples, foreign guests rides
with programming), and the thick families filed at the program's best real-
French per-topic numbers (venue 0.91–1.0, programming 0.85) — but the dev
filter still refused (agreement 0.29–0.32 vs 0.60) because the practice
meeting's own heavyweights (activities, security) remain thin: the label
passes never discussed them either. Supportive, not sufficient: the reopen is
now a conjunction — keep the merge, add matched-topic meetings toward ~100+
examples per family the tape uses. Tape still unburned, shipped default still
`rules`.

**The extractor code was removed (D-037, same date)**: judged against both
requirement sources — the brief needs no real-audio topic extraction, and a
component that never crosses its gate changes no shipped behaviour — the six
measured extractors were useless, so their modules, trainers, CLI commands,
reach gates, `make drawer` loop and refused checkpoints left the repository.
Resolvers are `rules | gold`; a removed name fails at startup naming D-037
(tested). What the blocker's evidence still stands on stays: every `runs/`
verdict artefact, the labelled gold (the real-data demos read their subject
keys from it — annotation supplied, extraction refused), and the shared
validator in `slots/subjects.py`. B-13 remains open as exactly what it
always was — a data-scale demand for real annotated meetings — now with no
dead code pretending otherwise.

**D-046 (2026-10-24) put a number on the remaining cause and refused its fix.**
Production subject resolution puts only **2/40** gold supersede pairs in front
of the guards (train+sel; 0/11 on the sealed holdout as measured by D-039), so
reach is the cap. A deterministic coarse-thread index + spoken-French value
witnesses (`link/thread.py`, `link/values.py`, `evaluate(..., same_thread=True)`)
raises reach to **9/40** — and the drawer upper bound is **39/40**, proving the
judge is fed 97.5% of the right olds if threads were perfect. Judgement stayed
at 1 true positive, and the XNLI head is at chance on the same pairs both bare
(AUC 0.5727) and with the thread prefix as premise (AUC 0.5411). The round is
refused and `link.thread_context` ships False; B-13's remaining demand is now
specifically *labelled threads plus a judge trained on (thread, new turn)*.

**D-047 moved the failure to the answer surface.** Since the relation cannot be
recovered here, the engine no longer presents a `mention` as the version in
force: it caveats the answer (or refuses when two live versions are visibly in
conflict). On train+sel that took "stale version shown confidently" from 40/40
to **0/40** while the synthetic demo corpus stays uncaveated (19/19). The cost is
stated: 87.9% of control segments on raw speech are caveated, because that is how
often the reader cannot ground currency. What B-13 still needs is unchanged.

**D-061 (2026-10-26) fixed the negation leak, not reach.** Clause-scoped
polarity plus `G13_polarity` stops a negated value from being read as a support
or replacement, and `G2_turn_order` stops a later turn from being superseded by
an earlier one. On the 40 train+sel gold, pair_local went 6 predictions / 1 TP
to **5 / 1** (precision 0.167 → 0.200, recall unchanged 0.025): one false
positive removed, no new true positive. Shipped reach stays 2/40; B-13's remaining demand is still labelled threads plus a judge trained
on `(thread, new turn)`.

**D-062 (2026-10-26) tested the change-first alternative and refused it.** The
new-turn definition is now `docs/CHANGE_DEFINITION.md`, but the chain does not
enforce it: cue coverage is 7/40 narrow and 22/40 broad with 6.7% precision; a
turn-only TF-IDF classifier has top-5 1/40; the 5-candidate ceiling is 21/40;
and the 7B accepts 35/40 gold new turns but also 25/40 sampled non-gold turns
(exact-pair P 0.165 / R 0.525). The next step is exhaustive binary change
labels, not another pair judge.

## v4 (2026-10-27)

- **v4 is not the default yet.** The append-only store, sentence spans, entity/
  event indexes, query interpreter and FlauBERT XNLI/STS linker exist; v3 still
  runs the shipped path until the real-data gates pass.
- **FlauBERT XNLI/STS is zero-shot on relations.** On 40 real SUMM-RE pairs it
  returns the old antecedent at rank 1 in **2/40**, rank ≤5 in **12/40**; it has
  not been trained on `supersedes`/status labels.
- **8/40 real pairs have no candidate.** These are long-range/weak-entity pairs;
  adding another retrieval source has not moved them.
- **The real labels are small.** 40 pairs, six meetings; the v4 reranker must
  not be selected on the same pairs it reports.
- **No LLM writes memory.** Answers are evidence-grounded; the verifier and
  abstention path remain mandatory.
