# Linker failure ledger

Snapshot: 2026-10-26. Scope: the relation/linker path on real French meeting
speech (`supersedes`, reach, pair judgement, threading, candidate selection).
Reader/retrieval failures are included only where they bound the linker.

## Bottom line

1. **Reach is the first wall.** The rules reader resolves a subject for ~2% of
   real ASR units (`17/781`), and calls `768/781` units `mention`. The guards
   refuse at G1 before any model is asked. Production `same_slot` reach is
   **2/40**; deterministic topic threads raise it to **22/40**; a perfect
   drawer threader is **39/40**. See `docs/BLOCKERS.md:493-502`,
   `docs/DECISIONS.md:1940-1959`, `docs/DECISIONS.md:2225-2249`.
   Reach is round-specific: **2/40 shipped**, **9/40 coarse thread** (D-046),
   **22/40 persistent topic** (D-052), **39/40 perfect drawers**. The
   2026-10-26 G13 rerun reports 23/40 on the current gold; TP is still 1.
2. **Once reach is fixed, judgement is the wall.** With 22 reachable gold pairs
   the candidate space explodes to 74-2,814 predictions for 1-7 true positives
   (`docs/DECISIONS.md:2234-2247`). The 7B judge is a good *verifier* when the
   pair is given (P 0.727-0.778 / R 0.636-0.727) but fails at natural
   prevalence (P 0.033 / R 0.417; `docs/DECISIONS.md:2256-2284`).
3. **The label supply is the binding constraint, not the architecture.** The
   real gold is 40 supersede pairs (28 in the spent test), 53 detector rows,
   2,673 subject-labelled segments. Pair classifiers trained on 23-25 positives
   reach precision 0.0009-0.028; the candidate ranker's AP is 0.008-0.022
   against a 0.004 base rate. More meetings did not move the threader
   (`docs/DECISIONS.md:1544-1574`, `:2292-2320`, `:1818-1853`).
4. **The release path that survives is v3 + review queue, not automatic
   supersede.** Nodes group 25/40 gold pairs, `at_turn` retraces 40/40, zero
   mutations; the detector works on value-rich 024c (P 1.000 / R 0.333) but not
   on discourse-heavy 047a without tuning (P 0.818 / R 0.571 after labels).
   `docs/DECISIONS.md:2470-2534`.

## 0. The implementations — the “6 or more” list

There are **three selectable `Linker` classes** in `build_linker`
(`src/engine/link/supervised.py:131`), plus candidate/thread layers and
measured pair judges that were never exposed as a backend. The failure table
below is larger than the backend list because the experiments lived in
`src/engine/tools/`, `src/engine/eval/`, and the ignored `runs/llm_judge/`
scripts.

### 0.1 Selectable backends (`link.backend`)

| # | implementation | code | mechanism | state |
|---|---|---|---|---|
| L1 | `RuleLinker` | `src/engine/link/rules.py:185` | lexical/slot proposal, then the deterministic guards | **shipped default** |
| L2 | `SupervisedLinker` (`logreg`) | `src/engine/link/supervised.py:26` | `PairClassifier` 21 features (`link/model.py:120`), trained by `link/train.py` or `tools/summre_linker.py` | wired; model optional, default remains rules |
| L3 | `NLILinker` (`nli`) | `src/engine/link/supervised.py:67` | XNLI-fine-tuned moderncamembert text head (`link/nli_finetune.py`) | wired; head at `models/linker_nli`; default remains rules |

### 0.2 Candidate/thread layers (not separate `build_linker` backends)

| # | implementation | code | role | state |
|---|---|---|---|---|
| L4 | `ThreadIndex` + value witnesses | `src/engine/link/thread.py:79`, `src/engine/link/values.py` | deterministic coarse thread, then `evaluate(..., same_thread=True)` | `link.thread_context=False`; reach 22/40, judge 1 TP |
| L5 | Session-first batch | `src/engine/ingest.py:343` | links the whole session before committing; uses L1 | `link.session_batch=False`; 19 edges / 1 TP |

### 0.3 Measured implementations that were never a shipped backend

| # | implementation | code | what it does | headline result |
|---|---|---|---|---|
| L6 | Real-pair logreg | `src/engine/tools/summre_linker.py` | same 21-feature classifier, fitted on real pairs | recall .4545, precision .0009; refused |
| L7 | e5 pair cosine | `src/engine/tools/summre_paircos.py` | sentence-embedding dot product | AUC .6329; refused |
| L8 | Cross-encoder topic tune | `src/engine/tools/finetune_reranker.py` | LoRA on the shipped reranker for same-topic pairs | AUC .6547 vs .80; refused |
| L9 | Graph-conditioned thread assigner | `src/engine/tools/finetune_threader.py` | sentence vs established thread history | top-1 .5038 vs .55; pair AUC .8169; refused |
| L10 | Whole-session 2B LLM | `src/engine/tools/summre_llmlink.py` | qwen3.5:2b proposes typed relations by segment index | 219 proposals, 0/40; refused |
| L11 | 7B verifier | `runs/llm_judge/llm_judge_v2.py` and siblings | verifies a given pair, then filters | P .727-.778 / R .636-.727 given; P .033 / R .417 at prevalence |
| L12 | Token attention | `src/engine/eval/token_attention.py`, `_links.py`, `_transfer.py` | attention/alignment signals between turns | sel AUC .564 vs .70; no head |
| L13 | Candidate ranker | `runs/llm_judge/train_candidate_ranker.py` | shortlist the gold old turn from 5,791 pairs | AP .014-.022 with 23 positives; refused |
| L14 | Review queue / detector | `src/engine/tools/session_changes.py` | groups 7B survivors by new turn and point, keeps latest old | 024c P 1.0 / R .333; 047a P .818 / R .571 after tuning |
| L15 | Decision-node view + detector | `src/engine/threads/`, `src/engine/eval/decision_nodes.py`, `src/engine/tools/session_nodes.py` | immutable node view; latest detected change becomes current | 25/40 pairs, `at_turn` 40/40, 0 mutations |

### 0.4 The six removed subject/candidate-key extractors

These were not `Linker` classes; they were the implementations that supplied
`slots.subject`, without which G1 refuses the pair. D-037 removed their code
after every one failed its gate (`docs/DECISIONS.md:1443-1467`):

| # | extractor | result | fate |
|---|---|---|---|
| E1 | rules lexicon | 17/781 keys; the only shipped resolver | **kept** |
| E2 | cosine nearest-attribute lookup | agreement .273/.286 vs .80 | removed |
| E3 | LLM fallback (`llm_subject.py`) | coverage .05 vs .90 | removed |
| E4 | span head v1 (`subject_model.py`) | external coverage .30 | removed |
| E5 | span head v2, dialogue context | anaphoric 0/6; gate .15 | removed |
| E6 | window-attention drawer classifier (`drawer_model.py`) | .5597 vs .60; no checkpoint | removed |
| E7 | graph projector (`graph_projector.py`) | separation 0.0 or agreement .002 | removed |

So: **3 selectable linkers + 2 thread/session layers + 10 measured judges +
7 subject extractors** (six removed, one retained). That is the “6 or more” —
the code was removed or kept out of `build_linker`, which is why the current
module list only shows three classes.

## 1. What the metrics mean

| metric | what it measures | what it does **not** measure |
|---|---|---|
| `reach` | gold-old units that get past G1 to the guards | whether the judge picks the right old |
| `pair coverage` | gold-thread units that receive a subject key | whether the key is correct |
| `same-thread agreement` | consistency of a drawer/thread classifier | correctness of the relation |
| `cross-thread separation` | how often different threads stay apart | supersede judgement |
| `null accuracy` | ability to abstain on empty/transition turns | agreement |
| `AUC` | ranking of positives above negatives | operating precision at natural prevalence |
| `AP` | precision over the whole ranking | top-k review quality |
| `precision/recall` | thresholded predictions vs gold | whether the gold definition is the product's |
| `temporal@1` | the in-force unit is rank 1 | linker recall by itself; retrieval always had the gold in pool (`gold_in_pool=1.0`) |
| exact-pair hit | the one canonical old turn was cited | whether the change was found at the new turn |
| new-turn hit | any proposed old for a gold new turn | exact citation |
| node pair co-membership | gold old+new in one node | which child is current |
| `at_turn` | replaying the node at the gold new turn | automatic detection |

The recurring measurement error was comparing a **balanced/offline** number to
a **natural-prevalence/end-to-end** number. D-056 is the explicit correction:
exact-pair scoring gave 2/12 on 024c while new-turn scoring gave 7/12 for the
same rows (`docs/DECISIONS.md:2362-2385`).

### 1.1 Gate protocol actually used

| round | split discipline | bar(s) declared before the run | outcome |
|---|---|---|---|
| subject extractors (D-026/D-028/D-029) | dev gold, spent test refused | coverage 0.90; agreement 0.80 / separation 0.90 | all fail |
| drawer classifier (D-030/D-032) | train/dev/measure declared per file; measure tape unread | 0.60 same-thread, 0.60 cross, 0.60 null | all fail |
| pair classifier (D-039) | train 5 / sel 1 / sealed 3, single read | B1 recall ≥.30, B2 precision ≥.70, B3 F1 > rules | B2 fail; B3 meaningless |
| topic identity (D-042/D-043) | sel, sealed holdout unread | control+0.10 and absolute floor .80 / .55 | fail |
| LLM judge (D-045) | six temporal-gold meetings, 40 pairs | pooled P ≥.70, R ≥.30 | 0/40 |
| pair-local thread (D-046) | train+sel 40; sealed unread | L1 reach ≥.50, L2 P ≥.70, L3 R ≥.30 | L1 fail then; L2/L3 fail |
| token attention (D-048) | train selection / sel scoring / sealed unread | T1 sel AUC ≥.70; T2 .75 if T1 | T1 .564 |
| 7B verifier (D-053) | 22 reachable gold + 40 background; then 024c natural prevalence | P ≥.70, R ≥.30 | passes given pair; fails prevalence |
| hybrid detector (D-059) | 2 labelled meetings; 4 unlabelled | P ≥.70, R ≥.30 | passes 2; 2/21 elsewhere |
| node view (D-060) | all six meetings | coverage, pair reach, at_turn, mutation | coverage .995; pair .625; at_turn 1.0; 0 mutations |

The consistent rules: labels frozen before scoring; one sealed read only;
spent test never re-read; train/dev/test separated by meeting; every refused
model must not exist on disk; config hash recorded; results rendered from
artefacts, not typed. The failures below are therefore mostly **capability
nulls**, not unmeasured claims.

## 2. Failure ledger

### 2.1 Input and candidate reach

| attempt | measured | gate / bar | outcome | cause | source |
|---|---|---|---|---|---|
| Rules linker, real ASR | 0/28 gold olds superseded; 994/1019 units `mention`; 0 store edges | mechanism reach | FAIL | no slots, no statuses; guards have no input | `docs/BLOCKERS.md:341-352` |
| Rules linker, train+sel | 2/40 `same_slot`; 0/11 sealed holdout | reach ≥ 0.50 | FAIL | subject resolver only 17/781 | `docs/DECISIONS.md:1940-1945` |
| Cosine nearest-attribute lookup | same-thread agreement 0.273/0.286; separation 0.854/0.849 | 0.80 agreement / 0.90 separation | FAIL | real sentences sit at 0.85-0.92 cosine to some centroid; no tau separates | `docs/DECISIONS.md:945-960` |
| LLM subject extractor (qwen3.5:4b) | pair coverage 0.05; 166/171 non-null; 47 self-echo rejected; 2/40 | 0.90 coverage | FAIL | answers with meeting titles/project ids, not keys; 16-51 s/call | `docs/DECISIONS.md:1001-1050` |
| Span-head v1 / v2 | v1 external coverage 0.30; v2 0.15; anaphoric 0.0/6 | external gate | FAIL | only 4 anaphoric training rows; templates always name the attribute | `docs/DECISIONS.md:1053-1113` |
| Window-attention drawer classifier, 1,625 segments / 57 drawers | best same-thread 0.5597; 3e 0.5597, 5e 0.4403 | dev 0.60 save filter | FAIL | 20 rows/class; no checkpoint saved; external gate never run | `docs/BLOCKERS.md:401-412`, `docs/DECISIONS.md:1115-1145` |
| Fifth meeting + tape repair | reach 110→129; same-thread 0.4403→0.2819 (5e), 0.5597→0.4149 (3e) | dev 0.60 | FAIL | meetings add classes faster than examples/class | `docs/BLOCKERS.md:414-429` |
| Final drawer retrain, 1,879 rows / 70 drawers | same-thread 0.2285/0.2453; separation 0.9247/0.9467; ~27 examples/drawer | 0.60/0.60/0.60 | FAIL | class support, not encoder | `docs/DECISIONS.md:1215-1223` |
| Graph projection of live threads | either one drawer (separation 0.0) or fragmentation (agreement 0.002) | dev bars | FAIL | e5 sits at 0.9+ between domain-adjacent French | `docs/DECISIONS.md:1774-1816` |
| More data, 3,237 rows / 72 drawers | same-thread 0.19/0.26 | 0.60 | FAIL | volume alone; matched topics missing | `docs/BLOCKERS.md:455-464` |
| Merged 72→16 families | 3e 0.3216 / cross 0.8605 / null 0.40; 5e 0.294 / 0.9079 / 0.3667 | 0.60/0.60/0.60 | FAIL | thick classes 0.85-1.0; thin activities 0.13, security 0.16 | `docs/DECISIONS.md:1400-1434` |
| Extractor code removed | six extractors measured | — | CLOSED | resolvers `rules|gold`; dead code deleted | `docs/DECISIONS.md:479-490` |
| Coarse thread index | reach 2/40→9/40; 1 TP; 39/40 drawer upper bound | L1 .50, L2 .70, L3 .30 | FAIL (L1 at the time; L2/L3 always) | thread supplies candidates, judge still chance | `docs/DECISIONS.md:1959-1974` |
| Persistent topic thread | reach 22/40 = 0.55; 1 TP | L2 .70, L3 .30 | FAIL | reach fixed; candidate explosion | `docs/DECISIONS.md:2225-2249` |
| Reader_v2 statuses | 510/781 assertive vs rules 9; over-firing caveat ~65% | B-11 | PARTIAL | the status half of the input is fixed; subjects remain at 17/781 | `docs/BLOCKERS.md:366-377` |

### 2.2 Pair judgement

| attempt | measured | gate / bar | outcome | cause | source |
|---|---|---|---|---|---|
| Synthetic logreg pair classifier | offline dev F1 0.6981 / test 0.7328 | offline F1 | PASS offline | never end-to-end: `SupervisedLinker.__init__` rejected the guard kwargs; D-019 | `docs/DECISIONS.md:484-499` |
| Same logreg on real French | XNLI contra-vs-all AUC 0.4544; rules answered `unrelated` 4906/5010 | external | FAIL | synthetic feature distribution does not travel | `docs/DECISIONS.md:700-708` |
| Real-pair logreg (SUMM-RE) | recall 0.4545 vs rules 0; precision 0.0009 (5 TP / 5,500 FP) | B1 recall .30, B2 precision .70 | FAIL | 25 training positives cannot train a pair classifier | `docs/DECISIONS.md:1550-1574` |
| XNLI fine-tune (moderncamembert) | external XNLI macro-F1 0.7986 | XNLI test | PASS external | not real supersede; static NLI | `docs/DECISIONS.md:710-719` |
| XNLI head on real pairs | bare AUC 0.5727, precision 0.0086, recall 0.2308; thread-prefix AUC 0.5411, precision 0.0064, recall 0.1795 | judge | FAIL | cross-lingual NLI does not transfer to ASR chat | `docs/DECISIONS.md:1976-1985` |
| bge reranker/cross-encoder as judge | restatement AUC 0.8295; contradiction ranked *below* unrelated (mean 0.061 vs 0.272) | compatibility | FAIL | relevance is not logical compatibility | `docs/DECISIONS.md:703-708` |
| Cross-encoder fine-tuned on drawer pairs | control 0.6097; tuned 0.6547 (+0.045) | G1 ≥ control+0.10 and ≥0.80 | FAIL | topic identity is weak; 5 meetings not enough | `docs/DECISIONS.md:1723-1753` |
| e5 dot product / cosine | AUC 0.6329; positives 0.9037 vs negatives 0.8898/0.8891 | AUC / same-thread | FAIL | knows the meeting, not the topic; symmetric, no direction | `docs/DECISIONS.md:1597-1605` |
| 2B whole-session LLM judge | 219 proposals, 0/40 exact hits; P/R 0.0; 016c 218/218 `supersedes` | P .70 / R .30 | FAIL | form correct, taxonomy unstable across meetings; no pair selection | `docs/DECISIONS.md:1860-1931` |
| 7B verifier, given pair | P 0.727 / R 0.727; skeptical pass P 0.778 / R 0.636 | P .70 / R .30 | PASS given pair | verifier, not discoverer | `docs/DECISIONS.md:2256-2284` |
| 7B at natural prevalence | 929 pairs / 12 gold; 153 kept, 5 TP / 148 FP: P 0.033 / R 0.417; fixed-point 67 pred, 4 TP: P 0.060 / R 0.333 | P .70 / R .30 | FAIL | candidate selection, not pair judgement | `docs/DECISIONS.md:2271-2284` |
| Candidate ranker | 5,791 pairs / 23 gold; logreg AUC 0.574 AP 0.022; RF 0.781/0.014; GB 0.698/0.008; top-20 precision 0.008-0.025 | learnable shortlister | FAIL | 23 positives; AP ~0.004 base rate | `docs/DECISIONS.md:2292-2320` |
| Token attention | 376 signals; best train AUC 0.747; selected sel 0.564 (95% CI 0.342-0.786); transfer mean 0.710, sd 0.134 | T1 sel ≥0.70 | INCONCLUSIVE/FAIL | signal present on 3 meetings, chance on 3; 7 positives; no head trained | `docs/DECISIONS.md:2076-2144` |
| Coreference (Coreferee) | 427/1192 pronouns = 0.358; chains cross 10/114; 4 bridged, 2 genuine | linker fix | FAIL (irrelevant) | both genuine links are action/choice, not supersedes | `docs/DECISIONS.md:2146-2170` |
| G13 polarity + turn order (D-061) | pair_local predictions 6→5; TP 1→1; P 0.167→0.200; R 0.025 | safety | PARTIAL | one false positive removed; no new reach/TP | `docs/DECISIONS.md:2536-2567` |
| Change-first: cue/turn + top-5 7B (D-062) | cue coverage 7/40 narrow, 22/40 broad (P .067); turn TF-IDF top-5 1/40; top-5 candidate recall 21/40; 7B exact-pair P .165 / R .525; non-gold turn selection 25/40 | P .70 / R .30 | FAIL | cues broad but low precision; turn classifier cannot rank; 7B over-inclusive; top-5 ceiling 52.5% | `docs/DECISIONS.md:2569-2630`, `docs/CHANGE_DEFINITION.md` |

### 2.3 Threading, graph, and session shape

| attempt | measured | gate / bar | outcome | cause | source |
|---|---|---|---|---|---|
| Graph-conditioned thread assigner | control top-1 0.440; tuned 0.5038 (+0.064); pair AUC 0.8169 | ≥control+0.10 and ≥0.55 | FAIL | context helps, but 878 rows cannot reach 0.55 | `docs/DECISIONS.md:1791-1807` |
| Scale test, 5→9 meetings | top-1 0.5038→0.4624; pair AUC 0.8121 | same | FAIL | no transfer; meeting-specific vocabulary; slope was noise | `docs/DECISIONS.md:1818-1853` |
| Session-first ingest / batch | candidate recall dense online 14/40, whole session 5/40, topic 22/40; end-to-end online 21 `supersedes` 0 TP; batch 19/1 TP/18 FP, P 0.053 / R 0.025 | candidate recall / P/R | FAIL (batch tiny gain) | dense retrieval loses candidates; thread is the only useful candidate kind; judge unchanged | `docs/DECISIONS.md:2322-2360` |
| `answers` relation | 19 gold answer/`answer_of` pairs, 0 edges by either reader; slot fallback 72 edges / 5 gold pairs, P 0.055 | relation | FAIL | meeting questions are `kind=info`, query units are never stored; all gold pairs span meetings | `src/engine/link/rules.py:104-125` |
| `completes` relation | 8 completion cues, 0 proposals, 0 plan→done pairs | relation | NULL | meetings are planning, not execution | `docs/DECISIONS.md:2172-2188` |
| G12 cardinality | 40/40 gold on unknown drawers; G12 refuses 0; marking sets would refuse 7/40 | safety | SAFE/INERT | label cardinality needs human review | `docs/DECISIONS.md:2190-2210` |
| Currency caveat (D-047) | 36 caveated + 4 refused = 40/40 honest; stale confidently 0/40; control caveat 116/132 = 0.879 | C1 ≥ 0.95 | PASS safety | the relation is still unknown; the caveat fires on most raw speech | `docs/DECISIONS.md:2035-2049` |
| Review queue, 024c | 17 rows / 17 true; P 1.0; 4/12 gold new turns = 0.333 | P .70 / R .30 | PASS | value-rich speech | `docs/DECISIONS.md:2395-2398` |
| Review queue, 047a | strict 20 rows, 1/7 R .14, P ~.60; permissive 36 rows, 3/7 R .43, P ~.5-.6; verifier 19 rows, 1/7 R .14, P ~.70 | P .70 / R .30 | FAIL | gold calls refinements "supersedes"; verifier calls them reformulations | `docs/DECISIONS.md:2400-2423` |
| Decision nodes | coverage .995; purity .809; V .505; gold pair 25/40 (.625); latest=gold new 2/22; 0 mutations | grouping | PARTIAL | node is a conversation view, not a change detector | `docs/DECISIONS.md:2430-2466` |
| Hybrid node+detector, 53 labels | 024c 17/17 P 1.0 / R .333; 047a 22/18 P .818 / R .571; node currents 20/18 P .900 / R .421; 8/19 | P .70 / R .30 | PASS on 2 meetings | 4 meetings unlabelled; targeted 2/21 = .095 | `docs/DECISIONS.md:2470-2529` |

### 2.4 Instrument and definition failures

These are not model failures; they either produced a wrong number or made a
real result unscoreable. Kept separate because several "linker failures" were
first instrument failures.

| instrumentation issue | effect | fix / status | source |
|---|---|---|---|
| `SupervisedLinker.__init__` rejected `word_only_supersedes` / `prefer_assertive_head` | trained linker could never be constructed; offline F1 0.6981/0.7328 was never an engine arm | signature fixed; D-019 records that no end-to-end trained-linker run exists | `docs/DECISIONS.md:484-499` |
| `unrelated` class averaged into macro-F1 | guaranteed-zero class made relation macro-F1 0.1544 look broken | score over five writable relations; false-edge rate reported separately | `docs/DECISIONS.md:626-632` |
| 25 positives against all pairs | 5 TP / 5,500 FP; precision 0.0009; B3 "F1 > rules" technically passed and meant nothing | refused; support-count law recorded | `docs/DECISIONS.md:1565-1574` |
| Threshold selected by argmax F1 on 6 positives | chosen point scored P 0.0014 / R 0.1667 on the same split; B2 unreachable by construction | `select_operating_point` fixed; old round not rerun because the sealed read is spent | `docs/DECISIONS.md:1997-2005` |
| Exact-pair scoring of a change | 2/12 exact vs 7/12 new-turn on the same 024c rows | new-turn level adopted; exact-pair retained as citation metric | `docs/DECISIONS.md:2362-2385` |
| LLM prompt/format/split bugs | prompt braces crashed; wrong split carried no temporal gold; `num_predict=4096` truncated a 14 KB answer and scored it as 0 | all fixed before verdict; clean six-meeting run remained 0/40 | `docs/DECISIONS.md:1909-1917` |
| Segmenter split/merge mapping | 5 turns became 6 units; gold indices drifted | recorded as an evaluation mapping issue, not a model defect | `docs/DECISIONS.md:2217-2223` |
| Same typed value + opposite polarity | "budget 2 M€" and "budget n'est pas 2 M€" shared a signature; one false positive was proposed as a support/replacement | `G13_polarity` added; one FP removed, no TP gained | `docs/DECISIONS.md:2536-2567` |
| "Supersedes" vs "refinement" definition | 047a's gold calls an open-option closing a change; the 7B calls it a reformulation | no threshold fixes it; needs relabelling to a product definition | `docs/DECISIONS.md:2411-2422` |

## 3. Why the linker failed, in dependency order

### 3.1 Reach: the guards never see the pair

`supersede.evaluate` starts with G1: both sides need the same non-empty
`slots.subject`. On real ASR the rules resolver supplies one for **17/781**
units; the rules reader calls **768/781** units `mention`, so G3 would block
what G1 lets through (`docs/DECISIONS.md:1940-1949`). Production reach is
**2/40**; the sealed holdout is **0/11**. No model can recover a pair the
guards refuse before consulting it.

The deterministic thread layer was the successful part of the fix: it replaces
a missing key with a coarse subject/attribute/date/amount/number slot, then a
persistent topic (`billetterie`, `programmation`, ...) carried for 12 turns.
Reach moved **2/40 → 9/40 → 22/40**; the perfect-drawer upper bound is
**39/40**. That proves the guards and the chain work when identity is supplied;
it also proves identity is the first choke.

### 3.2 Judgement: once reach is fixed, the candidate space is too wide

With 22 reachable pairs, the pair-local judge writes 74-2,814 edges for 1-7
true positives; every rule is at precision 0.000-0.014 (`docs/DECISIONS.md:2234-2247`).
The 7B judge is the exception, but only as a **verifier**: given the pair,
P 0.727-0.778 / R 0.636-0.727; at natural prevalence it keeps 153 of 929
candidates for 5 TP, P 0.033 / R 0.417 (`docs/DECISIONS.md:2256-2284`). It
cannot discover *which* candidate is the change.

Why the pair judges fail:

- **Cross-lingual NLI is static.** XNLI fine-tune reaches 0.7986 on XNLI, then
  AUC 0.5727 bare / 0.5411 with thread context on real pairs. It learns
  entailment/contradiction, not "this later turn replaces that earlier value."
- **Relevance is not compatibility.** The bge cross-encoder separates
  restatement at 0.8295 but ranks a contradiction *below* unrelated
  (0.061 vs 0.272). Wiring it into `propose()` would mis-rank supersedes.
- **Embedding geometry knows the meeting, not the topic.** e5 pairwise AUC
  0.6329; positives average cosine 0.9037, negatives 0.8898 — the distributions
  overlap almost entirely. The score is also symmetric, so it cannot give a
  direction.
- **Attention sees the anaphora, not the relation.** The token probe finds
  "les" → "pizzas" at 0.7021 vs 0.0083 runner-up, but the aggregate signal is
  meeting-dependent: sel AUC 0.564 (CI 0.342-0.786), 0.564-0.876 across
  meetings, chance on 016c/047c/060a. Raw attention is not a relation score.
- **Coreference does not bridge the gold pairs.** 427/1192 pronouns resolve,
  but of 41 no-content-overlap pronoun cases only 4 bridge and 2 are genuine;
  both are action/choice, not supersedes. The 40 gold pairs mostly repeat their
  topic noun.

### 3.3 Labels: the models are data-bound, not architecture-bound

The support-count law appears at every layer:

- pair classifier: 25 positives → precision 0.0009;
- candidate ranker: 23 positives over 5,791 pairs → AP 0.014-0.022 vs 0.004
  base rate; top-20 precision 0.008-0.025;
- drawer classifier: ~27 examples/drawer → 0.23-0.34 agreement vs 0.60;
- threader: 5→9 meetings → no gain (0.5038→0.4624), because the extra rows teach
  meeting-specific vocabulary.

Template corpora do not fix it: the ledger always names the attribute in the
sentence, so the corpus carries only **4 anaphoric training rows**; template
French scores 0.95/0.97 while the 6 dev anaphoric rows score **0.0**
(`docs/DECISIONS.md:1084-1089`). More unlabelled segments do not create
supersede labels; they only create more candidates.

The real supply is small and partly spent: 40 train+sel supersede pairs, 28 in
the spent external test, 11 in the sealed holdout (read once by D-039), 53
detector rows (17 + 36), 2,673 subject-labelled segments / 10 meetings. Every
attempt to train a pair ranker from 23-25 positives reproduces the same result.

### 3.4 Representation: the real distribution is discourse-level

D-046's anatomy is the clearest explanation: 18% of the 51 pairs carry a
parseable typed value; 31% of the *new* turn is ≤8 words; 37% opens with a
discourse marker (`"sinon le onze"`, `"oui donc on part sur ces deux là"`,
`"ben on fait ça à marseille"`). The state change lives in the thread and its
anaphora, not in two isolated sentences. This is why:

- `same_slot` works on synthetic boilerplate and fails on speech;
- the thread context moves the score (sentence-vs-thread AUC 0.8169 vs
  sentence-vs-sentence 0.6547) but not enough;
- the 7B can verify a pair but cannot scan 929 candidates;
- the review queue works on value-rich 024c and fails on discourse-heavy 047a
  because "refinement of an open option" is not the same product definition as
  "value replacement."

### 3.5 Measurement: several failures were ruler failures first

The record separates these deliberately:

- **D-019**: the trained linker could not be constructed; the offline F1 was
  never an end-to-end number.
- **D-020/D-021**: a guaranteed-zero `unrelated` class in macro-F1; false-edge
  rate diluted by unscoreable new edges.
- **D-039**: 25 positives against all pairs; 5,500 FP; B3 "F1 > rules" passed
  and meant nothing.
- **D-046(d)**: argmax-F1 threshold on 6 positives produced P 0.0014; fixed
  selector, old round not rerun because the sealed read is spent.
- **D-045**: prompt/format/split bugs; killed before any verdict; clean run
  still 0/40.
- **D-052**: segment split/merge mapping, not a model defect.
- **D-056**: exact-pair scoring undercounted a change found at the new turn
  (2/12 vs 7/12).
- **D-057**: the 047a failure is a definition mismatch, not a threshold.
- **D-061**: a negated value shares the typed value with its affirmative;
  G13 removed one false positive, no true positive.

## 4. What the metrics actually established

- The guards are not the problem: with gold subjects the real-data demos pass
  **10/10** and write **9** supersede edges; the v3 node view retraces
  **40/40** at `at_turn` with **0** mutations (`docs/DECISIONS.md:1245-1300`,
  `docs/DECISIONS.md:2513-2529`).
- Retrieval is not the problem for the synthetic headline: `gold_in_pool = 1.0`
  on every arm; temporal@1 misses are ranking misses, not lost candidates
  (`docs/DECISIONS.md:1576-1582`).
- The v3 hybrid is the useful product shape: node grouping 25/40 (.625),
  detector currents P 0.900 / R 0.421 on the two labelled meetings, coverage
  .995, zero mutation. On the other four meetings the targeted scan reaches
  only **2/21 = .095**; that is candidate discovery, not node-view failure.
- The 7B verifier plus deterministic filters is the only judge that clears the
  declared pair bars, and only when the pair is supplied. Candidate selection
  is the remaining task.

## 5. What would move it

1. **Label binary new-turn changes exhaustively on a sample, then the old
   citations.** D-062 showed the turn-first chain cannot be scored from 40
   positives: use 7B proposals as a high-recall queue, but a human must mark
   both positives and negatives. Label 100-200 same-thread pairs per unlabelled
   meeting only after the turn-level set exists. This is the supply D-053/D-054
   name.
2. **Train a candidate ranker on those labels**, with a grouped meeting split;
   keep the 7B verifier as the second stage. Do not train on the 40 train+sel
   or the spent 11-pair holdout.
3. **Keep the deterministic guards**: G13 polarity, G2 turn order, G12
   cardinality, G7 speech acts. They are cheap and prevent wrong deletions.
4. **Do not ship `thread_context`** until a judge clears precision at natural
   prevalence; current pair-local predictions are 74-2,814 for 1-7 TP.
5. **Do not call the node's latest child "the current decision"** when the
   product needs the latest detected change; that is the review queue
   (D-056/D-057), not the node view.

## 6. Source index

- `docs/BLOCKERS.md:330-520` — B-13 real-corpus record, D-026 to D-047.
- `docs/DECISIONS.md:688-742` — D-022 XNLI/cross-encoder.
- `docs/DECISIONS.md:909-960` — D-026 cosine / status.
- `docs/DECISIONS.md:1001-1145` — D-028/D-029/D-030 subject extraction.
- `docs/DECISIONS.md:1400-1434` — D-036 merge.
- `docs/DECISIONS.md:1544-1931` — D-039 to D-045 real-pair linker, LLM.
- `docs/DECISIONS.md:1933-2070` — D-046/D-047 thread ceiling and caveat.
- `docs/DECISIONS.md:2076-2360` — D-048 to D-055 token attention, coreference,
  cardinality, 7B verifier, candidate ranker, session batch.
- `docs/DECISIONS.md:2362-2534` — D-056 to D-060 review queue, nodes, hybrid,
  v3.
- `docs/DECISIONS.md:2536-2567` — D-061 polarity/turn guard.
- `docs/DECISIONS.md:2569-2630` — D-062 change-first test (cues, turn classifier,
  top-5 candidates, 7B).
- `docs/CHANGE_DEFINITION.md` — the product definition used by D-062.
- `docs/REPORT.md:176-430` — rendered §6e-§6o tables.
- Code: `src/engine/link/{rules,supersede,values,thread,supervised,model,nli_finetune}.py`,
  `src/engine/eval/{linker_local,linker_probe,decision_nodes,token_attention}.py`,
  `src/engine/tools/session_changes.py`.
