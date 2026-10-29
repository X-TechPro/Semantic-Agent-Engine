PY      ?= .venv/bin/python
PIP     ?= uv pip
PYTHONPATH := src
export PYTHONPATH
SEEDS   ?= 42,43,44
PROJECTS?= 20
NPROC   ?= 4

# Python 3.14 has no CUDA torch wheel to re-download here, and this host already
# has torch+cu130 in the user site. The venv inherits system packages instead of
# duplicating a 2.5 GB download (docs/DECISIONS.md D-001).
.PHONY: setup
setup:
	@command -v uv >/dev/null || { echo "uv not found; use: python -m venv --system-site-packages .venv"; exit 1; }
	uv venv --python /usr/bin/python3.14 --allow-existing --system-site-packages .venv
	$(PIP) install --python .venv/bin/python -e .
	$(PY) -c "import faiss, bm25s, chunklet, pydantic, hypothesis, dateparser; print('deps ok')"
	@$(PY) -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
	@echo "run: make test && make meetings && make steps"
	@echo "legacy v3: make demo && make demo-real"

# 12.3 golden minimal-pair suites (>= 30 pairs per category). The files under
# tests/golden/*.jsonl are generated; a test asserts they still match, so they
# cannot silently drift from their generator.
.PHONY: golden
golden:
	$(PY) -m engine.tools.gen_golden_pairs

# M3 (6.4): fine-tune the embedder on synthetic ledger triples + constructed
# minimal-pair negatives, scored against a disjoint general-French probe.
# Writes models/embed/qwen3-ledger-s42 and runs/embed_finetune/result_42.json.
.PHONY: embed
embed:
	$(PY) -m engine.embed.finetune --epochs 6 --projects $(PROJECTS) --seed 42

# Does the tuned encoder earn its place? frozen vs tuned, same arms, DEV split.
.PHONY: embed-eval
embed-eval:
	@for m in Qwen/Qwen3-Embedding-0.6B models/embed/qwen3-ledger-s42; do \
	  echo "== $$m"; \
	  $(PY) -m engine.cli bench --split dev --seeds 42 --projects $(PROJECTS) \
	    --embed st --model "$$m" 2>&1 | grep -E "temporal@1|recall@5" | tail -3; done

.PHONY: test
test:
	$(PY) -m pytest tests -q

.PHONY: test-unit
test-unit:
	$(PY) -m pytest tests/unit tests/golden -q

.PHONY: cov
cov:
	$(PY) -m pytest tests -q --cov=engine --cov-report=term-missing

# Formatting exists as an explicit choice, not a commit hook: measured, 111 of 126
# Python files are not black-clean, so a pre-commit formatter would silently rewrite
# whatever anyone stages and bury the actual change (see .pre-commit-config.yaml).
# `make format` is the one deliberate commit that adopts it; `format-check` reports
# the current state without touching anything and is expected to fail until then.
.PHONY: format format-check
format:
	$(PY) -m black --line-length 100 src tests api ui

format-check:
	$(PY) -m black --check --line-length 100 src tests api ui

# 12.2 property tests + 12.5 safety tests, the two tiers CI must not skip
.PHONY: gates
gates:
	$(PY) -m pytest tests -q -m "property or e2e"

.PHONY: schemas
schemas:
	$(PY) -m engine.cli export-schemas

# M0 gate: verified model cards against the live HF API
.PHONY: models
models:
	$(PY) -m engine.cli verify-models

.PHONY: ledger
ledger:
	$(PY) -m engine.cli ledger --projects $(PROJECTS) --seed 42

.PHONY: render
render: ledger
	$(PY) -m engine.cli render --ledger data/ledger/ledger-42.json --seed 42

# M1 gate: Baseline A (plain hybrid RAG) numbers, then System B (typed relations)
.PHONY: baseline
baseline:
	$(PY) -m engine.cli bench --projects $(PROJECTS) --seeds $(SEEDS)

.PHONY: stages
stages:
	@for s in segment reader slots linker retrieve; do \
	  echo "== $$s"; $(PY) -m engine.cli stage $$s --projects $(PROJECTS) --seed 42 | tail -1; done

# 9.5 CI gate: small fixed set, must fail if a headline metric drops > 2 pts
.PHONY: eval-fast
eval-fast:
	$(PY) -m eval.regression_check

# 9.1 system metrics: per-stage p50/p95, peak RAM/VRAM, index size, throughput.
# `make eval` runs it because 10 requires latency reported beside quality.
.PHONY: latency
latency:
	# rows are explicit since D-041: "none" is the ablation, cross_encoder is
	# the shipped default — a bare row would silently duplicate the other
	$(PY) -m engine.cli latency --split test --projects $(PROJECTS) --seed 42 \
	  --rerank none
	$(PY) -m engine.cli latency --split test --projects $(PROJECTS) --seed 42 \
	  --rerank cross_encoder

.PHONY: eval
eval: stages bench latency

# --- current pipeline: five hand-written meetings, synthetic-only data -------
# Everything below uses the v4 path (ingest_v4 → store/v4 → search → answer).
# `make demo` and `make demo-real` are the legacy v3 scenarios kept as rollback.

.PHONY: meetings
meetings:
	# step 7 end to end: reranker + retrieved context + memory state + LLM
	$(PY) -m engine.eval.meetings_eval --rerank --answer hybrid

.PHONY: steps
steps:
	# the per-step table (store, chunk, label, link, memory, search, answer)
	$(PY) -m engine.eval.steps_eval --rerank

.PHONY: step7
step7:
	# the funnel inside step 7: parse, retrieve, state, match, select, direct
	$(PY) -m engine.eval.step7_debug

.PHONY: rerank
rerank:
	# Qwen3-Reranker-0.6B vs the pipeline order, on the same candidates
	$(PY) -m engine.eval.rerank_qwen

.PHONY: tag
tag:
	# fine-tune Laya on the piscine tags (GPU, ~1.2 GB output)
	$(PY) -m engine.read.laya_tags --epochs 2 --lr-enc 2e-5

.PHONY: validate-data
validate-data:
	$(PY) -m engine.tools.validate_dataset data/mine/*.meeting.json
	$(PY) -m engine.tools.validate_dataset data/mine/*.questions.json \
	  --meetings data/mine/*.meeting.json

# --- legacy v3 demo and report ----------------------------------------------
.PHONY: demo
demo:
	$(PY) -m engine.cli demo

.PHONY: demo-real
demo-real:
	$(PY) -m engine.cli demo-real

.PHONY: ask4
ask4:
	@test -n "$(M)" || { echo 'usage: make ask4 M=data/mine/nova.meeting.json Q="..."'; exit 1; }
	@test -n "$(Q)" || { echo 'usage: make ask4 M=data/mine/nova.meeting.json Q="..."'; exit 1; }
	$(PY) -m engine.cli_v4 "$(M)" "$(Q)" $(if $(ANSWER),--answer $(ANSWER)) $(if $(NORERANK),--no-rerank)

.PHONY: ask
ask:
	@test -n "$(Q)" || { echo 'usage: make ask Q="..." [PROJECT=Atlas]'; exit 1; }
	$(PY) -m engine.cli ask "$(Q)" $(if $(PROJECT),--project $(PROJECT))

.PHONY: api
api:
	$(PY) -m uvicorn api.app:app --host 127.0.0.1 --port 8080

.PHONY: ui
ui:
	$(PY) -m streamlit run ui/app.py

.PHONY: offline
offline:
	$(PY) -m engine.offline

.PHONY: killtest
killtest:
	$(PY) -m eval.killtest

# 6.13 concept attention (M5). Train writes models/concept/concept.pt; refine
# writes vectors.refined into a store. Both are opt-in: nothing here changes the
# shipped configuration, and `row9` below only becomes measurable once both ran.
.PHONY: concept-train
concept-train:
	$(PY) -m engine.cli concept-train --projects $(PROJECTS) --seed 42 --verbose

.PHONY: concept-refine
concept-refine:
	@test -n "$(STORE)" || { echo 'usage: make concept-refine STORE=path/to/store.db [--dry-run] [--use]'; exit 1; }
	$(PY) -m engine.cli concept-refine --store $(STORE) $(if $(DRY),--dry-run,) $(if $(USE),--use,)

# 10 row 9: System B over the refined column. Guarded so it cannot be mistaken for
# the headline — a refined run is barred from `canonical` inside run_bench_seeds,
# and this target fails loudly if the checkpoint is missing rather than quietly
# measuring the frozen base twice.
.PHONY: row9
row9:
	@test -f models/concept/concept.pt || { echo "no models/concept/concept.pt: run make concept-train"; exit 1; }
	$(PY) -m engine.cli bench --vectors refined --split dev --seeds $(SEEDS) --projects $(PROJECTS)

# The 6.13 decision itself: base vs refined from ONE code path, so the arms cannot
# differ in anything but the column. Exits non-zero when the like-for-like guard
# fails, which is a void comparison rather than a result.
.PHONY: concept-gate
concept-gate:
	$(PY) -m eval.concept_gate --split $(or $(SPLIT),dev) --projects $(PROJECTS) \
	  --seeds $(SEEDS)

# 9.3 oracle swaps: give one stage its gold and read the end-to-end gain, so the
# remaining error is attributed to a stage instead of guessed at.
.PHONY: oracle
oracle:
	$(PY) -m eval.oracle_swap --split dev --projects $(PROJECTS) --seed 42

# 9.4 adversarial end-to-end suite: unanswerables, ambiguous recipients, stale
# values, hypothesis-action refusal. Exits non-zero while any case fails.
.PHONY: e2e
e2e:
	$(PY) -m eval.e2e

# 7.2 real-French external check. `frdata` fetches the pinned corpora (idempotent,
# writes PROVENANCE.json beside each); `frbench` scores linker vs XNLI-fr, reader
# intent vs MASSIVE-fr and the retrieval stack vs MIRACL-fr. Verdicts live in
# runs/frbench/ and the report reads them. COMPONENT=xnli|xnli-rerank|massive|miracl
.PHONY: frdata frbench
frdata:
	$(PY) -m engine.cli frdata

frbench:
	$(PY) -m engine.cli frbench --component $(or $(COMPONENT),all) \
	  --limit $(or $(LIMIT),5000) --device $(or $(DEVICE),cuda) \
	  $(if $(READER),--reader-model $(READER),)

# 6.8b / 7.2: train the linker's text relation head on XNLI-fr. Selection is on
# validation; the test split is read ONCE, by `make frbench COMPONENT=xnli-nli`
# (the harness refuses a second read).
.PHONY: nli
nli:
	$(PY) -m engine.link.nli_finetune --epochs 2

# B-12's reader half: MASSIVE-fr warm-up of the trained reader (the "bit of
# data" the guide asks for). Only kind/intent heads train -- the encoder and
# the guard-input type/status heads are frozen and the invariance is asserted.
# Selection on massive validation + ledger dev; revalidation reads ledger test
# ONCE and the harness refuses a second read. Score the external side with
# `make frbench COMPONENT=massive READER=models/reader_warm`.
.PHONY: readwarm
readwarm:
	$(PY) -m engine.cli reader-warm

# 6.4 "must not forget general French". Unmeasured while mteb is absent; the
# artefact then carries a labelled proxy instead of a substituted number.
.PHONY: mteb-gate
mteb-gate:
	$(PY) -m eval.regression_mteb --tuned models/embed/qwen3-ledger-s42

.PHONY: report
report:
	$(PY) -m eval.report

.PHONY: clean
clean:
	rm -rf .pytest_cache runs/tmp data/engine.db* .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
