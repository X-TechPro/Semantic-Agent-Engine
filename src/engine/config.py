"""Config loading — one YAML per experiment/component (BUILD_GUIDE 0.6).

No hard-coded paths: every path is resolved against the repo root, which is
discovered from this file's location or the `SAE_ROOT` env var.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "SAE_"


def repo_root() -> Path:
    env = os.environ.get(_ENV_PREFIX + "ROOT")
    if env:
        return Path(env).resolve()
    # src/engine/config.py -> repo root
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
DATA_DIR = Path(os.environ.get(_ENV_PREFIX + "DATA", ROOT / "data"))
RUNS_DIR = Path(os.environ.get(_ENV_PREFIX + "RUNS", ROOT / "runs"))
SCHEMAS_DIR = ROOT / "schemas"
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "seeds": [42, 43, 44],
    "language": "fr",
    "device": "auto",  # auto | cpu | cuda
    # The one switch that separates Baseline A from System B (10). It disables
    # both the ingest-time link pass and the retrieval-side validity weighting +
    # supersede-following, so the two rows differ by exactly one flag.
    "linker_enabled": True,
    "paths": {
        "data": str(DATA_DIR),
        "runs": str(RUNS_DIR),
        "schemas": str(SCHEMAS_DIR),
    },
    "segment": {
        "backend": "chunklet",          # chunklet | turns | fixed
        "lang": "fr",
        "max_turn_tokens": 25,          # a short turn stays one unit (6.1)
        "asr_max_tokens": 40,           # unpunctuated fallback window
        "asr_min_tokens": 8,
        "tau_merge": 0.55,              # doc chunker merge threshold
        "doc_max_tokens": 256,
        "doc_min_tokens": 40,
        "fixed_window_words": 200,      # baseline_fixed.py
        "fixed_stride_words": 100,
        "max_unit_tokens": 60,          # guard: a unit stays sentence-sized
    },
    "reader": {
        "backend": "rules",             # rules | model (see read/infer.py)
        "model_dir": str(MODELS_DIR / "reader"),
        "context_units": 0,             # 6.2.3 ablation knob
        "abstain_threshold": 0.0,       # calibrated tau (model backend only)
        "max_length": 96,               # reader token budget (6.2.3)
    },
    "slots": {
        "backend": "rules",             # rules | gliner | llm
        "persons_backend": "rules",     # rules | gliner
        "subject_resolver": "rules",    # rules | gold
        # 6.3's trained subject extractors (b lookup, c LLM, d/e span, f drawer
        # classifier, g graph projection) were each built, wired and measured
        # against one pre-registered gate (D-026…D-036); none crossed it on
        # real French speech and the code was removed (D-037). Their verdicts
        # stay in runs/ and REPORT §6e. `gold` supplies hand-labelled keys for
        # the real-data demos only — it annotates, it extracts nothing; asking
        # for a removed resolver is a startup error, never a silent fall-back.
        # subject_model_dir: for resolver=gold, the labelled-gold directory
        # (None -> data/gold).
        "subject_model_dir": None,
    },
    "embed": {
        "backend": "hash",              # hash (baseline) | sentence_transformers
        "model_id": None,               # set by config; verified in MODELCARDS
        "dim": 256,                     # hash baseline only
        "batch_size": 64,
        "max_seq_length": 512,
        "normalize": True,
        "use_fp16_store": False,
    },
    "store": {
        "path": str(DATA_DIR / "engine.db"),
        "faiss": {"index_key": "base", "max_neighbors": 20},
    },
    "lexical": {
        "backend": "bm25s",             # bm25s | fts5
        "strip_accents": True,
        "stem": False,
        "over": "sentences",            # 6.6: BM25 runs over sentences
    },
    "retrieve": {
        "top_k_dense": 50,
        "top_k_lexical": 50,
        "rrf_k": 60,
        "weights": {"dense": 1.0, "bm25": 1.0},
        "final_k": 30,
        "slot_boost": 1.25,             # explicit subject = soft boost (6.7)
        # owner questions: "risk: under control" is not the answer to "who
        # owns the risk?"; a row carrying a person/owner gets this additive
        # prior before validity. Measured on the corrected 20-project test set:
        # 0.780 -> 0.828 mean over seeds 42/43/44.
        "owner_boost": 0.5,
        "node_view": True,              # v3: decision nodes beside retrieval
        # v3.1: additive state-point nodes. Topic reach is 25/40; topic+state
        # reach is 30/40 on the six gold meetings. The fast gate is re-recorded
        # when this changes; the temporal@1 path is retrieval, not node view.
        "node_state": True,
        # D-041 (2026-10-23, owner decision): the cross-encoder was promoted to
        # the shipped default. It is the only lever that has ever moved the
        # headline (temporal@1 0.5518 -> 0.6616, 3/3 seeds on dev AND test,
        # D-015); the accepted price is ~9 s cold start and ~4x steady-state
        # latency (12 ms p50). `none`/`overlap` remain selectable ablations.
        "rerank": {"backend": "cross_encoder", "candidates": 30},
        "validity": {
            "alpha_superseded": 0.0,    # 6.10 "current" queries
            "alpha_history": 0.3,       # 6.10 "history" queries
            "beta_soft_status": 0.5,    # hypothesis / proposed
            "beta_mention": 1.0,        # 6.10 extension, D-040: mention units
                                        # (name a subject, assert no state) at
                                        # rank 1 caused 8/17 census misses; the
                                        # AB run decides 1.0 (off) vs 0.5
            "hard_filter": False,
            "graph_expansion": True,
        },
    },
    "concept": {
        # 6.13 optional experiment. Off by default on purpose: 0.4 says the
        # measured baseline stays in force until the trained version beats it,
        # and the gate has not been passed yet (docs/BLOCKERS.md B-06).
        "enabled": False,
        "k_neighbors": 16,              # 6.13: unit + its k nearest neighbours
        "d_model": 256,                 # projected down from the embedder dim
        "layers": 2,                    # 6.13 allows 2-4
        "heads": 4,                     # 6.13 allows 4-8
        "epochs": 8,
        "lr": 3e-4,
        "batch_size": 8,
        "temperature": 0.07,
        "triplet_margin": 0.3,
        "max_drift": 0.25,              # how far refined may move from base
        "weights": {"contrastive": 1.0, "validity": 1.0, "incremental": 0.5,
                    "state": 0.5, "drift": 1.0},
        "checkpoint": str(MODELS_DIR / "concept" / "concept.pt"),
    },
    "link": {
        "backend": "rules",             # rules | logreg | nli (see link/supervised.py)
        "model_path": str(MODELS_DIR / "linker_lr.json"),
        # 6.8b text relation head (D-022/D-023): answers only the lexical
        # fall-through where rules have no structural signal; guards unchanged.
        "nli_model_dir": str(MODELS_DIR / "linker_nli"),
        "k_neighbors": 12,
        "supersede_min_score": 0.55,
        "guards": True,                 # 6.8.3 deterministic guards
        "same_scope_only": True,
        # D-046 pair-local threads: when True, units sharing a deterministic
        # coarse slot (resolved subject, else attribute/date/amount/number from
        # their own words) may be judged by `evaluate(..., same_thread=True)`
        # instead of being refused at G1 for a missing subject key. OFF: the
        # measured round on train+sel is refused (runs/linker_local), so the
        # shipped behaviour and every recorded number stay unchanged.
        "thread_context": False,
        "thread_window": 8,
        # Session-first ingest (D-055): annotate the whole source, link against
        # session-local candidates, then commit units+edges once. OFF: the
        # shipped online path stores each unit before linking, and every
        # recorded number stays comparable.
        "session_batch": False,
        # Value-comparability thresholds for G9/G4. Tuned on dev, not invented.
        # The sweep lives in the artefacts, not in prose: `runs/stage_linker_dev_*`
        # and `runs/g9_ablation/` (reproduce with `sae stage linker --split dev
        # --config <file>`). Recorded: 0.5 gave P 0.4545 / R 0.2632, 0.0 gave
        # P 0.52 / R 0.6842 -- 0.0 dominates on BOTH axes, so the topical floor
        # was only ever suppressing the edges the validity filter needs. Re-measured
        # 2026-10-21: 0.0 -> P 0.5172 / R 0.7143, 0.5 -> P 0.4375 / R 0.3333.
        # Tuning it on the test holdout was my mistake (9.5 says tune on dev).
        # Consequence worth stating: at 0.0 the topical branch below can never
        # fire (`topical < 0.0` is false by construction), so comparability is
        # carried by "both sides have a value signature" plus the later guards.
        # An earlier version of this comment pointed at a REPORT.md section that
        # does not exist.
        # When False, a pair whose values differ only in free text (vendor,
        # scope, stack) is recorded as a conflict to be surfaced rather than a
        # replacement that demotes evidence: 6.8's "prefer contradicts + ask".
        "word_only_supersedes": True,
        # G12: never supersede a soft (hypothesis/proposed/conditional) unit
        # while an assertive current statement exists for the same slot.
        "prefer_assertive_head": True,
        "value_topical_min": 0.0,
        "value_equal_max": 0.8,
    },
    "agent": {
        "llm": {"backend": "template"},  # template | ollama | openai-compatible
        # the diagram's last box, opt-in: an LLM may REWRITE the template
        # answer (whose claims are already unit-cited), and the rewrite must
        # pass agent/verify — any invented name/number bounces to the template.
        # Default off: every shipped artefact is template-first (deterministic,
        # and the LLM-off ablation row). `make demo-real --compose` tests it.
        "llm_answer": False,
        "contacts_path": str(DATA_DIR / "contacts.json"),
        # Abstention is CONTENT-based (engine/retrieve/relevance.py). RRF scores
        # live in ~[0.016, 0.033] for two lists at k=60, so a threshold of 0.05
        # is above the maximum achievable score and abstains on every query —
        # measured: 100% abstention on answerable questions.
        "no_answer_threshold": 0.0,     # fused-score floor (ranking-scale units)
        "no_answer_overlap": 0.06,      # content-word Jaccard floor
        # D-047 unresolved-currency guard: two live statements that disagree
        # about the same thing, with no relation deciding between them, make the
        # answer say "I cannot tell which is in force" instead of leading with
        # rank 1. Measured on real speech in runs/currency_guard; gated here so
        # the shipped behaviour is a recorded choice.
        "currency_guard": True,
    },
    "ledger": {
        "n_projects": 20,
        "meetings_per_project": 3,
        "seed": 42,
        "distractor_rate": 0.35,
        "attribute_vocab_version": "v0",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """Dict subclass with attribute access and dotted `get("retrieve.rrf_k")`."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def hash(self) -> str:
        blob = json.dumps(self, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config(copy.deepcopy(DEFAULTS))
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = CONFIGS_DIR / p
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = Config(_deep_merge(cfg, loaded))
    if overrides:
        cfg = Config(_deep_merge(cfg, overrides))
    return cfg


def path_of(cfg: Config, key: str) -> Path:
    p = Path(cfg.get_path(key))
    return p if p.is_absolute() else ROOT / p
