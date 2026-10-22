"""Baseline A vs System B — the measurement of the thesis (BUILD_GUIDE 9 / 10).

    Baseline A = segment + store + BM25 + dense + RRF        (plain hybrid RAG)
    System B   = A + reader + slots + linker + supersede + validity filter

Instrument honesty, learned the hard way elsewhere (a harness bug once faked a
scaling result):

  * `gold_in_pool` is reported first. If the gold unit is never retrieved,
    temporal-correctness@1 cannot detect a relation-layer gain and a null result
    says nothing about the thesis.
  * Every headline delta carries a **paired bootstrap CI over projects** (§9.6)
    and is only called a win when the CI excludes zero.
  * Both systems embed and retrieve the same corpus with the same candidate
    budget; the only difference is `linker_enabled` (§10 fairness rule).
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config, repo_root
from engine.embed.base import build_embedder
from engine.engine import Engine
from engine.retrieve.rerank import build_reranker
from engine.schemas import Query, utcnow
from eval import metrics as M
from gen.hard_negs import build_hard_negatives
from gen.ledger import Ledger, build_ledger
from gen.qa_from_ledger import build_questions
from gen.render import render_meeting
from gen.splice import assign, write_manifest

EXP_ID = "m1_m2_temporal"

#: the corpus size the headline is defined at (§16: 3 seeds x 20 projects).
CANONICAL_PROJECTS = 20


def run_dir(root: Path, split: str, seed: int, n_projects: int, cfg) -> Path:
    """The artefact directory for one bench run, named by what it actually ran.

    The project count belongs in the name and not only in the file's `meta`:
    `make eval-fast` runs 8 projects at seed 42 on the test split, which used to
    resolve to exactly the canonical 20-project headline directory and overwrite
    it. A tripwire had been replacing headline evidence with a smaller, easier
    corpus and nothing complained.
    """
    reader_tag = str(cfg.get_path("reader.backend", "rules")).replace(" ", "")
    rerank_tag = str(cfg.get_path("retrieve.rerank.backend", "none")).replace(" ", "")
    # 6.13's A/B compares the same system over two vector columns, and the column
    # would otherwise be the one thing missing from the name: a refined run would
    # land on the base run's directory and overwrite the recorded headline — the
    # same identity failure as the 8-project tripwire above. Tagged only when it is
    # not the default, so every artefact already on disk keeps its path and the
    # report and the gate keep reading it.
    column = str(cfg.get_path("store.faiss.index_key", "base")).replace(" ", "")
    column_tag = "" if column == "base" else f"_v-{column}"
    return (root / "runs" /
            f"{EXP_ID}_{split}_{seed}_p{n_projects}_r-{reader_tag}"
            f"_k-{rerank_tag}{column_tag}")


def gold_unit_map(store) -> dict[str, list[str]]:
    """event_id -> unit ids that carry it (the ledger->render->segment chain)."""
    out: dict[str, list[str]] = {}
    for u in store.all_units():
        for eid in u.gold_event_ids:
            out.setdefault(eid, []).append(u.unit_id)
    return out


def _build_engine(cfg, *, linker_enabled: bool, path: Path, roster=None,
                  embedder=None, reranker=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in path.parent.glob(path.name + "*"):
        p.unlink()
    return Engine(cfg, path=path, linker_enabled=linker_enabled, roster=roster,
                  embedder=embedder, reranker=reranker)


def run_bench(n_projects: int = 20, meetings_per_project: int = 3, seed: int = 42,
              split: str = "test", out_dir: Path | None = None, cfg=None,
              embed_backend: str | None = None, embed_model: str | None = None,
              reader_backend: str | None = None, rerank_backend: str | None = None,
              vector_column: str | None = None,
              limit_questions: int | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    # every selectable stage goes through ONE override dict, so adding a backend
    # switch cannot quietly reach one entry point and not the other (the
    # `link.backend` inert-knob bug, D-011).
    overrides: dict[str, Any] = {}
    if embed_backend:
        overrides["embed"] = {"backend": embed_backend, "model_id": embed_model}
    if reader_backend:
        overrides["reader"] = {"backend": reader_backend}
    if rerank_backend:
        overrides["retrieve"] = {"rerank": {"backend": rerank_backend}}
    if vector_column:
        # 10 row 9: the vector column is a stage selector like any other, and it
        # belongs in this dict rather than in a separate knob, so `run_dir` sees it
        # (it reads the built cfg) and the refined arm cannot silently land in the
        # base arm's directory.
        overrides.setdefault("store", {})["faiss"] = {"index_key": vector_column}
    if overrides:
        cfg = load_config(overrides=overrides)
    root = repo_root()
    # split, reader AND reranker belong in the path: a dev run used to overwrite
    # the test artefacts in the same directory, and the first trained-reader run
    # then overwrote the rules run beside it — letting a stale number look fresh.
    out_dir = out_dir or run_dir(root, split, seed, n_projects, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger(n_projects=n_projects, meetings_per_project=meetings_per_project,
                          seed=seed)
    manifest = write_manifest(root / "data/splits/ledger.json", ledger.projects, seed=seed)
    holdout = set(manifest["split"][split])
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})

    t_ingest = time.perf_counter()
    # One embedder and one reranker for the whole run, injected into both arms.
    # Built per arm before, which put two copies of a 568M-parameter
    # cross-encoder on an 8 GB card and OOMed the §10 rerank row — and would have
    # made the two arms differ by allocator pressure rather than by
    # `linker_enabled`, breaking the fairness rule in the module docstring.
    shared_embedder = build_embedder(cfg)
    shared_reranker = build_reranker(cfg)
    engines: dict[str, Engine] = {}
    for name, linker in (("baseline_A", False), ("system_B", True)):
        engines[name] = _build_engine(cfg, linker_enabled=linker,
                                      path=out_dir / f"store_{name}.db", roster=roster,
                                      embedder=shared_embedder,
                                      reranker=shared_reranker)
        for meeting in ledger.meetings:
            if meeting.project not in holdout:
                continue
            src = render_meeting(ledger, meeting.meeting_id, seed=seed)
            engines[name].ingest(src)
    ingest_s = round(time.perf_counter() - t_ingest, 2)

    # 10 row 9 (6.13): the SAME two arms, the same corpus and budgets, with only the
    # vector column changed. Refining here — after ingest, before any scoring — is
    # what keeps the comparison single-ruler: the metric code, the gold-by-lookup
    # map, the candidate budget and the scoping flags are all untouched, so a
    # refined-vs-base difference cannot be attributed to anything but the geometry.
    # The e5 row is the cautionary case in `docs/BLOCKERS.md`: an arm that differed
    # in more than the thing under test could not be compared to the floor.
    column = str(cfg.get_path("store.faiss.index_key", "base"))
    concept: dict[str, Any] = {"requested_column": column}
    if column == "refined":
        from engine.concept.refine_job import refine
        from engine.concept.train import load as load_concept

        ckpt = Path(str(cfg.get_path("concept.checkpoint")))
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        concept["checkpoint"] = str(ckpt)
        concept["arms"] = {}
        for name, eng in engines.items():
            # the embed-version guard inside `load` is what stops a checkpoint
            # trained on one encoder's space from repointing another's store
            model, meta = load_concept(ckpt, expect_embed_version=eng.store.embed_version)
            rep = refine(eng.store, model,
                         k=int(cfg.get_path("concept.k_neighbors", 16)),
                         batch_size=int(cfg.get_path("concept.batch_size", 8)),
                         max_drift=float(cfg.get_path("concept.max_drift", 0.25)),
                         writes=True)
            # raises if the anchoring gate refused the write: a run must not be able
            # to report `refined` while the frozen base is what was actually searched
            eng.use_vector_column("refined")
            concept["arms"][name] = rep
        concept["vector_column_read_back"] = {
            name: eng.vector_column for name, eng in engines.items()}
        concept["trained_embed_version"] = meta.get("embed_model_version")
        concept["labels"] = meta.get("labels")
        concept["best_epoch"] = meta.get("best_epoch")

    qa = build_questions(ledger, seed=seed)
    items = [q for q in qa.items if q["project"] in holdout]
    if limit_questions:
        items = items[:limit_questions]
    scored_types = ("current_decision", "history", "proposal_vs_decision", "owner",
                    "date", "unanswerable")
    items = [q for q in items if q["kind"] in scored_types]

    # 10 rows 4, 6, 7. Row 6 is scored on row 4's store, so the only thing that
    # changes between them is the scoping flag: that separates "metadata filters"
    # from "typed relations" instead of bundling both into one delta. The first
    # headline (+0.31) was measured against an unscoped row 4, which inflated the
    # relation layer's share.
    results: dict[str, Any] = {}
    for name, eng in engines.items():
        gmap = gold_unit_map(eng.store)
        results[name] = _score_system(eng, items, gmap, ledger, cfg, arm=name)
    results["a_scoped"] = _score_system(engines["baseline_A"], items,
                                        gold_unit_map(engines["baseline_A"].store),
                                        ledger, cfg, arm="a_scoped")
    results["ingest_seconds"] = ingest_s
    results["meta"] = {"seed": seed, "n_projects": n_projects,
                       "holdout_projects": sorted(holdout),
                       "n_questions": len(items), "split": split,
                       "embed_version": engines["system_B"].embedder.version,
                       # a `link.backend` that silently falls back to rules made
                       # two arms byte-identical while looking "trained"; the
                       # artefact now states which linker actually ran (D-011).
                       "linker_backend": getattr(engines["system_B"].ingest_pipeline,
                                                 "linker_backend", "unknown"),
                       # the same trap for the reader: `reader.backend: model`
                       # with a missing checkpoint falls back to rules, and a run
                       # must not get to claim the trained head it did not use.
                       "reader_backend": getattr(engines["system_B"].ingest_pipeline,
                                                 "reader_backend", "unknown"),
                       # and the same for the reranker: reading it back off the
                       # built pipeline means a row cannot claim `cross_encoder`
                       # while the config actually resolved to `none`.
                       "rerank_backend": engines["system_B"].rerank_backend,
                       # which vector space was actually searched. `base` is the
                       # frozen encoder's space and `refined` is 6.13's; a row that
                       # does not say cannot be compared to the recorded floor, and
                       # the refined arm is only legitimate if the refine step really
                       # wrote the column (see `concept` below).
                       "vector_column": engines["system_B"].vector_column,
                       "concept": concept,
                       "config_hash": cfg.hash(),
                       "kinds": _tally(items, "kind"),
                       "generated_at": utcnow().isoformat()}
    hard = _hard_negative_eval(engines, cfg, ledger, holdout, seed)
    results["hard_negatives"] = hard

    cmp = _compare(results["baseline_A"], results["system_B"])
    # saturation guard: report the arm's headroom next to its delta, so "no
    # gain" can never again be read as "no effect" without checking that the
    # split contained the failure the layer is supposed to fix.
    cmp["headroom_pct"] = {name: results[name]["headroom_pct"]
                           for name in ("baseline_A", "a_scoped", "system_B")
                           if name in results}
    hr = cmp["headroom_pct"].get("a_scoped")
    if hr is not None and hr < 0.10:
        cmp["instrument_warning"] = (
            f"LOW HEADROOM: only {hr:.1%} of scored questions had stale rank-1 "
            "after scoping, so this split cannot measure the relation layer in "
            "either direction (neither a win nor a null is evidence here)")
    cmp["decomposition"] = _decompose(results, cmp)
    results["comparison"] = cmp
    (out_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(results), encoding="utf-8")
    results["_out_dir"] = str(out_dir)
    _release(engines)
    engines.clear()
    shared_embedder = shared_reranker = None
    return results


def _release(engines: dict[str, Engine]) -> None:
    """Close every arm's store and hand the GPU blocks back.

    Without this, `run_bench_seeds` left each seed's models alive while the next
    seed allocated its own, so a 3-seed cross-encoder run stacked loads on an
    8 GB card instead of reusing one.
    """
    for eng in engines.values():
        eng.close()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


ARMS: dict[str, tuple[bool, bool]] = {
    "baseline_A": (False, False),   # 10 row 4: dense+BM25+RRF, no metadata filter
    "a_scoped": (False, True),       # 10 row 6: + project/status filters, no linker
    "system_B": (True, True),        # 10 row 7: + typed relations
}


def _score_system(eng: Engine, items: list[dict], gmap: dict[str, list[str]],
                  ledger: Ledger, cfg, *, arm: str = "system_B") -> dict[str, Any]:
    linker_on, scoping_on = ARMS[arm]
    # Headroom must be judged against the LEDGER's supersede truth, not against
    # the arm's own store: the row-4/row-6 store is built with the linker off, so
    # no unit is ever marked superseded there and headroom would read 0.0 by
    # construction — the first version of this metric did exactly that and made a
    # split look untestable when it was only measured wrong.
    gold_stale = {u for e in ({ev.supersedes for ev in ledger.events
                               if ev.supersedes}) for u in gmap.get(e, [])}
    per: dict[str, list[float]] = {k: [] for k in
                                   ("recall@1", "recall@5", "recall@10", "mrr@10",
                                    "ndcg@10", "temporal@1", "stale@10",
                                    "hyp_leak", "gold_in_pool", "abstain_correct",
                                    # headroom: questions where a STALE unit holds
                                    # rank-1, i.e. the only cases a relation layer
                                    # can improve. 0 headroom => the metric is
                                    # saturated and says nothing about the thesis.
                                    "headroom")}
    groups: list[str] = []
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in items:
        gold_units = {u for e in q["gold_event_ids"] for u in gmap.get(e, [])}
        superseded = {u for e in q.get("gold_superseded_event_ids", [])
                      for u in gmap.get(e, [])}
        t0 = time.perf_counter()
        hits = eng.retrieve(Query(text=q["question"], project=q["project"],
                                  mode="history" if q["kind"] == "history" else "current"),
                            linker_enabled=linker_on, scoping=scoping_on)
        latencies.append(time.perf_counter() - t0)
        ranked = [h.unit.unit_id for h in hits]
        if hits and gold_units:
            per["headroom"].append(1.0 if (ranked[0] in gold_stale
                                           and ranked[0] not in gold_units)
                                   else 0.0)
        else:
            per["headroom"].append(float("nan"))
        soft = {h.unit.unit_id for h in hits
                if h.unit.status in ("proposed", "hypothesis", "conditional")}
        is_cur = q["kind"] in ("current_decision", "proposal_vs_decision", "owner", "date")
        rec1 = M.recall_at_k(gold_units, ranked, 1) if gold_units else float("nan")
        per["recall@1"].append(rec1)
        per["recall@5"].append(M.recall_at_k(gold_units, ranked, 5) if gold_units else float("nan"))
        per["recall@10"].append(M.recall_at_k(gold_units, ranked, 10) if gold_units else float("nan"))
        per["mrr@10"].append(M.mrr_at_10(gold_units, ranked) if gold_units else float("nan"))
        per["ndcg@10"].append(M.ndcg_at_k(gold_units, ranked, 10) if gold_units else float("nan"))
        per["gold_in_pool"].append(M.gold_in_pool(gold_units, ranked) if gold_units else float("nan"))
        per["stale@10"].append(M.stale_rate_at_k(ranked, superseded, 10) if superseded
                               else float("nan"))
        t1 = M.temporal_correctness_at_1(gold_units, ranked, superseded) if is_cur else None
        per["temporal@1"].append(t1 if t1 is not None else float("nan"))
        leak = M.hypothesis_leak(ranked, soft, bool(gold_units)) if q["kind"] == "proposal_vs_decision" else None
        per["hyp_leak"].append(leak if leak is not None else float("nan"))
        if q["kind"] == "unanswerable":
            # ABSTENTION INSTRUMENT — deliberately not an absolute RRF threshold.
            # RRF scores live around 1/(60+rank) ~ 0.016-0.033, so any fixed
            # `no_answer_threshold` above that would score 1.000 on every
            # question and fake a perfect abstention rate. Instead: abstention is
            # correct when nothing retrieved *shares content* with the question
            # (top-1 content-word Jaccard below tau). Non-saturated, and
            # documented in docs/LIMITS.md.
            top = hits[0] if hits else None
            if top is None:
                per["abstain_correct"].append(1.0)
            else:
                from engine.store.fts import tokenize

                qw = {t for t in tokenize(q["question"]) if len(t) > 4}
                tw = {t for t in tokenize(top.unit.text) if len(t) > 4}
                jac = len(qw & tw) / len(qw | tw) if (qw | tw) else 0.0
                per["abstain_correct"].append(1.0 if jac < 0.15 else 0.0)
        details.append({"qid": q["qid"], "kind": q["kind"], "project": q["project"],
                        "question": q["question"], "gold": sorted(gold_units),
                        "top": ranked[:5],
                        "temporal@1": t1, "gold_in_pool": (1.0 if gold_units & set(ranked) else 0.0)
                        if gold_units else None})
    out = {k: _mean(v) for k, v in per.items()}
    out["_groups"] = [i["project"] for i in items]
    # one keyed bag of per-question values; _compare reads exactly these names
    out["headroom_pct"] = _mean(per["headroom"])
    out["_values"] = {key: list(per[key]) for key in
                      ("temporal@1", "recall@5", "gold_in_pool")}
    out["latency_ms_p50"] = round(float(np.percentile(latencies, 50)) * 1000, 2) if latencies else 0.0
    out["latency_ms_p95"] = round(float(np.percentile(latencies, 95)) * 1000, 2) if latencies else 0.0
    out["per_question"] = details
    out["by_kind"] = _by_kind(details)
    return out


def _by_kind(details: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    kinds = {d["kind"] for d in details}
    for k in sorted(kinds):
        rows = [d for d in details if d["kind"] == k]
        t = [d["temporal@1"] for d in rows if d["temporal@1"] is not None]
        g = [d["gold_in_pool"] for d in rows if d["gold_in_pool"] is not None]
        out[k] = {"n": len(rows),
                  "temporal@1": _mean(t),
                  "gold_in_pool": _mean(g),
                  "mrr@10": _mean([1.0 if d["gold"] and d["top"] and d["gold"][0] == d["top"][0]
                                   else 0.0 for d in rows])}
    return out


def _hard_negative_eval(engines: dict[str, Engine], cfg, ledger: Ledger,
                        holdout: set[str], seed: int) -> dict[str, Any]:
    """9.1 status-discrimination: does the positive outrank its minimal pair?

    Only pairs whose anchor contains the discriminating cue are scored
    (`meta.scoreable is not False`); a `repeat` pair asserts the same value on
    both sides, so its gold relation is `supports` and any ranking result is a
    coin flip that would drag the headline number toward chance.
    """
    pairs = [p for p in build_hard_negatives(ledger, seed=seed, per_project=4)
             if p.project in holdout]
    scoreable = [p for p in pairs if p.meta.get("scoreable", True) and p.cue]
    out: dict[str, Any] = {"n_pairs": len(pairs), "n_scoreable": len(scoreable),
                           "by_flip": {}, "by_flip_n": {}}
    eng = engines["system_B"]
    if not scoreable:
        out["note"] = "no scoreable pairs in this split"
        return out
    embed = eng.embedder
    pos = embed.embed([p.positive for p in scoreable], role="doc")
    neg = embed.embed([p.negative for p in scoreable], role="doc")
    anchors = embed.embed([p.anchor for p in scoreable], role="query")
    pos_hit = np.sum(pos * anchors, axis=1)
    neg_hit = np.sum(neg * anchors, axis=1)
    out["mean_cosine_positive_negative"] = round(float(np.mean(
        np.sum(pos * neg, axis=1))), 4)
    ties = int(np.sum(np.isclose(pos_hit, neg_hit, atol=1e-6)))
    out["ties"] = ties
    out["anchor_pairwise_accuracy"] = round(float(np.mean(pos_hit > neg_hit)), 4)
    by_flip: dict[str, list[float]] = {}
    for p, a, b in zip(scoreable, pos_hit, neg_hit):
        by_flip.setdefault(p.flip, []).append(1.0 if a > b else 0.0)
    out["by_flip"] = {k: _mean(v) for k, v in sorted(by_flip.items())}
    out["by_flip_n"] = {k: len(v) for k, v in sorted(by_flip.items())}
    # chance level with n<8 per flip is not evidence either way; say so
    out["min_n_per_flip"] = min(out["by_flip_n"].values(), default=0)
    out["interpret_with_caution"] = bool((out["min_n_per_flip"] or 0) < 12)
    return out


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    groups = a["_groups"]
    out: dict[str, Any] = {}
    for key in ("temporal@1", "recall@5", "gold_in_pool"):
        va, vb = a["_values"][key], b["_values"][key]
        pt, lo, hi = M.bootstrap_ci(vb, groups, n=1000)
        paired = M.paired_bootstrap_diff(va, vb, groups, n=1000)
        out[key] = {"baseline_A": a[key], "system_B": b[key],
                    "B_mean": pt, "B_ci95": [lo, hi],
                    "delta": paired["delta"], "delta_ci95": [paired["lo"], paired["hi"]],
                    "significant_win": bool(paired["significantly_better"] > 0)}
    out["stale@10"] = {"baseline_A": a["stale@10"], "system_B": b["stale@10"]}
    out["hyp_leak"] = {"baseline_A": a["hyp_leak"], "system_B": b["hyp_leak"]}
    out["latency_ms_p95"] = {"baseline_A": a["latency_ms_p95"],
                             "system_B": b["latency_ms_p95"]}
    out["instrument_warning"] = (
        "gold_in_pool too low to read temporal@1" if (a["gold_in_pool"] < 0.5
                                                      or b["gold_in_pool"] < 0.5) else "")
    return out


def _dec_str(d: dict | None) -> str:
    """Render one decomposition line from pack()'s shape (delta + ci95)."""
    if not d:
        return "not measured"
    ci = d.get("ci95") or [None, None]
    return (f"\u0394 {d.get('delta')} (95% CI {ci[0]}\u2026{ci[1]}, "
            f"significant={d.get('significant_win')}, over {d.get('n_projects')} "
            f"projects)")


def _mean(vals: list[float | None]) -> float:
    clean = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(sum(clean) / len(clean), 4) if clean else float("nan")


def _tally(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i[key]] = out.get(i[key], 0) + 1
    return out


def _summary(res: dict[str, Any]) -> str:
    meta = res["meta"]
    a, b = res["baseline_A"], res["system_B"]
    cmp = res["comparison"]
    L = [f"# runs/{EXP_ID} — Baseline A vs System B", "",
         f"- seed {meta['seed']} | {meta['n_projects']} projects | "
         f"{meta['n_questions']} questions on holdout ({meta['split']}) split",
         f"- embedder `{meta['embed_version']}` | linker `{meta.get('linker_backend')}` "
         f"| config hash `{meta['config_hash']}` | "
         f"ingest {res['ingest_seconds']} s",
         f"- question kinds: {meta['kinds']}", "",
         "## Instrument check (read this before the headline)", "",
         f"- gold_in_pool: A={a['gold_in_pool']} B={b['gold_in_pool']} "
         f"(temporal@1 is only meaningful if this is high)",
         f"- {cmp['instrument_warning'] or 'instrument has headroom'}", "",
         "## Headline (mean over projects, 95% CI paired bootstrap)", "",
         "| metric | Baseline A | System B | Δ | CI95 | win? |",
         "|---|---|---|---|---|---|"]
    for key in ("temporal@1", "recall@5", "gold_in_pool"):
        c = cmp[key]
        L.append(f"| {key} | {c['baseline_A']} | {c['system_B']} | {c['delta']:+} | "
                 f"[{c['delta_ci95'][0]:+}, {c['delta_ci95'][1]:+}] | "
                 f"{'YES' if c['significant_win'] else 'no'} |")
    dec = (cmp.get("decomposition") or {})
    if dec:
        dt = dec.get("temporal@1") or {}
        L += ["", "## Decomposition — is it the filters or the relations? (10 rows 4/6/7)",
              "",
              "temporal@1 across the three arms, same corpus and candidate budget:",
              "",
              f"| arm | temporal@1 | recall@5 |", "|---|---|---|",
              f"| row 4 · dense+BM25+RRF (Baseline A) | {dt.get('baseline_A')} | "
              f"{(dec.get('recall@5') or {}).get('baseline_A')} |",
              f"| row 6 · + metadata scoping, **no linker** | {dt.get('a_scoped')} | "
              f"{(dec.get('recall@5') or {}).get('a_scoped')} |",
              f"| row 7 · + typed relations (System B) | {dt.get('system_B')} | "
              f"{(dec.get('recall@5') or {}).get('system_B')} |",
              "",
              "- attributed to **metadata scoping**: "
              f"{_dec_str(dec.get('from_metadata_scoping'))}",
              "- attributed to **the typed-relation layer itself**: "
              f"{_dec_str(dec.get('from_typed_relations'))}",
              "",
              f"- **headroom** (stale unit at rank-1): row 4 "
              f"{(cmp.get('headroom_pct') or {}).get('baseline_A')}, row 6 "
              f"{(cmp.get('headroom_pct') or {}).get('a_scoped')}, row 7 "
              f"{(cmp.get('headroom_pct') or {}).get('system_B')} — the second "
              "figure bounds what the relation layer could possibly fix; if it is "
              "~0, the split cannot test this claim at all.",
              "The bundled A→B number overstates the relation layer, because it also "
              "contains the win from simply filtering by project. The claim worth "
              "defending is the second line, read only where headroom is non-zero.", ""]
    L += ["", "## Secondary", "",
          f"- stale@10: A={a['stale@10']} → B={b['stale@10']}",
          f"- hypothesis-leak (proposal_vs_decision): A={a['hyp_leak']} → B={b['hyp_leak']}",
          f"- latency p95 (ms, LLM off): A={a['latency_ms_p95']} → B={b['latency_ms_p95']}",
          f"- retrieval MRR@10: A={a['mrr@10']} → B={b['mrr@10']}",
          f"- nDCG@10: A={a['ndcg@10']} → B={b['ndcg@10']}", "",
          "## Hard negatives (constructed minimal pairs)", ""]
    hn = res["hard_negatives"]
    L += [f"- pairs: {hn['n_pairs']} | mean cos(pos,neg) {hn.get('mean_cosine_positive_negative')}",
          f"- anchor pairwise accuracy: {hn.get('anchor_pairwise_accuracy')}",
          "- by flip: " + ", ".join(f"{k}={v}" for k, v in hn["by_flip"].items()), "",
          "## Per question kind", "",
          "| kind | n | temporal@1 A | temporal@1 B | gold_in_pool A | gold_in_pool B |",
          "|---|---|---|---|---|---|"]
    for k in sorted(set(a["by_kind"]) | set(b["by_kind"])):
        ka, kb = a["by_kind"].get(k, {}), b["by_kind"].get(k, {})
        L.append(f"| {k} | {ka.get('n', kb.get('n', 0))} | {ka.get('temporal@1')} | "
                 f"{kb.get('temporal@1')} | {ka.get('gold_in_pool')} | {kb.get('gold_in_pool')} |")
    return "\n".join(L) + "\n"


def run_bench_seeds(seeds: list[int], *, n_projects: int = 20, split: str = "test",
                    embed_backend: str | None = None,
                    embed_model: str | None = None,
                    reader_backend: str | None = None,
                    rerank_backend: str | None = None,
                    vector_column: str | None = None) -> dict[str, Any]:
    """8: three seeds for every headline number (mean +/- spread).

    Shared by `sae bench` and `python -m eval.temporal` so the two entry points
    cannot drift apart.
    """
    per: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        res = run_bench(n_projects=n_projects, seed=seed, split=split,
                        embed_backend=embed_backend, embed_model=embed_model,
                        reader_backend=reader_backend,
                        rerank_backend=rerank_backend,
                        vector_column=vector_column)
        per[seed] = res
    keys = ("baseline_A", "system_B", "delta")
    mean = {k: round(sum(r["comparison"]["temporal@1"][k] for r in per.values())
                     / len(per), 4) for k in keys}
    wins = sum(1 for r in per.values()
               if r["comparison"]["temporal@1"]["significant_win"])
    spread = round(max(r["comparison"]["temporal@1"]["system_B"] for r in per.values())
                   - min(r["comparison"]["temporal@1"]["system_B"] for r in per.values()), 4)
    #: provenance read back off each run's own meta, not from the arguments:
    #: `rerank_backend=None` means "whatever the config said", so only the
    #: artefact knows whether this was the shipped configuration.
    metas = [r.get("meta") or {} for r in per.values()]
    readers = sorted({str(m.get("reader_backend", "?")) for m in metas})
    reranks = sorted({str(m.get("rerank_backend", "?")) for m in metas})
    columns = sorted({str(m.get("vector_column", "base")) for m in metas})
    provenance = {"split": split, "seeds": list(seeds),
                  "reader_backend": readers[0] if len(readers) == 1 else readers,
                  "rerank_backend": reranks[0] if len(reranks) == 1 else reranks,
                  "vector_column": columns[0] if len(columns) == 1 else columns,
                  "n_projects": n_projects}
    # only the shipped configuration may overwrite the headline artefacts. An
    # ablation (a reranker, the trained reader, a dev split) writes a variant
    # file and leaves the gate alone: `last_good.json` was briefly left holding a
    # dev + cross-encoder number, which `make eval-fast` would have graded every
    # later run against.
    # The project count is part of "shipped" too: the 8-project tripwire is a
    # different, easier corpus, and a 3-seed run of it used to qualify as
    # canonical and rewrite the headline with its own smaller numbers.
    canonical = (split == "test" and len(seeds) >= 3
                 and n_projects == CANONICAL_PROJECTS
                 and readers == ["rules"] and reranks == ["none"]
                 # 6.13's row 9 is a different system, not the shipped one: a
                 # refined run at test x 3 seeds x 20 projects with rules+none
                 # satisfies every other clause, and without this it would rewrite
                 # `temporal_mean.json` and `last_good.json` with its own numbers and
                 # the recorded floor would silently become the experiment.
                 and columns == ["base"])
    out = {"seeds": list(seeds), "n_projects": n_projects, "split": split,
           "provenance": provenance, "canonical": canonical,
           "temporal@1_mean": mean, "temporal@1_spread": spread,
           "seeds_with_significant_win": wins,
           "per_seed": {str(s): r["comparison"] for s, r in per.items()},
           "per_seed_recall@5": {str(s): r["comparison"]["recall@5"]
                                 for s, r in per.items()}}
    runs_dir = repo_root() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tag = f"__{split}__r-{'-'.join(readers)}__k-{'-'.join(reranks)}"
    if columns != ["base"]:
        tag += f"__v-{'-'.join(columns)}"
    path = (runs_dir / "temporal_mean.json" if canonical
            else runs_dir / f"temporal_mean{tag}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # a copy that `make eval-fast` can diff against as the regression gate
    gate = repo_root() / "baselines" / "last_good.json"
    prev = json.loads(gate.read_text(encoding="utf-8")) if gate.exists() else None
    if canonical:
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(json.dumps({"temporal@1_mean": mean, "seeds": list(seeds),
                                    "provenance": provenance,
                                    "per_seed_recall@5": out["per_seed_recall@5"]},
                                   indent=1), encoding="utf-8")
    out["_previous_gate"] = prev if canonical else None
    out["_out"] = str(path)
    return out


def _decompose(results: dict[str, Any], cmp: dict[str, Any]) -> dict[str, Any]:
    """Split the headline delta into metadata-scoping vs typed-relations.

    Reported because the two are different claims: "filters help" is not
    "the relation layer is what makes temporal correctness work", and bundling
    them overstates the second one.
    """
    vals = {k: results[k]["_values"]["temporal@1"]
            for k in ("baseline_A", "a_scoped", "system_B")}
    groups = results["system_B"]["_groups"]
    filters = M.paired_bootstrap_diff(vals["baseline_A"], vals["a_scoped"], groups,
                                      n=1000)
    relations = M.paired_bootstrap_diff(vals["a_scoped"], vals["system_B"], groups,
                                        n=1000)
    def pack(d: dict[str, float]) -> dict[str, Any]:
        # paired_bootstrap_diff returns significantly_better/worse; asking it for
        # "significant_win" raised KeyError and killed the whole decomposition.
        return {"delta": d.get("delta"), "ci95": [d.get("lo"), d.get("hi")],
                "significant_win": bool(d.get("significantly_better")),
                "n_projects": d.get("n_projects")}

    return {
        "temporal@1": {k: _mean(v) for k, v in vals.items()},
        "from_metadata_scoping": pack(filters),
        "from_typed_relations": pack(relations),
        "recall@5": {k: results[k]["recall@5"] for k in
                     ("baseline_A", "a_scoped", "system_B")},
        "headline_is": "temporal@1 of row 4 -> row 7; the relation layer's own "
                       "share is `from_typed_relations`, and is only interpretable "
                       "where headroom > 0",
        "headroom_pct": {k: v for k, v in (results.get("headroom_pct", {}) or {}).items()}
        if isinstance(results.get("headroom_pct"), dict) else results.get("headroom_pct"),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="", help="comma list; overrides --seed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embed", default=None, help="hash | st")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reader", default=None, choices=["rules", "model"],
                    help="reader backend; 'model' needs models/reader to exist, "
                         "and meta.reader_backend says which one actually ran")
    ap.add_argument("--rerank", default=None,
                    choices=["none", "overlap", "cross_encoder"],
                    help="reranker backend; meta.rerank_backend says which one "
                         "the built pipeline actually holds")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        agg = run_bench_seeds(seeds, n_projects=args.projects, split=args.split,
                              embed_backend=args.embed, embed_model=args.model,
                              reader_backend=args.reader,
                              rerank_backend=args.rerank)
        print(json.dumps({"temporal@1_mean": agg["temporal@1_mean"],
                          "spread": agg["temporal@1_spread"],
                          "seeds_with_significant_win":
                              agg["seeds_with_significant_win"]}, indent=1))
        print("wrote", agg["_out"])
    else:
        res = run_bench(n_projects=args.projects, seed=args.seed, split=args.split,
                        embed_backend=args.embed, embed_model=args.model,
                        reader_backend=args.reader, rerank_backend=args.rerank,
                        limit_questions=args.limit)
        print(json.dumps({k: res[k] for k in ("meta", "comparison")}, indent=1)[:2000])
        print("\nwrote", res["_out_dir"])
