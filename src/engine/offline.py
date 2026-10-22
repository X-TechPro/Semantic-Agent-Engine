"""Offline jobs (BUILD_GUIDE 6.14).

The online path stays cheap (segment -> read -> slots -> embed -> store -> link).
This module owns batch work: consistency audit, index rebuild, the
re-embed decision against `embed_model_version`, clustering for browsing, and
the threshold re-calibration audit (B-11) — six steps, six reported outcomes.

Clustering is explicitly *not* a retrieval dependency (6.14), so it is reported
here and never consulted at query time. Neither is the recalibration audit a
silent writer: it rebuilds the dev split the reader's calibration is *supposed*
to live on and reports drift; an `apply` writes a CANDIDATE checkpoint only,
never a shipped one.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import repo_root


def consistency_audit(store) -> dict[str, Any]:
    """Check the 6.5 invariants over the live store."""
    problems: list[str] = []
    units = store.all_units()
    ids = {u.unit_id for u in units}
    heads: dict[tuple[str, str | None], list[str]] = {}
    for u in units:
        if u.valid_to and u.valid_from and u.valid_to < u.valid_from:
            problems.append(f"{u.unit_id}: valid_to < valid_from")
        if u.superseded_by and u.superseded_by not in ids:
            problems.append(f"{u.unit_id}: superseded_by dangles -> {u.superseded_by}")
        if u.valid_to and not u.superseded_by:
            problems.append(f"{u.unit_id}: valid_to without superseded_by")
        if u.is_current and u.status == "confirmed" and u.kind == "info":
            heads.setdefault((u.project or "", u.slots.subject), []).append(u.unit_id)
    for row in store.db.execute("SELECT from_id, to_id FROM edges").fetchall():
        for side in (row["from_id"], row["to_id"]):
            if side not in ids:
                problems.append(f"edge endpoint missing: {side}")
    # >1 in-force confirmed for a slot is only legal when the pair is an explicit
    # conflict (6.5 "unless explicitly marked parallel"), so subtract those.
    # Witnesses of the SAME value are also legal: a restatement is labelled
    # `supports`, not superseded, so two rows can legitimately hold one state.
    # Counting them separately flagged 22 clean slots as violations (measured:
    # "On valide 180 k€" + "je remets une couche, le tarif = 180 k€").
    conflicts = {frozenset((r["from_id"], r["to_id"])) for r in store.db.execute(
        "SELECT from_id, to_id FROM edges WHERE relation='contradicts'")}
    supports = {frozenset((r["from_id"], r["to_id"])) for r in store.db.execute(
        "SELECT from_id, to_id FROM edges WHERE relation='supports'")}
    by_id = {u.unit_id: u for u in units}
    multi_head: list[dict[str, Any]] = []
    for key, head_ids in heads.items():
        if len(head_ids) <= 1:
            continue
        unexplained = [h for h in head_ids
                       if not any(h in c for c in conflicts)]
        # collapse heads that agree on the value signature
        reps: list[str] = []
        for h in unexplained:
            if any(h in s and _same_value(by_id.get(h), by_id.get(r)) for r in reps
                   for s in supports if {h, r} <= s):
                continue
            if any(_same_value(by_id.get(h), by_id.get(r)) for r in reps):
                continue
            reps.append(h)
        if len(reps) > 1:
            multi_head.append({"project": key[0], "subject": key[1],
                               "heads": sorted(reps)})
    return {"n_units": len(units), "problems": sorted(problems),
            "multi_head_slots": multi_head,
            "ok": not problems and not multi_head}


def _same_value(a, b) -> bool:
    if a is None or b is None:
        return False
    for f in ("date_value", "amount"):
        va, vb = getattr(a.slots, f), getattr(b.slots, f)
        if va and vb:
            return va == vb
    if a.slots.owner or b.slots.owner:
        return (a.slots.owner or "") == (b.slots.owner or "")
    import re

    na = set(re.findall(r"\d+(?:[.,]\d+)?", a.text))
    nb = set(re.findall(r"\d+(?:[.,]\d+)?", b.text))
    if na and nb:
        return na == nb
    return False


def reembed_needed(store, embedder) -> dict[str, Any]:
    rows = store.db.execute(
        "SELECT DISTINCT embed_model_version v FROM units WHERE v IS NOT NULL"
    ).fetchall()
    missing = store.db.execute(
        """SELECT COUNT(*) c FROM units u
           LEFT JOIN vectors v ON v.unit_id = u.unit_id WHERE v.unit_id IS NULL"""
    ).fetchone()["c"]
    versions = [r["v"] for r in rows]
    return {"stored_versions": versions, "current_version": embedder.version,
            "units_without_vectors": missing,
            "reembed_required": embedder.version not in versions or missing > 0}


def cluster(store, threshold: float = 0.35, n_clusters: int | None = None
            ) -> dict[str, Any]:
    """Agglomerative (cosine, average linkage) over base vectors, scored against
    the gold project labels (6.14).

    Both partition styles are reported on purpose. A distance threshold on weak
    vectors produces hundreds of near-singleton clusters, and ARI against a
    20-class labelling is ~0 for *any* such partition — reporting that alone
    would look like "clustering fails" when the metric only measured partition
    size. `n_clusters` fixes the granularity so the geometry is what varies.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)

    ids, mat = store.vector_matrix("base")
    if len(ids) < 3:
        return {"n": len(ids), "note": "too few vectors to cluster"}
    project_of = {u.unit_id: (u.project or "") for u in store.all_units()}
    labels_true = [project_of.get(i, "") for i in ids]
    n_true = len({p for p in labels_true if p})
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=None if n_clusters else threshold,
        metric="cosine", linkage="average")
    labels_pred = [int(x) for x in model.fit_predict(norm.astype("float32"))]

    by_cluster: dict[int, list[str]] = {}
    for lab, proj in zip(labels_pred, labels_true):
        by_cluster.setdefault(lab, []).append(proj)
    purity_hits = 0
    for members in by_cluster.values():
        top = max(set(members), key=members.count)
        purity_hits += members.count(top)
    k = len(set(labels_pred))
    return {
        "n": len(ids), "n_clusters": k, "n_true_projects": n_true,
        "mode": f"n_clusters={n_clusters}" if n_clusters else f"threshold={threshold}",
        "ari": round(adjusted_rand_score(labels_true, labels_pred), 4),
        "nmi": round(normalized_mutual_info_score(labels_true, labels_pred), 4),
        "purity": round(purity_hits / max(1, len(ids)), 4),
        "note": "browsing/dossier grouping aid only — not used at query time",
    }


def concept_refine(eng, *, writes: bool = True) -> dict[str, Any]:
    """6.14's optional concept-attention refine, reached from the one offline entry.

    Reports `skipped` with a reason rather than being absent, so a run that was
    *supposed* to refine and silently did not is distinguishable from one that was
    never asked to — the inert-`link.backend` pattern (D-011) in a new place.

    The retrieval column is reported beside it because `store.faiss.index_key` is
    what actually decides whether any of this is used: writing refined vectors while
    retrieval reads `base` is a no-op that looks like a completed experiment.
    """
    cc = dict(eng.cfg.get_path("concept", {}) or {})
    column = str(eng.cfg.get_path("store.faiss.index_key", "base"))
    out: dict[str, Any] = {"enabled": bool(cc.get("enabled")), "retrieval_column": column}
    if not out["enabled"]:
        out["skipped"] = "concept.enabled is false (6.13 stays opt-in until it " \
                         "beats the measured baseline, 0.4)"
        return out
    ckpt = cc.get("checkpoint") or str(repo_root() / "models" / "concept" / "concept.pt")
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root() / ckpt_path
    if not ckpt_path.exists():
        out["skipped"] = f"no checkpoint at {ckpt_path}: run `sae concept-train` first"
        out["status"] = "needs_confirmation"
        return out
    from engine.concept.refine_job import refine
    from engine.concept.train import load

    # the version guard lives in `load`: a checkpoint trained against another
    # embedder's base vectors is refused rather than allowed to write a space that
    # is anchored to nothing.
    model, meta = load(ckpt_path, expect_embed_version=eng.store.embed_version)
    report = refine(eng.store, model, k=int(cc.get("k_neighbors", 16)),
                    batch_size=int(cc.get("batch_size", 8)),
                    max_drift=float(cc.get("max_drift", 0.25)), writes=writes)
    report["checkpoint"] = str(ckpt_path)
    report["trained_embed_version"] = meta.get("embed_model_version")
    report["labels"] = meta.get("labels")
    report["best_epoch"] = meta.get("best_epoch")
    if out["retrieval_column"] != "refined":
        report["warning"] = (
            f"refined vectors written but retrieval reads `{column}`: this job "
            f"changed nothing observable until store.faiss.index_key is `refined`")
    out.update(report)
    return out


def rebuild_dev_rows(projects: int, seed: int, run_dir: Path) -> dict[str, Any]:
    """The dev half of B-11's open decision (where dev logits come from):
    REBUILD the reader's own dev split — same ledger, same seed, same
    `build_split` the trainer ran — because that is the only distribution the
    calibration is defined against. The store holds no gold, and a cached copy
    of an old split would hide exactly the drift this audit exists to catch:
    `models/reader`'s temperatures were fitted before casual variants entered
    the corpus, and the rebuild is what makes that *visible* as a number.
    """
    from engine.read.dataset import build_split
    from gen.ledger import build_ledger
    from gen.splice import write_manifest

    led = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(run_dir / "ledger_splits.json", led.projects,
                        seed=seed)["split"]
    return build_split(led, set(sp["dev"]), seed, run_dir,
                       embed_backend="hash", tag="dev")


def abstain_scan(probs: dict[str, Any], gold: dict[str, Any],
                 grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9,
                                             0.95)) -> list[dict[str, Any]]:
    """The curve a default flip would be chosen from.

    `reader.abstain_threshold` is a human-set config knob — an offline job does
    not flip engine-wide behavioural defaults — so the pass reports the number
    instead: per candidate floor, the share of dev rows the reader would refuse
    and the accuracy on what it keeps, measured on the min of the gated heads'
    top probabilities, which is exactly what `ModelReader.read` compares. Empty
    denominators are None, never 0.0."""
    import numpy as np

    from engine.read.infer import GATED_HEADS

    heads = [h for h in GATED_HEADS if h in probs and h in gold]
    if not heads:
        return []
    base = np.ones(len(np.asarray(probs[heads[0]])), dtype=bool)
    ok = np.ones_like(base)
    conf = np.full(len(base), np.inf)
    for h in heads:
        g = np.asarray(gold[h])
        p = np.asarray(probs[h])
        row_ok = (p.argmax(axis=1) == g) | (g < 0)
        ok &= row_ok
        base &= g >= 0
        conf = np.minimum(conf, p.max(axis=1))
    n = int(base.sum())
    out = []
    for t in grid:
        kept = base & (conf >= t)
        k = int(kept.sum())
        out.append({"threshold": t,
                    "abstain_rate": round((n - k) / n, 4) if n else None,
                    "kept": k,
                    "accuracy_kept": round(float(ok[kept].mean()), 4) if k else None})
    return out


def recal_accepts(current: dict, refit: dict, *, ece_gate: float = 0.08,
                  gap_tol: float = 0.05) -> list[str]:
    """Reasons not to write the refit; an empty list means accept.

    Per gated head (the ones `ModelReader.read` compares, so the ones the
    guards can feel): the refit must not be WORSE on dev ECE than the shipped
    calibration, must land inside the trainer's own honesty gates (ECE < 0.08
    and |confidence − accuracy| ≤ 0.05 — the same numbers `read/train.py`
    ships under), and a head with no gold dev rows ABSTAINS from the decision
    rather than passing it silently. The NLL non-regression invariant is
    `fit_temperature`'s own, upstream of this.
    """
    from engine.read.infer import GATED_HEADS

    reasons: list[str] = []
    for head in GATED_HEADS:
        cur, ref = current.get(head) or {}, refit.get(head) or {}
        cur_ece, new_ece = cur.get("ece"), ref.get("ece_after")
        if cur_ece is None or new_ece is None:
            reasons.append(f"{head}: unmeasurable on dev (no gold rows) — abstain")
            continue
        if new_ece > cur_ece + 1e-9:
            reasons.append(f"{head}: refit ECE {new_ece} worse than shipped {cur_ece}")
        if new_ece >= ece_gate:
            reasons.append(f"{head}: refit ECE {new_ece} still breaches gate {ece_gate}")
        gap = ref.get("confidence_gap")
        if gap is None or abs(gap) > gap_tol:
            reasons.append(f"{head}: refit confidence gap {gap} over tolerance "
                           f"{gap_tol}")
    return reasons


def recalibrate_thresholds(eng, *, projects: int = 20, seed: int = 42,
                           batch_size: int = 64, apply: bool = False,
                           run_dir: Path | None = None,
                           dev: dict[str, Any] | None = None,
                           model=None, model_dir: str | None = None
                           ) -> dict[str, Any]:
    """6.14's sixth step (B-11): re-fit the reader temperatures on the dev split
    as it exists NOW, and report whether the shipped calibration still lives on
    it. Default projects/seed match `read/train.py`'s so an untouched audit
    reproduces the training corpus's dev half exactly.

    Nothing here is a silent writer. The report names what is stale and why;
    `apply` writes a CANDIDATE checkpoint (`models/<name>_recal`) only when
    every gate in `recal_accepts` passes — a shipped checkpoint,
    `reader.backend` and `reader.abstain_threshold` are never touched by an
    offline job, because D-014 already showed what an unexamined reader knob
    does at scale. Temperature scaling cannot change argmax labels (asserted in
    the report's numbers: `current` and `refit` accuracies are equal), so
    calibration drift is about confidence and abstention — never about the
    guard-input labels themselves.
    """
    import copy as _copy
    import numpy as np

    from engine.read.infer import GATED_HEADS
    from engine.read.model import (apply_temperature, evaluate_heads,
                                   mean_confidence)

    out: dict[str, Any] = {"projects": projects, "seed": seed,
                           "reader_backend": getattr(eng, "reader_backend",
                                                     "unknown")}
    model = model if model is not None else getattr(
        getattr(eng, "reader", None), "model", None)
    if model is None:
        out["skipped"] = ("no model reader is loaded — rules has no calibration "
                          "to refresh. Audit one with `sae offline --reader "
                          "model`: B-11 measures the CHECKPOINT's calibration, "
                          "not the store's")
        out["model_name"] = None
        return out
    if model_dir is None:
        cfg_reader = (getattr(eng, "cfg", None) or {}).get("reader") or {}
        model_dir = cfg_reader.get("model_dir") or "models/reader"
    name = Path(str(model_dir)).name
    shipped = {h: float(t) for h, t in
               (getattr(model, "temperatures", {}) or {}).items()}
    out["model_name"] = name
    out["shipped_temperatures"] = shipped
    out["train_meta"] = {k: v for k, v in (getattr(model, "train_meta", {}) or {}).items()
                         if k in ("n_train", "encoder", "seed", "freeze_encoder")}

    if dev is None:
        run_dir = Path(run_dir or repo_root() / "runs" / "offline" / "recal")
        run_dir.mkdir(parents=True, exist_ok=True)
        dev = rebuild_dev_rows(projects, seed, run_dir)
    out["n_dev_rows"] = int(len(dev["texts"]))

    logits = model.logits_batch(dev["texts"], dev["speakers"], dev["contexts"],
                                batch_size=batch_size)
    probs = {h: apply_temperature(lg, shipped.get(h, 1.0))
             for h, lg in logits.items()}
    pred = {h: p.argmax(axis=1) for h, p in probs.items()}
    current = evaluate_heads(dev["labels"], pred, probs)
    for h, entry in current.items():
        if h in probs:
            keep = np.asarray(dev["labels"][h]) >= 0
            mc = round(mean_confidence(probs[h][keep]), 4) if keep.any() else None
            entry["mean_conf"] = mc
            entry["confidence_gap"] = (None if mc is None
                                        or entry.get("accuracy") is None
                                        else round(mc - entry["accuracy"], 4))
            entry["temperature"] = shipped.get(h, 1.0)
    out["current"] = current

    from engine.read.train import ECE_GATE, GAP_TOLERANCE
    out["stale"] = bool(any(
        (current.get(h) or {}).get("ece") is not None
        and ((current[h]["ece"] >= ECE_GATE)
             or (current[h].get("confidence_gap") is not None
                 and abs(current[h]["confidence_gap"]) >= GAP_TOLERANCE))
        for h in GATED_HEADS if current.get(h) is not None))

    probe = _copy.copy(model)
    probe.temperatures = {}
    refit = probe.calibrate(logits, dev["labels"])
    out["refit"] = refit
    reasons = recal_accepts(current, refit)
    out["accept"] = not reasons
    out["refuse_reasons"] = reasons
    out["abstain_scan"] = abstain_scan(probs, dev["labels"])
    out["policy"] = ("candidate-only: apply writes models/<name>_recal and "
                     "nothing else; shipped checkpoint, reader.backend and "
                     "reader.abstain_threshold are never touched here")
    if apply:
        if reasons:
            out["apply"] = "refused (see refuse_reasons): nothing written"
        else:
            probe.temperatures = {h: float(r["temperature"]) for h, r in refit.items()
                                  if r.get("temperature") is not None}
            cand = repo_root() / "models" / f"{name}_recal"
            probe.save(cand)
            out["apply"] = {"candidate": str(cand), "temperatures": probe.temperatures}
    return out


def run_offline(eng, *, writes: bool = True, threshold: float = 0.35,
                recalibrate: bool = True, apply_recalibration: bool = False,
                model_dir: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"date": date.today().isoformat()}
    out["audit"] = consistency_audit(eng.store)
    out["reembed"] = reembed_needed(eng.store, eng.embedder)
    out["concept"] = concept_refine(eng, writes=writes)
    out["rebuilt_vectors"] = eng.store.rebuild_index()
    # Report BOTH partition styles. ARI on a threshold-built partition is
    # confounded by cluster count: hundreds of near-singleton clusters score
    # ~0 against a k-class label for *any* geometry, including a perfect one.
    # The k-matched run is the fair test of whether the space groups projects.
    n_proj = len({u.project for u in eng.store.all_units() if u.project}) or None
    clusters: dict[str, Any] = {}
    try:
        clusters["threshold"] = cluster(eng.store, threshold=threshold)
        clusters["k_matched"] = cluster(eng.store, n_clusters=n_proj)
    except Exception as exc:            # sklearn missing or degenerate vectors
        clusters["error"] = f"{type(exc).__name__}: {exc}"
    out["clusters"] = clusters
    if recalibrate:
        # the sixth 6.14 step (B-11): a rules reader reports `skipped` with a
        # reason, never an absent section — an offline pass that quietly does
        # not check calibration is how a stale abstain threshold survives.
        out["recalibrate"] = recalibrate_thresholds(
            eng, apply=apply_recalibration, model_dir=model_dir)
    out["store_counts"] = eng.store.count()
    if writes:
        mname = (out.get("recalibrate") or {}).get("model_name")
        tag = f"-{mname}" if mname else ""
        path = (repo_root() / "runs" / "offline"
                / f"offline-{date.today().isoformat()}{tag}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        out["_written"] = str(path)
    return out


def main(argv=None) -> int:
    import argparse

    from engine.config import load_config
    from engine.engine import Engine

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--reader", choices=["rules", "model"], default=None)
    ap.add_argument("--reader-model", default=None,
                    help="checkpoint dir to audit (implies --reader model)")
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="skip the 6.14 sixth step (the audit is the point of "
                         "B-11; this flag exists for a broken dev corpus, not "
                         "for convenience)")
    ap.add_argument("--apply-recal", action="store_true",
                    help="write an accepted refit to models/<name>_recal — a "
                         "CANDIDATE, never the shipped checkpoint")
    args = ap.parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.store:
        overrides["store"] = {"path": args.store}
    if args.reader or args.reader_model:
        rd: dict[str, Any] = {"backend": args.reader or "model"}
        if args.reader_model:
            rd["model_dir"] = args.reader_model
        overrides["reader"] = rd
    cfg = load_config(args.config, overrides=overrides or None)
    eng = Engine(cfg, path=cfg["store"]["path"])
    try:
        print(json.dumps(run_offline(eng, writes=not args.no_write,
                                     threshold=args.threshold,
                                     recalibrate=not args.no_recalibrate,
                                     apply_recalibration=args.apply_recal,
                                     model_dir=args.reader_model),
                         indent=1, default=str))
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
