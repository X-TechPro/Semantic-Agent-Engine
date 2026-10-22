"""The refine job (6.14): write `vectors.refined` and rebuild that index.

Runs offline, never online. The online path embeds with the frozen encoder; this
job reads the base vectors, forms each unit's context set, and stores the
transformer's topic vector beside them. `vectors.base` is never touched, so
switching back to the frozen space is a config value, not a re-embed.

**Determinism comes from the base space.** A context set is the unit's k nearest
neighbours *in the base vectors*, so the refined vector of a unit is a function of
the base index alone — not of the order units were ingested, and not of the
refined vectors themselves. A self-referential refine (neighbours taken from the
refined space) would drift a little further on every nightly run and make two
identical runs incomparable, which is the failure mode the drift bound exists for.

Reports `drift` and `n_without_base` on purpose: those are the two ways this job
can produce a healthy-looking index that retrieval cannot actually use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from engine.concept.data import build_sets
from engine.concept.model import ConceptAttention
from engine.concept.train import _pad


def refine(store, model: ConceptAttention, *, k: int = 16, batch_size: int = 16,
           device: str | None = None, writes: bool = True,
           max_drift: float = 0.25) -> dict[str, Any]:
    """Compute and store refined topic vectors for every unit that has a base one.

    `writes=False` measures the job without touching the store — used by the
    ablation and by tests, so "what would this do" can be answered before it is
    done to the index.

    `max_drift` is the same bound the training loss hinges on, and it is the number
    that decides whether the anchoring gate below can ever pass: a job asked to
    refine with a 0.25 bound after training against 10.0 would refuse every
    checkpoint it was handed.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ids, base = store.vector_matrix("base")
    if not ids:
        return {"n_units": 0, "n_refined": 0, "note": "no base vectors to refine"}
    model.to(dev).eval()
    sets = build_sets(store, k=k)
    by_centre = {s.centre_id: s for s in sets}
    refined: dict[str, np.ndarray] = {}
    queue = [by_centre[uid] for uid in ids if uid in by_centre]
    for start in range(0, len(queue), batch_size):
        chunk = queue[start:start + batch_size]
        batch = {kk: vv.to(dev) for kk, vv in _pad(chunk).items()}
        with torch.no_grad():
            out = model(batch)
        # slot 0 is the centre of its own context set, so take column 0 per row
        vec = out["topic"][:, 0].to("cpu").numpy()
        for s, v in zip(chunk, vec):
            refined[s.centre_id] = v.astype("float32")

    missing = [uid for uid in ids if uid not in refined]
    report: dict[str, Any] = {"n_units": len(ids), "n_refined": len(refined),
                              "n_without_context": len(missing)}
    if refined:
        ordered = np.stack([refined[uid] for uid in ids if uid in refined])
        norms = np.linalg.norm(ordered, axis=1)
        base_ordered = np.stack([base[i] for i, uid in enumerate(ids) if uid in refined])
        drift = np.linalg.norm(ordered - base_ordered, axis=1)
        cos_floor = 1.0 - (max_drift ** 2) / 2.0
        report.update({
            "unit_length_min": round(float(norms.min()), 4),
            "unit_length_max": round(float(norms.max()), 4),
            "cosine_to_base_mean": round(float((ordered * base_ordered).sum(1).mean()), 4),
            "cosine_to_base_floor_required": round(cos_floor, 4),
            "max_drift_allowed": max_drift,
            "drift_mean": round(float(drift.mean()), 4),
            "drift_p95": round(float(np.percentile(drift, 95)), 4),
            "drift_max": round(float(drift.max()), 4),
        })
        if not np.allclose(norms, 1.0, atol=1e-3):
            report["warning"] = ("refined vectors are not unit length: IndexFlatIP "
                                 "over them is no longer cosine, so the base and "
                                 "refined arms would be measured on different metrics")
        # The anchoring gate. Retrieval embeds the query with the frozen base
        # embedder and searches this column, so a refined vector that has drifted to
        # (near) orthogonality of its base still returns k hits — a ranking that
        # looks alive and is meaningless. Measured on a 2-epoch CPU checkpoint:
        # cosine_to_base_mean 0.0105 with drift_mean 1.4063 against a 0.25 bound,
        # while both columns returned 5 hits for the same query. Refusing the write
        # is the only thing that makes that state un-ignorable.
        #
        # p95, not the mean. A 60-step toy run measured drift_mean 0.1055 (passes)
        # with drift_max 0.5310 — 2.1x the bound — so a mean-only gate lets a
        # handful of units move arbitrarily far provided the average stays quiet.
        # The hinge in `losses.drift` is per-row, so the contract is per-row too.
        anchored = (report["drift_p95"] <= max_drift
                    and report["cosine_to_base_mean"] >= cos_floor)
        report["anchored"] = bool(anchored)
        if not anchored and writes:
            report["status"] = "refused_unanchored"
            report["warning"] = (
                f"refined space is not anchored to base (cosine "
                f"{report['cosine_to_base_mean']} < required {round(cos_floor, 4)}, "
                f"drift {report['drift_mean']} > bound {max_drift}): writing this "
                f"column would serve a ranking computed against vectors the query "
                f"cannot compare to. Nothing was written; raise the drift weight or "
                f"train further (6.13).")
            writes = False
    if writes and refined:
        store.add_vectors(list(refined), np.stack([refined[u] for u in refined]),
                          kind="refined")
        report["rebuilt_index"] = store.rebuild_index("refined")
    elif not refined:
        report["rebuilt_index"] = 0
    return report


def clear_refined(store) -> int:
    """Drop every refined vector, returning the store to the frozen space.

    §0.4 keeps the measured baseline available, so undoing a refine has to be a
    command rather than a rebuild-from-scratch procedure.
    """
    with store._lock, store.db:
        n = store.db.execute("SELECT COUNT(*) c FROM vectors WHERE refined IS NOT NULL"
                             ).fetchone()["c"]
        store.db.execute("UPDATE vectors SET refined=NULL WHERE refined IS NOT NULL")
    store._index_dirty.add("refined")
    store.faiss.indexes.pop("refined", None)
    p = store.faiss.path.with_suffix(".refined.faiss")
    if p.exists():
        p.unlink()
    return int(n)


__all__ = ["refine", "clear_refined"]
