"""Training loop for concept attention (6.13) — train split only, dev-selected.

§9.5 forbids tuning or selecting on test, and §0.5 splits by *project*, so a
context set's centre is drawn only from train projects. Neighbours still come
from the whole corpus, because that is what the live index will contain: filtering
them to train projects would train against a smaller space than the one being
measured and the A/B would no longer be the same experiment.

**The checkpoint stores the embedder version, and refinement refuses a mismatch.**
A refined vector is a small perturbation of the *base* vector it was computed
from. Re-running a checkpoint against a store embedded by a different model would
write vectors that are anchored to a space that no longer exists, and retrieval —
which embeds the query with the live embedder and searches the refined index —
would degrade while every offline number still looked healthy. The guard is the
only thing that makes `refined` trustworthy, so it is an error rather than a
warning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engine.concept import losses as L
from engine.concept.data import ContextSet, build_sets, label_counts
from engine.concept.model import ConceptAttention, ConceptConfig, log_gap_days
from engine.config import repo_root

CHECKPOINT_VERSION = 1


def _pad(sets: list[ContextSet], *, max_n: int | None = None) -> dict[str, Any]:
    """Collage context sets into one padded batch.

    Padding index is always -1 ("no such neighbour"), never 0: index 0 is the
    centre, so padding with 0 would silently pair every set with itself and the
    losses would report an easy, meaningless win.
    """
    n = max_n or max(s.n for s in sets)
    m = max(1, max((len(s.positives) for s in sets), default=1))
    k = max(1, max((len(s.hard_negatives) for s in sets), default=1))
    dim = sets[0].base.shape[1]
    base = np.zeros((len(sets), n, dim), dtype="float32")
    mask = np.zeros((len(sets), n), dtype="int64")
    src = np.zeros((len(sets), n), dtype="int64")
    kind = np.zeros((len(sets), n), dtype="int64")
    typ = np.zeros((len(sets), n), dtype="int64")
    stat = np.zeros((len(sets), n), dtype="int64")
    bucket = np.zeros((len(sets), n), dtype="int64")
    loggap = np.zeros((len(sets), n), dtype="float32")
    posn = np.zeros((len(sets), n), dtype="int64")
    pos_idx = np.full((len(sets), m), -1, dtype="int64")
    neg_idx = np.full((len(sets), k), -1, dtype="int64")
    valid = np.full((len(sets), n), -1.0, dtype="float32")
    for b, s in enumerate(sets):
        take = min(s.n, n)
        base[b, :take] = s.base[:take]
        mask[b, :take] = 1
        src[b, :take] = s.src[:take]
        kind[b, :take] = s.kind[:take]
        typ[b, :take] = s.type[:take]
        stat[b, :take] = s.status[:take]
        posn[b, :take] = np.arange(take)
        for i, g in enumerate(s.gap_days[:take]):
            loggap[b, i] = log_gap_days(g)
            # bucketed age: 0 = unknown, then doubling day-windows (1,2,4,8,...)
            if g is None:
                bucket[b, i] = 0
            else:
                bucket[b, i] = min(8, int(np.floor(np.log1p(abs(g)) / np.log(2))) + 1)
        for i, j in enumerate(s.positives[:m]):
            if j < take:
                pos_idx[b, i] = j
        for i, j in enumerate(s.hard_negatives[:k]):
            if j < take:
                neg_idx[b, i] = j
        for j, v in s.validity.items():
            if j < take:
                valid[b, j] = v
    t = torch.as_tensor
    return {"x": t(base), "src": t(src), "kind": t(kind), "type": t(typ),
            "status": t(stat), "bucket": t(bucket), "log_gap": t(loggap),
            "pos": t(posn), "mask": t(mask), "positives": t(pos_idx),
            "negatives": t(neg_idx), "validity": t(valid)}


def _prefix(batch: dict[str, Any]) -> dict[str, Any]:
    """The same batch with the last neighbour slot removed.

    6.13's incremental-vs-full consistency: a corpus grows meeting by meeting, and
    retrieval reads whatever the last refine job wrote. If the centre's vector
    moves when an unrelated neighbour arrives, the stored space depends on the
    order it was built in.
    """
    out = dict(batch)
    n = batch["x"].shape[1]
    if n <= 2:
        return batch
    keep = n - 1
    out["x"] = batch["x"][:, :keep]
    for key in ("src", "kind", "type", "status", "bucket", "log_gap", "pos", "mask"):
        out[key] = batch[key][:, :keep]
    # a neighbour that no longer exists in the shortened set cannot be a pair:
    # re-map it to -1 rather than letting the index point at a padded row
    for key in ("positives", "negatives"):
        idx = batch[key].clone()
        idx[idx >= keep] = -1
        out[key] = idx
    out["validity"] = batch["validity"][:, :keep]
    return out


def train(store, *, cfg: ConceptConfig, seed: int = 42, epochs: int = 8,
          lr: float = 3e-4, batch_size: int = 8, k_neighbors: int = 16,
          weights: dict[str, float] | None = None, temperature: float = 0.07,
          triplet_margin: float = 0.3, max_norm: float = 0.25,
          train_projects: set[str] | None = None,
          dev_projects: set[str] | None = None,
          device: str | None = None, verbose: bool = False) -> dict[str, Any]:
    """Fit on train, select the epoch by dev loss, return the model + report."""
    from engine.seeding import seed_all

    from engine.concept.losses import TERMS

    # The real guard here is the unknown-key check below, not this line. Measured:
    # `dict(w) if w else {}` and `dict(w) if w is not None else {}` agree on every
    # case that can arrive (None, {}, a dict of zeros — the last is non-empty, hence
    # truthy), so this is spelled `is not None` only to say "absent means defaults".
    #
    # A weights dict is honoured only if every key names a real term: `total()`
    # looks weights up with `.get(name, 0.0)`, so a typo (`driftt: 5.0`) would
    # otherwise train with the drift hinge silently disabled -- the inert-knob bug
    # (D-011) in a config's clothes, and here it disables the very constraint that
    # keeps the refined space usable by base-space queries.
    weights = dict(weights) if weights is not None else {}
    unknown = sorted(set(weights) - set(TERMS))
    if unknown:
        raise ValueError(f"unknown concept loss term(s) {unknown}; valid: {list(TERMS)}")
    defaults = {"contrastive": 1.0, "validity": 1.0, "incremental": 0.5,
                "state": 0.5, "drift": 1.0}
    missing = sorted(set(TERMS) - set(weights))
    for name in missing:
        weights[name] = defaults[name]
    seed_all(seed)
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    restrict = None
    if train_projects is not None:
        restrict = {u.unit_id for u in store.all_units()
                    if (u.project or "") in train_projects}
        # A name that matches nothing must stop the run. Falling through to
        # `restrict=None` would train on every project including test, and the
        # curve would still look healthy — this project already had a split gate
        # silently bypassed by a project-count mismatch in a directory name.
        if not restrict:
            raise ValueError(
                f"train_projects matched no stored units: {sorted(train_projects)} "
                f"(projects present: "
                f"{sorted({u.project for u in store.all_units() if u.project})})")
    sets = build_sets(store, k=k_neighbors, restrict=restrict)
    counts = label_counts(sets)
    if not sets:
        raise RuntimeError("no context sets: the store has no vectors to refine")
    # built once, outside the epoch loop: reconstructing dev context sets per epoch
    # re-runs kNN over the whole corpus for no information, and because the metric
    # it feeds is the one that selects `best_epoch`, a moving dev set would make the
    # selection depend on how many epochs were requested.
    dev_sets: list[ContextSet] = []
    if dev_projects is not None:
        dev_restrict = {u.unit_id for u in store.all_units()
                        if (u.project or "") in dev_projects}
        if not dev_restrict:
            raise ValueError(
                f"dev_projects matched no stored units: {sorted(dev_projects)}")
        dev_sets = build_sets(store, k=k_neighbors, restrict=dev_restrict)
        if not dev_sets:
            raise ValueError(
                "dev_projects produced no context sets, so `best_epoch` would be "
                "selected on train metrics while the report says it was dev")
    rng = np.random.default_rng(seed)
    order = list(range(len(sets)))
    curve: list[dict[str, float]] = []
    model = ConceptAttention(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def run_batches(batch_sets: list[ContextSet], *, fit: bool) -> dict[str, float]:
        if fit:
            model.train()
        else:
            model.eval()
        acc: dict[str, float] = {}
        n = 0
        idx = list(range(len(batch_sets)))
        for start in range(0, len(idx), batch_size):
            chunk = [batch_sets[i] for i in idx[start:start + batch_size]]
            batch = _pad(chunk)
            batch = {k: v.to(dev) for k, v in batch.items()}
            ctx = torch.enable_grad() if fit else torch.no_grad()
            with ctx:
                out = model(batch)
                parts = {
                    "contrastive": L.contrastive(out["topic"], batch["x"], batch["mask"],
                                                 batch["positives"], batch["negatives"],
                                                 temperature=temperature),
                    "validity": L.validity(out["validity"], batch["validity"],
                                           batch["mask"]),
                    "incremental": L.incremental(out["topic"][:, :1],
                                                 _incremental_prefix(model, batch, dev),
                                                 batch["mask"][:, :1]),
                    "state": L.triplet(out["state"], batch["mask"], batch["positives"],
                                       batch["negatives"], margin=triplet_margin),
                    "drift": L.drift(out["topic"], batch["x"], batch["mask"],
                                     max_norm=max_norm)}
                tot, seen = L.total(parts, weights)
                if fit:
                    opt.zero_grad()
                    tot.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            for key in ("contrastive", "validity", "incremental", "state", "drift",
                        "total"):
                acc[key] = acc.get(key, 0.0) + seen[key]
            n += 1
        return {k: round(v / max(1, n), 5) for k, v in acc.items()}

    for epoch in range(epochs):
        rng.shuffle(order)
        train_metrics = run_batches([sets[i] for i in order], fit=True)
        dev_metrics = (run_batches(dev_sets, fit=False) if dev_sets else train_metrics)
        curve.append({"epoch": epoch, "train": train_metrics, "dev": dev_metrics})
        if verbose:
            print(f"epoch {epoch} train={train_metrics['total']:.4f} "
                  f"dev={dev_metrics['total']:.4f}")
    best = min(curve, key=lambda c: c["dev"]["total"])
    return {"model": model, "device": str(dev), "config": cfg,
            "labels": counts, "curve": curve, "best_epoch": best["epoch"],
            "weights": weights, "seed": seed, "epochs": epochs,
            "train_projects": sorted(train_projects or set()),
            "dev_projects": sorted(dev_projects or set()),
            "n_train_sets": len(sets), "n_dev_sets": len(dev_sets),
            "embed_model_version": store.embed_version}


def _incremental_prefix(model, batch: dict[str, Any], dev) -> torch.Tensor:
    """The centre's topic vector from the shortened context set."""
    short = _prefix(batch)
    with torch.no_grad():
        out = model(short)
    return out["topic"][:, :1].detach().to(dev)


def save(model: ConceptAttention, report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg: ConceptConfig = report["config"]
    # `config` and `model` are dropped from meta and re-added as plain data below:
    # a dataclass instance inside the blob is what forces `weights_only=False`,
    # which turns every checkpoint load into arbitrary code execution.
    meta = {k: v for k, v in report.items() if k not in ("model", "config")}
    meta["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    from dataclasses import asdict

    # `asdict`, not a hand-written field list: the manual list predated
    # `max_step`, so a checkpoint would have silently restored a *different*
    # movement bound than the one it was trained under — and `max_step` is the
    # constraint the whole anchoring guarantee rests on, so losing it loses the
    # property that makes the refined column trustworthy at all.
    torch.save({"version": CHECKPOINT_VERSION,
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg),
                "meta": meta},
               path)
    return path


def load(path: Path, *, expect_embed_version: str | None = None
         ) -> tuple[ConceptAttention, dict[str, Any]]:
    """Load a checkpoint, refusing one anchored to a different base space."""
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {blob.get('version')!r} != "
                         f"{CHECKPOINT_VERSION}")
    cfg = ConceptConfig(**blob["config"])
    model = ConceptAttention(cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    meta = blob.get("meta", {})
    trained_on = meta.get("embed_model_version")
    if expect_embed_version and trained_on and trained_on != expect_embed_version:
        raise ValueError(
            f"refusing to refine: the checkpoint was trained against base vectors "
            f"from {trained_on!r} but this store uses {expect_embed_version!r}. "
            f"Refined vectors are perturbations of the base they came from, so "
            f"mixing the two would write an anchored-to-nothing space. Retrain "
            f"against the current embedder (6.13).")
    return model, meta


def default_checkpoint_path(name: str = "concept") -> Path:
    return repo_root() / "models" / "concept" / f"{name}.pt"


__all__ = ["train", "save", "load", "_pad", "_prefix", "default_checkpoint_path",
           "CHECKPOINT_VERSION"]
