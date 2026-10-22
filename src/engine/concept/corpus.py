"""The corpus concept attention trains on — built from the non-test splits only.

§9.5 says tune and select on dev, never on test. For a *generative* stage that
rule is sharper than "don't look at test numbers": if any test project were
present in the training store, its units would become k-nearest-neighbour context
for the units being fitted, and the checkpoint would carry information about test
content even though no test label was ever read. So the store is built from
`train` and `dev` projects and **test is absent from it entirely** — not filtered
at loss time, absent from the corpus.

That leaves one honest gap to state rather than hide: at refine time the context
sets are built over whatever store is being indexed (the benchmark's holdout,
which *is* test projects). Those neighbours are used unsupervised — a unit's
neighbourhood is a function of the frozen base vectors, and no test label, gold
edge or validity target reaches the forward pass, because the model is frozen by
then. Transductive, in other words, and said out loud in the report rather than
glossed as "no leakage by construction".

The split partition itself comes from `gen.splice.write_manifest`, the same call
`eval.temporal` uses, so the projects this module trains on cannot drift away from
the projects the benchmark scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root


def build_corpus(*, n_projects: int = 20, seed: int = 42,
                 embed_backend: str | None = None,
                 embed_model: str | None = None,
                 reader_backend: str | None = None,
                 store_path: Path | None = None,
                 device: str | None = None) -> dict[str, Any]:
    """Ingest every non-test meeting into a scratch store and return it open.

    The caller closes the engine; `train_store` below is the normal path.
    """
    from gen.ledger import build_ledger
    from gen.render import render_meeting
    from gen.splice import write_manifest
    from engine.engine import Engine

    root = repo_root()
    ledger = build_ledger(n_projects=n_projects, seed=seed)
    manifest = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                              seed=seed)
    split = manifest["split"]
    train = set(split["train"])
    dev = set(split["dev"])
    test = set(split["test"])
    if train & test or dev & test:
        raise ValueError("split sets overlap test; refusing to train")
    keep = train | dev
    path = Path(store_path or (root / "runs" / "concept_corpus" / f"c{seed}.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(path.name + "*"):
        stale.unlink()
    # `model_id` must travel with the backend: the st embedder raises
    # "embed.model_id must be set for the st backend" on None (hf_embedder.py),
    # so `--embed st --model X` accepted the model and crashed anyway -- the
    # inert-knob bug (D-011) in argument clothes.
    cfg = load_config(overrides={"embed": {"backend": embed_backend or "hash",
                                           "model_id": embed_model},
                                 "reader": {"backend": reader_backend or "rules"},
                                 "device": device or "cpu",
                                 "store": {"path": str(path)}})
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    eng = Engine(cfg, path=path, roster=roster)
    n = 0
    dropped = 0
    for m in ledger.meetings:
        if m.project not in keep:
            continue
        src = render_meeting(ledger, m.meeting_id, seed=seed)
        # Turn-level scrub. Skipping test *meetings* is not enough: the ledger
        # renders ~35% distractor turns, so a meeting of a train project carries
        # turns about a test project, and the segmenter deliberately gives those
        # units the discussed project (6.8.3's cross-project guard depends on it).
        # Measured: with meeting-level filtering only, `{'Atlas'}` -- a test project
        # -- was present in the training store, so the claim this module's docstring
        # makes was false until this line existed.
        kept = [t for t in src.turns if ((t.project or src.project) or "") not in test]
        dropped += len(src.turns) - len(kept)
        if not kept:
            continue
        eng.ingest(src.model_copy(update={"turns": kept}))
        n += 1
    present = {u.project for u in eng.store.all_units() if u.project}
    if present & test:
        eng.close()
        raise ValueError(f"test projects reached the training corpus: "
                         f"{sorted(present & test)}")
    return {"engine": eng, "ledger": ledger, "train": train, "dev": dev,
            "test": test, "n_meetings_ingested": n,
            "n_distractor_turns_dropped": dropped, "path": path,
            "counts": eng.store.count()}


def train_store(*, n_projects: int = 20, seed: int = 42, epochs: int | None = None,
                checkpoint: Path | None = None, embed_backend: str | None = None,
                embed_model: str | None = None,
                reader_backend: str | None = None, device: str | None = None,
                verbose: bool = False) -> dict[str, Any]:
    """Build the corpus, fit, select on dev, save the checkpoint, report."""
    from engine.concept.model import ConceptConfig
    from engine.concept import train as T

    cfg = load_config()
    cc = dict(cfg.get_path("concept", {}) or {})
    corpus = build_corpus(n_projects=n_projects, seed=seed,
                          embed_backend=embed_backend, embed_model=embed_model,
                          reader_backend=reader_backend, device=device)
    eng = corpus["engine"]
    dim = int(eng.embedder.dim)
    # ONE knob. `max_drift` is the published contract (the refine gate and the loss
    # hinge both read it); the model's structural cap is derived from it at 80% so
    # the geometry always satisfies the bound with margin, instead of being a second
    # number someone can set to 0.2 while asking the gate for 0.1 — which would let
    # every checkpoint violate the contract it was measured against.
    max_drift = float(cc.get("max_drift", 0.25))
    cc_cfg = ConceptConfig(
        dim=dim, d_model=min(int(cc.get("d_model", 256)), dim),
        layers=int(cc.get("layers", 2)), heads=int(cc.get("heads", 4)),
        max_context=int(cc.get("k_neighbors", 16)) + 1,
        max_step=round(0.8 * max_drift, 6))
    report = T.train(eng.store, cfg=cc_cfg, seed=seed,
                     epochs=int(epochs or cc.get("epochs", 8)),
                     lr=float(cc.get("lr", 3e-4)),
                     batch_size=int(cc.get("batch_size", 8)),
                     k_neighbors=int(cc.get("k_neighbors", 16)),
                     weights=dict(cc.get("weights", {})) or None,
                     temperature=float(cc.get("temperature", 0.07)),
                     triplet_margin=float(cc.get("triplet_margin", 0.3)),
                     max_norm=float(cc.get("max_drift", 0.25)),
                     train_projects=corpus["train"], dev_projects=corpus["dev"],
                     device=device or "cpu", verbose=verbose)
    path = Path(checkpoint or cc.get("checkpoint") or T.default_checkpoint_path())
    if not path.is_absolute():
        path = repo_root() / path
    saved = T.save(report["model"], report, path)
    # the anchoring number is reported here, at training time, because it is the
    # cheapest place to discover that the model cannot be used at all: a checkpoint
    # whose refined space is orthogonal to base will be refused by the refine job's
    # gate, and finding that out after an hours-long bench run is wasted work.
    from engine.concept.refine_job import refine

    probe = refine(eng.store, report["model"],
                   k=int(cc.get("k_neighbors", 16)),
                   batch_size=int(cc.get("batch_size", 8)),
                   max_drift=float(cc.get("max_drift", 0.25)),
                   device=device or "cpu", writes=False)
    out = {k: v for k, v in report.items() if k != "model"}
    out.update({"checkpoint": str(saved), "bytes": saved.stat().st_size,
                "corpus": {"n_meetings": corpus["n_meetings_ingested"],
                           "train_projects": sorted(corpus["train"]),
                           "dev_projects": sorted(corpus["dev"]),
                           "test_projects_absent": sorted(corpus["test"]),
                           "counts": corpus["counts"]},
                "embed_version": eng.store.embed_version,
                "anchor_probe": {kk: probe.get(kk) for kk in
                                 ("anchored", "cosine_to_base_mean", "drift_mean",
                                  "drift_p95", "cosine_to_base_floor_required",
                                  "max_drift_allowed", "n_refined")}})
    eng.close()
    return out


__all__ = ["build_corpus", "train_store"]
