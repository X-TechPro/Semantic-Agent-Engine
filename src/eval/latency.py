"""`make latency` — the system metrics 9.1 asks for, measured on the real path.

    p50/p95 per retrieval stage and end-to-end (LLM off and on), peak RAM/VRAM,
    index size, ingest throughput (units/s), cost per 1k queries.

Why this module exists at all: the §10 reranker rows were decided on quality
alone, and §10's own fairness rule says "report latency/VRAM beside quality". The
numbers that module writes are the ones the reranker decision has to survive.

Two instrument choices, both deliberate:

* **Per-stage timings come from the pipeline itself** (`RetrievalStats.timings`),
  not from wrapping calls in this script. Wrapping would have attributed the
  dense+lexical block to whichever stage happened to be timed last.
* **The reranker's model-load cost is measured separately** from its per-query
  cost. A 568M-parameter cross-encoder is ~2 GB of VRAM that appears once, at
  construction; charging that to every query would make the per-query number
  meaningless, and hiding it would make the VRAM number one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query, utcnow
from gen.ledger import build_ledger
from gen.qa_from_ledger import build_questions
from gen.render import render_meeting
from gen.splice import write_manifest

#: stages the pipeline times; `rerank_s` is absent when no reranker ran, and that
#: absence must be reported as "not measured", not as 0 ms.
STAGES = ("dense_s", "lexical_s", "rerank_s", "validity_s")


def _peak_ram_mb() -> float:
    """Peak resident set of this process, in MB.

    Two traps fixed here: `psutil.Process().memory_info().rss` is the CURRENT set,
    which after `eng.close()` had already released the store and the models, so it
    printed a post-run number under a field named "peak"; and `ru_maxrss` is
    **kilobytes on Linux** but **bytes on Darwin**, so dividing by 1e6 on Linux
    reported a 3.9 GB peak as "3.9 MB".
    """
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e3 if sys.platform != "darwin" else peak / 1e6


def _vram_mb() -> tuple[float, float]:
    """(current, peak) allocator MiB, or (0, 0) with no CUDA.

    `max_memory_allocated` is the allocator's high-water mark, which is the
    number that predicts OOM. RSS-resident VRAM is larger and would overstate
    what a second component can be given.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        torch.cuda.synchronize()
        return (round(torch.cuda.memory_allocated() / 2**20, 1),
                round(torch.cuda.max_memory_allocated() / 2**20, 1))
    except Exception:
        return 0.0, 0.0


def _pct(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"p50": round(float(np.percentile(values, 50)) * 1000, 2),
            "p95": round(float(np.percentile(values, 95)) * 1000, 2),
            "mean": round(float(np.mean(values)) * 1000, 2),
            "n": len(values)}


def measure(n_projects: int = 20, meetings_per_project: int = 3, seed: int = 42,
            split: str = "test", cfg=None, limit_queries: int | None = None,
            llm: bool = False, out_dir: Path | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    root = repo_root()
    rerank_tag = str(cfg.get_path("retrieve.rerank.backend", "none")).replace(" ", "")
    reader_tag = str(cfg.get_path("reader.backend", "rules")).replace(" ", "")
    out_dir = out_dir or (root / "runs" / "latency"
                          / f"{split}_{seed}_p{n_projects}_r-{reader_tag}_k-{rerank_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger(n_projects=n_projects, meetings_per_project=meetings_per_project,
                          seed=seed)
    manifest = write_manifest(root / "data/splits/ledger.json", ledger.projects, seed=seed)
    holdout = set(manifest["split"][split])
    train = [m for m in ledger.meetings if m.project not in holdout]
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})

    db = out_dir / "store.db"
    for stale in out_dir.glob("store.db*"):
        stale.unlink()
    vram_before = _vram_mb()
    t0 = time.perf_counter()
    eng = Engine(cfg, path=db, roster=roster)
    build_s = time.perf_counter() - t0
    # the cost of LOADING the models, kept separate from per-query cost so a
    # one-off 2 GB weight upload is not charged to every query
    vram_after_build = _vram_mb()

    units = 0
    t1 = time.perf_counter()
    for m in train:
        units += eng.ingest(render_meeting(ledger, m.meeting_id, seed=seed)).n_stored
    ingest_s = max(1e-9, time.perf_counter() - t1)

    qa = build_questions(ledger, seed=seed)
    items = [q for q in qa.items if q["project"] in holdout]
    items = [q for q in items if q["kind"] in ("current_decision", "history", "owner",
                                               "date", "proposal_vs_decision",
                                               "unanswerable")]
    if limit_queries:
        items = items[:limit_queries]

    stage_ms: dict[str, list[float]] = {s: [] for s in STAGES}
    e2e: list[float] = []
    agent: list[float] = []
    for q in items:
        t = time.perf_counter()
        eng.retrieve(Query(text=q["question"], project=q["project"],
                           mode="history" if q["kind"] == "history" else "current"))
        e2e.append(time.perf_counter() - t)
        # read back, do not re-time: the pipeline already clocked its own stages
        for k, v in (eng.pipeline.last_stats.timings or {}).items():
            if k in stage_ms:
                stage_ms[k].append(v)
        if llm:
            t = time.perf_counter()
            eng.ask(q["question"], intent="answer", project=q["project"])
            agent.append(time.perf_counter() - t)

    vram_peak = _vram_mb()
    # index size means everything the running system has to carry on disk, so the
    # FAISS files are counted beside the database rather than assumed identical
    index_bytes = {p.name: p.stat().st_size for p in out_dir.glob("store.db*")}
    store_dir = Path(cfg["store"]["path"]).parent
    if store_dir.exists():
        for p in store_dir.glob(f"{Path(cfg['store']['path']).name}*"):
            index_bytes.setdefault(p.name, p.stat().st_size)
    eng.close()

    out: dict[str, Any] = {
        "measured_at": utcnow().isoformat(),
        "config_hash": cfg.hash(),
        "seed": seed, "split": split, "n_projects": n_projects,
        "n_queries": len(items), "n_units_ingested": units,
        "backends": {"embed": eng.embedder.version,
                     "reader": eng.reader_backend, "rerank": eng.rerank_backend,
                     "linker": getattr(eng.ingest_pipeline, "linker_backend", "unknown")},
        "models_loaded_once": {"build_s": round(build_s, 2),
                               "vram_after_build_mib": vram_after_build[1],
                               "vram_before_build_mib": vram_before[1]},
        "ingest": {"units_per_s": round(units / ingest_s, 1),
                   "seconds": round(ingest_s, 2)},
        "end_to_end_ms": _pct(e2e),
        # The first query in a process pays CUDA context init and cuDNN kernel
        # selection, which showed up here as one ~10 s sample inside a p95 of
        # 41 ms. Averaged into the stage table it once made a reranker look 78x
        # slower than it is, so the cold sample is reported on its own and the
        # percentiles below it exclude it.
        "cold_start_ms": round(e2e[0] * 1000, 1) if e2e else None,
        "end_to_end_steady_ms": _pct(e2e[1:]) if len(e2e) > 1 else None,
        "stages_ms": {k: _pct(v) for k, v in stage_ms.items()},
        "with_llm_ms": _pct(agent) if llm else None,
        "peak": {"ram_mb": round(_peak_ram_mb(), 1), "vram_alloc_mib": vram_peak[1]},
        "index_bytes": index_bytes,
        # 9.1 asks for cost per 1k queries; GPU-seconds is only meaningful when a
        # CUDA model actually ran, so the field states what it is a cost OF.
        "cost_per_1k_queries": {
            "wall_s": round(float(np.mean(e2e)) * 1000, 1) if e2e else None,
            "gpu_seconds": (round(float(np.mean(e2e)) * 1000, 1)
                            if e2e and vram_peak[1] > 0 else None),
            "on_gpu": vram_peak[1] > 0},
        "not_measured": [k for k, v in ((s, _pct(stage_ms[s]) or {}) for s in STAGES)
                         if not v],
    }
    (out_dir / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(out), encoding="utf-8")
    out["_out_dir"] = str(out_dir)
    return out


def _summary(o: dict[str, Any]) -> str:
    b = o["backends"]
    L = [f"# latency — seed {o['seed']} | {o['n_projects']} projects | "
         f"{o['n_queries']} queries ({o['split']})", "",
         f"- embed `{b['embed']}` | reader `{b['reader']}` | rerank `{b['rerank']}` | "
         f"linker `{b['linker']}` | config `{o['config_hash']}`",
         f"- models loaded once in {o['models_loaded_once']['build_s']} s, "
         f"{o['models_loaded_once']['vram_after_build_mib']} MiB allocator peak after "
         f"(before: {o['models_loaded_once']['vram_before_build_mib']} MiB)",
         f"- ingest {o['ingest']['units_per_s']} units/s over "
         f"{o['ingest']['seconds']} s", ""]

    def row(name: str, v: dict | None) -> str:
        if not v:
            return f"| {name} | not measured | not measured | not measured |"
        return f"| {name} | {v['p50']} | {v['p95']} | {v['mean']} |"

    L += ["| stage | p50 ms | p95 ms | mean ms |", "|---|---|---|---|",
          row("**end to end (LLM off)**", o["end_to_end_ms"])]
    labels = {"dense_s": "dense (FAISS)", "lexical_s": "lexical (BM25)",
              "rerank_s": "rerank", "validity_s": "validity + ranking"}
    for k in STAGES:
        L.append(row(labels[k], o["stages_ms"].get(k)))
    if o["with_llm_ms"]:
        L.append(row("end to end (LLM on)", o["with_llm_ms"]))
    L += ["",
          f"- peak RAM {o['peak']['ram_mb']} MB, VRAM allocator peak "
          f"{o['peak']['vram_alloc_mib']} MiB",
          f"- cost per 1k queries: {o['cost_per_1k_queries']['wall_s']} s wall"
          + ("" if o["cost_per_1k_queries"]["on_gpu"] else " (CPU: no GPU-seconds)"),
          f"- index on disk: {sum(o['index_bytes'].values()) / 1e6:.2f} MB",
          "\nStages listed as `not measured` did not run for this configuration; they "
          "are not zeros (a reranker that never executed must not appear as 0 ms)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--rerank", choices=["none", "overlap", "cross_encoder"],
                    default=None, help="override retrieve.rerank.backend")
    ap.add_argument("--embed", choices=["hash", "st"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reader", choices=["rules", "model"], default=None)
    ap.add_argument("--llm", action="store_true",
                    help="also time the agent path end to end")
    args = ap.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.rerank:
        overrides["retrieve"] = {"rerank": {"backend": args.rerank}}
    if args.embed:
        overrides["embed"] = {"backend": args.embed, "model_id": args.model}
    if args.reader:
        overrides["reader"] = {"backend": args.reader}
    cfg = load_config(overrides=overrides or None)
    out = measure(args.projects, seed=args.seed, split=args.split, cfg=cfg,
                  limit_queries=args.limit_queries, llm=args.llm)
    print(_summary(out))
    print("wrote", out["_out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
