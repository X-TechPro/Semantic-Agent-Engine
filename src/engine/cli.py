"""`sae` — the command-line surface of the engine (M2 gate: "end-to-end CLI works").

argparse (not typer) so the CLI has no dependency on an optional import.
Every subcommand is config-driven; nothing here is a notebook-only path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query, Source


def _emit(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    else:
        print(_pretty(obj))


def _pretty(obj) -> str:
    if isinstance(obj, dict):
        return "\n".join(f"{k}: {v}" for k, v in obj.items())
    return str(obj)


def _open_engine(args, linker_enabled: bool = True) -> Engine:
    cfg = load_config(getattr(args, "config", None))
    overrides = {}
    if getattr(args, "store", None):
        overrides["store"] = {"path": args.store}
    if getattr(args, "embed", None):
        overrides["embed"] = {"backend": args.embed,
                              "model_id": getattr(args, "model", None)}
    reader = _reader_of(args)
    if reader:
        overrides["reader"] = {"backend": reader}
    rerank = _rerank_of(args)
    if rerank:
        overrides["retrieve"] = {"rerank": {"backend": rerank}}
    cfg = load_config(getattr(args, "config", None), overrides=overrides or None)
    return Engine(cfg, path=cfg["store"]["path"], linker_enabled=linker_enabled)


def cmd_ledger(args) -> int:
    from gen.ledger import build_ledger, summarise

    led = build_ledger(n_projects=args.projects, meetings_per_project=args.meetings,
                       seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = led.save(out / f"ledger-{args.seed}.json")
    audit = out / "audit.md"
    audit.write_text("\n".join(led.audit_lines()), encoding="utf-8")
    s = summarise(led)
    print(f"wrote {path}\n {s}\n audit sample -> {audit}")
    return 0


def cmd_render(args) -> int:
    from gen.ledger import Ledger
    from gen.render import render_ledger

    led = Ledger.from_dict(json.loads(Path(args.ledger).read_text(encoding="utf-8")))
    sources = render_ledger(led, styles=args.styles, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in sources:
        (out / f"{s.source_id}.json").write_text(
            s.model_dump_json(indent=1), encoding="utf-8")
    n_turns = sum(len(s.turns) for s in sources)
    print(f"rendered {len(sources)} meetings ({n_turns} turns) -> {out}")
    return 0


def cmd_ingest(args) -> int:
    eng = _open_engine(args, linker_enabled=not args.no_linker)
    total = None
    for raw in args.paths:
        p = Path(raw)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            src = Source.model_validate_json(f.read_text(encoding="utf-8"))
            rep = eng.ingest(src)
            total = rep if total is None else total
            print(f"{rep.source_id}: units={rep.n_units} stored={rep.n_stored} "
                  f"skipped={rep.n_skipped} queries={rep.n_queries} "
                  f"edges={rep.n_edges} superseded={rep.n_superseded} {rep.relations}")
    print("store:", eng.stats())
    eng.close()
    return 0


def cmd_query(args) -> int:
    eng = _open_engine(args, linker_enabled=not args.no_linker)
    ru = eng.read_text(args.text, project=args.project)
    q = Query(text=args.text, project=args.project, slots=ru.unit.slots,
              intent=args.intent or ru.intent, mode=args.mode)
    hits = eng.retrieve(q, k=args.k)
    rows = [{"rank": h.rank, "unit_id": h.unit.unit_id, "score": round(h.score, 5),
             "status": h.unit.status, "type": h.unit.type,
             "current": h.unit.is_current, "date": h.unit.source_date.isoformat(),
             "speaker": h.unit.speaker, "source": h.unit.source_id,
             "path": h.path, "evidence": h.evidence, "score_parts": h.score_parts}
            for h in hits]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        for r in rows:
            flag = "" if r["current"] else "  [SUPERSEDED]"
            print(f"{r['rank']:>3} {r['score']:.5f} {r['status']:<11} "
                  f"{r['date']} {str(r['speaker'] or '-'):<8} "
                  f"{r['evidence'][:70]}{flag}")
    eng.close()
    return 0


def cmd_ask(args) -> int:
    eng = _open_engine(args, linker_enabled=not args.no_linker)
    res = eng.ask(args.text, intent=args.intent, project=args.project, mode=args.mode)
    payload = {"intent": res.intent, "operation": res.operation,
               "used_llm": res.used_llm, "notes": res.notes,
               "hits": res.hit_unit_ids, "payload": res.payload}
    if res.clarification:
        payload["clarification"] = res.clarification.model_dump(mode="json")
    _emit(payload, args.json)
    eng.close()
    return 0


def cmd_history(args) -> int:
    eng = _open_engine(args)
    rows = [eng.store.with_sentences(u) for u in
            eng.store.history(args.subject, args.project)]
    out = [{"unit_id": u.unit_id, "date": u.source_date.isoformat(),
            "status": u.status, "valid_to": u.valid_to.isoformat() if u.valid_to else None,
            "superseded_by": u.superseded_by, "text": u.text} for u in rows]
    _emit(out, args.json)
    eng.close()
    return 0


def cmd_demo(args) -> int:
    from engine.cli_demo import run_demos

    return run_demos(args)


def cmd_demo_real(args) -> int:
    from engine.cli_demo_real import run_real_demos

    return run_real_demos(args)


def cmd_bench(args) -> int:
    from eval.temporal import run_bench_seeds

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    agg = run_bench_seeds(seeds, n_projects=args.projects, split=args.split,
                          embed_backend=_embed_of(args), embed_model=_model_of(args),
                          reader_backend=_reader_of(args),
                          rerank_backend=_rerank_of(args),
                          vector_column=getattr(args, "vectors", None))
    m = agg["temporal@1_mean"]
    print(f"temporal@1 mean over {len(seeds)} seeds: "
          f"A={m['baseline_A']} B={m['system_B']} delta={m['delta']:+}")
    # the column is printed because a refined row and a base row are different
    # systems; without this line a reader of the terminal output cannot tell which
    # of §10's rows they are looking at.
    print(f"vector column: {agg['provenance']['vector_column']} | "
          f"canonical={agg['canonical']} (only a base run may overwrite the headline)")
    print(f"system-B spread across seeds: {agg['temporal@1_spread']}")
    print(f"seeds with a significant win (CI excludes 0): "
          f"{agg['seeds_with_significant_win']}/{len(seeds)}")
    for s, c in agg["per_seed"].items():
        t = c["temporal@1"]
        print(f"  seed {s}: A={t['baseline_A']} B={t['system_B']} "
              f"delta={t['delta']:+} ci={t['delta_ci95']} "
              f"win={'yes' if t['significant_win'] else 'no'}")
    print("wrote", agg["_out"])
    return 0


def cmd_frbench(args) -> int:
    """Real-French evaluation (7.2): external gold, my sentences not authored by me.

    Reachable from the CLI because the §7.4 promise is that the real-data check is
    re-runnable when the code changes; a harness only callable via `python -m` is a
    number that quietly stops being re-measured.
    """
    from eval.frbench import run

    # every flag the parser advertises must reach run(); `--pool`/`--concept`
    # accepted-and-discarded here would make the 6.13 arm unreachable from `sae`
    # while the module-level entry point had it -- the D-011 inert-knob class.
    comps = (["xnli", "massive", "miracl"] if args.component == "all"
             else [args.component])
    rc = 0
    for c in comps:
        try:
            res = run(c, split=args.split, limit=args.limit, device=args.device,
                      rerank=not args.no_rerank,
                      pool=getattr(args, "pool", "qrels"),
                      concept=getattr(args, "concept", "") or None,
                      reader_model=getattr(args, "reader_model", "") or None)
            r = res.get("result", res)
            print(json.dumps(r, indent=1, ensure_ascii=False)[:2500], flush=True)
        except Exception as exc:
            print(f"{c} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def cmd_frdata(args) -> int:
    """Fetch the pinned French corpora (idempotent; writes PROVENANCE.json)."""
    from engine.tools.fetch_french import fetch

    out = fetch(args.only)
    print(json.dumps({k: v["revision"] for k, v in out.items()}, indent=1))
    return 0


def cmd_latency(args) -> int:
    """9.1 system metrics, measured on the same corpus the quality rows use.

    Wired as a subcommand rather than only a module because 10 requires latency
    reported beside quality: a number reachable only by `python -m eval.latency`
    is a number that quietly stops being re-measured when the code changes.
    """
    from eval.latency import measure

    overrides = {}
    if _embed_of(args):
        overrides["embed"] = {"backend": _embed_of(args), "model_id": _model_of(args)}
    if _reader_of(args):
        overrides["reader"] = {"backend": _reader_of(args)}
    if _rerank_of(args):
        overrides.setdefault("retrieve", {})["rerank"] = {"backend": _rerank_of(args)}
    cfg = load_config(getattr(args, "config", None), overrides=overrides or None)
    out = measure(n_projects=args.projects, seed=args.seed, split=args.split,
                  cfg=cfg, limit_queries=args.limit_queries, llm=args.with_llm)
    e = out["end_to_end_ms"]
    if not e:
        # Nothing to time: an empty holdout, or --limit-queries 0. Printing
        # p50=None would read as "0 ms", the exact silent-zero failure the
        # latency numbers exist to avoid, so say what happened and fail.
        print(f"no queries to time ({out['n_queries']} on split "
              f"{out['split']!r} of {args.projects} projects) — widen --projects "
              f"or --split", file=sys.stderr)
        return 1
    print(f"end to end (LLM off): p50={e['p50']} ms "
          f"p95={e['p95']} ms | cold start "
          f"{out['cold_start_ms']} ms")
    for name, v in out["stages_ms"].items():
        print(f"  {name:12s} {v}")
    print(f"  peak: RAM {out['peak']['ram_mb']} MB | VRAM "
          f"{out['peak']['vram_alloc_mib']} MiB | ingest "
          f"{out['ingest']['units_per_s']} units/s")
    if out["not_measured"]:
        print(f"  not measured: {', '.join(out['not_measured'])}")
    print("wrote", out["_out_dir"])
    return 0


def cmd_export_schemas(args) -> int:
    from engine.tools.export_schemas import export_all

    n = export_all(Path(args.out))
    print(f"wrote {n} JSON schemas -> {args.out}")
    return 0


def cmd_verify_models(args) -> int:
    from engine.tools.verify_models import main as verify_main

    return verify_main([])


def cmd_stage(args) -> int:
    from eval.run_stage import run_stage

    return run_stage(args.stage, args)


def cmd_reader_warm(args) -> int:
    """B-12's reader half, reachable from `sae` -- a harness only callable via
    `python -m` is a number that quietly stops being re-measured (7.4)."""
    from engine.read import warm

    if not args.revalidate_only:
        rep = warm.train(epochs=args.epochs, max_train=args.max_train or None,
                         batch=args.batch, head_lr=args.head_lr, seed=args.seed,
                         projects=args.projects, device=args.device,
                         base=Path(args.base) if args.base else None,
                         out=Path(args.out) if args.out else None)
        print(json.dumps({"selected": rep["selected"],
                          "frozen_head_drift": rep["frozen_head_drift"],
                          "temperatures": rep["temperatures"]},
                         indent=1, default=str), flush=True)
    if args.train_only:
        print("ledger test left unread (run `reader-warm --revalidate-only` to "
              "spend the one read)", flush=True)
        return 0
    rep = warm.revalidate(seed=args.seed, projects=args.projects,
                          device=args.device,
                          warm=Path(args.out) if args.out else None)
    print(json.dumps({"compare_dev": rep["compare_dev"],
                      "compare_test": rep["compare_test"]}, indent=1), flush=True)
    return 0


def cmd_concept_train(args) -> int:
    """Fit concept attention (6.13) on the non-test corpus and save a checkpoint.

    Thin over `concept.corpus.train_store`, which builds the store from train+dev
    projects only; the test projects are absent from that corpus rather than
    filtered at loss time, so §9.5 cannot be violated by an argument slip here.
    """
    from engine.concept.corpus import train_store

    rep = train_store(n_projects=args.projects, seed=args.seed, epochs=args.epochs,
                      checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                      embed_backend=_embed_of(args), embed_model=_model_of(args),
                      reader_backend=_reader_of(args),
                      device=args.device, verbose=args.verbose)
    # The run report's name carries the embed identity: an e5-space fit landing
    # on `concept_train_42_p20.json` would have overwritten the published
    # hash-space report with the same seed/projects, and the two differ in the
    # one thing (the base space) that makes them separate experiments.
    eb, em = _embed_of(args), _model_of(args)
    etag = ""
    if eb and eb != "hash":
        import re as _re
        etag = "-" + _re.sub(r"[^A-Za-z0-9._-]+", "-", str(em or eb))
    out = repo_root() / "runs" / f"concept_train_{args.seed}_p{args.projects}{etag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    lab = rep.get("labels", {})
    print(f"trained {rep['epochs']} epochs on {rep['n_train_sets']} train sets "
          f"({rep['n_dev_sets']} dev sets) | best_epoch={rep['best_epoch']}")
    print(f"  labels: positives={lab.get('positives')} hard_negatives="
          f"{lab.get('hard_negatives')} validity_rate={lab.get('validity_supervised_rate')}")
    a = rep.get("anchor_probe", {})
    print(f"  anchoring: cosine_to_base={a.get('cosine_to_base_mean')} "
          f"required>={a.get('cosine_to_base_floor_required')} drift_p95={a.get('drift_p95')} "
          f"bound={a.get('max_drift_allowed')} -> anchored={a.get('anchored')}")
    print(f"  checkpoint -> {rep['checkpoint']} ({rep['bytes']} bytes)")
    if not a.get("anchored"):
        print("  NOT ANCHORED: the refine job will refuse to write this column "
              "(6.13). Raise the drift weight or train longer.")
    print("wrote", out)
    return 0


def cmd_concept_refine(args) -> int:
    """Write `vectors.refined` for an existing store (6.14's batch step)."""
    from engine.concept.refine_job import clear_refined, refine
    from engine.concept.train import load as load_concept

    eng = _open_engine(args)
    try:
        if args.clear:
            print(f"cleared {clear_refined(eng.store)} refined vectors; "
                  f"retrieval stays on `{eng.vector_column}`")
            return 0
        ckpt = Path(args.checkpoint or eng.cfg.get_path("concept.checkpoint"))
        if not ckpt.is_absolute():
            ckpt = repo_root() / ckpt
        model, meta = load_concept(ckpt, expect_embed_version=eng.store.embed_version)
        rep = refine(eng.store, model,
                     k=int(eng.cfg.get_path("concept.k_neighbors", 16)),
                     batch_size=int(eng.cfg.get_path("concept.batch_size", 8)),
                     max_drift=float(eng.cfg.get_path("concept.max_drift", 0.25)),
                     device=args.device, writes=not args.dry_run)
        _emit(rep, args.json)
        if rep.get("status") == "refused_unanchored":
            print("refused: nothing written (see the warning above)", file=sys.stderr)
            return 1
        if not args.dry_run and args.use:
            eng.use_vector_column("refined")
            print(f"retrieval now reads `{eng.vector_column}` "
                  f"(read back from the engine, not the flag)")
        return 0
    finally:
        eng.close()


def cmd_offline(args) -> int:
    """The nightly batch pass (6.14) from `sae`, B-11's audit included.

    Delegates to `engine.offline.main` so the flag surface, the report and the
    artefact naming live in exactly one place: `sae offline` and `make offline`
    cannot drift apart, and the recalibration step is not reachable only by
    `python -m` (the pattern this repo already called out for frbench).
    """
    from engine.offline import main as offline_main

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "store", None):
        argv += ["--store", args.store]
    reader = _reader_of(args)
    if reader:
        argv += ["--reader", reader]
    if getattr(args, "reader_model", None):
        argv += ["--reader-model", args.reader_model]
    if getattr(args, "no_recalibrate", False):
        argv += ["--no-recalibrate"]
    if getattr(args, "apply_recal", False):
        argv += ["--apply-recal"]
    if getattr(args, "no_write", False):
        argv += ["--no-write"]
    if getattr(args, "threshold", None) is not None:
        argv += ["--threshold", str(args.threshold)]
    return offline_main(argv)


def _concept_flags(s) -> None:
    """Flags the two concept subcommands genuinely both read.

    Deliberately *not* the whole shared set: `--projects`/`--seed` would mean
    nothing to `concept-refine`, which opens a store that is already built, and an
    accepted-but-ignored flag is the `link.backend` inert-knob bug (D-011) wearing
    an argument's clothes.
    """
    s.add_argument("--device", default=None, help="cpu | cuda (default: cpu)")
    s.add_argument("--checkpoint", default=None,
                   help="override concept.checkpoint from config")


def _embed_of(args) -> str | None:
    return getattr(args, "embed_sc", None) or getattr(args, "embed", None)


def _model_of(args) -> str | None:
    return getattr(args, "model_sc", None) or getattr(args, "model", None)


def _reader_of(args) -> str | None:
    return getattr(args, "reader_sc", None) or getattr(args, "reader", None)


def _reader_flags(s) -> None:
    """Backend switches on the subcommands that open an engine.

    `--rerank` rides along here on purpose: this helper is already attached to
    every retrieval-opening subcommand, so adding the flag once wires all of them.
    Declaring it in only one place is what keeps `sae bench --rerank` from being
    the sole route to a reranker while `sae query` silently ignores the stage —
    the `link.backend` inert-knob bug (D-011) in a new costume.
    """
    s.add_argument("--reader", choices=["rules", "model"], default=None,
                   dest="reader_sc")
    s.add_argument("--rerank", choices=["none", "overlap", "cross_encoder"],
                   default=None, dest="rerank_sc")


def _rerank_of(args) -> str | None:
    return getattr(args, "rerank_sc", None) or getattr(args, "rerank", None)


def _embed_flags(s) -> None:
    """Embedder flags, accepted after the subcommand too.

    They are defined on the top-level parser as well, so `sae --embed st bench`
    and `sae bench --embed st` both work; a subcommand-level value overrides the
    global one. Without this, `sae bench --embed st` failed with
    "unrecognized arguments" and the frozen-embedder rows were unreachable.
    """
    s.add_argument("--embed", choices=["hash", "st"], default=None,
                   dest="embed_sc", metavar="BACKEND")
    s.add_argument("--model", default=None, dest="model_sc", metavar="MODEL_ID")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sae", description=__doc__)
    p.add_argument("--config", help="YAML config (configs/*.yaml)")
    p.add_argument("--store", help="sqlite path override")
    p.add_argument("--embed", choices=["hash", "st"], help="embedder backend")
    p.add_argument("--model", help="embedder model id when --embed=st")
    p.add_argument("--reader", choices=["rules", "model"],
                   help="reader backend; models/reader must exist for `model`")
    p.add_argument("--rerank", choices=["none", "overlap", "cross_encoder"],
                   help="reranker backend (6.9); meta.rerank_backend records the "
                        "one that actually ran")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ledger", help="generate a synthetic ledger (gold by construction)")
    s.add_argument("--projects", type=int, default=20)
    s.add_argument("--meetings", type=int, default=3)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default=str(repo_root() / "data/ledger"))
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("render", help="render a ledger into French meeting JSON")
    s.add_argument("--ledger", required=True)
    s.add_argument("--styles", default="ABC")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default=str(repo_root() / "data/rendered"))
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("ingest", help="segment/read/embed/store/link a source")
    s.add_argument("paths", nargs="+")
    s.add_argument("--no-linker", action="store_true")
    _reader_flags(s)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("query", help="retrieval only (no agent operation)")
    s.add_argument("text")
    s.add_argument("--project")
    s.add_argument("--intent")
    s.add_argument("--mode", default="current", choices=["current", "history", "compare"])
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--no-linker", action="store_true")
    _reader_flags(s)
    s.set_defaults(fn=cmd_query)

    s = sub.add_parser("ask", help="full pipeline: read -> retrieve -> act")
    s.add_argument("text")
    s.add_argument("--project")
    s.add_argument("--intent")
    s.add_argument("--mode", default="current", choices=["current", "history", "compare"])
    s.add_argument("--no-linker", action="store_true")
    _reader_flags(s)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("history", help="every assertion made about a subject slot")
    s.add_argument("subject")
    s.add_argument("--project")
    s.set_defaults(fn=cmd_history)

    s = sub.add_parser("demo", help="run the 7 mandatory scenarios (BUILD_GUIDE 13)")
    s.add_argument("--fixture", default=str(repo_root() / "tests/golden/demo_scenarios.json"))
    s.add_argument("--contacts", default=str(repo_root() / "data/contacts.json"))
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("demo-real",
                       help="the 7 mandatory scenarios on REAL French meetings "
                            "(subject keys annotated — B-13 open; everything "
                            "else runs live)")
    s.add_argument("--meetings",
                   default="024c_EBPH,047a_EDRH,012c_EBPZ")
    s.add_argument("--reader-model", default="models/reader_v2", dest="reader_model")
    s.add_argument("--contacts", default=str(repo_root() / "data/contacts.json"))
    s.add_argument("--compose", action="store_true",
                   help="the diagram's last box: ollama rewrites the template "
                        "answer, agent/verify must approve every rewrite")
    s.add_argument("--compose-model", default="qwen3.5:4b", dest="compose_model")
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_demo_real)

    s = sub.add_parser("bench", help="Baseline A vs System B (temporal correctness)")
    s.add_argument("--projects", type=int, default=20)
    s.add_argument("--seeds", default="42")
    s.add_argument("--split", default="test")
    s.add_argument("--vectors", choices=["base", "refined"], default=None,
                   help="10 row 9: search the frozen `base` column or 6.13's "
                        "`refined` one; a refined run can never overwrite the "
                        "headline artefacts")
    _embed_flags(s)
    _reader_flags(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("concept-train",
                       help="fit 6.13 concept attention on the non-test corpus")
    s.add_argument("--projects", type=int, default=20)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--epochs", type=int, default=None,
                   help="override concept.epochs from config")
    s.add_argument("--verbose", action="store_true", help="print the per-epoch curve")
    _concept_flags(s)
    _embed_flags(s)
    _reader_flags(s)
    s.set_defaults(fn=cmd_concept_train)

    s = sub.add_parser("concept-refine",
                       help="write vectors.refined into an existing store (6.14)")
    s.add_argument("--dry-run", action="store_true",
                   help="measure what the job would write without touching the store")
    s.add_argument("--use", action="store_true",
                   help="point retrieval at the refined column after a successful write")
    s.add_argument("--clear", action="store_true",
                   help="drop every refined vector, returning to the frozen space")
    _concept_flags(s)
    s.set_defaults(fn=cmd_concept_refine)

    s = sub.add_parser("latency",
                       help="9.1 system metrics: per-stage p50/p95, RAM/VRAM, index size")
    s.add_argument("--projects", type=int, default=20)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--split", default="test", choices=["train", "dev", "test"])
    s.add_argument("--limit-queries", type=int, default=None)
    s.add_argument("--with-llm", action="store_true",
                   help="also time the agent path end to end")
    _embed_flags(s)
    _reader_flags(s)
    s.set_defaults(fn=cmd_latency)

    s = sub.add_parser("frdata",
                       help="fetch the pinned French corpora (XNLI/MASSIVE/MIRACL/SUMM-RE)")
    s.add_argument("--only", default=None,
                   choices=["xnli", "massive", "miracl", "miracl_corpus", "summre", "all"])
    s.set_defaults(fn=cmd_frdata)

    s = sub.add_parser("frbench",
                       help="real-French eval (7.2): linker vs XNLI, reader vs "
                            "MASSIVE, retrieval vs MIRACL, temporal vs SUMM-RE")
    s.add_argument("--component", default="all",
                   choices=["all", "xnli", "xnli-rerank", "xnli-nli", "massive",
                            "summre", "miracl"])
    s.add_argument("--split", default="test")
    s.add_argument("--limit", type=int, default=5000,
                   help="MIRACL passages to index (identity goes in the artefact dir)")
    s.add_argument("--pool", default="qrels", choices=["qrels", "slice0"],
                   help="miracl candidate pool (qrels = judged set; slice0 is the "
                        "invalid 5k-head slice kept for the record)")
    s.add_argument("--concept", default="",
                   help="concept checkpoint in THIS embed space: adds the "
                        "dense_refined/hybrid_refined arms (6.13)")
    s.add_argument("--reader-model", default="", dest="reader_model",
                   help="massive arm: trained-reader dir to score, e.g. "
                        "models/reader_warm; the artefact names the backend that "
                        "ACTUALLY ran, fallback included")
    s.add_argument("--device", default="cuda")
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(fn=cmd_frbench)

    s = sub.add_parser("stage", help="per-stage evaluation on gold input (9.2)")
    s.add_argument("stage", choices=["segment", "reader", "slots", "linker", "retrieve"])
    s.add_argument("--projects", type=int, default=10)
    s.add_argument("--seed", type=int, default=42)
    # without this, `stage X --split dev` was rejected by argparse and the
    # evaluator silently fell back to the test split (9.5: tune on dev)
    s.add_argument("--split", default="test", choices=["train", "dev", "test"])
    s.add_argument("--linker", default=None, choices=["rules", "logreg", "nli"],
                   help="override link.backend; a non-rules run writes a suffixed "
                        "artefact dir so the rules baseline is never clobbered")
    s.add_argument("--reader", default=None, choices=["rules", "model", "warm"],
                   help="override reader.backend (warm = models/reader_warm); a "
                        "non-rules run gets its own suffixed artefact dir")
    s.add_argument("--reader-model", default=None, dest="reader_model",
                   help="with --reader model: any checkpoint dir (e.g. "
                        "models/reader_v2); the artefact dir is named after it "
                        "so no existing row can be clobbered")
    s.set_defaults(fn=cmd_stage)

    s = sub.add_parser("reader-warm",
                       help="MASSIVE-fr warm-up of the trained reader (B-12): "
                            "kind/intent heads only; encoder and the guard-input "
                            "type/status heads are frozen and the invariance is "
                            "asserted, not promised")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--max-train", type=int, default=4000,
                   help="massive-fr train subset (the 'bit of data'; all rows if 0)")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--head-lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--projects", type=int, default=20)
    s.add_argument("--device", default=None)
    s.add_argument("--base", default=None, help="checkpoint to warm up (models/reader)")
    s.add_argument("--out", default=None, help="where to save (models/reader_warm)")
    s.add_argument("--revalidate-only", action="store_true",
                   help="skip training; run the guarded ledger re-validation")
    s.add_argument("--train-only", action="store_true",
                   help="stop after training; the revalidation reads ledger test "
                        "ONCE, so verify the curve before spending that read")
    s.set_defaults(fn=cmd_reader_warm)

    s = sub.add_parser("offline",
                       help="the nightly batch pass (6.14): audit, re-embed "
                            "decision, concept refine, rebuild, clusters, and "
                            "the calibration audit (B-11)")
    s.add_argument("--reader", choices=["rules", "model"], default=None,
                   dest="reader_sc")
    s.add_argument("--reader-model", default=None, dest="reader_model",
                   help="checkpoint dir to audit (implies --reader model)")
    s.add_argument("--threshold", type=float, default=None,
                   help="cluster distance threshold (module default 0.35)")
    s.add_argument("--no-write", action="store_true")
    s.add_argument("--no-recalibrate", action="store_true")
    s.add_argument("--apply-recal", action="store_true",
                   help="write an accepted refit to models/<name>_recal — a "
                        "CANDIDATE, never the shipped checkpoint")
    s.set_defaults(fn=cmd_offline)

    s = sub.add_parser("export-schemas", help="write schemas/*.json")
    s.add_argument("--out", default=str(repo_root() / "schemas"))
    s.set_defaults(fn=cmd_export_schemas)

    s = sub.add_parser("verify-models", help="live HF licence/existence check -> docs/MODELCARDS.md")
    s.set_defaults(fn=cmd_verify_models)

    # Accept the global output/config flags *after* the subcommand as well.
    # `--json` was only defined on the top-level parser, so `sae query x --json`
    # died with "unrecognized arguments" while `sae --json query x` worked -- a
    # README example that could not be run. Applied by looping over every
    # subparser, so a command added later inherits it instead of re-introducing
    # the gap.
    for s in sub.choices.values():
        _global_flags(s)
    return p


def _global_flags(s) -> None:
    """`--config` / `--store` / `--json`, re-declared with `SUPPRESS` defaults.

    SUPPRESS is load-bearing, not style: a subparser-level `default=False` would
    *overwrite* a parent-level `sae --json ...` with False, so the flag would be
    accepted and then silently ignored -- the inert-knob bug (D-011) in a
    boolean's clothes. With SUPPRESS the attribute is simply not set unless the
    user typed it after the subcommand, so the parent's value survives.
    """
    s.add_argument("--config", default=argparse.SUPPRESS,
                   help="YAML config (configs/*.yaml)")
    s.add_argument("--store", default=argparse.SUPPRESS, help="sqlite path override")
    s.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except BrokenPipeError:  # `sae query ... | head`
        return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
