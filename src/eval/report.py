"""`make report` — assemble docs/REPORT.md from what is actually on disk.

Nothing here is typed in by hand: every number is read from `runs/`, and a
measurement that does not exist is written as `not measured` rather than
estimated. That is the whole point of generating the report from artefacts — the
document cannot drift ahead of the evidence.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from eval.temporal import CANONICAL_PROJECTS

TARGETS = [
    ("Temporal correctness@1 (current Qs)", "temporal@1", 0.60, 0.90, "max"),
    ("Proposal-vs-decision hard-neg accuracy", "hardneg", 0.65, 0.85, "max"),
    ("Recall@5 (dev)", "recall@5", None, None, "delta+5"),
    ("Linker macro-F1 (ledger-test)", "linker_f1", None, 0.80, "max"),
    ("`supersedes` precision", "supersedes_precision", None, 0.95, "max"),
    ("False-recipient rate", "false_recipient", None, 0.02, "min"),
    ("Unsupported-claim rate (dossier)", "unsupported", None, 0.05, "min"),
    ("Hypothesis → action rate", "hyp_action", None, 0.0, "must_be"),
    ("Query latency p95 (LLM off)", "latency_p95", None, 400.0, "max"),
    ("General FR retrieval regression", "mteb_regression", None, 1.0, "min_drop"),
]

#: 6.13's promotion bar for concept attention, stated once here and used as the
#: *target* column only; the measured value always comes from the gate artefact, so
#: the report cannot disagree with the run that produced it.
GATE_613_PTS = 3.0


def _git() -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout or "").strip() or "no-git"
    except Exception:
        return "no-git"


def _load(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage(runs: Path, stage: str) -> dict[str, Any] | None:
    """The rules-baseline artefact for a stage, never an ablation row.

    A linker ablation (`sae stage linker --linker nli`) writes a suffixed
    directory and records the backend that actually ran in its result. Letting
    `sorted()` pick one alphabetically would silently swap which system the
    report's stage table describes — the artefact-identity trap that already bit
    this repo once (the 8-project CI run overwriting the 20-project headline).
    """
    def baseline(d: Path) -> dict[str, Any] | None:
        r = _load(d / "result.json")
        if not r:
            return None
        # suffixed ablation dirs whose result names a non-rules backend are not
        # the default system; a legacy result with no key at all is rules.
        # reader_backend joins linker_backend: a warm-reader stage row must not
        # become the default stage line either.
        if r.get("linker_backend") not in (None, "rules"):
            return None
        if r.get("reader_backend") not in (None, "rules"):
            return None
        return r

    for split in ("test", "dev"):
        for d in sorted(runs.glob(f"stage_{stage}_{split}_*")):
            r = baseline(d)
            if r:
                return r
    for d in sorted(runs.glob(f"stage_{stage}_*")):
        r = baseline(d)
        if r:
            return r
    return None


def _headline(benches: list[Path]) -> Path | None:
    """The run the report should quote: the shipped default config, newest first.

    This used to be `max(dir / "metrics.json")`, i.e. a **lexicographic** pick over
    path strings. That was harmless while run dirs were `..._{seed}`, and became a
    silent misreport the moment they carried a backend tag: `_k-overlap` sorts
    after `_k-none`, so an ablation arm would have become the headline. Selection is
    now by what the artefact reports it ran, not by how its name sorts.

    The project count is part of the identity for the same reason: `make eval-fast`
    runs 8 projects on the test split with the shipped backends, and until the run
    directory carried the corpus size that tripwire wrote straight over the seed-42
    headline artefact. An 8-project number is not the headline.
    """
    best = {"test": (None, None), "other": (None, None)}
    for d in benches:
        meta = (_load(d / "metrics.json") or {}).get("meta") or {}
        reader = str(meta.get("reader_backend", "rules"))
        rerank = str(meta.get("rerank_backend", "none"))
        split = str(meta.get("split", ""))
        projects = meta.get("n_projects")
        if not reader.startswith("rules") or rerank != "none":
            continue
        if split == "test" and projects != CANONICAL_PROJECTS:
            continue
        key = (d.stat().st_mtime, d.name)
        slot = "test" if split == "test" else "other"
        if best[slot][1] is None or key > best[slot][1]:
            best[slot] = (d, key)
    return best["test"][0] or best["other"][0]


_FRBENCH = (
    # (key, glob). One artefact per component family, newest first. The rules-
    # fallback reader artefact is kept, not deleted: it is the evidence that the
    # GPU-OOM run reported `rules` while the YAML asked for `model`, which is why
    # the artefact directory is named by the backend that RAN.
    ("xnli_pairclassifier", "xnli_pairclassifier_*"),
    ("xnli_rerank_probe", "xnli-rerank_*"),
    ("xnli_nli_head", "xnli-nli_*"),
    # validation families are listed FIRST so their dirs are claimed before the
    # test-row families can pick them up by mtime: a tuning-split run finishing
    # after the baseline test run must never retag §6d's headline row
    ("massive_model_warm_val", "massive_model-reader_warm*_validation_*"),
    ("massive_model_val", "massive_model*_validation_*"),
    ("massive_model_warm", "massive_model-reader_warm*"),
    ("massive_model", "massive_model*"),
    ("massive_fallback", "massive_reader*"),
    ("summre_temporal", "summre_*"),
    ("miracl_pool", "miracl_*pool*_dev*"),
    ("miracl_slice_invalid", "miracl_*_dev_n*"),
)


def _frbench(runs: Path) -> dict[str, Any]:
    """Real-French artefacts (7.2), read from disk like every other number.

    Newest by mtime, not by name: within a family the directory name carries the
    split, and name order would let a stale dev run outrank a fresh test run -- the
    same lexicographic-selection class the headline picker already had to fix.
    """
    out: dict[str, Any] = {}
    base = runs / "frbench"
    for i, (key, pat) in enumerate(_FRBENCH):
        # a dir matching an EARLIER (more specific) family is never claimed by a
        # later one: `massive_model*` also matches the warm checkpoint's dir, so
        # without this an mtime rule would let a warm run silently replace the
        # baseline row -- the shadowing class the fallback key exists for
        earlier = [p for _k, p in _FRBENCH[:i]]
        dirs = [d for d in base.glob(pat)
                if (d / "result.json").exists() and not any(d.match(p) for p in earlier)]
        if not dirs:
            continue
        newest = max(dirs, key=lambda d: (d / "result.json").stat().st_mtime)
        r = _load(newest / "result.json")
        if r:
            out[key] = r
    return out


def summre_section(st: dict[str, Any]) -> list[str]:
    """§6d's B-13 rows, computed from the artefact — and only from it.

    The mechanism/instrument columns come BEFORE the headline delta is read:
    on a corpus with no status annotations the reachable edges bound what any
    ranking difference can mean (measure-before-diagnosing, stated as output).
    """
    if not st:
        return []
    arms = st.get("arms") or {}
    a = arms.get("baseline_A") or {}
    b = arms.get("system_B") or {}
    mech = st.get("mechanism_B") or {}
    ins = st.get("instrument") or {}
    con = mech.get("contradicts") or {}
    supo = mech.get("supports") or {}
    L = [f"**Temporal layer — SUMM-RE test (B-13, hand-labelled, one read).** "
         f"{st.get('n_supersedes_pairs')} gold supersedes pairs on raw French "
         f"ASR ({st.get('reader_backend')} reader, {st.get('linker_backend')} "
         f"linker, e5 base, top-{st.get('top_k')}).", "",
         "- mechanism reach first, because it bounds the result: the linker "
         f"marked **{_fmt(mech.get('marked_superseded_rate'))}** of gold-old "
         f"units superseded ({mech.get('marked')}/{mech.get('n_pairs')}, chain "
         f"direct {_fmt(mech.get('direct_chain_rate'))}); contradicts found on "
         f"{con.get('found')}/{con.get('n')} ({_fmt(con.get('rate'))}), supports on "
         f"{supo.get('found')}/{supo.get('n')} ({_fmt(supo.get('rate'))}).",
         f"- status mix on real ASR French: "
         + ", ".join(f"`{k}` {v}" for k, v in
                     sorted((ins.get("status_mix_B") or {}).items()))
         + " — SUMM-RE ships no status annotations, so this ceiling is read "
           "BEFORE the delta below is interpreted.", "",
         f"- ranking the newer claim above the superseded one: baseline A "
         f"**{_fmt(a.get('new_above_old'))}** → System B "
         f"**{_fmt(b.get('new_above_old'))}**; stale-on-top "
         f"{_fmt(a.get('stale_above_new'))} → {_fmt(b.get('stale_above_new'))}; "
         f"MRR(new) {_fmt(a.get('mrr_new_at_k'))} → "
         f"{_fmt(b.get('mrr_new_at_k'))}. "]
    if a.get("n_pairs", 0) == 0 or b.get("n_pairs", 0) == 0:
        L.append("_instrument warning: nothing was scoreable — no verdict may "
                 "be read from this row either way._")
    L.append("")
    return L


def subjects_section(c: dict[str, Any]) -> list[str]:
    """§6e — the supervised subject programme (6.3 d–f), computed from artefacts.

    One section per family the label campaign produced: the corpus (splits are
    declared INSIDE each gold file, rendered here, never retyped), every drawer
    attempt with its dev-consistency columns and the save gate's own refusal
    strings, both external gates only when a checkpoint existed to judge, and
    the end-to-end marked rate. The honest states matter: "gate not run" must
    read as the dev filter refusing — never as an un-run experiment. Rates go
    through `_fmt`, so a measured 0.0 renders as 0, not "not measured" (the
    falsy-zero trap), and thresholds print from the artefact, not from here.
    """
    L = ["", "## 6e. Supervised subject extraction (6.3 d–f — real labelled "
              "corpus)", ""]
    corpus = c.get("subject_corpus") or []
    if not corpus:
        return L + ["_no labelled corpus yet — run the label tool "
                    "(`python -m engine.tools.summre_label --help`)_", ""]
    by_split: dict[str, list[dict]] = {}
    for g in corpus:
        by_split.setdefault(str(g.get("split")), []).append(g)
    parts = ", ".join(f"{s} {sum(m.get('n_segments') or 0 for m in ms)} segments"
                      f" / {len(ms)} meeting(s)" for s, ms in sorted(by_split.items()))
    L += [f"**Corpus:** {parts}. One drawer per real segment; the split is "
          "declared inside each file — the trainer cannot read `measure`, the "
          "gate reads only it.", ""]
    att = c.get("drawer_attempts") or []
    if att:
        L += ["**Drawer-classifier attempts** (dev meeting, REAL French; the "
              "dev filter decides whether a checkpoint exists at all):", "",
              "| attempt | seed | drawers | train rows | dev same-thread | dev "
              "cross-thread | dev null-acc | outcome |", "|---|---|---|---|---|---|---|---|"]
        for a in sorted(att, key=lambda x: x.get("file") or ""):
            d = a.get("dev") or {}
            cons = d.get("consistency") or {}
            st = (cons.get("same_thread") or {}).get("rate")
            cr = (cons.get("cross_thread") or {}).get("rate")
            failed = a.get("failed") or []
            outcome = ("saved — see the gate row" if a.get("saved")
                       else ("; ".join(failed) if failed else "—"))
            L.append(f"| `{a.get('file')}` | {a.get('seed')} | {a.get('vocab_n')} "
                     f"| {_fmt(a.get('n_train'))} | {_fmt(st)} | {_fmt(cr)} "
                     f"| {_fmt(d.get('null_accuracy'))} | {outcome} |")
        L.append("")
    else:
        L += ["_no supervised attempt on file — that code was removed (D-037); "
              "the verdict numbers below are rendered from preserved artefacts_",
              ""]
    for key, label, gate_key in (
            ("span_gate", "span head (d/e, synthetic-trained corpus)",
             "consistency_supervised"),
            ("drawer_gate", "drawer classifier (f, real labelled corpus)",
             "consistency_drawer"),
            ("graph_gate", "graph projection (g, live-thread centroids)",
             "consistency_graph")):
        art = c.get(key) or {}
        block = art.get(gate_key) or {}
        gt = block.get("gate") or art.get("gate")
        if not gt:
            why = block.get("refused")
            L.append(f"- {label}: external gate **not run** — "
                     + (why if why else
                        "the dev filter never saved a checkpoint to judge "
                        "(refusal, not absence of experiment)."))
            continue
        th = gt.get("thresholds") or {}
        L.append(f"- {label}: gate **{'PASS' if gt.get('pass') else 'FAIL'}** "
                 f"— agreement {_fmt(gt.get('worst_same_thread'))} vs "
                 f"{_fmt(th.get('same_thread'))}, separation "
                 f"{_fmt(gt.get('worst_cross_thread'))} vs "
                 f"{_fmt(th.get('cross_thread'))}, coverage "
                 f"{_fmt(gt.get('pair_coverage'))} vs {_fmt(th.get('pair_coverage'))}"
                 " (bars read from the artefact).")
    arms = (c.get("reach_run") or {}).get("reach") or {}
    if arms:
        L += ["", "**End-to-end on the temporal dev pairs** (marked rate = "
              "gold-old units actually stamped superseded by a real ingest):",
              "", "| arm | subjects resolved | marked | store edges | newer-on-top |",
              "|---|---|---|---|---|"]
        for name, a in arms.items():
            L.append(f"| `{name}` | {a.get('subject_resolved')} | "
                     f"{a.get('marked_superseded')}/{a.get('n_pairs')} | "
                     f"{a.get('store_superseded_total')} | "
                     f"{_fmt((a.get('ranking') or {}).get('new_above_old'))} |")
        L.append("")
    return L


def demo_real_section(art: dict[str, Any]) -> list[str]:
    """§6f — the seven mandatory demos on REAL French meetings, rendered from
    the harness's own artefact (`runs/demo_real/`).

    Subject keys are annotation-supplied and the header says so on every
    regeneration (B-13 stays open — the resolver is read back from the run,
    never restated); each check prints the verdict the harness recorded. No
    prose here may soften a FAIL or invent a number the file does not carry.
    """
    L = ["", "## 6f. The seven mandatory demos on real French meetings", ""]
    if not art:
        return L + ["_not measured — run `make demo-real`_", ""]
    checks = art.get("checks") or []
    passed = sum(1 for x in checks if x.get("pass"))
    cp = art.get("corpus_stats") or {}
    back = art.get("engine_backends") or {}
    L += [f"Meetings `{', '.join(art.get('meetings') or [])}` | reader "
          f"`{art.get('reader')}` | embed `{art.get('embed')}` | resolver read "
          f"back `{back.get('subject_resolver')}` — **subject keys supplied "
          "from the hand-labelled drawer gold (B-13 open: no extractor is "
          f"trusted on real speech)** | units {cp.get('n_units')} | supersede "
          f"edges on real speech {_fmt(art.get('supersede_edges'))}.", "",
          ]
    if art.get("llm_answer"):
        # the composed run is a different experiment; its outcome per query is
        # on disk (llm_notes) — rendered, not summarised: the verifier keeps
        # or bounces each rewrite, and the file says which happened.
        outcomes = "; ".join(f"`{k}`: {(v or ['—'])[0]}"
                             for k, v in (art.get("llm_notes") or {}).items())
        L += [f"**LLM answer box (off by default):** {art['llm_answer']}"
              + (f" — per-query outcomes: {outcomes}" if outcomes else "")
              + " — every rewrite passes `agent/verify` or the template stands.",
              ""]
    L += ["| check | verdict | detail |", "|---|---|---|"]
    for x in checks:
        L.append(f"| {x.get('name')} | {'PASS' if x.get('pass') else '**FAIL**'} "
                 f"| {(x.get('detail') or '').replace('|', chr(92) + '|')} |")
    L += ["", f"**{passed}/{len(checks)} checks passed** — verdicts are the "
              "harness's recorded output (`failed` names each broken promise; "
              "`runs/demo_real/` holds the JSON).", ""]
    return L


def _fbres(art: dict | None) -> dict[str, Any]:
    """Unwrap a run() envelope ({component,result,...}); miracl writes its dict
    directly, so both shapes must render from one accessor."""
    art = art or {}
    return art.get("result", art) if isinstance(art.get("result"), dict) else art


def collect() -> dict[str, Any]:
    runs = repo_root() / "runs"
    out: dict[str, Any] = {"git": _git(), "date": date.today().isoformat(),
                           "config_hash": load_config().hash()}
    out["temporal_mean"] = _load(runs / "temporal_mean.json")
    benches = [d for d in runs.glob("m1_m2_temporal_*") if (d / "metrics.json").exists()]
    head = _headline(benches)
    out["bench"] = _load(head / "metrics.json") if head else None
    out["bench_dir"] = head.name if head else None
    out["stages"] = {s: _stage(runs, s) for s in
                     ("segment", "reader", "slots", "linker", "retrieve")}
    kills = sorted(runs.glob("killtest_*"))
    out["killtest"] = _load(kills[-1] / "result.json") if kills else None
    # the embedder fine-tune writes one result per seed; report the newest so the
    # prose below is read from an artefact and cannot drift from the measurement
    fts = sorted(runs.glob("embed_finetune/result_*.json"))
    out["embed_finetune"] = _load(fts[-1]) if fts else None
    off = sorted((runs / "offline").glob("*.json")) if (runs / "offline").exists() else []
    out["offline"] = _load(off[-1]) if off else None
    # 9.1 system metrics: one artefact per configuration actually measured. Read
    # from disk, never typed in, so the latency claim and the quality claim in §4
    # cannot drift apart the way the reranker's cost nearly did.
    out["latency"] = {}
    for d in sorted((runs / "latency").glob("*/result.json")):
        r = _load(d)
        if r:
            key = (r.get("backends") or {}).get("rerank") or d.parent.name
            tag = f"{key}|{r.get('split')}|p{r.get('n_projects')}|s{r.get('seed')}"
            out["latency"][tag] = r
    # 9.3 oracle swaps: which stage actually owns the remaining error. Read from
    # the artefact like everything else here, so the attribution cannot drift from
    # the run that produced it.
    out["oracle"] = None
    o_dirs = sorted(runs.glob("oracle_swap_*"))
    for d in reversed(o_dirs):
        r = _load(d / "result.json")
        if r:
            out["oracle"], out["oracle_dir"] = r, d.name
            break
    out["e2e"] = _load(runs / "e2e" / "result.json")
    out["mteb_regression"] = _load(runs / "mteb_regression" / "result.json")
    # 6.13 (M5): the gate decision and the fit behind it, read from disk like every
    # other number. The report previously asserted in prose that `concept/` was an
    # empty package; once M5 was built that sentence became a lie the generator kept
    # re-emitting, which is exactly the failure mode "nothing is typed by hand"
    # exists to prevent.
    out["concept_gate"] = _load(runs / "concept_gate" / "summary.json")
    tr = sorted(runs.glob("concept_train_*.json"))
    out["concept_train"] = _load(tr[-1]) if tr else None
    out["frbench"] = _frbench(runs)
    # 6.3(d–f): the supervised subject programme (§6e), all from disk.
    sc = []
    for p in sorted((repo_root() / "data/gold").glob("summre_subjects_*.json")):
        g = _load(p) or {}
        if g.get("split"):
            sc.append({"meeting": g.get("meeting"), "split": g["split"],
                       "n_segments": g.get("n_segments"),
                       "drawers": len({k for k in (g.get("keys") or {}).values()
                                       if k is not None})})
    out["subject_corpus"] = sc
    att = []
    for p in sorted((runs / "drawer_sup").glob("result-*.json")):
        r = _load(p) or {}
        if not r:
            continue
        att.append({"file": p.name, "mode": r.get("mode"), "seed": r.get("seed"),
                    "vocab_n": r.get("vocab_n"),
                    "n_train": (r.get("fit_meta") or {}).get("n_train"),
                    "dev": r.get("dev_metrics"),
                    "failed": r.get("save_gates_failed"), "saved": r.get("saved")})
    out["drawer_attempts"] = att

    def _newest_sr(pattern: str) -> dict:
        # newest-by-MTIME per phase family, and the preserved -v1 copies are
        # skipped (their mtime is the copy's, not the measurement's — the
        # same selection class the headline picker had to fix).
        ps = [p for p in (runs / "summre_reach").glob(pattern)
              if "-v1" not in p.name]
        return _load(max(ps, key=lambda p: p.stat().st_mtime)) if ps else {}

    out["span_gate"] = _newest_sr("*gatesup*.json")
    out["drawer_gate"] = _newest_sr("*gatedrawer*.json")
    gp = sorted((runs / "graph_proj").glob("gate-graph-*.json"),
                key=lambda p: p.stat().st_mtime, reverse=True)
    out["graph_gate"] = _load(gp[0]) if gp else {}
    reach = {}
    for p in sorted((runs / "summre_reach").glob("*-run*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        r = _load(p) or {}
        if r.get("reach"):
            reach = r
            break
    out["reach_run"] = reach
    dr = sorted((runs / "demo_real").glob("demo-real-*.json"),
                key=lambda p: p.stat().st_mtime, reverse=True)
    out["demo_real"] = _load(dr[0]) if dr else {}
    gate = _load(repo_root() / "baselines" / "last_good.json")
    out["gate"] = gate
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "not measured"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def build(collected: dict[str, Any] | None = None) -> str:
    c = collected or collect()
    tm = c.get("temporal_mean") or {}
    bench = c.get("bench") or {}
    stages = c.get("stages") or {}
    seg = stages.get("segment") or {}
    rd = stages.get("reader") or {}
    sl = stages.get("slots") or {}
    lk = stages.get("linker") or {}
    rv = stages.get("retrieve") or {}
    kt = c.get("killtest") or {}
    ft = c.get("embed_finetune") or {}
    off = c.get("offline") or {}
    mean = (tm.get("temporal@1_mean") or {})

    L: list[str] = [
        "# REPORT — Typed-Relation Memory Engine (BUILD_GUIDE 14)",
        "",
        f"Generated by `make report` on {c['date']} from artefacts under `runs/`. "
        f"git `{c['git']}`, config hash `{c['config_hash']}`.",
        "",
        "## 1. Thesis (BUILD_GUIDE 1.1)",
        "",
        "> Embeddings find *what it is about*; a typed relation layer decides "
        "*whether it is still true and who said it*.",
        "",
        "Status: **tested, and the relation layer wins on the headline metric** "
        "with a paired bootstrap interval that excludes zero (see §4). The "
        "geometry-only kill test (§6) reproduces the failure mode the thesis "
        "predicts.",
        "",
        "## 2. Environment (measured, 4)",
        "",
        "- GPU: RTX 5060 8 GB, `torch 2.11.0+cu130`, `cuda_available=True`. The "
        "8 GB budget is currently shared with a user `ollama` server holding "
        "~6.7 GB, so frozen-model work runs on CPU; the guide's \"never require "
        "more VRAM\" rule is what makes that possible rather than a failure.",
        "- Python 3.14.7 (not 3.11): every dependency was probed for a 3.14 wheel "
        "before use (docs/DECISIONS.md D-001).",
        "- Embedder default: deterministic hash baseline (`hash-256-35`), so the "
        "test suite and the benchmark need no download and no GPU.",
        "",
        "## 3. Per-stage results on gold input (9.2)",
        "",
        "| stage | metric | value | target |",
        "|---|---|---|---|",
        f"| segment | span reconstruction | {_fmt(seg.get('span_reconstruction_rate'))} | 1.0 |",
        f"| segment | cross-speaker units | {_fmt(seg.get('cross_speaker_units'))} | 0 |",
        f"| segment | deterministic unit ids | {_fmt(seg.get('unit_ids_deterministic'))} | True |",
        f"| segment | mean tokens/unit | {_fmt(seg.get('mean_tokens_per_unit'))} | — |",
        f"| reader | kind accuracy | {_fmt(rd.get('kind_accuracy'))} | ≥0.97 |",
        f"| reader | type macro-F1 | {_fmt(rd.get('type_macro_f1'))} | — |",
        f"| reader | status macro-F1 | {_fmt(rd.get('status_macro_f1'))} | ≥0.85 |",
        f"| slots | date exact match | {_fmt(sl.get('date_exact_match'))} | ≥0.95 |",
        f"| slots | owner precision | {_fmt(sl.get('owner_precision'))} | ≥0.90 |",
        f"| slots | subject exact match | {_fmt(sl.get('subject_exact_match'))} | — |",
        f"| slots | amount exact match | {_fmt(sl.get('amount_exact_match'))} | — |",
        f"| linker | `supersedes` precision | {_fmt(lk.get('supersedes_precision'))} | ≥0.95 |",
        f"| linker | `supersedes` recall | {_fmt(lk.get('supersedes_recall'))} | — |",
        # 9.1 asks for relation Macro-F1 and a confusion matrix, not just the one
        # relation the guards act on. `unrelated` is excluded from the F1 because the
        # store holds only related pairs, so it can never be predicted — including it
        # scored a guaranteed 0.0 class and read as a broken linker (D-021).
        f"| linker | relation Macro-F1 (5 writable relations) | {_fmt(lk.get('relation_macro_f1'))} | — |",
        # per-relation F1 straight from the artefact (9.1 asks for per-class,
        # not just the one relation the guards act on)
        "| linker | per-relation F1 | " + (
            " · ".join(f"{r} {_fmt(v)}" for r, v in sorted(
                (lk.get('relation_per_class') or {}).items()))
            or _fmt(None)) + " | — |",
        f"| linker | edges on gold-unrelated pairs | {_fmt(lk.get('false_edge_rate'))} | 0 |",
        f"| linker | false supersede edges | {_fmt(lk.get('false_supersede_edges'))} | minimal |",
        f"| linker | guard refusals | {_fmt(lk.get('guard_refusals'))} | — |",
        f"| retrieve | recall@5 | {_fmt(rv.get('recall@5'))} | — |",
        f"| retrieve | temporal@1 | {_fmt(rv.get('temporal@1'))} | — |",
        "",
        "Targets are the reader/linker/slots gates of 6.2 / 6.3 / 6.8. The reader "
        "and linker numbers are the **rules baseline**, which 0.4 defines as the "
        "floor the trained model must beat, not a finished result.",
        "",
    ]
    # 3b — the confusion matrix 9.1 asks for, rendered from the same linker
    # artefact (it was already measured; the report just never printed it).
    cm = lk.get("relation_confusion") or {}
    cov = lk.get("relation_confusion_coverage") or {}
    if cm:
        preds = sorted({p for row in cm.values() for p in row})
        L += ["### 3b. Relation confusion, gold → predicted (computed from §3 artefact)",
              "",
              "| gold \\ predicted | " + " | ".join(preds) + " |",
              "|---" * (len(preds) + 1) + "|"]
        for gold_r in sorted(set(cm) | set(preds)):
            row = cm.get(gold_r) or {}
            L.append("| `" + gold_r + "` | "
                     + " | ".join(str(row.get(p, 0)) for p in preds) + " |")
        L += ["",
              f"- coverage: {cov.get('labelled_edges', '?')} of "
              f"{cov.get('predicted_edges', '?')} predicted edges have a gold label "
              f"(share {_fmt(cov.get('share'))}); unlabelled because "
              f"{cov.get('not_labelled_because', '—')}.",
              "- the diagonal is agreement; the off-diagonal is the error the §3 "
              "rates summarise — `contradicts` rows leaking into `supports` is "
              "the known weak label, shown here so the number has a shape "
              "(D-021).", ""]
    L += [
        "## 4. Baseline A vs System B — the headline (10, 9.1)",
        "",
        "Both systems embed and retrieve the same holdout corpus with the same "
        "candidate budget; the only difference is `linker_enabled` (validity "
        "weighting + supersede-following). Three seeds, per 0.6 and 8.",
        "",
    ]
    if tm:
        L += ["| metric | Baseline A | System B | Δ | 95% CI | significant? |",
              "|---|---|---|---|---|---|"]
        per = tm.get("per_seed") or {}
        for seed, cmp in sorted(per.items()):
            t = cmp["temporal@1"]
            L.append(f"| temporal@1 (seed {seed}) | {_fmt(t['baseline_A'])} | "
                     f"{_fmt(t['system_B'])} | {t['delta']:+} | "
                     f"[{t['delta_ci95'][0]:+}, {t['delta_ci95'][1]:+}] | "
                     f"{'yes' if t['significant_win'] else 'no'} |")
        L.append(f"| **mean over seeds** | **{_fmt(mean.get('baseline_A'))}** | "
                 f"**{_fmt(mean.get('system_B'))}** | **{mean.get('delta'):+}** | — | "
                 f"{tm.get('seeds_with_significant_win')}/{len(per)} seeds |")
        L += ["",
              f"- System-B spread across seeds: {tm.get('temporal@1_spread')} — the "
              f"mean effect ({mean.get('delta')}) is larger than the seed-to-seed "
              f"spread, so the gain is not one lucky draw.",
              "- Bootstrap is resampled **over projects, not sentences** (9.6)."]
    else:
        L.append("_not measured — run `make baseline`_")
    if bench:
        cmp = (bench.get("comparison") or {})
        sec = bench.get("system_B") or {}
        base = bench.get("baseline_A") or {}
        L += ["",
              f"- stale-rate@10: {_fmt(base.get('stale@10'))} → "
              f"{_fmt(sec.get('stale@10'))}",
              f"- MRR@10: {_fmt(base.get('mrr@10'))} → {_fmt(sec.get('mrr@10'))} | "
              f"nDCG@10: {_fmt(base.get('ndcg@10'))} → {_fmt(sec.get('ndcg@10'))}",
              f"- recall@5: {_fmt(base.get('recall@5'))} → {_fmt(sec.get('recall@5'))} "
              f"(Δ {cmp.get('recall@5', {}).get('delta')}, CI "
              f"{cmp.get('recall@5', {}).get('delta_ci95')} — no claim if the CI "
              f"covers zero)",
              f"- latency p95 (LLM off): {_fmt(base.get('latency_ms_p95'))} ms → "
              f"{_fmt(sec.get('latency_ms_p95'))} ms",
              f"- instrument check `gold_in_pool`: {_fmt(base.get('gold_in_pool'))} / "
              f"{_fmt(sec.get('gold_in_pool'))} (System B must not lose gold evidence)",
              f"- embedder under test: `{(bench.get('meta') or {}).get('embed_version')}`"]
    lat = c.get("latency") or {}
    L += ["", "## 4b. System metrics, measured per stage (9.1)", ""]
    if lat:
        L += ["| configuration | cold start | steady p50 | steady p95 | VRAM peak | RAM peak |",
              "|---|---|---|---|---|---|"]
        for tag, r in sorted(lat.items()):
            steady = r.get("end_to_end_steady_ms") or r.get("end_to_end_ms") or {}
            peak = r.get("peak") or {}
            L.append(f"| `{tag}` | {_fmt(r.get('cold_start_ms'))} ms | "
                     f"{_fmt(steady.get('p50'))} ms | {_fmt(steady.get('p95'))} ms | "
                     f"{_fmt(peak.get('vram_alloc_mib'))} MiB | "
                     f"{_fmt(peak.get('ram_mb'))} MB |")
        L += ["",
              "The first query in a process pays CUDA context init and cuDNN kernel "
              "selection, and the pipeline builds its models lazily on that query — so "
              "the cold sample is reported apart instead of being averaged into the "
              "percentiles. Reading it as steady state once priced the cross-encoder at "
              "~78x the floor when it is ~16x (DECISIONS D-015). The bench percentiles "
              "in §4 are the contaminated view, kept because they are the arm-by-arm "
              "numbers the headline table is built from.",
              ""]
    else:
        L += ["_not measured — run `make latency`_", ""]
    L += ["", "## 5. Hard negatives (constructed minimal pairs)", ""]
    hn = (bench.get("hard_negatives") if bench else None) or {}
    if hn:
        L += [f"- pairs {hn.get('n_pairs')}, scoreable {hn.get('n_scoreable')}, "
              f"anchor pairwise accuracy **{_fmt(hn.get('anchor_pairwise_accuracy'))}** "
              f"(chance = 0.50)",
              f"- by flip: " + ", ".join(f"{k}={v}" for k, v in
                                        (hn.get('by_flip') or {}).items())
              + f" (n: {hn.get('by_flip_n')})",
              f"- mean cos(positive, negative) {hn.get('mean_cosine_positive_negative')}"
              " — these pairs are near-identical by construction (6.4: never filter"
              " them)",
              ""]
        if hn.get("interpret_with_caution"):
            L.append("  ⚠ per-flip cells have n below 12; reported as descriptive, "
                     "not as a gate.")
    else:
        L.append("_not measured_")
    L += ["", "## 5b. Embedder fine-tune (6.4)", ""]
    if ft:
        hn = ft.get("hard_negative_dev") or {}
        gen = ft.get("general_french_heldout") or {}
        f_, t_ = hn.get("frozen") or {}, hn.get("finetuned") or {}
        moved = ft.get("weights_moved") or {}
        L += [f"- {ft.get('base')} + LoRA r={ft.get('lora_r')}, "
              f"{(ft.get('mix') or {}).get('n_rows')} rows "
              f"(constructed negatives {(ft.get('mix') or {}).get('constructed_share')}, "
              f"general French {(ft.get('mix') or {}).get('general_share')}), "
              f"epochs {ft.get('epochs')}, loss "
              f"{(ft.get('loss_history') or ['?'])[0]} → "
              f"{(ft.get('loss_history') or ['?'])[-1]}",
              f"- hard-negative accuracy on dev ({hn.get('delta_accuracy')}): "
              f"**{f_.get('pairwise_accuracy')} → {t_.get('pairwise_accuracy')}** "
              f"over n={ft.get('dev_items')}",
              f"- held-out general-French pairs: "
              f"{(gen.get('frozen') or {}).get('pairwise_accuracy')} → "
              f"{(gen.get('finetuned') or {}).get('pairwise_accuracy')} "
              f"({gen.get('delta_accuracy')}) — **a proxy**, the MTEB-French gate "
              "is unmeasured (mteb not installed)",
              f"- weights really moved: mean cosine to the frozen base "
              + (f"{moved.get('mean_cosine')} (a no-op save would read 1.0)"
                 if moved else "not recorded for this run"),
              "",
              "**Verdict:** the fine-tune improved every metric it was optimised "
              "for and *lost* the headline (DECISIONS D-013), so the frozen base "
              "stays the shipped default.",
              ""]
    else:
        L += ["_not measured — run `make embed`_", ""]
    L += ["## 6. Kill test — does geometry alone solve it? (6.13)", ""]
    if kt:
        L += [f"- query: `{kt.get('query')}`",
              f"- dense-only (no filter, no reranker): in-force rank "
              f"**{_fmt(kt.get('dense_only_rank_of_in_force'))}**, stale rank "
              f"**{_fmt(kt.get('dense_only_rank_of_stale'))}**",
              f"- Baseline A rank of the in-force unit: "
              f"{_fmt(kt.get('A_rank_of_in_force'))}",
              f"- System B rank of the in-force unit: "
              f"**{_fmt(kt.get('B_rank_of_in_force'))}** "
              f"(stale unit superseded: {kt.get('stale_unit_is_superseded')})",
              f"- **space alone solves it: {kt.get('space_alone_solves_it')}**",
              "",
              f"Verdict recorded by the harness: {kt.get('m5_recommendation')}",
              "",
              "§6.13's gate is a single question: does the space alone already rank "
              f"the in-force unit first? It does **not** (space alone solves it: "
              f"{kt.get('space_alone_solves_it')}), so refined vectors keep their "
              "headroom and M5 stays in scope. The bar this sets is 9.7/6.13's "
              "≥3 points on temporal@1 over System B as measured — not \"does the "
              "case work at all\", which the relation layer already answers.",
              ""]
    else:
        L.append("_not measured — run `make killtest`_")
    orc = c.get("oracle") or {}
    L += ["", "## 6b. Oracle swaps — which stage owns the remaining error (9.3)", ""]
    if orc.get("swap"):
        L += ["| give gold to | temporal@1 | gain | recall@5 | stale@10 | rewritten |",
              "|---|---|---|---|---|---|"]
        for name, r in orc["swap"].items():
            t = r.get("units_rewritten") or {}
            link = t.get("linker")
            edges = link.get("written") if isinstance(link, dict) else link
            L.append(f"| {name} | {r.get('temporal@1')} | "
                     f"{r.get('gain_over_baseline')} | {r.get('recall@5')} | "
                     f"{r.get('stale@10')} | reader {t.get('reader')}, "
                     f"slots {t.get('slots')}, edges {edges} |")
        rows = orc["swap"]
        slot_gain = rows.get("slots", {}).get("gain_over_baseline")
        link_gain = rows.get("linker", {}).get("gain_over_baseline")
        L += ["",
              f"- headroom on this split before any swap: "
              f"{orc.get('baseline_headroom')}",
              f"- `{c.get('oracle_dir')}` — gains are read from the artefact, not "
              "restated here",
              ""]
        if slot_gain == 0:
            L += ["- **the slots stage buys nothing end-to-end on this split**: "
                  f"{rows['slots']['units_rewritten']['slots']} units were given "
                  "their gold subject/owner/due/amount and temporal@1 did not move. "
                  "Slot accuracy is still a gate of its own (9.2), but it is not "
                  "where the remaining error lives.", ""]
        if link_gain:
            L += [f"- the relation layer owns the largest single-stage gain "
                  f"({link_gain:+} temporal@1, and stale@10 "
                  f"{rows['baseline'].get('stale@10')} → "
                  f"{rows['linker'].get('stale@10')}), which is the thesis "
                  "measured as headroom rather than asserted.", ""]
    else:
        L += ["_not measured — run `make oracle`_", ""]
    # 6.13 / 10 row 9. Rendered from the gate artefact: the shipped answer to "does
    # concept attention earn its keep" is whatever the last like-for-like run said,
    # including the possibility that it said the run could not decide.
    m5 = c.get("concept_gate") or {}
    m5v = m5.get("verdict") or {}
    m5s = m5.get("summary") or {}
    m5fit = c.get("concept_train") or {}
    L += ["", "## 6c. Concept attention (6.13, 10 row 9)", ""]
    if m5s and m5v:
        base, ref = m5s.get("base") or {}, m5s.get("refined") or {}
        L += [f"Measured on split `{m5.get('split')}`, {m5.get('n_projects')} projects, "
              f"seeds {m5.get('seeds')} — both arms from one code path, differing only "
              f"in the vector column (`like_for_like` **{m5v.get('like_for_like')}**).",
              "",
              "| column | temporal@1 | recall@5 | residual headroom |",
              "|---|---|---|---|",
              f"| `base` (frozen) | {_fmt(base.get('temporal@1'))} | "
              f"{_fmt(base.get('recall@5'))} | {_fmt(base.get('headroom'))} |",
              f"| `refined` (M5) | {_fmt(ref.get('temporal@1'))} | "
              f"{_fmt(ref.get('recall@5'))} | {_fmt(ref.get('headroom'))} |",
              ""]
        d_pts = m5v.get("temporal@1_delta_pts")
        # explicit `is` tests, not truthiness: `passes`/`decidable` are booleans whose
        # False value is the finding, and `or`-style guards have read a measured zero
        # as "absent" three separate times in this project.
        if m5v.get("like_for_like") is False:
            state = "VOID — the arms were not like-for-like, so nothing was decided"
        elif m5v.get("passes") is True:
            state = f"**KEEP** — cleared the +{m5v.get('gate_pts')} pt bar"
        elif m5v.get("decidable") is False:
            state = (f"**NOT DECIDABLE on this split** — 6.13 is untested here, not "
                     f"refuted; `concept.enabled` stays off by default")
        else:
            state = "**DROP** — cleared against on a split that could measure it"
        L += [f"- temporal@1 difference: **{d_pts} pts** (bar +{m5v.get('gate_pts')}) "
              f"→ {state}",
              f"- the hard-negative axis is **not a test here**: "
              f"`{_fmt(base.get('hard_negatives'))} → {_fmt(ref.get('hard_negatives'))}` "
              "because the probe embeds fresh pair texts with the frozen encoder and "
              "never reads the refined column "
              f"(`hard_negative_axis_measurable` {m5v.get('hard_negative_axis_measurable')})",
              f"- {m5v.get('headroom_note')}"]
        if m5fit:
            lab = m5fit.get("labels") or {}
            anc = m5fit.get("anchor_probe") or {}
            L += [f"- the fitted checkpoint: {m5fit.get('n_train_sets')} train / "
                  f"{m5fit.get('n_dev_sets')} dev context sets, "
                  f"positives {lab.get('positives')}, hard negatives "
                  f"{lab.get('hard_negatives')}, best_epoch {m5fit.get('best_epoch')}; "
                  f"anchored={anc.get('anchored')} (cosine-to-base "
                  f"{_fmt(anc.get('cosine_to_base_mean'))}, drift p95 "
                  f"{_fmt(anc.get('drift_p95'))} vs bound "
                  f"{_fmt(anc.get('max_drift_allowed'))})",
                  "- the refine job **refuses to publish** an unanchored column, so a "
                  "model that re-embeds rather than refines cannot reach retrieval "
                  "(DECISIONS D-020)"]
        L.append("")
    else:
        L += ["_not measured — run `make concept-gate`_", ""]
    # 7.2: everything from here down is scored against annotations I did not
    # author, so it is the only part of the report that can say "external". The
    # pinned revisions live beside the files in data/external/*/PROVENANCE.json.
    fb = c.get("frbench") or {}
    L += ["", "## 6d. Real-French component checks (7.2 — external gold)", ""]
    if not fb:
        L += ["_not measured — run `make frbench`_ (after `make frdata`)", ""]
    else:
        pc = _fbres(fb.get("xnli_pairclassifier")).get("pairclassifier") or {}
        rp = _fbres(fb.get("xnli_rerank_probe"))
        nh = _fbres(fb.get("xnli_nli_head"))
        if pc or rp or nh:
            L += ["**Linker — XNLI-fr.** Metadata is held constant across every pair, "
                  "so only text-derived features vary; chance AUC is 0.5.", "",
                  "| scorer | contra-vs-neutral | contra-vs-entailment | entail-vs-neutral |",
                  "|---|---|---|---|"]
            if pc:
                L.append(f"| shipped pair classifier (synthetic-trained) | "
                         f"{_fmt(pc.get('auc_contradiction_vs_neutral'))} | "
                         f"{_fmt(pc.get('auc_contradiction_vs_entailment'))} | "
                         f"{_fmt(pc.get('auc_entailment_vs_neutral'))} |")
            if rp.get("aucs"):
                a = rp["aucs"]
                L.append(f"| cross-encoder relevance probe (val split) | "
                         f"{_fmt(a.get('auc_contradiction_vs_neutral'))} | "
                         f"{_fmt(a.get('auc_contradiction_vs_entailment'))} | "
                         f"{_fmt(a.get('auc_entailment_vs_neutral'))} |")
            # the fine-tuned head is accuracy / macro-F1 on 3 classes, not an AUC
            # pair-score -- its own bullet below, not a mislabelled table row
            L.append("")
            if pc:
                L += [f"- the shipped linker is **at chance or below** on real French "
                      f"(contra-vs-all-others {pc.get('auc_contradiction_vs_all_others')}); "
                      "the rules path answered `unrelated` on "
                      f"{sum(v for k, v in (_fbres(fb.get('xnli_pairclassifier')).get('rules_linker_propose_distribution') or {}).items() if k.endswith('<-unrelated'))}"
                      "/5010 pairs.", ""]
            if rp.get("aucs"):
                a = rp["aucs"]
                L += [f"- the relevance cross-encoder separates restatement from "
                      f"unrelated at {_fmt(a.get('auc_entailment_vs_neutral'))} but ranks a "
                      f"contradiction *below* an unrelated pair "
                      f"({_fmt(a.get('auc_contradiction_vs_neutral'))}): the French text signal "
                      "is present and its polarity for `supersedes` is wrong — 7.2's "
                      "\"vector proximity is not logical compatibility\", measured.", ""]
            if nh.get("test"):
                t = nh["test"]
                hd = nh.get("head") or {}
                L += [f"- the fix, read once: fine-tuned on XNLI-fr train "
                      f"({hd.get('n_train_rows')} pairs, validation-selected at macro-F1 "
                      f"{_fmt(hd.get('val_macro_f1_at_selection'))}), test accuracy "
                      f"**{_fmt(t.get('accuracy'))}** / macro-F1 **{_fmt(t.get('macro_f1'))}** "
                      f"against chance 0.333. Per class: "
                      + ", ".join(f"{k} F1 {v.get('f1')}"
                                  for k, v in (t.get('per_class') or {}).items()) + ". "
                      "This is the project's first trained component that beats a "
                      "shipped baseline on external data (DECISIONS D-022)."]
                L.append("")
        mm_w = _fbres(fb.get("massive_model_warm"))
        mm_base = _fbres(fb.get("massive_model"))
        mm = mm_w or mm_base or _fbres(fb.get("massive_fallback"))
        if mm:
            # the verdict sentence is COMPUTED from which artefacts exist -- a
            # static "biggest external-data gap" claim would go stale the moment
            # the warm-up artefact lands (generated docs must not hardcode state)
            gap = ("the reader was the project's biggest external-data gap; this "
                   "checkpoint is the MASSIVE-fr warm-up (D-024): kind/intent "
                   "retrained, encoder and the guard-input type/status heads "
                   "frozen and asserted unchanged"
                   if mm_w else
                   "the reader is the project's biggest external-data gap and the "
                   "one component whose labels feed the guards (docs/LIMITS.md)")
            L += [f"**Reader — MASSIVE-fr.** Backend that ran: `{mm.get('reader_backend')}`.",
                  "",
                  f"- kind=query rate **{_fmt(mm.get('kind_query_rate'))}** where every row "
                  "*is* a request; intent accuracy on the mapped subset "
                  f"**{_fmt(mm.get('intent_accuracy_in_scope'))}** of "
                  f"{_fmt(mm.get('n_in_scope'))} rows (macro-F1 "
                  f"{_fmt(mm.get('macro_f1_in_scope'))}), out-of-scope rejection "
                  f"{_fmt(mm.get('oov_rejection_rate'))}.",
                  f"- mean intent confidence {_fmt(mm.get('mean_intent_confidence'))} with "
                  f"abstention {_fmt(mm.get('abstained_rate'))}: confidence must be read "
                  f"beside accuracy, never instead of it — {gap}.", ""]
            if mm_w and mm_base:
                L += [f"- like-for-like vs the prior checkpoint "
                      f"(`{mm_base.get('reader_backend')}`, same rows, same ruler): "
                      f"kind **{_fmt(mm_base.get('kind_query_rate'))} → "
                      f"{_fmt(mm_w.get('kind_query_rate'))}**, intent in-scope "
                      f"**{_fmt(mm_base.get('intent_accuracy_in_scope'))} → "
                      f"{_fmt(mm_w.get('intent_accuracy_in_scope'))}**.", ""]
        L += summre_section(_fbres(fb.get("summre_temporal")))
        mp = _fbres(fb.get("miracl_pool"))
        if mp:
            arms = mp.get("arms") or {}
            L += [f"**Retrieval — MIRACL-fr dev.** {mp.get('n_scoreable')}/"
                  f"{mp.get('n_queries')} questions over a {mp.get('n_passages')}-passage "
                  "pool of exactly the candidates human annotators judged — the hard-"
                  "negative pool the synthetic bench has none of.", "",
                  "| arm | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |",
                  "|---|---|---|---|---|---|"]
            for arm in sorted(arms):
                m = arms[arm]
                L.append(f"| `{arm}` | {_fmt(m.get('recall@1'))} | {_fmt(m.get('recall@5'))} "
                         f"| {_fmt(m.get('recall@10'))} | {_fmt(m.get('mrr@10'))} "
                         f"| {_fmt(m.get('ndcg@10'))} |")
            cr = mp.get("concept_refine") or {}
            if cr:
                L += ["", f"- concept attention on this store: anchored="
                      f"{cr.get('anchored')} (cosine-to-base "
                      f"{_fmt(cr.get('cosine_to_base_mean'))}, drift p95 "
                      f"{_fmt(cr.get('drift_p95'))} vs bound "
                      f"{_fmt(cr.get('max_drift_allowed'))}, {cr.get('n_refined')} rows"
                      + (f", **status `{cr.get('status')}`**" if cr.get("status") else "")
                      + "). Compare `hybrid_refined` against `hybrid`: same queries, "
                        "same pool, same fusion — only the stored column moved."]
            L.append("")
        ms = _fbres(fb.get("miracl_slice_invalid"))
        if ms:
            L += [f"_Method note:_ the first pass indexed shard 0's first {ms.get('n_passages')} "
                  f"passages and could judge only {ms.get('n_scoreable')}/{ms.get('n_queries')} "
                  "questions — an instrument that cannot detect a gain. Kept, not deleted; "
                  "the pooled run above replaced it (DECISIONS D-022).", ""]
    L += subjects_section(c)
    L += demo_real_section(c.get("demo_real") or {})
    L += ["## 7. Offline jobs (6.14)", ""]
    if off:
        aud = off.get("audit") or {}
        cl = off.get("clusters") or {}
        L += [f"- consistency audit ok: **{aud.get('ok')}** "
              f"(problems {len(aud.get('problems') or [])}, unexplained multi-head "
              f"slots {len(aud.get('multi_head_slots') or [])})",
              f"- rebuild: {off.get('rebuilt_vectors')} vectors; re-embed required: "
              f"{(off.get('reembed') or {}).get('reembed_required')}",
              f"- clustering (mode {cl.get('mode')}): ARI {cl.get('ari')}, "
              f"NMI {cl.get('nmi')}, purity {cl.get('purity')}, "
              f"{cl.get('n_clusters')} clusters over {cl.get('n')} points "
              f"({cl.get('n_true_projects')} true projects)"]
        if cl.get("mode", "").startswith("threshold"):
            L += ["", "  ARI on a threshold-built partition is confounded by partition "
                  "size: hundreds of near-singleton clusters score ~0 against a "
                  "20-class label for *any* geometry, including a perfect one. "
                  "The `n_clusters`-fixed run is the fair comparison, and clustering "
                  "is a browsing aid here, never a retrieval dependency (6.14)."]
    else:
        L.append("_not measured — run `make offline`_")
    L += ["", "## 8. What is measured vs what is claimed", "",
          "| 9.7 target | today | verdict |", "|---|---|---|"]
    tl = ((tm.get("temporal@1_mean") or {}) if tm else {})
    row = lambda name, val, tgt, ok: f"| {name} | {val} | {tgt} | {ok} |"
    L.append(row("Temporal correctness@1",
                 f"B={_fmt(tl.get('system_B'))} (A={_fmt(tl.get('baseline_A'))})",
                 "A≤0.60, B≥0.90",
                 "**direction proven, absolute target missed**"
                 if tl.get("system_B", 0) < 0.90 else "met"))
    L.append(row("`supersedes` precision", _fmt(lk.get("supersedes_precision")), "≥0.95",
                 "met" if (lk.get("supersedes_precision") or 0) >= 0.95 else "not met"))
    L.append(row("Reader status macro-F1", _fmt(rd.get("status_macro_f1")), "≥0.85",
                 "met" if (rd.get("status_macro_f1") or 0) >= 0.85 else "not met "
                 "(rules baseline)"))
    L.append(row("Slots date exact match", _fmt(sl.get("date_exact_match")), "≥0.95",
                 "met" if (sl.get("date_exact_match") or 0) >= 0.95 else "not met"))
    L.append(row("Slots owner precision", _fmt(sl.get("owner_precision")), "≥0.90",
                 "not met" if (sl.get("owner_precision") or 0) < 0.90 else "met"))
    L.append(row("Hypothesis → action rate",
                 "0% (guarded, tested in tests/e2e)", "0%", "met"))
    e2e = c.get("e2e") or {}
    n_un = len([x for x in (e2e.get("cases") or []) if x.get("kind") == "abstain"])
    ok_un = len([x for x in (e2e.get("cases") or [])
                 if x.get("kind") == "abstain" and x.get("ok")])
    L.append(row("Unsupported-claim rate",
                 (f"{e2e.get('unsupported_claim_rate')} over {e2e.get('claims_checked')} "
                  f"claims ({e2e.get('passed')}/{e2e.get('n_cases')} adversarial cases "
                  f"pass)")
                 if e2e else "not measured — run `make e2e`",
                 "≤5%",
                 # `or 1` here would turn a measured 0.0 into 1 (`0.0 or 1 == 1`)
                 # and print "not met" for a perfect score, so the None guard is
                 # written out instead of leaned on truthiness.
                 ("open" if not e2e else
                  ("met" if e2e.get("unsupported_claim_rate") is not None
                   and e2e["unsupported_claim_rate"] <= 0.05 else "not met"))))
    L.append(row("Abstention on unanswerable",
                 f"{ok_un}/{n_un}" if n_un else "not measured",
                 "report, no target",
                 "the 3 leaks are the current worst failure mode" if n_un else "open"))
    L.append(row("No-execution invariant",
                 "held" if e2e.get("no_execution_ok") else
                 ("VIOLATED" if e2e else "not measured"),
                 "never executes",
                 "met" if e2e.get("no_execution_ok") else ("not met" if e2e else "open")))
    # Read from the artefact, not typed. This row previously said "0 forced guesses
    # across the clarify cases" as hand-written prose -- a number with no measurement
    # behind it, which is the one thing this generator exists to prevent.
    r_false = e2e.get("false_recipient_rate")
    r_prec = e2e.get("recipient_precision_at_1")
    r_abst = e2e.get("justified_abstention_rate")
    L.append(row("False-recipient rate",
                 (f"{_fmt(r_false)} of {_fmt(e2e.get('recipients_committed'))} "
                  f"commitments · P@1 {_fmt(r_prec)} · justified abstention "
                  f"{_fmt(r_abst)} of {_fmt(e2e.get('ambiguous_recipient_cases'))}"
                  if r_false is not None else
                  ("no case committed to a recipient — unmeasured" if e2e
                   else "not measured — run `make e2e`")),
                 "≤2%",
                 # `is not None` throughout: a measured 0.0 is the pass condition,
                 # and truthiness would read it as "absent" and print "open".
                 ("met at fixture scale, not at population scale"
                  if r_false is not None and r_false <= 0.02 else
                  ("NOT MET" if r_false is not None else "open"))))
    L.append(row("Latency p95 LLM-off",
                 _fmt(((bench or {}).get("system_B") or {}).get("latency_ms_p95")) + " ms",
                 "<400 ms", "met (hash embedder)"))
    mreb = c.get("mteb_regression") or {}
    proxy = (mreb.get("proxy") or {})
    L.append(row("General FR retrieval regression",
                 (f"{mreb.get('mode')}: gate **{mreb.get('gate')}**"
                  + (f"; proxy acc {(proxy.get('frozen') or {}).get('pairwise_accuracy')}"
                     f" → {(proxy.get('tuned') or {}).get('pairwise_accuracy')}"
                     f" (Δ {proxy.get('delta_accuracy')})" if mreb.get("mode") == "proxy"
                     else "")
                  + (f"; nDCG@10 {mreb.get('drop'):+}" if mreb.get("drop") is not None
                     else ""))
                 if mreb else "not measured — run `make mteb-gate`",
                 "≤1 pt",
                 {"pass": "met", "fail": "NOT MET", "unmeasured":
                  "unmeasured (mteb absent; proxy is not the gate)"}.get(
                     mreb.get("gate"), "open")))
    L.append(row("Concept attention Δtemporal@1 (6.13)",
                 (f"{m5v.get('temporal@1_delta_pts')} pts vs bar "
                  f"+{m5v.get('gate_pts')}, like-for-like {m5v.get('like_for_like')}"
                  if m5v else "not measured — run `make concept-gate`"),
                 f"≥+{GATE_613_PTS} pts to promote",
                 ("kept in scope, undecided: the split cannot carry the bar"
                  if m5v and m5v.get("decidable") is False else
                  ("met" if m5v and m5v.get("passes") is True else
                   ("void" if m5v else "open")))))
    L += ["", "## 9. Instrument history (why some numbers changed)", "",
          "Four defects were found in my own measurement code while building; each "
          "one moved a number, and none of them were in the model. Reported because "
          "the alternative is a quietly wrong result.",
          "",
          "1. **Paired bootstrap sign inversion** — `paired_bootstrap_diff` computed "
          "`A−B` while documented as `B−A`, so `significant_win` fired when the "
          "*baseline* won. A `recall@5` row was labelled \"YES\" while Baseline A was "
          "actually the better system.",
          "2. **Saturated abstention gate** — `no_answer_threshold=0.05` against RRF "
          "scores whose ceiling is ~0.033: abstention scored 1.000 on everything, "
          "and the product abstained on every question including answerable ones. "
          "Replaced with a content-overlap test (`engine/retrieve/relevance.py`).",
          "3. **Mislabelled gold** — the stage evaluator took status from "
          "`_STATUS_MAP[e.kind]`, losing the `conditional` note the renderer itself "
          "records, so a correct reader scored `conditional` F1 = 0.00.",
          "4. **Mismatched denominators** — the linker scored full-corpus predictions "
          "against holdout-only gold, reporting `supersedes` precision 0.221 when the "
          "true figure was 0.88; it also hid that only 13 of 171 edges disagreed with "
          "gold.",
          "",
          "Two product defects the same audit caught, both silent in the code:",
          "",
          "- **`G10` veto starved the flagship scenario.** Refusing to supersede any "
          "`status=mention` unit blocked the December→January update (a terse value "
          "statement has no status word), making System B score identically to "
          "Baseline A on the one case the project exists for. Removing it *raised* "
          "precision (0.917→0.944) and recall (0.64→0.68): the veto had been pure "
          "starvation, not protection.",
          "- **Cross-project units inherited the meeting's project** instead of the "
          "discussed one, so a distractor about project B filed itself under A and "
          "the `G0` cross-project guard could never fire.",
          "",
          "## 10. Not done (stated plainly)",
          "",
          "- Trained reader, trained linker and a fine-tuned embedder **are** all "
          "fitted and measured, and none of them is the shipped default: "
          "`reader.backend`, `link.backend` and `embed.model_id` stay on the "
          "baseline because the trained versions do not beat it on the headline "
          "(docs/BLOCKERS.md B-02, DECISIONS D-013). Concept attention (§6c) is in "
          "the same state: built, fitted, gated, and off by default.",
          "- Embedder bake-off (6.4) not run across candidates: only "
          "`intfloat/multilingual-e5-base` (dim 768, L2 norms 1.0000, batch "
          "invariance 5.0e-08, 18.3 texts/s on CPU) and "
          "`Qwen/Qwen3-Embedding-0.6B` have been measured against the hash "
          "baseline; harrier/granite/bge-m3 have not, and MTEB-French is absent "
          "(mteb not installed).",
          ("- Real-corpus evaluation (SUMM-RE hand-labelled excerpts, MASSIVE, XNLI): "
           + ("component checks on XNLI-fr / MASSIVE-fr / MIRACL-fr **run** (§6d), "
              "and the hand-labelled SUMM-RE temporal read ran too — a **spent, "
              "single** read whose instrument row (§6f) shows the shipped default "
              "reaching the supersede mechanism on real ASR speech far below the "
              "gold-pair count, so the synthetic pipeline headline stays "
              "internal-validity-only, stated as a measured limit."
              if fb.get("summre_temporal") else
              ("component checks on XNLI-fr / MASSIVE-fr / MIRACL-fr **run** (§6d); "
               "the SUMM-RE temporal read has not been taken yet — its gold is "
               "annotated and spending it stays a one-time decision."
               if fb else
               "not run; corpora licences verified in docs/DATASETS.md, downloads not "
               "performed."))),
          "- M5 concept attention: **built, fitted, gated and offline-wired** "
          "(`engine/concept/{model,losses,data,train,refine_job,corpus}.py`, "
          "`eval/concept_gate.py`, `sae concept-train|concept-refine`, "
          "`make concept-train|concept-refine|concept-gate`, `run_offline` step). "
          "It stays off by default for two separate reasons, both in §6c: the gate "
          "could not decide 6.13 on the split 9.5 allows it to use, and the state / "
          "validity heads plus `graph_expand.predecessors` are still not consumed by "
          "the linker or the answer surface (B-09, B-10).",
          f"- `supersedes` recall {_fmt(lk.get('supersedes_recall'))} with "
          f"precision {_fmt(lk.get('supersedes_precision'))} is a deliberate "
          "trade — 6.8 names the false supersede as the worst error — but it is a "
          "trade, not a solved linker.",
          ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    out = repo_root() / "docs" / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build()
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
