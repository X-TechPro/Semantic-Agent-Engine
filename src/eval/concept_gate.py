"""Concept-attention gate (6.13 / 10 row 9) — does refining the space earn its keep?

The guide's bar is explicit: keep M5 only if it beats the plain version by >= 3
points on temporal@1 or on hard-negative accuracy; otherwise drop it and say so.
Deciding that needs the two arms to differ in **one** thing, so this module runs
`eval.temporal.run_bench` twice over the same seeds from the same code, asking for
`base` and then `refined`, and compares the recorded numbers.

Why not compare against the artefacts already on disk: `docs/BLOCKERS.md` carries
the e5 lesson — a run made before independent scoping existed is a different
experiment even when its config looks the same. A stale base artefact and a fresh
refined one can disagree for reasons that have nothing to do with the vector column,
and a +/-0.0 result is exactly the kind of claim that has to survive that objection.

Why dev, and why that is not the whole story: 9.5 forbids selecting on test, and
keep-or-drop *is* a selection, so the decision is made here. But the bench prints
its own headroom for a reason — on dev only ~2% of questions had a stale rank-1
after scoping, so dev is a weak probe of this particular layer. The run therefore
reports headroom beside every number, and a null here is labelled as a null on a
low-headroom split rather than as a refutation of the hypothesis.

`out_dir` is passed explicitly so the scratch runs cannot touch the canonical
headline artefacts or the regression gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from eval.temporal import run_bench

GATE_PTS = 3.0          # 6.13: temporal@1 or hard-negative accuracy
COLUMNS = ("base", "refined")


def _pick(res: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparable numbers out of one run's artefact.

    `hard_negatives` is read from the FLAT keys, not from an arm sub-dict — the
    first version asked for `hard["system_B"]`, which does not exist, so it would
    have reported `None` for both columns and turned a missing key into a clean
    zero difference on the gate's second axis.

    And it must be read with its limitation attached: `_hard_negative_eval` embeds
    the *pair texts* with the frozen embedder and never consults `vectors.refined`.
    Concept attention refines stored unit vectors only, so this probe is insensitive
    to the treatment by construction. Reporting it as though a null there were
    evidence would be the e5 mistake again — a number that cannot move is not a test.
    """
    c, m = res["comparison"], res["meta"]
    hard = res.get("hard_negatives") or {}
    sysb = res.get("system_B") or {}
    return {
        "column": m.get("vector_column", "base"),
        "temporal@1": c["temporal@1"]["system_B"],
        "temporal@1_baseline_A": c["temporal@1"]["baseline_A"],
        "delta_vs_A": c["temporal@1"]["delta"],
        "ci95": c["temporal@1"].get("delta_ci95"),
        "significant_vs_A": c["temporal@1"].get("significant_win"),
        "recall@5": c["recall@5"]["system_B"],
        "gold_in_pool": sysb.get("gold_in_pool"),
        "headroom": c.get("headroom_pct", {}).get("system_B"),
        "hard_negatives": hard.get("anchor_pairwise_accuracy"),
        "hard_negatives_n_scoreable": hard.get("n_scoreable"),
        "hard_negatives_sensitive_to_column": False,
        "n_questions": m.get("n_questions"),
        "embed_version": m.get("embed_version"),
        "reader_backend": m.get("reader_backend"),
        "rerank_backend": m.get("rerank_backend"),
        "config_hash": m.get("config_hash"),
        "concept": m.get("concept"),
    }


def _normalised_hash(cfg) -> str:
    """A config hash with the knob under test erased.

    The first version compared `config_hash` between the two arms and therefore
    could never pass: the refined arm's config differs in `store.faiss.index_key`
    *by construction*, that being the whole experiment. What has to be proven is
    the exact-opposite claim -- that the two configs differ in that one key and
    nowhere else -- so the key is set to a sentinel before hashing and the two
    hashes must then be equal.
    """
    import copy
    import hashlib
    import json

    d = copy.deepcopy(dict(cfg))
    d.setdefault("store", {}).setdefault("faiss", {})["index_key"] = "<knob>"
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _pairwise_like_for_like(runs: dict[str, dict[str, Any]], seeds: list[int]) -> tuple[bool, list[str]]:
    """Prove, per seed, that only the vector column differs.

    Like-for-like is a *within-seed* property: different seeds have different
    ledgers and so different question counts (measured: 52, 50, 51), and comparing
    across seeds made a sound experiment look unsound. The per-seed pairs must agree
    on embedder, reader, reranker, question count and the column-erased config hash.
    """
    problems: list[str] = []
    for seed in seeds:
        b = runs.get(f"base/s{seed}")
        r = runs.get(f"refined/s{seed}")
        if not b or not r:
            problems.append(f"seed {seed}: missing an arm")
            continue
        for key in ("embed_version", "reader_backend", "rerank_backend", "n_questions",
                    "normalised_hash"):
            if b.get(key) != r.get(key):
                problems.append(f"seed {seed}: {key} differs "
                                f"({b.get(key)!r} vs {r.get(key)!r})")
        if b.get("column") != "base" or r.get("column") != "refined":
            problems.append(f"seed {seed}: arms did not read the columns asked for")
    return (not problems), problems


def gate(seeds: list[int], *, n_projects: int = 20, split: str = "dev",
         out_root: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    base_dir = Path(out_root or (root / "runs" / "concept_gate"))
    base_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}
    cfgs = {c: load_config(overrides={"store": {"faiss": {"index_key": c}}})
            for c in COLUMNS}
    for column in COLUMNS:
        for seed in seeds:
            d = base_dir / f"{column}_s{seed}_{split}_p{n_projects}"
            res = run_bench(n_projects=n_projects, seed=seed, split=split, out_dir=d,
                            cfg=cfgs[column])
            p = _pick(res)
            p["normalised_hash"] = _normalised_hash(cfgs[column])
            runs[f"{column}/s{seed}"] = p
            print(f"{column:8} s{seed}: temporal@1={p['temporal@1']} "
                  f"recall@5={p['recall@5']} hard_neg={p['hard_negatives']} "
                  f"headroom={p['headroom']} nq={p['n_questions']}", flush=True)

    def col_mean(column: str, key: str) -> float | None:
        vals = [v[key] for k, v in runs.items()
                if k.startswith(f"{column}/") and v.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    summary = {k: {"temporal@1": col_mean(k, "temporal@1"),
                   "recall@5": col_mean(k, "recall@5"),
                   "hard_negatives": col_mean(k, "hard_negatives"),
                   "headroom": col_mean(k, "headroom"),
                   "gold_in_pool": col_mean(k, "gold_in_pool")}
               for k in COLUMNS}
    t_b, t_r = summary["base"]["temporal@1"], summary["refined"]["temporal@1"]
    h_b, h_r = summary["base"]["hard_negatives"], summary["refined"]["hard_negatives"]
    d_t = None if (t_b is None or t_r is None) else round((t_r - t_b) * 100, 2)
    d_h = None if (h_b is None or h_r is None) else round((h_r - h_b) * 100, 2)
    ok, problems = _pairwise_like_for_like(runs, list(seeds))
    # 6.13 allows either axis to carry the gate, but here only temporal@1 can move:
    # `_hard_negative_eval` embeds fresh pair texts with the frozen encoder and never
    # reads `vectors.refined`, so a zero on that axis is an insensitive instrument,
    # not a null result. Counting it as a failed test would be the e5 mistake again.
    verdict = {
        "gate_pts": GATE_PTS,
        "temporal@1_delta_pts": d_t,
        "hard_negative_delta_pts": d_h,
        "hard_negative_axis_measurable": False,
        "like_for_like": ok,
        "like_for_like_problems": problems,
        "passes": bool(ok and (d_t or 0) >= GATE_PTS),
        # one shared helper decides headroom, per-seed room and the note for both
        # this path and `verdict_from_summary`, so the two cannot drift into
        # different rules -- and `_verdict_word` defaults a missing `decidable` to
        # True, which would print "DROP" on a corpus that cannot measure the bar.
        **_decidability(runs, list(seeds), GATE_PTS),
    }
    out = {"seeds": list(seeds), "n_projects": n_projects, "split": split,
           "runs": runs, "summary": summary, "verdict": verdict}
    (base_dir / "summary.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (base_dir / "table.md").write_text(_table(out), encoding="utf-8")
    return out


def _headroom_pts(runs: dict[str, Any], seeds: list[int]) -> dict[int, float | None]:
    """Residual headroom of the **base arm** per seed, in points.

    The base arm is the only correct denominator: 6.13's bar is a gain *over System
    B*, so the quantity that bounds a possible gain is System B's own remaining
    failure. An earlier version of this file cited the unscoped arm's 22.1% headroom
    to argue the bar could be met on test -- a denominator error that made an
    unmeasurable comparison look merely forbidden.
    """
    out: dict[int, float | None] = {}
    for seed in seeds:
        h = (runs.get(f"base/s{seed}") or {}).get("headroom")
        out[seed] = None if h is None else round(h * 100, 2)
    return out


def _decidability(runs: dict[str, Any], seeds: list[int], gate_pts: float
                  ) -> dict[str, Any]:
    """Whether this corpus can carry the bar -- by seed, not by mean.

    One rule, shared by `gate()` and `verdict_from_summary()`: the same
    two-copies-of-a-table failure inverted a reader verdict elsewhere in this
    project (DECISIONS, reader gold). Measured reason it must be per-seed: test
    leaves 2.04 / 10.42 / 2.27 pts of headroom, so its 4.91 pt mean clears a +3 pt
    bar while **two of three seeds cannot measure it at all**. A mean-based check
    would print `decidable: true` for a comparison that is really one lucky seed.
    """
    pts = _headroom_pts(runs, seeds)
    known = [v for v in pts.values() if v is not None]
    with_room = [s for s, v in sorted(pts.items()) if v is not None and v >= gate_pts]
    mean_frac = round(statistics.mean(known) / 100, 4) if known else None
    # strict majority of the seeds that reported a headroom at all
    decidable = bool(known) and len(with_room) * 2 > len(known)
    return {
        "residual_headroom": mean_frac,
        "headroom_pts_per_seed": pts,
        "seeds_with_room_for_the_bar": len(with_room),
        "seeds_measured": len(known),
        "decidable": decidable,
        "headroom_note": _headroom_note(pts, gate_pts, mean_frac),
    }


def _headroom_note(pts: dict[int, float | None], gate_pts: float,
                   mean_frac: float | None) -> str:
    """State the corpus's carrying capacity beside the verdict.

    A bar of +3 pts where <1 pt of failure remains is not a refutation of 6.13; it is
    a corpus that cannot answer the question. Naming the difference is what keeps a
    null result from being filed as a negative finding, which this project has done
    before (the saturated-split warning in `eval.temporal`).
    """
    shown = ", ".join(f"{s}={'n/a' if v is None else v}" for s, v in sorted(pts.items()))
    room = sum(1 for v in pts.values() if v is not None and v >= gate_pts)
    total = sum(1 for v in pts.values() if v is not None)
    return (f"System B's own residual headroom, per seed: {shown} pts "
            f"(mean {mean_frac}). {room} of {total} seeds leave >= {gate_pts} pts for a "
            f"retrieval-side change to win, and the bar is judged on that majority "
            f"rather than on the mean, which one seed can carry alone. "
            + ("This corpus cannot decide 6.13 either way."
               if room * 2 <= total else
               "This corpus can carry the comparison."))


def verdict_from_summary(path: Path, seeds: list[int]) -> dict[str, Any]:
    """Re-derive the verdict from stored runs, without re-benchmarking.

    The first gate run measured six benches correctly and then voided itself with a
    guard that demanded equality on the very knob under test. Re-running 20 minutes
    of compute to re-judge the same numbers would be wasteful; this recomputes the
    verdict against the corrected guard so a fixed instrument can be shown to work on
    the data it already collected.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = d["runs"]
    cfgs = {c: load_config(overrides={"store": {"faiss": {"index_key": c}}})
            for c in COLUMNS}
    for key, p in runs.items():
        p["normalised_hash"] = _normalised_hash(cfgs[key.split("/")[0]])
    ok, problems = _pairwise_like_for_like(runs, seeds)
    s = d["summary"]
    d_t = None if (s["base"]["temporal@1"] is None or s["refined"]["temporal@1"] is None) \
        else round((s["refined"]["temporal@1"] - s["base"]["temporal@1"]) * 100, 2)
    v = d["verdict"]
    h_b, h_r = s["base"].get("hard_negatives"), s["refined"].get("hard_negatives")
    d_h = None if (h_b is None or h_r is None) else round((h_r - h_b) * 100, 2)
    # The recompute writes exactly the keys `gate()` does, through the same
    # `_decidability` helper, because `_table` indexes them and this path calls it:
    # a first version filled `temporal@1_delta_pts` but not
    # `hard_negative_delta_pts`, so re-judging an artefact crashed on its own output,
    # and a second version kept its own mean-based `decidable` rule that the fresh
    # run did not have -- two copies of one rule, drifting (see the reader-gold
    # failure in DECISIONS). Headroom is now read from the per-seed runs, which is
    # the only denominator that can show a mean carried by a single lucky seed.
    v.update({"like_for_like": ok, "like_for_like_problems": problems,
              "temporal@1_delta_pts": d_t, "hard_negative_delta_pts": d_h,
              "hard_negative_axis_measurable": False,
              "passes": bool(ok and (d_t or 0) >= v["gate_pts"]),
              **_decidability(runs, seeds, v["gate_pts"])})
    d["verdict"] = v
    Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    (Path(path).parent / "table.md").write_text(_table(d), encoding="utf-8")
    return d


def _table(out: dict[str, Any]) -> str:
    v = out["verdict"]
    L = [f"# Concept-attention gate (6.13, 10 row 9)", "",
         f"split={out['split']}  projects={out['n_projects']}  "
         f"seeds={out['seeds']}  bar=+{v['gate_pts']} pts", "",
         "| column | temporal@1 | recall@5 | hard-neg acc | headroom | gold_in_pool |",
         "|---|---|---|---|---|---|"]
    for col, s in out["summary"].items():
        L.append(f"| {col} | {s['temporal@1']} | {s['recall@5']} | "
                 f"{s['hard_negatives']} | {s['headroom']} | {s['gold_in_pool']} |")
    L += ["", f"**temporal@1 difference:** {v['temporal@1_delta_pts']} pts | "
              f"**hard-negative difference:** {v['hard_negative_delta_pts']} pts "
              f"(axis not sensitive to the column: the probe embeds fresh pair texts "
              f"with the frozen encoder) | **like-for-like:** {v['like_for_like']} | "
              f"**verdict:** {_verdict_word(v)}", "",
          v["headroom_note"], ""]
    if v.get("like_for_like_problems"):
        L += [f"- guard problem: {p}" for p in v["like_for_like_problems"]] + [""]
    for k, p in out["runs"].items():
        ann = (p.get("concept") or {}).get("arms", {}).get("system_B") or {}
        if ann:
            L.append(f"- `{k}` anchoring: anchored={ann.get('anchored')} "
                     f"cos={ann.get('cosine_to_base_mean')} "
                     f"drift_p95={ann.get('drift_p95')} refined={ann.get('n_refined')}")
    return "\n".join(L) + "\n"


def _verdict_word(v: dict[str, Any]) -> str:
    """Three states, because two of them are not the same claim.

    `DROP` is a refutation; `NOT DECIDABLE` says the split had nothing left to fix
    and the hypothesis was never tested. Printing the first when the second is true
    is the mistake this project already made with the saturated split, and it is how
    a measured architecture gets retired by an instrument rather than by evidence.
    """
    if not v.get("like_for_like"):
        return "VOID (the arms are not like-for-like; see the guard problems above)"
    if v.get("passes"):
        return "KEEP (6.13 bar cleared)"
    if not v.get("decidable", True):
        return ("NOT DECIDABLE on this split (6.13 untested -- the residual failure "
                "is smaller than the bar), so the default stays off and the "
                "hypothesis is not refuted")
    return "DROP (6.13 bar cleared against, on a split that could measure it)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    args = ap.parse_args(argv)
    out = gate([int(s) for s in args.seeds.split(",") if s.strip()],
               n_projects=args.projects, split=args.split)
    v = out["verdict"]
    print("\nverdict:", json.dumps(v, indent=1))
    print("wrote", repo_root() / "runs" / "concept_gate" / "summary.json")
    return 0 if v["like_for_like"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
