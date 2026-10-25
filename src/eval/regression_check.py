"""`make eval-fast` — the 9.5 CI regression gate.

Runs a small fixed set and fails the build if any headline metric drops more
than 2 points against its OWN baseline, `baselines/fast_gate.json`.

That file is deliberately separate from `baselines/last_good.json`: the canonical
gate is written only by the 3-seed × 20-project test run, and comparing an
8-project single-seed tripwire against it would be comparing two different
experiments. They previously shared one file with two different schemas, which is
how this gate ended up reading keys nobody ever wrote.

Fails loudly when a baseline key is missing: the old code `continue`d on `None`,
so a schema mismatch turned the tripwire into an unconditional PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.config import repo_root
from eval.temporal import run_bench

DROP_LIMIT = 2.0          # percentage points (9.5)
GATE_NAME = "baselines/fast_gate.json"
#: metrics compared against the baseline, all read from a single run
GATED = (("temporal@1", "System B temporal@1"),
         ("recall@5", "System B recall@5"),
         ("gold_in_pool", "System B gold_in_pool"))


def _read_gate(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check(projects: int = 8, seed: int = 42, *, write_gate: bool = False) -> int:
    gate_path = repo_root() / GATE_NAME
    prev = _read_gate(gate_path)
    res = run_bench(n_projects=projects, seed=seed, split="test")
    cmp = res["comparison"]
    meta = res.get("meta") or {}
    a = cmp["temporal@1"]["baseline_A"]
    b = cmp["temporal@1"]["system_B"]
    r5 = (res["system_B"] or {}).get("recall@5")
    gold = (res["system_B"] or {}).get("gold_in_pool")
    prov = {"reader_backend": meta.get("reader_backend"),
            "rerank_backend": meta.get("rerank_backend"),
            "linker_backend": meta.get("linker_backend")}
    cur = {"temporal@1": b, "baseline_temporal@1": a, "recall@5": r5,
           "gold_in_pool": gold, "seed": seed, "n_projects": projects,
           "provenance": prov,
           "embed_version": meta.get("embed_version"),
           "config_hash": meta.get("config_hash")}

    print(f"eval-fast  seed={seed} projects={projects}")
    print(f"  temporal@1  A={a}  B={b}  (delta {cmp['temporal@1']['delta']:+})")
    print(f"  recall@5    B={r5}")
    print(f"  gold_in_pool B={gold}")
    print(f"  backends    {prov}")

    failures: list[str] = []
    if write_gate or prev is None:
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        print(("gate updated" if prev else "gate created") + f" -> {gate_path}")
    else:
        # a baseline from another configuration cannot detect a drop in this one,
        # and it must not be able to report PASS either
        if (prev.get("provenance") or {}) != prov:
            failures.append(
                f"gate was written under a different configuration: "
                f"{prev.get('provenance')} != {prov} — re-run with --update-gate")
        for key, label in GATED:
            old, new = prev.get(key), cur.get(key)
            if old is None or new is None:
                # previously this `continue`d, so a schema mismatch made the gate
                # pass forever without comparing anything
                failures.append(f"{label}: cannot compare, baseline has "
                                f"{old!r} and this run has {new!r}")
                continue
            drop = (old - new) * 100.0
            flag = "DROP" if drop > DROP_LIMIT else "ok"
            print(f"  {label}: {old} -> {new}  ({drop:+.2f} pts) {flag}")
            if drop > DROP_LIMIT:
                failures.append(f"{label} dropped {drop:.2f} pts (> {DROP_LIMIT})")
        # D-041/D-042: with the reranker on BOTH arms, the 8-project tripwire
        # corpus measures A = B (0.4444 = 0.4444) — plain re-scoring already
        # picks the right unit in a small pool, so "B beats A" is a property
        # of the CLAIM corpus (20 projects), not of this drift alarm. Here B
        # must not LOSE; the strict thesis claim is asserted below from the
        # recorded canonical numbers, where it is actually made — free, and
        # it fails loudly if that file is missing or no longer positive.
        if cmp["temporal@1"]["delta"] < 0:
            failures.append(f"System B lost to Baseline A on the tripwire set "
                            f"(delta {cmp['temporal@1']['delta']})")
        lg = _read_gate(repo_root() / "baselines" / "last_good.json")
        lg_delta = ((lg or {}).get("temporal@1_mean") or {}).get("delta")
        if lg_delta is None:
            failures.append("baselines/last_good.json missing or has no "
                            "temporal@1_mean.delta — the thesis claim is "
                            "ungated; run `make baseline`")
        elif lg_delta <= 0:
            failures.append(f"the canonical 20-project thesis delta is no "
                            f"longer positive (last_good.json: {lg_delta}) "
                            "— the relation layer lost to plain search on "
                            "the claim corpus")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--update-gate", action="store_true",
                    help=f"rewrite {GATE_NAME} from this run")
    args = ap.parse_args(argv)
    return check(args.projects, args.seed, write_gate=args.update_gate)


if __name__ == "__main__":
    sys.exit(main())
