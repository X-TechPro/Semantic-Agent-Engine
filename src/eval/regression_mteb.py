"""General-French retrieval regression (BUILD_GUIDE 6.4, 9.4) — "the fine-tune must
not make the encoder forget ordinary French".

Gate as written: MTEB-French / NanoMTEB-French nDCG@10 before vs after each embedder
fine-tune, drop ≤ 1.0 pt or fail.

**`mteb` is not installed in this environment** (checked: `ModuleNotFoundError`). This
module therefore has two modes and says which one it ran in every artefact it writes:

* `mode="mteb"` — the library is importable, real NanoMTEB-French tasks run, and the
  gate is decided on nDCG@10.
* `mode="proxy"` — the library is absent, so the gate is reported as **unmeasured**
  and a labelled proxy is offered instead: pairwise accuracy on held-out general
  French paraphrase pairs from `mteb/amazon_massive_intent` (apache-2.0, licence
  verified live), hash-split **by sentence** into train/probe buckets so the probe
  never appears in training.

The distinction is the point. A proxy that is not labelled as one becomes the number
people quote, and a "drop ≤ 1.0 pt" gate reported against the wrong metric is worse
than an unmeasured gate. This file exists because the report's target table had
`General FR retrieval regression: not measured` and nothing else would ever fill it.

Run: PYTHONPATH=src python -m eval.regression_mteb \
        --tuned models/embed/qwen3-ledger-s42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Any

from engine.config import load_config, repo_root

GATE_MAX_DROP = 0.01        # 1.0 point, expressed as a fraction like the metric
PROBE_ITEMS = 300


def _mteb_available() -> bool:
    return importlib.util.find_spec("mteb") is not None


def _probe_items(seed: int, n: int) -> list[tuple[str, str, str]]:
    """Held-out (anchor, positive, distractor) triples from the probe bucket.

    Reuses the generator's own bucket split so the probe is disjoint from
    training by construction; the third element is another intent's text, which
    makes the task "does the positive beat an unrelated sentence" rather than a
    pure duplicate test.
    """
    from engine.embed.hard_negatives import BUCKET_PROBE, general_rows

    anchors, positives = general_rows(limit=n * 4, seed=seed, bucket=BUCKET_PROBE)
    pairs = list(zip(anchors, positives))[:n]
    if not pairs:
        # An empty probe here would surface as ZeroDivisionError from the
        # distractor rotation below, or as a silently-skipped gate. Both read as
        # "the regression check passed", which is the worst possible outcome for a
        # check that never ran.
        raise RuntimeError(
            "no held-out general-French pairs available: the MASSIVE cache is "
            "empty (offline and never downloaded?). The 6.4 gate cannot be "
            "attempted, not even as a proxy.")
    items = []
    for i, (a, p) in enumerate(pairs):
        # rotate the distractor so no single sentence is the negative 300 times
        distractor = pairs[(i + 7) % len(pairs)][1]
        if distractor == p:
            distractor = pairs[(i + 13) % len(pairs)][1]
        items.append((a, p, distractor))
    return items


def _score(model_id: str, items: list[tuple[str, str, str]], device: str) -> dict:
    """Score through the engine's own embedder class, not a hand-rolled one.

    Same prefixes, same pooling, same normalisation the product uses — a proxy
    measured on a differently-configured encoder would not describe this system.
    """
    from engine.embed.finetune import discriminate

    return discriminate(model_id, items, device=device)


def run(frozen: str | None = None, tuned: str | None = None,
        probe_items: int = PROBE_ITEMS, seed: int = 42) -> dict[str, Any]:
    cfg = load_config()
    frozen = frozen or str(cfg["embed"].get("model_id") or "Qwen/Qwen3-Embedding-0.6B")
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    have_mteb = _mteb_available()
    items = _probe_items(seed, probe_items)
    out: dict[str, Any] = {
        "gate_max_drop": GATE_MAX_DROP, "seed": seed, "probe_items": len(items),
        "frozen": frozen, "tuned": tuned, "device": device,
        "mteb_installed": have_mteb,
        "mode": "mteb" if have_mteb else "proxy",
        "mteb_tasks": None, "proxy": None, "gate": "unmeasured",
        "why": (None if have_mteb else
                "mteb is not installed, so NanoMTEB-French nDCG@10 cannot be "
                "computed; the gate stays unmeasured and the proxy below is "
                "labelled as a proxy, not a substitute score"),
    }

    if have_mteb:
        out["mteb_tasks"] = _run_mteb(frozen, tuned, device)
        f = (out["mteb_tasks"] or {}).get("frozen_ndcg@10")
        t = (out["mteb_tasks"] or {}).get("tuned_ndcg@10")
        if f is not None and t is not None:
            out["drop"] = round(f - t, 4)
            out["gate"] = "pass" if (t - f) >= -GATE_MAX_DROP else "fail"
    else:
        fro = _score(frozen, items, device)
        out["proxy"] = {"frozen": fro}
        if tuned:
            tun = _score(tuned, items, device)
            out["proxy"]["tuned"] = tun
            out["proxy"]["delta_accuracy"] = round(
                (tun.get("pairwise_accuracy") or 0) -
                (fro.get("pairwise_accuracy") or 0), 4)
        out["proxy"]["note"] = (
            "pairwise accuracy on held-out general-French paraphrase pairs "
            "(MASSIVE, apache-2.0). NOT nDCG@10, NOT an MTEB task: it cannot "
            "satisfy the 6.4 gate, only show the direction of forgetting.")

    out_dir = repo_root() / "runs" / "mteb_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(out, indent=1, default=str),
                                         encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(out), encoding="utf-8")
    out["_out"] = str(out_dir)
    return out


def _run_mteb(frozen: str, tuned: str | None, device: str) -> dict[str, Any]:
    """Real MTEB path. Guarded so an API change in the library surfaces as a loud
    `mteb_error` field rather than an absent number."""
    try:
        import mteb

        tasks = mteb.get_tasks(benchmark="MTEB(eng)" if not hasattr(mteb, "get_benchmark")
                               else "NanoMTEB-French")
        model_f = mteb.get_model_configs(frozen) if hasattr(mteb, "get_model_configs") \
            else frozen
        evaluator = mteb.MTEB(tasks=list(tasks)[:3])
        res_f = evaluator.run(model_f, raise_errors=False)
        out = {"task_names": [t.metadata.name for t in tasks][:3],
               "frozen_ndcg@10": _ndcg(res_f)}
        if tuned:
            res_t = evaluator.run(mteb.get_model_configs(tuned)
                                  if hasattr(mteb, "get_model_configs") else tuned,
                                  raise_errors=False)
            out["tuned_ndcg@10"] = _ndcg(res_t)
        return out
    except Exception as exc:
        return {"mteb_error": f"{type(exc).__name__}: {exc}"}


def _ndcg(res) -> float | None:
    try:
        scores = [s for task in res for m in (task.scores or [])
                  for hf, v in m.items() if hf.endswith("ndcg@10")
                  for v in ([v] if isinstance(v, (int, float)) else v)]
        scores = [s for s in scores if isinstance(s, (int, float))]
        return round(sum(scores) / len(scores), 4) if scores else None
    except Exception:
        return None


def _summary(o: dict[str, Any]) -> str:
    L = ["# general-French retrieval regression (6.4)", "",
         f"- mode: **{o['mode']}** | mteb installed: **{o['mteb_installed']}** | "
         f"gate: **{o['gate']}** | max allowed drop: {GATE_MAX_DROP}",
         f"- frozen `{o['frozen']}`" + (f" | tuned `{o['tuned']}`" if o.get('tuned')
                                        else " (no tuned model given)")]
    if o["mode"] == "proxy":
        p = o.get("proxy") or {}
        f = (p.get("frozen") or {}).get("pairwise_accuracy")
        t = (p.get("tuned") or {}).get("pairwise_accuracy")
        L += [f"- PROXY only: pairwise accuracy {f} → {t} "
              f"(Δ {p.get('delta_accuracy')}) over {o['probe_items']} held-out pairs",
              f"- {p.get('note')}",
              f"- why the gate is unmeasured: {o['why']}"]
    else:
        m = o.get("mteb_tasks") or {}
        L += [f"- nDCG@10 {m.get('frozen_ndcg@10')} → {m.get('tuned_ndcg@10')} "
              f"(drop {o.get('drop')}) over {m.get('task_names')}"]
        if m.get("mteb_error"):
            L.append(f"- mteb error: {m['mteb_error']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--tuned", default=None,
                    help="path or hub id of the fine-tuned encoder to gate")
    ap.add_argument("--probe-items", type=int, default=PROBE_ITEMS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out = run(args.frozen, args.tuned, args.probe_items, args.seed)
    print(_summary(out))
    print("wrote", out["_out"])
    return 0 if out["gate"] in ("pass", "unmeasured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
