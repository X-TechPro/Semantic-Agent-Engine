"""Train the multi-head reader (BUILD_GUIDE 6.2.3 / 6.2.6, 9.5 tuning rule).

Discipline, in the order the code runs it:

1. fit on **train** projects only;
2. early-stop and temperature-scale on **dev** — 9.5 forbids tuning on test, and
   the first pass of this file's sibling (`link/train.py`) shipped a dev-selected
   threshold that inference never read, so the operating point is applied here
   through `ReaderModel.calibrate`, which writes it into the saved `meta.json`;
3. only then report **test**, with the rules reader run over the *same rows* as
   the reference (0.4: the trained version must beat the baseline it replaces).

The §6.2.6 gates are asserted, not admired: `kind` macro-F1 >= 0.97, `status`
macro-F1 >= 0.85, `intent` >= 0.85 over the classes that have gold, calibrated
`ECE < 0.08`. A gate that cannot be measured (an intent class with no gold rows)
is reported as `unmeasurable`, never as a pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config, repo_root
from engine.read.dataset import UNTRAINABLE_INTENTS, build_split, predict_rules
from engine.read.infer import ModelReader, build_reader
from engine.read.model import HEADS, MASK, ReaderModel, evaluate_heads
from gen.ledger import build_ledger
from gen.splice import write_manifest

GATES = {"kind": 0.97, "status": 0.85, "intent": 0.85, "type": 0.80}
ECE_GATE = 0.08
#: how far mean confidence may sit from measured accuracy before the head is
#: called untrustworthy, even if its ECE passes. Without this, driving a head
#: one-hot manufactures a passing ECE out of an arbitrary head.
GAP_TOLERANCE = 0.05


def _metrics(model: ReaderModel, split: dict[str, Any], batch_size: int
             ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Metrics over the *calibrated* probabilities the engine will actually see.

    Applying `model.temperatures` here is the whole point: an ECE gate measured on
    raw softmax while inference uses calibrated softmax is the same class of bug
    as the linker's dev-selected threshold that nothing read.
    """
    from engine.read.model import apply_temperature

    logits = model.logits_batch(split["texts"], split["speakers"],
                                split["contexts"], batch_size=batch_size)
    probs = {name: apply_temperature(lg, model.temperatures.get(name, 1.0))
             for name, lg in logits.items()}
    pred = {name: p.argmax(axis=1) for name, p in probs.items()}
    return evaluate_heads(split["labels"], pred, probs), logits, split["labels"]


def train(projects: int = 20, seed: int = 42, encoder: str | None = None,
          out: Path | None = None, epochs: int = 8, batch_size: int = 16,
          lr: float = 2e-5, head_lr: float = 1e-3, max_length: int = 96,
          with_context: bool = False, freeze_encoder: bool = False,
          embed_backend: str = "hash", patience: int = 2, verbose: bool = True
          ) -> dict[str, Any]:
    root = repo_root()
    out_dir = Path(out) if out else root / "models" / "reader"
    # the run directory is named after the checkpoint being written: training
    # models/reader_v2 must not overwrite the v1 report or scratch `best/`
    # (artefact identity — the CI-tripwire class this repo already paid for);
    # the default out keeps the historical `runs/reader` path exactly.
    run_dir = root / "runs" / (out_dir.name if out else "reader")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best"

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    ledger = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                        seed=seed)["split"]
    parts = {tag: build_split(ledger, set(sp[tag]), seed, run_dir,
                              embed_backend=embed_backend, tag=tag)
             for tag in ("train", "dev", "test")}
    for tag, part in parts.items():
        say(f"{tag}: {len(part['texts'])} rows ({part['n_units']} units ingested)")

    model = ReaderModel(encoder_id=encoder or _default_encoder(),
                        max_length=max_length, with_context=with_context)
    model._ensure_backend()

    # ---- early stop on DEV status+type macro-F1 (never on test) ------------
    best, best_epoch, stale = -1.0, -1, 0
    for epoch in range(1, epochs + 1):
        model.fit(parts["train"]["texts"], parts["train"]["labels"],
                  speakers=parts["train"]["speakers"],
                  contexts=parts["train"]["contexts"], epochs=1,
                  batch_size=batch_size, lr=lr, head_lr=head_lr, seed=seed,
                  freeze_encoder=freeze_encoder)
        dev, _, _ = _metrics(model, parts["dev"], batch_size * 4)
        signal = float(np.mean([dev[h]["macro_f1"] or 0.0
                                for h in ("kind", "type", "status")]))
        say(f"epoch {epoch}: dev composite {signal:.4f} "
            + " ".join(f"{h}={dev[h]['macro_f1']}" for h in
                       ("kind", "type", "status") if dev[h]["macro_f1"] is not None))
        if signal > best + 1e-4:
            best, best_epoch, stale = signal, epoch, 0
            model.save(ckpt)
        else:
            stale += 1
            if stale >= patience:
                say(f"early stop at epoch {epoch} (best was {best_epoch})")
                break
    if best_epoch >= 0:
        model = ReaderModel.load(ckpt, device=model.device)

    # ---- temperature fitted on DEV, then applied for every later metric ----
    _dev_before, dev_logits, dev_gold = _metrics(model, parts["dev"], batch_size * 4)
    calibration = model.calibrate(dev_logits, dev_gold)
    temperatures = {k: v["temperature"] for k, v in calibration.items()}
    model.save(out_dir)
    dev_metrics, _, _ = _metrics(model, parts["dev"], batch_size * 4)
    test_metrics, _, _ = _metrics(model, parts["test"], batch_size * 4)

    # ---- the reference: the shipped rules reader on the identical rows ------
    rules = build_reader(load_config(overrides={"reader": {"backend": "rules"}}))[0]
    ref_test = evaluate_heads(parts["test"]["labels"],
                              predict_rules(rules, parts["test"]["rows"]))
    ref_dev = evaluate_heads(parts["dev"]["labels"],
                             predict_rules(rules, parts["dev"]["rows"]))

    gates = _gates(test_metrics, ref_test, calibration)
    result = {
        "seed": seed, "encoder": model.encoder_id, "epochs_ran": epochs,
        "best_dev_epoch": best_epoch, "max_length": max_length,
        "with_context": with_context, "freeze_encoder": freeze_encoder,
        "temperatures": temperatures, "calibration": calibration,
        "n_rows": {k: len(v["texts"]) for k, v in parts.items()},
        "label_mix_train": parts["train"]["mix"],
        "dev": dev_metrics, "test": test_metrics,
        "rules_reference_test": ref_test, "rules_reference_dev": ref_dev,
        "gates": gates, "ece_gate": ECE_GATE,
        "intent_coverage": {"trained_on": [i for i in HEADS["intent"]
                                           if i not in UNTRAINABLE_INTENTS],
                            "unmeasurable": UNTRAINABLE_INTENTS},
        "out_dir": str(out_dir),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1),
                                         encoding="utf-8")
    shutil.rmtree(ckpt, ignore_errors=True)
    say(json.dumps({"test": test_metrics, "rules_ref": ref_test,
                    "gates": gates}, indent=1, ensure_ascii=False))
    return result


def _default_encoder() -> str:
    from engine.read.model import DEFAULT_ENCODER

    return DEFAULT_ENCODER


def _gates(test: dict[str, Any], ref: dict[str, Any],
           calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Each §6.2.6 gate, plus whether the trained head beat the rules baseline.

    The ECE gate is paired with a confidence-gap check rather than judged alone.
    ECE on its own can be made small by driving a head one-hot, so `pass` also
    requires mean confidence to sit within GAP_TOLERANCE of measured accuracy on
    the same rows. A temperature pinned at the search bound is recorded but is
    NOT a failure: these heads start under-confident (label smoothing caps the
    attainable probability), so T < 1 sharpening is the correct direction, and a
    1.6-point gap at 97.8% accuracy is honest certainty, not an artifact.
    """
    out: dict[str, Any] = {}
    for head, threshold in GATES.items():
        got = (test.get(head) or {}).get("macro_f1")
        baseline = (ref.get(head) or {}).get("macro_f1")
        if got is None:
            out[head] = {"gate": threshold, "measured": None,
                         "pass": False, "reason": "no gold rows for this head"}
            continue
        entry: dict[str, Any] = {"gate": threshold, "measured": got,
                                 "rules_baseline": baseline,
                                 "pass": bool(got >= threshold)}
        if baseline is not None:
            entry["beats_rules"] = bool(got > baseline)
        if head == "intent" and entry["pass"]:
            entry["caveat"] = ("measured over answer/compare/build_dossier only; "
                               "the other 5 intent labels have no gold in the "
                               "ledger corpus and are not counted")
        out[head] = entry
    eces = [(test.get(h) or {}).get("ece") for h in ("kind", "type", "status")]
    eces = [e for e in eces if e is not None]
    worst = max(eces) if eces else None
    gaps = {h: c["confidence_gap"] for h, c in (calibration or {}).items()
            if h in ("kind", "type", "status")}
    worst_gap = max(gaps.values()) if gaps else None
    entry = {"gate": ECE_GATE, "measured": worst, "gap_tolerance": GAP_TOLERANCE,
             "worst_confidence_gap": worst_gap,
             "pass": bool(worst is not None and worst < ECE_GATE
                          and worst_gap is not None
                          and abs(worst_gap) < GAP_TOLERANCE),
             "uncalibrated_ece": {h: (calibration or {}).get(h, {}).get("ece_before")
                                  for h in ("kind", "type", "status")},
             "pinned_at_bound": sorted(h for h, c in (calibration or {}).items()
                                       if c.get("pinned_at_bound"))}
    out["ece"] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--with-context", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--embed", default="hash", choices=["hash", "st"])
    args = ap.parse_args(argv)
    res = train(args.projects, args.seed, args.encoder,
                Path(args.out) if args.out else None, args.epochs,
                args.batch_size, args.lr, max_length=args.max_length,
                with_context=args.with_context, freeze_encoder=args.freeze_encoder,
                embed_backend=args.embed)
    failed = [h for h, g in res["gates"].items() if not g["pass"]]
    print(f"\nsaved {res['out_dir']}  (dev best epoch {res['best_dev_epoch']}, "
          f"temperatures {res['temperatures']})")
    print("gates failed: " + (", ".join(failed) if failed else "none"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
