"""MASSIVE-fr warm-up for the trained reader (B-12's remaining half).

Measured reason (D-022, artefacts `runs/frbench/massive_model*`): the synthetic
checkpoint on real French requests calls **12.2%** of them `query` (they are
*all* requests), scores **16.2%** mapped-subset intent accuracy, and averages
**0.9601 intent confidence with 0.0000 abstention** — its dev-fitted
temperatures (kind 0.0573, status 0.05) sit at the clamp floor, so it is
one-hot by construction and the calibration does not travel.

The safety design is the whole module: reader labels feed the guards
(`type`/`status` decide assertiveness, supersede eligibility), and this project
already paid for letting a model move a guard's input quietly. So —

* the **encoder is frozen** and **`type`/`status` are masked on every row**:
  those two heads receive zero gradient by construction, and their dev
  predictions are asserted bit-identical after training. If they moved at all,
  this module REFUSES to save;
* only `kind` and `intent` train, on MASSIVE-fr **train** rows plus a
  **replay** of the synthetic ledger train rows — without replay, `kind`
  (every MASSIVE row is a query) would collapse to always-query and destroy
  the meeting statements the guards read;
* selection uses MASSIVE **validation** + ledger **dev** only (§9.5). Ledger
  **test** is read once, after the checkpoint is frozen, through a refuse-
  on-second-read artefact (`runs/reader_warm/ledger_test_once.json`); the
  massive-test read belongs to `sae frbench --reader-model models/reader_warm`
  with its own one-artefact-per-arm identity.

Usage:
    PYTHONPATH=src python -m engine.read.warm                # train + revalidate
    PYTHONPATH=src python -m engine.read.warm --max-train 4000 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import MODELS_DIR, repo_root
from engine.read.dataset import build_split
from engine.read.massive import massive_rows, row_label_arrays
from engine.read.model import HEADS, MASK, ReaderModel, evaluate_heads

BASE_DIR = MODELS_DIR / "reader"
WARM_DIR = MODELS_DIR / "reader_warm"
RUN_DIR = repo_root() / "runs" / "reader_warm"
#: heads this warm-up may change; the others must come out bit-identical
TRAIN_HEADS = ("kind", "intent")
FROZEN_HEADS = tuple(h for h in HEADS if h not in TRAIN_HEADS)


def _ledger_parts(seed: int, projects: int) -> dict[str, dict]:
    """Synthetic train/dev/test rows through the ONE builder train.py uses."""
    from gen.ledger import build_ledger
    from gen.splice import write_manifest

    ledger = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(repo_root() / "data/splits/ledger.json",
                        ledger.projects, seed=seed)["split"]
    run = RUN_DIR / "splits"
    return {tag: build_split(ledger, set(sp[tag]), seed, run,
                             embed_backend="hash", tag=tag)
            for tag in ("train", "dev", "test")}, ledger


def _concat(parts: list[dict]) -> dict[str, Any]:
    """Stack packed row-groups (ledger `stack()` shape or massive arrays)."""
    out: dict[str, Any] = {"texts": [], "speakers": [], "contexts": []}
    labels: dict[str, list[np.ndarray]] = {h: [] for h in HEADS}
    for p in parts:
        out["texts"] += list(p["texts"])
        out["speakers"] += list(p["speakers"])
        out["contexts"] += list(p["contexts"])
        for h in HEADS:
            labels[h].append(np.asarray(p["labels"][h]))
    out["labels"] = {h: np.concatenate(v) for h, v in labels.items()}
    return out


def _massive_packed(rows: dict) -> dict:
    m = row_label_arrays(rows["rows"])
    # the two heads that feed the guards are masked HERE, in the one place this
    # module assembles massive rows -- not left to a caller's memory
    m = dict(m, type=np.full_like(m["type"], MASK),
             status=np.full_like(m["status"], MASK))
    return {"texts": [r["text"] for r in rows["rows"]],
            "speakers": [None] * rows["n"],
            "contexts": [None] * rows["n"],
            "labels": m}


def _predict(model: ReaderModel, packed: dict, batch: int = 64
             ) -> tuple[dict, dict, dict]:
    preds, probs = model.predict_batch(packed["texts"], packed["speakers"],
                                       packed["contexts"], batch_size=batch)
    logits = model.logits_batch(packed["texts"], packed["speakers"],
                                packed["contexts"], batch_size=batch)
    return preds, probs, logits


def _num(v: float | None) -> float:
    """A measured 0.0 and an absent value are different facts; for the mean the
    absent one contributes 0.0 explicitly (never via truthiness -- the falsy-zero
    trap this project has paid for three times)."""
    return 0.0 if v is None else v


def _freeze_guard_heads(labels: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Mask `type`/`status` on EVERY training row -- ledger replay included.

    The first warm-up run masked only the MASSIVE rows; the 613 replay rows came
    with real type/status gold (measured: 613/613 unmasked), so `fit()` trained
    the very heads this module promises to freeze, and the drift assertion
    caught it red-handed (logit drift 1.59 / 1.51) BEFORE any save. A masked
    head never contributes a loss term, its grad stays None, AdamW skips
    None-grad params -- so frozen-ness here is structural, and this one function
    is where the promise lives.
    """
    out = dict(labels)
    for h in FROZEN_HEADS:
        out[h] = np.full(n, MASK, dtype=np.asarray(labels[h]).dtype)
    return out


def _signal(massive_val: dict, ledger_dev: dict) -> tuple[float, dict]:
    """Selection number on the two TUNING splits only — never test (§9.5)."""
    mv = evaluate_heads(massive_val["labels"], massive_val["preds"],
                        massive_val["probs"])
    ld = evaluate_heads(ledger_dev["labels"], ledger_dev["preds"],
                        ledger_dev["probs"])
    parts = {
        "massive_kind_acc": _num(mv["kind"]["accuracy"]),
        "massive_intent_f1": _num(mv["intent"]["macro_f1"]),
        "ledger_kind_f1": _num(ld["kind"]["macro_f1"]),
        "ledger_intent_f1": _num(ld["intent"]["macro_f1"]),
    }
    return round(sum(parts.values()) / len(parts), 4), {"massive_val": mv,
                                                        "ledger_dev": ld,
                                                        **parts}


def train(*, epochs: int = 2, max_train: int | None = 4000, batch: int = 32,
          head_lr: float = 1e-3, lr: float = 2e-5, seed: int = 42,
          projects: int = 20, device: str | None = None,
          base: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    base = Path(base or BASE_DIR)
    out = Path(out or WARM_DIR)
    if base.resolve() == out.resolve():
        raise ValueError("refusing to overwrite the base checkpoint: "
                         f"out={out} resolves to base={base}")
    if not (base / "meta.json").exists():
        raise FileNotFoundError(f"no trained reader at {base} -- run "
                                "`python -m engine.read.train` first")
    dev = device or ("cuda" if _cuda() else "cpu")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    parts, _ledger = _ledger_parts(seed, projects)
    mtrain = massive_rows("train", limit=max_train, seed=seed)
    mval = massive_rows("validation", seed=seed)
    mp_train, mp_val = _massive_packed(mtrain), _massive_packed(mval)
    train_set = _concat([parts["train"], mp_train])
    # the freeze is enforced on the ASSEMBLED set, not row-source by row-source:
    # one place to audit, no way for a replay branch to reintroduce gold here
    train_set["labels"] = _freeze_guard_heads(train_set["labels"],
                                              len(train_set["texts"]))
    print(f"rows: replay {len(parts['train']['texts'])} ledger + "
          f"{mtrain['n']} massive ({mtrain['n_mapped']} mapped, "
          f"none-share {mtrain['none_share']}) | val {mval['n']}", flush=True)

    model = ReaderModel.load(base, device=dev)
    ledger_dev_p = _concat([parts["dev"]])
    before_preds, before_probs, before_logits = _predict(model, ledger_dev_p)
    base_temps = dict(model.temperatures)
    # captured HERE, before the warm run's own fit() overwrites train_meta --
    # lineage means the checkpoint this one was warmed FROM, not this run's
    # own hparams echoed back under a misleading key
    base_train_meta = dict(model.train_meta or {})
    # the BASE model's own tuning-split signal, computed once before any epoch:
    # a curve without the starting point cannot show that the warm-up moved
    # anything rather than merely landing somewhere
    bv_p, bv_pr, _bv_l = _predict(model, mp_val)
    base_signal, _base_report = _signal(
        {"labels": mp_val["labels"], "preds": bv_p, "probs": bv_pr},
        {"labels": ledger_dev_p["labels"], "preds": before_preds,
         "probs": before_probs})
    print(f"base signal (dev+val, pre-warm): {base_signal:.4f}", flush=True)

    ck = RUN_DIR / "best"
    best = {"signal": -1.0}
    curve = []
    for ep in range(1, epochs + 1):
        model.fit(train_set["texts"], train_set["labels"],
                  speakers=train_set["speakers"], contexts=train_set["contexts"],
                  epochs=1, batch_size=batch, lr=lr, head_lr=head_lr, seed=seed,
                  freeze_encoder=True)
        vp, vpr, _ = _predict(model, mp_val)
        dp, dpr, _ = _predict(model, ledger_dev_p)
        sig, rep = _signal({"labels": mp_val["labels"], "preds": vp, "probs": vpr},
                           {"labels": ledger_dev_p["labels"], "preds": dp,
                            "probs": dpr})
        curve.append({"epoch": ep, "signal": sig,
                      "massive_intent_f1": rep["massive_intent_f1"],
                      "massive_kind_acc": rep["massive_kind_acc"],
                      "ledger_kind_f1": rep["ledger_kind_f1"]})
        print(f"epoch {ep}: signal {sig:.4f} " + json.dumps(
            {k: rep[k] for k in ("massive_intent_f1", "massive_kind_acc",
                                 "ledger_kind_f1")}), flush=True)
        if sig > best["signal"]:
            best = {"signal": sig, "epoch": ep}
            model.save(ck)
    if best["signal"] >= 0:
        model = ReaderModel.load(ck, device=dev)
        shutil.rmtree(ck, ignore_errors=True)

    # -- the guard-input invariance is checked, not promised ------------------
    after_preds, after_probs, after_logits = _predict(model, ledger_dev_p)
    drift = {h: float(np.abs(after_logits[h] - before_logits[h]).max())
             for h in FROZEN_HEADS}
    ident = all(bool(np.array_equal(after_preds[h], before_preds[h]))
                and drift[h] < 1e-4 for h in FROZEN_HEADS)
    if not ident:
        raise RuntimeError(
            f"guard-input heads moved (type/status): argmax-equal="
            f"{ {h: bool(np.array_equal(after_preds[h], before_preds[h])) for h in FROZEN_HEADS} }, "
            f"logit-drift={drift}; refusing to save a warm-up that changes "
            f"what the guards read")

    # -- refit ALL four temperatures on the combined tuning dev ---------------
    # (calibrate() resets the dict, so it must see every head; massive rows are
    # masked for type/status and contribute no gold there, ledger dev does)
    cal_dev = _concat([parts["dev"], mp_val])
    _p, _pr, cal_logits = _predict(model, cal_dev)
    calibration = model.calibrate(cal_logits, cal_dev["labels"])
    model.train_meta = {
        "warmup": {"base": str(base), "encoder_frozen": True,
                   "trained_heads": list(TRAIN_HEADS),
                   "masked_frozen_heads": list(FROZEN_HEADS),
                   "massive": {"splits": "train/validation", **{
                       k: mtrain[k] for k in ("n", "n_mapped", "intent_mix")},
                       "limit": max_train, "seed": seed},
                   "replay": {"n_ledger_train": len(parts["train"]["texts"])},
                   "hparams": {"epochs": epochs, "batch": batch,
                               "head_lr": head_lr, "max_length": model.max_length},
                   "curve": curve, "selected": best,
                   "frozen_head_drift": drift,
                   "temperatures_before": base_temps,
                   "temperatures_after": dict(model.temperatures),
                   "calibration": calibration,
                   "base_train_meta": base_train_meta,
                   "device": dev,
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    model.save(out)
    art = {"out": str(out), "base_signal": base_signal,
           "curve": curve, "selected": best,
           "frozen_head_drift": drift,
           "temperatures": {"before": base_temps,
                            "after": dict(model.temperatures)},
           "calibration": calibration}
    (RUN_DIR / "warm_result.json").write_text(
        json.dumps(art, indent=1, default=str), encoding="utf-8")
    print(f"saved {out}", flush=True)
    return art


def revalidate(*, seed: int = 42, projects: int = 20, device: str | None = None,
               warm: Path | None = None) -> dict[str, Any]:
    """Ledger dev+test on base vs warm; TEST is read exactly once.

    The ledger-test numbers already exist for the base checkpoint
    (`runs/reader/result.json`: kind 1.0 / type 0.8733 / status 0.9255 /
    intent 1.0), so only the warm side needs a fresh test read — and this
    guard is what keeps that one read from becoming a tunable second one.
    """
    once = RUN_DIR / "ledger_test_once.json"
    if once.exists():
        raise RuntimeError(f"ledger test already read once ({once}); "
                           "refusing to re-read it (9.5)")
    warm = Path(warm or WARM_DIR)
    dev = device or ("cuda" if _cuda() else "cpu")
    parts, _ = _ledger_parts(seed, projects)
    base = ReaderModel.load(BASE_DIR, device=dev)
    wmodel = ReaderModel.load(warm, device=dev)
    out: dict[str, Any] = {"base": {}, "warm": {}}
    for tag in ("dev", "test"):
        packed = _concat([parts[tag]])
        for name, m in (("base", base), ("warm", wmodel)):
            preds, probs, _ = _predict(m, packed)
            out[name][tag] = evaluate_heads(packed["labels"], preds, probs)
    out["compare_dev"] = {h: {"base": out["base"]["dev"][h]["macro_f1"],
                              "warm": out["warm"]["dev"][h]["macro_f1"]}
                          for h in HEADS}
    out["compare_test"] = {h: {"base": out["base"]["test"][h]["macro_f1"],
                               "warm": out["warm"]["test"][h]["macro_f1"]}
                           for h in HEADS}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    once.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _cuda() -> bool:
    import torch
    return bool(torch.cuda.is_available())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=4000,
                    help="massive-fr train subset (the 'bit of data'; all if 0)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--revalidate-only", action="store_true",
                    help="skip training; run the guarded ledger re-validation")
    ap.add_argument("--train-only", action="store_true",
                    help="stop after training: the revalidation reads ledger "
                         "test ONCE, so verify the curve before spending that read")
    args = ap.parse_args(argv)
    if not args.revalidate_only:
        train(epochs=args.epochs, max_train=args.max_train or None,
              batch=args.batch, head_lr=args.head_lr, seed=args.seed,
              projects=args.projects, device=args.device,
              base=Path(args.base) if args.base else None,
              out=Path(args.out) if args.out else None)
    if args.train_only:
        print("trained; ledger test left unread -- run --revalidate-only when "
              "the curve is accepted", flush=True)
        return 0
    rep = revalidate(seed=args.seed, projects=args.projects, device=args.device,
                     warm=Path(args.out) if args.out else None)
    print(json.dumps({"compare_dev": rep["compare_dev"],
                      "compare_test": rep["compare_test"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
