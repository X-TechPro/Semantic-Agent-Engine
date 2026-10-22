"""Trained multi-head reader (BUILD_GUIDE 6.2.3, 6.2.6).

One encoder, four classification heads: `kind` (2), `type` (9), `status` (6),
`intent` (8). Loss is the sum of the four cross-entropies with class weights and
label smoothing 0.05. Class weighting is not cosmetic: `type` and `status` are
badly imbalanced on ledger text (most turns are `fact`/`mention`), and an
unweighted head learns to predict the majority class and still scores decent
accuracy — which is the failure 6.2.4 warns about.

Label provenance decides how much a number means (9.2):

* kind / type / status are looked up from the **ledger event** the unit was
  rendered from (`gen.ledger.reader_gold`), so they are gold by construction.
  A miss here is a real labelling error, not annotation noise;
* intent is gold only for **query** rows, which the QA generator writes with a
  known intent. Meeting statements are masked out of the intent loss entirely —
  training them on `none` would teach the head nothing and would inflate its
  measured accuracy with free majority-class hits.

Calibration is its own step (6.2.5): a temperature fitted on **dev**, never on
test, because the linker consumes these confidences as thresholds and 6.2.6
requires ECE < 0.08.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from engine.schemas import INTENTS, KINDS, STATUSES, TYPES, Unit

#: canonical label order per head — taken from the schema constants so the model
#: can never drift from the contract, and persisted into `meta.json` on save.
HEADS: dict[str, list[str]] = {
    "kind": list(KINDS),
    "type": list(TYPES),
    "status": list(STATUSES),
    "intent": list(INTENTS),
}
#: a head whose gold is unknown for a row gets this label, and is masked out of
#: both the loss and the metrics
MASK = -1
LABEL_SMOOTHING = 0.05
#: MIT-licensed ModernBERT-French, verified in docs/MODELCARDS.md. Laya is *not*
#: usable here: its root config declares `LayaTypedDecisions` with no root
#: tokenizer, so it cannot initialise a token encoder (see project memory).
DEFAULT_ENCODER = "almanach/moderncamembert-base"


def gold_vector(kind: str, type_: str, status: str,
                intent: str | None = None) -> dict[str, int]:
    """Label indices from schema label names, with -1 where gold is unknown.

    Raises on an unknown label rather than defaulting: a silent 0 here would
    train every bad row into the first class of the head.
    """
    def idx(head: str, value: str | None) -> int:
        if value is None:
            return MASK
        try:
            return HEADS[head].index(value)
        except ValueError:
            raise KeyError(f"{value!r} is not a {head} label "
                           f"({HEADS[head]})") from None
    return {"kind": idx("kind", kind), "type": idx("type", type_),
            "status": idx("status", status), "intent": idx("intent", intent)}


def encode_inputs(tokenizer, texts: list[str], speakers: list[str | None],
                  max_length: int, context: list[str | None] | None = None,
                  with_context: bool = False):
    """`[speaker] text` (6.2.3), optionally prefixed by the previous turn (6.2.4).

    The speaker prefix is a *knob*, not a given: 6.2.3 asks for it and the ablation
    must be able to switch it off, which is why `with_context` lives on the model
    and is persisted in `meta.json` rather than being baked into the strings.
    """
    styled = []
    for i, t in enumerate(texts):
        sp = speakers[i] if speakers else None
        head = f"[{sp}] " if sp else ""
        if with_context and context and context[i]:
            head = f"{context[i].strip()} | {head}"
        styled.append(head + t)
    return tokenizer(styled, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")


def _class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels[labels >= 0], minlength=n_classes).astype("float64")
    seen = counts.sum()
    if seen == 0:
        return np.ones(n_classes)
    # only the classes that actually occur are reweighted. The first version
    # divided by the full `counts` vector, so unseen classes came out `inf` and
    # were patched afterwards — same answer, plus a divide-by-zero warning on
    # every masked head (intent on statement-only rows) in every test run.
    nz = counts > 0
    w = np.ones(n_classes)
    w[nz] = seen / (int(nz.sum()) * counts[nz])
    return w / w.mean()


def _ece(probs: np.ndarray, gold: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == gold).astype("float64")
    n = int(len(conf))
    if n == 0:
        return 0.0
    idx = np.floor(conf * bins).astype(int).clip(0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += float(m.sum()) / n * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(total)


def evaluate_heads(gold: dict[str, np.ndarray], pred: dict[str, np.ndarray],
                   probs: dict[str, np.ndarray] | None = None
                   ) -> dict[str, dict[str, Any]]:
    """Per-head macro-F1 / accuracy / ECE over the *unmasked* rows only."""
    from sklearn.metrics import accuracy_score, f1_score

    out: dict[str, dict[str, Any]] = {}
    for name, classes in HEADS.items():
        if name not in gold:
            continue
        keep = gold[name] >= 0
        if not keep.any():
            out[name] = {"macro_f1": None, "accuracy": None, "n": 0,
                         "classes_seen": 0, "ece": None}
            continue
        g, p = gold[name][keep], pred[name][keep]
        entry: dict[str, Any] = {
            "macro_f1": round(float(f1_score(g, p, average="macro",
                                             zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(g, p)), 4),
            "n": int(keep.sum()),
            "classes_seen": int(len(set(g.tolist()) | set(p.tolist()))),
            "n_classes_total": len(classes),
        }
        if probs and name in probs:
            entry["ece"] = round(_ece(probs[name][keep], g), 4)
        out[name] = entry
    return out


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted / max(temperature, 1e-6))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, gold: np.ndarray) -> float:
    """One-parameter temperature scaling by NLL minimisation (6.2.5).

    Bounded to [0.05, 20]. The bound is not cosmetic: with the first version's
    `min=1e-2` the status head drove T to the floor, which makes the softmax a
    near-one-hot and produces a *tiny ECE as an artifact of over-confidence*
    (conf 1.0, acc 0.98 => ECE 0.02) rather than a calibrated number. A
    temperature pinned at a bound is reported as degenerate by `calibrate`, and
    the linker must not read those confidences as evidence strength.
    """
    import torch
    from torch.nn import functional as F

    keep = torch.tensor(gold >= 0, dtype=torch.bool)
    lg = torch.tensor(logits[keep.numpy()], dtype=torch.float32)
    tg = torch.tensor(gold[keep.numpy()], dtype=torch.long)
    if lg.shape[0] == 0:
        return 1.0
    t = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    # strong_wolfe is not decoration. Without a line search LBFGS takes unit steps
    # on this one parameter and can land on the clamp floor, which sharpens a
    # head it should be flattening: a toy confidently-wrong head measured NLL 3.89
    # at the fitted T against 1.72 at T = 1.0, i.e. "calibration" that made the
    # confidence strictly worse than doing nothing.
    opt = torch.optim.LBFGS([t], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / t.clamp(min=0.05, max=20.0), tg)
        loss.backward()
        return loss

    opt.step(closure)
    fitted = float(t.detach().clamp(min=0.05, max=20.0).item())
    # the invariant the artefact reports: temperature scaling is an optimisation
    # of dev NLL, so it must never leave dev NLL worse than the identity.
    if nll(logits, gold, fitted) > nll(logits, gold, 1.0) + 1e-9:
        return 1.0
    return fitted


def nll(logits: np.ndarray, gold: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold label at a given temperature."""
    keep = gold >= 0
    if not keep.any():
        return 0.0
    probs = apply_temperature(logits[keep], temperature)
    p = probs[np.arange(len(gold[keep])), gold[keep]]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def mean_confidence(probs: np.ndarray) -> float:
    """Average top-class probability.

    Read together with accuracy this is what distinguishes a calibrated head
    from a one-hot one: a mean confidence of ~1.00 against an accuracy of 0.98 is
    not calibration, it is a degenerate temperature making ECE trivially small.
    """
    return float(probs.max(axis=1).mean()) if len(probs) else 0.0


class ReaderModel:
    """Encoder + 4 heads. Saved as a directory: weights + tokenizer + meta.

    `meta.json` carries the label order, so loading can *refuse* a checkpoint
    whose heads no longer match the schema instead of mis-indexing labels — the
    same discipline `PairClassifier.load` applies to feature order.
    """

    def __init__(self, encoder=None, tokenizer=None, heads: dict[str, Any] | None = None,
                 encoder_id: str = DEFAULT_ENCODER, max_length: int = 96,
                 with_context: bool = False, temperatures: dict[str, float] | None = None,
                 device: str | None = None, hidden_size: int | None = None):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.heads = heads or {}
        self.encoder_id = encoder_id
        self.max_length = max_length
        self.with_context = with_context
        self.temperatures = temperatures or {}
        self.device = device
        self.hidden_size = hidden_size
        self.train_meta: dict[str, Any] = {}

    def _ensure_backend(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_id)
            self.encoder = AutoModel.from_pretrained(self.encoder_id,
                                                     dtype=torch.float32)
            self.encoder.to(self.device)
            self.hidden_size = self.encoder.config.hidden_size

    def _pool(self, batch) -> Any:
        """Mean pooling over non-masked tokens."""
        enc = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        return (enc * mask).sum(1) / mask.sum(1).clamp(min=1)

    # ---------------------------------------------------------------- training
    def fit(self, texts: list[str], labels: dict[str, np.ndarray],
            speakers: list[str | None] | None = None,
            contexts: list[str | None] | None = None,
            epochs: int = 4, batch_size: int = 16, lr: float = 2e-5,
            head_lr: float = 1e-3, weight_decay: float = 0.01,
            seed: int = 42, log: Any = None, freeze_encoder: bool = False
            ) -> dict[str, Any]:
        import torch
        from torch.nn import functional as F

        self._ensure_backend()
        torch.manual_seed(seed)
        np.random.seed(seed)
        for name in HEADS:
            if name not in labels:
                continue
            existing = self.heads.get(name)
            if existing is None or existing.out_features != len(HEADS[name]):
                self.heads[name] = torch.nn.Linear(self.hidden_size,
                                                   len(HEADS[name])).to(self.device)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = [p for h in self.heads.values() for p in h.parameters()]
        groups = []
        if enc_params:
            groups.append({"params": enc_params, "lr": lr})
        groups.append({"params": head_params, "lr": head_lr})
        opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
        weights = {n: torch.tensor(_class_weights(labels[n], len(HEADS[n])),
                                   dtype=torch.float32, device=self.device)
                   for n in labels}
        n, order = len(texts), np.arange(len(texts))
        rng = np.random.default_rng(seed)
        steps = 0
        for ep in range(epochs):
            rng.shuffle(order)
            self.encoder.train()
            for h in self.heads.values():
                h.train()
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                batch = encode_inputs(self.tokenizer, [texts[i] for i in idx],
                                      [speakers[i] for i in idx]
                                      if speakers is not None else None,
                                      self.max_length,
                                      [contexts[i] for i in idx]
                                      if contexts is not None else None,
                                      self.with_context)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pooled = self._pool(batch)
                opt.zero_grad()
                loss = 0.0
                for name, head in self.heads.items():
                    tg = torch.tensor(labels[name][idx], dtype=torch.long,
                                      device=self.device)
                    keep = tg >= 0
                    if not bool(keep.any()):
                        continue        # every row masked for this head
                    lg = head(pooled)[keep]
                    loss = loss + F.cross_entropy(
                        lg, tg[keep], weight=weights[name],
                        label_smoothing=LABEL_SMOOTHING)
                if isinstance(loss, float):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
                opt.step()
                total += float(loss.detach())
                steps += 1
            if log:
                log(f"epoch {ep + 1}/{epochs} loss {total / max(1, steps):.3f}")
        self.encoder.eval()
        for h in self.heads.values():
            h.eval()
        self.train_meta = {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "head_lr": head_lr, "seed": seed, "steps": steps,
                           "n_train": n, "encoder": self.encoder_id,
                           "freeze_encoder": freeze_encoder,
                           "with_context": self.with_context,
                           "label_smoothing": LABEL_SMOOTHING,
                           "device": self.device}
        return self.train_meta

    # ------------------------------------------------------------- inference
    def _batches(self, texts, speakers, contexts, batch_size):
        for start in range(0, len(texts), batch_size):
            idx = list(range(start, min(start + batch_size, len(texts))))
            batch = encode_inputs(self.tokenizer, [texts[i] for i in idx],
                                  [speakers[i] for i in idx]
                                  if speakers is not None else None,
                                  self.max_length,
                                  [contexts[i] for i in idx]
                                  if contexts is not None else None,
                                  self.with_context)
            yield idx, {k: v.to(self.device) for k, v in batch.items()}

    def logits_batch(self, texts: list[str], speakers: list[str | None] | None = None,
                     contexts: list[str | None] | None = None,
                     batch_size: int = 32) -> dict[str, np.ndarray]:
        """Raw head outputs. No autocast: the metric must match the deployed path."""
        import torch

        out: dict[str, list[np.ndarray]] = {k: [] for k in self.heads}
        self.encoder.eval()
        with torch.no_grad():
            for _idx, batch in self._batches(texts, speakers, contexts, batch_size):
                pooled = self._pool(batch)
                for name, head in self.heads.items():
                    out[name].append(head(pooled).float().cpu().numpy())
        return {k: (np.concatenate(v) if v else np.zeros((0, len(HEADS[k]))))
                for k, v in out.items()}

    def predict_batch(self, texts: list[str], speakers: list[str | None] | None = None,
                      contexts: list[str | None] | None = None,
                      batch_size: int = 32
                      ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """(argmax per head, calibrated probabilities per head)."""
        logits = self.logits_batch(texts, speakers, contexts, batch_size)
        preds, probs = {}, {}
        for name, lg in logits.items():
            temperature = self.temperatures.get(name, 1.0)
            probs[name] = apply_temperature(lg, temperature)
            preds[name] = probs[name].argmax(axis=1)
        return preds, probs

    def calibrate(self, dev_logits: dict[str, np.ndarray],
                  dev_gold: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
        """Fit one temperature per head on dev, and report what it did.

        Returns a per-head diagnostic rather than just the number, because the
        number alone is what fooled me once: a temperature at the clamp bound
        turns the softmax one-hot, and a near-one-hot head with 98% accuracy
        shows ECE ~0.02 *without being calibrated at all*. `degenerate` and the
        NLL/mean-confidence before-and-after make that visible in the artefact.
        """
        report: dict[str, dict[str, Any]] = {}
        self.temperatures = {}
        for name in dev_logits:
            if name not in dev_gold:
                continue
            lg, tg = dev_logits[name], dev_gold[name]
            if not (tg >= 0).any():
                # No labelled dev row for this head: there is nothing to calibrate
                # against. Fitting anyway produced `accuracy: nan` and
                # `confidence_gap: nan` in the artefact, and a temperature of 1.0
                # that looked like a measured choice rather than an abstention.
                continue
            t = fit_temperature(lg, tg)
            self.temperatures[name] = t
            keep = tg >= 0
            before = apply_temperature(lg, 1.0)[keep]
            after = apply_temperature(lg, t)[keep]
            tg = tg[keep]
            report[name] = {
                "temperature": round(t, 4),
                "accuracy": round(float((after.argmax(axis=1) == tg).mean()), 4),
                "nll_before": round(nll(lg, dev_gold[name], 1.0), 4),
                "nll_after": round(nll(lg, dev_gold[name], t), 4),
                "mean_conf_before": round(mean_confidence(before), 4),
                "mean_conf_after": round(mean_confidence(after), 4),
                "ece_before": round(_ece(before, tg), 4),
                "ece_after": round(_ece(after, tg), 4),
                # a temperature at the bound means the optimum is outside the
                # search range — worth recording, NOT a failure by itself. These
                # heads are UNDER-confident (label smoothing caps achievable
                # probability near 1-e_alpha), so T < 1 sharpening is the correct
                # direction; the gap below is what separates honest certainty
                # from a one-hot covering a wrong head.
                # the bound that matters is fit_temperature's clamp (0.05), not
                # the 1e-3 it once started from: the warm-up refitted `kind` to
                # exactly 0.05 and the old test reported `pinned False`.
                "pinned_at_bound": bool(t <= 0.05 + 1e-6 or t >= 20.0 - 1e-6),
                "confidence_gap": round(float(mean_confidence(after))
                                        - float((after.argmax(axis=1) == tg).mean()), 4),
            }
        return report

    # ------------------------------------------------------------- persistence
    def save(self, out_dir: Path | str) -> Path:
        import torch

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder_state": {k: v.detach().cpu() for k, v in
                                      self.encoder.state_dict().items()},
                    "heads": {k: {n: p.detach().cpu() for n, p in
                                  head.state_dict().items()}
                              for k, head in self.heads.items()}}, d / "weights.pt")
        self.tokenizer.save_pretrained(d / "tokenizer")
        (d / "meta.json").write_text(json.dumps({
            "encoder_id": self.encoder_id, "max_length": self.max_length,
            "with_context": self.with_context, "heads": HEADS,
            "temperatures": self.temperatures, "hidden_size": self.hidden_size,
            "train_meta": self.train_meta,
        }, indent=1), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: Path | str, device: str | None = None) -> "ReaderModel":
        import torch
        from transformers import AutoModel, AutoTokenizer

        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["heads"] != HEADS:
            raise ValueError(
                f"{d}: head labels differ from the current schema; retrain rather "
                f"than load a checkpoint whose label order no longer matches")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        enc = AutoModel.from_pretrained(meta["encoder_id"], dtype=torch.float32)
        blob = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
        enc.load_state_dict(blob["encoder_state"])
        enc.to(dev).eval()
        heads = {}
        for name, state in blob["heads"].items():
            lin = torch.nn.Linear(meta["hidden_size"], len(HEADS[name]))
            lin.load_state_dict(state)
            heads[name] = lin.to(dev)
        model = cls(enc, tok, heads, meta["encoder_id"], meta["max_length"],
                    meta["with_context"], meta.get("temperatures") or {}, dev,
                    meta["hidden_size"])
        model.train_meta = meta.get("train_meta") or {}
        return model


__all__ = ["ReaderModel", "HEADS", "MASK", "gold_vector", "encode_inputs",
           "evaluate_heads", "fit_temperature", "apply_temperature", "nll",
           "mean_confidence", "DEFAULT_ENCODER", "LABEL_SMOOTHING"]
