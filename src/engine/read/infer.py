"""The trained reader behind the same 2.2 contract as the rules baseline.

`build_reader` is the **only** place a reader is constructed. Two sites used to
do `RuleReader()` directly, which is the bug class that already bit three times
in this project: a config key that nothing reads, so an experiment reports
"trained" while running rules. Both `IngestPipeline` and `Engine` now call this
and keep the returned backend string, exactly like `build_linker`.

The knobs that were inert are honoured here:
  * `reader.backend` — rules | model, with a fallback that *says* it fell back;
  * `reader.context_units` — 6.2.3's ablation: how many previous units of the
    same source to prefix. 0 keeps the `[speaker] text` form;
  * `reader.abstain_threshold` — a calibrated confidence floor. It sets
    `Reading.abstained` and nothing else: the reader never silently relabels a
    low-confidence unit, because a wrong label is still written to the unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from engine.config import repo_root
from engine.read.baseline_rules import RuleReader
from engine.read.model import HEADS, ReaderModel
from engine.schemas import Reading, Slots, Unit

DEFAULT_MODEL_DIR = "models/reader"
#: heads whose confidence the abstain threshold looks at. `intent` is excluded:
#: it is only gold for query rows, so its confidence on a statement is not a
#: measure of anything the engine acts on.
GATED_HEADS = ("kind", "type", "status")


class ModelReader:
    """Trained 4-head reader exposing `read(unit, ctx) -> Reading`."""

    name = "model"

    def __init__(self, model: ReaderModel, *, context_units: int = 0,
                 abstain_threshold: float = 0.0):
        self.model = model
        self.context_units = max(0, int(context_units))
        self.abstain_threshold = float(abstain_threshold)
        #: last reading's per-head top probability, for calibration reporting
        self.last_confidence: dict[str, float] = {}

    def read(self, unit: Unit, ctx: Iterable[Unit] | None = None) -> Reading:
        context = None
        if self.context_units > 0 and ctx:
            prior = [c.text for c in list(ctx)[-self.context_units:]]
            context = " | ".join(prior) if prior else None
        preds, probs = self.model.predict_batch([unit.text], [unit.speaker],
                                                [context])
        labels = {name: HEADS[name][int(preds[name][0])] for name in HEADS}
        top = {name: float(probs[name][0].max()) for name in HEADS}
        self.last_confidence = top
        kind = labels["kind"]
        intent = labels["intent"] if kind == "query" else "none"
        confidence = min(top[h] for h in GATED_HEADS if h in top)
        probs_out = {name: {HEADS[name][i]: round(float(p), 5)
                            for i, p in enumerate(probs[name][0])}
                     for name in HEADS}
        return Reading(unit_id=unit.unit_id, kind=kind, type=labels["type"],
                       status=labels["status"], intent=intent, slots=Slots(),
                       probabilities=probs_out,
                       # 6.2.5: abstain means "do not trust this reading", which
                       # the caller must see — never a silent relabel.
                       abstained=bool(self.abstain_threshold > 0.0
                                      and confidence < self.abstain_threshold))


def build_reader(cfg, *, device: str | None = None
                 ) -> tuple[RuleReader | ModelReader, str]:
    """Return (reader, backend_actually_in_use).

    A missing or unloadable checkpoint falls back to rules and *reports* the
    fallback in the returned string, so `meta.reader_backend` can never read as
    "model" while the rules baseline ran.
    """
    reader_cfg = dict(cfg["reader"]) if "reader" in cfg else {}
    common = dict(context_units=int(reader_cfg.get("context_units", 0)),
                  abstain_threshold=float(reader_cfg.get("abstain_threshold", 0.0)))
    want = str(reader_cfg.get("backend", "rules")).lower()
    if want in ("model", "trained", "bert"):
        raw = reader_cfg.get("model_dir") or DEFAULT_MODEL_DIR
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root() / path
        if (path / "meta.json").exists():
            try:
                model = ReaderModel.load(path, device=device)
            except Exception as exc:                       # noqa: BLE001
                return (RuleReader(),
                        f"rules[fallback: {path.name} failed to load: "
                        f"{type(exc).__name__}: {exc}]")
            return ModelReader(model, **common), f"model({path.name})"
        return RuleReader(), f"rules[fallback: no reader at {path}]"
    return RuleReader(), "rules"


def reader_meta(reader: Any) -> dict[str, Any]:
    """What this reader actually is — recorded in every run's meta."""
    return {"backend": getattr(reader, "name", "rules"),
            "encoder": getattr(getattr(reader, "model", None), "encoder_id", None),
            "with_context": bool(getattr(getattr(reader, "model", None),
                                         "with_context", False)),
            "context_units": getattr(reader, "context_units", 0),
            "abstain_threshold": getattr(reader, "abstain_threshold", 0.0)}


__all__ = ["ModelReader", "build_reader", "reader_meta", "DEFAULT_MODEL_DIR",
           "GATED_HEADS"]
