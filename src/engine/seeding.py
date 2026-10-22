"""Seeding — every experiment runs under fixed seeds 42/43/44 (BUILD_GUIDE 0.6)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int = 42, deterministic_torch: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional (docs/DECISIONS.md D-002)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def subseed(seed: int, *parts: object) -> int:
    """Deterministic child seed from a parent seed + labels (split/renderer)."""
    from hashlib import sha256

    key = f"{seed}|" + "|".join(str(p) for p in parts)
    return int(sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def utcnow_stamp() -> str:
    """Second-precision UTC stamp for generated docs (never microseconds, so a
    regenerated MODELCARDS.md diff is about content, not clock noise)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
