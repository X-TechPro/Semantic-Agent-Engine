"""Keep the Hugging Face cache out of the repository.

A repo-local HF_HOME duplicates the user's cache (8.7 GB once) and is easy to
create by accident: any tool started from this directory with
HF_HOME=./models/hf will download a second copy. The environment is read by
huggingface_hub at import time, so this runs from `engine/__init__` before
anything imports it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _global_hf_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "huggingface"


def guard_hf_cache() -> None:
    """Never keep a second copy of the HF cache inside the repository.

    A repo-local HF_HOME (models/hf, 8.7 GB once) duplicates the user's cache.
    The environment is read by huggingface_hub at import time, so this runs
    before sentence_transformers is imported and redirects the variable when it
    points inside the checkout. One warning, once per process.
    """
    target = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if not target:
        return
    root = Path(__file__).resolve().parents[3]
    try:
        inside = root in Path(target).resolve().parents
    except OSError:
        inside = False
    if not inside:
        return
    fallback = _global_hf_home()
    os.environ["HF_HOME"] = str(fallback)
    os.environ.pop("HF_HUB_CACHE", None)
    print(f"hf cache: {target} is inside the repo; using {fallback} "
          f"(unset HF_HOME to silence this)", file=sys.stderr)


