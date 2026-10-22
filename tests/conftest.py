import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # for `import api.app` in integration tests
sys.path.insert(0, str(ROOT / "src"))  # engine, gen, eval packages

from engine.schemas import (  # noqa: E402
    Edge, Slots, Span, Unit, utcnow,
)


@pytest.fixture
def cfg():
    from engine.config import load_config

    return load_config()


@pytest.fixture
def tmp_store(tmp_path):
    from engine.store.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_path / "engine.db")
    yield store
    store.close()


def make_unit(text: str, **kw):
    """Re-exported from helpers so fixtures and modules share one factory."""
    from helpers import make_unit as _mk

    return _mk(text, **kw)


def make_edge(from_id: str, to_id: str, relation: str = "supports",
              score: float = 0.9, guard: str | None = None) -> Edge:
    from helpers import make_edge as _mk

    return _mk(from_id, to_id, relation, score, guard)
