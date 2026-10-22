"""FAISS `IndexFlatIP` over L2-normalised vectors, keyed by the SQLite rowid.

2.4: flat is enough up to ~100k units; persist to disk; rebuild is a
deterministic function of the `vectors` table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - faiss-cpu is a hard dep in pyproject
    faiss = None


class FaissVectors:
    """One live index per vector column (`base`, `refined`).

    Retrieval is exact (flat inner product), so a filtered query uses
    over-fetch-then-mask rather than an IDSelector, which keeps results
    identical across rebuilds (idempotence gate in 6.5).
    """

    OVERFETCH = 8

    def __init__(self, path: Path):
        self.path = Path(path)
        self.indexes: dict[str, object] = {}
        self.ids: dict[str, list[str]] = {}          # use -> ordinal -> unit_id
        self.rowid_of_unit: dict[str, int] = {}
        self.dim = 0
        if faiss is None:
            raise ImportError("faiss-cpu is required (docs/DECISIONS.md D-002)")

    # --- build -------------------------------------------------------------
    def rebuild(self, unit_ids: list[str], mat: np.ndarray, *,
                rowid_of: dict[str, int], use: str = "base") -> None:
        self.rowid_of_unit = dict(rowid_of)
        self.ids[use] = list(unit_ids)
        if mat.size == 0:
            self.indexes.pop(use, None)
            self._save(use)
            return
        mat = np.ascontiguousarray(mat, dtype="float32")
        self.dim = mat.shape[1]
        idx = faiss.IndexFlatIP(self.dim)
        # ordinal order = unit_id order (sorted by caller) => deterministic
        idx.add(mat)
        self.indexes[use] = idx
        self._save(use)

    # --- search ------------------------------------------------------------
    def search(self, q: np.ndarray, k: int, *, allow_rowids: set[int] | None = None,
               use: str = "base") -> list[tuple[str, float]]:
        idx = self.indexes.get(use)
        if idx is None or idx.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(q, dtype="float32").reshape(1, -1))
        fetch = k if allow_rowids is None else min(idx.ntotal, max(k * self.OVERFETCH, 32))
        scores, ords = idx.search(q, int(fetch))
        out: list[tuple[str, float]] = []
        ids = self.ids[use]
        for s, o in zip(scores[0], ords[0]):
            if o < 0 or o >= len(ids):
                continue
            uid = ids[o]
            if allow_rowids is not None and self.rowid_of_unit.get(uid) not in allow_rowids:
                continue
            out.append((uid, float(s)))
            if len(out) >= k:
                break
        if allow_rowids is not None and len(out) < k and fetch < idx.ntotal:
            # over-fetch was not enough: exact brute force over the allowed set
            out = self._brute(q[0], k, allow_rowids, use) or out
        return out

    def _brute(self, q: np.ndarray, k: int, allow_rowids: set[int] | None,
               use: str) -> list[tuple[str, float]]:
        idx = self.indexes.get(use)
        if idx is None:
            return []
        mat = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d).reshape(idx.ntotal, idx.d)
        scores = mat @ np.asarray(q, dtype="float32")
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for o in order:
            uid = self.ids[use][int(o)]
            if allow_rowids is not None and self.rowid_of_unit.get(uid) not in allow_rowids:
                continue
            out.append((uid, float(scores[int(o)])))
            if len(out) >= k:
                break
        return out

    # --- persistence -------------------------------------------------------
    def _save(self, use: str) -> None:
        idx = self.indexes.get(use)
        if idx is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(self.path.with_suffix(f".{use}.faiss")))

    def load(self, use: str = "base") -> bool:
        p = self.path.with_suffix(f".{use}.faiss")
        if not p.exists():
            return False
        idx = faiss.read_index(str(p))
        self.indexes[use] = idx
        self.dim = idx.d
        return True

    def close(self) -> None:
        self.indexes.clear()

    def __len__(self) -> int:
        return sum(i.ntotal for i in self.indexes.values())
