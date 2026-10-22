"""SQLite store (BUILD_GUIDE 2.4 + 6.5).

Invariants enforced here and property-tested in tests/property/test_store.py:
  * a unit is never deleted (2.2) — supersede only sets valid_to/superseded_by
  * valid_to >= valid_from
  * every edge endpoint exists
  * rebuild_index() is idempotent and returns identical top-k

Because units are never deleted, SQLite's implicit rowid is a stable integer key
for FAISS, which is exactly what 2.4 asks for.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engine.schemas import Edge, Filters, Slots, Span, Unit, utcnow
from engine.store import migrations as M
from engine.store.faiss_index import FaissVectors
from engine.store.fts import FtsIndex


def _d(v: date | datetime | str | None) -> str | None:
    if v is None:
        return None
    return v.isoformat() if isinstance(v, (date, datetime)) else str(v)


def _day(v: str | None) -> date | None:
    return date.fromisoformat(v[:10]) if v else None


def _dt(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


class SQLiteStore:
    """Thread-safe by construction.

    FastAPI runs `sync def` endpoints on a worker threadpool, so sharing one
    store across requests hit sqlite3's default `check_same_thread=True` and
    raised ProgrammingError on the first /query (caught by
    tests/integration/test_api.py). Writes take an RLock — one instance serves
    CLI, API, UI and offline jobs, so the guard lives here, not in the API layer.
    Reads rely on WAL + sqlite's own serialisation; a FAISS rebuild swaps a whole
    index object, so a concurrent reader sees the old or the new index, never a
    half-built one.
    """

    def __init__(self, path: str | Path, embed_version: str = "unversioned"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_version = embed_version
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA busy_timeout = 5000")
        for stmt in M.DDL + M.FTS_DDL:
            self.db.execute(stmt)
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                        (str(M.SCHEMA_VERSION),))
        self.faiss = FaissVectors(self.path.with_suffix(".faiss"))
        self.fts = FtsIndex(self.db)
        self._rowid_of: dict[str, int] = {}
        self._index_dirty: set[str] = set()
        self.last_search_ms = 0.0
        self._load_rowids()

    # --- writes ------------------------------------------------------------
    def add_units(self, units: Iterable[Unit]) -> int:
        n = 0
        with self._lock, self.db:
            for u in units:
                if self.has(u.unit_id):
                    self._write_unit(u)      # idempotent re-ingest, same transaction
                    continue
                if u.valid_from and u.valid_to and u.valid_to < u.valid_from:
                    raise ValueError(
                        f"{u.unit_id}: valid_to {u.valid_to} < valid_from {u.valid_from}")
                cur = self.db.execute(
                    """INSERT INTO units(unit_id, source_id, source_type, source_date,
                       ingested_at, text, speaker, project, turn_index, kind, type,
                       status, slots_json, valid_from, valid_to, superseded_by,
                       embed_model_version, gold_event_ids_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (u.unit_id, u.source_id, u.source_type, _d(u.source_date),
                     _d(u.ingested_at), u.text, u.speaker, u.project, u.turn_index,
                     u.kind, u.type, u.status, json.dumps(u.slots.model_dump()),
                     _d(u.valid_from), _d(u.valid_to), u.superseded_by,
                     u.embed_model_version, json.dumps(u.gold_event_ids)))
                self._rowid_of[u.unit_id] = cur.lastrowid
                self.db.executemany(
                    "INSERT INTO sentences(unit_id, start, end, text) VALUES(?,?,?,?)",
                    [(u.unit_id, s.start, s.end, s.surface) for s in u.sentences])
                n += 1
        self.fts.sync_from_sentences()
        self._index_dirty.add("lexical")
        return n

    def update_unit(self, unit: Unit) -> None:
        with self._lock, self.db:
            self._write_unit(unit)

    def _write_unit(self, unit: Unit) -> None:
        """Validate and write. The caller already holds the lock and the
        transaction, so opening a second `with self.db` here would commit
        add_units' batch early (sqlite has one transaction per connection)."""
        if not self.has(unit.unit_id):
            raise KeyError(f"unknown unit {unit.unit_id}")
        if unit.valid_from and unit.valid_to and unit.valid_to < unit.valid_from:
            raise ValueError(
                f"{unit.unit_id}: valid_to {unit.valid_to} < valid_from {unit.valid_from}")
        self.db.execute(
                """UPDATE units SET source_id=?, source_type=?, source_date=?,
                   ingested_at=?, text=?, speaker=?, project=?, turn_index=?, kind=?,
                   type=?, status=?, slots_json=?, valid_from=?, valid_to=?,
                   superseded_by=?, embed_model_version=? WHERE unit_id=?""",
                (unit.source_id, unit.source_type, _d(unit.source_date),
                 _d(unit.ingested_at), unit.text, unit.speaker, unit.project,
                 unit.turn_index, unit.kind, unit.type, unit.status,
                 json.dumps(unit.slots.model_dump()), _d(unit.valid_from),
                 _d(unit.valid_to), unit.superseded_by, unit.embed_model_version,
                 unit.unit_id))

    def add_vectors(self, unit_ids: list[str], vecs, kind: str = "base") -> None:
        arr = np.asarray(vecs, dtype="float32")
        if arr.shape[0] != len(unit_ids):
            raise ValueError("unit_ids / vectors length mismatch")
        with self._lock, self.db:
            for uid, v in zip(unit_ids, arr):
                if not self.has(uid):
                    raise KeyError(f"vector for unknown unit {uid}")
                blob = np.asarray(v, dtype="float32").tobytes()
                if kind == "base":
                    self.db.execute(
                        """INSERT INTO vectors(unit_id, base, refined, model_version, dim)
                           VALUES(?,?,NULL,?,?)
                           ON CONFLICT(unit_id) DO UPDATE SET base=excluded.base,
                           model_version=excluded.model_version, dim=excluded.dim""",
                        (uid, blob, self.embed_version, int(len(v))))
                else:
                    self.db.execute(
                        """INSERT INTO vectors(unit_id, base, refined, model_version, dim)
                           VALUES(?,NULL,?,?,?)
                           ON CONFLICT(unit_id) DO UPDATE SET refined=excluded.refined""",
                        (uid, blob, self.embed_version, int(len(v))))
        self._index_dirty.add(kind)

    def add_edges(self, edges: Iterable[Edge]) -> int:
        n = 0
        with self._lock, self.db:
            for e in edges:
                for side in (e.from_id, e.to_id):
                    if not self.has(side):
                        raise KeyError(f"edge endpoint {side} is not a stored unit")
                self.db.execute(
                    """INSERT INTO edges(from_id,to_id,relation,score,evidence_source_id,
                       created_at,guard) VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(from_id,to_id,relation) DO UPDATE SET
                       score=excluded.score, created_at=excluded.created_at,
                       guard=excluded.guard""",
                    (e.from_id, e.to_id, e.relation, e.score, e.evidence_source_id,
                     _d(e.created_at), e.guard))
                n += 1
        return n

    def mark_superseded(self, old_id: str, new_id: str, at: date) -> Unit:
        """6.8.4 — sets valid_to/superseded_by, never deletes, never rewrites text."""
        old = self.get(old_id)
        if old is None:
            raise KeyError(old_id)
        if not self.has(new_id):
            raise KeyError(f"supersede target {new_id} not stored")
        if old.valid_from and at < old.valid_from:
            raise ValueError(
                f"cannot supersede {old_id} at {at}: earlier than valid_from "
                f"{old.valid_from}")
        upd = old.model_copy(update={"valid_to": at, "superseded_by": new_id})
        self.update_unit(upd)
        return upd

    # --- reads -------------------------------------------------------------
    def has(self, unit_id: str) -> bool:
        return unit_id in self._rowid_of

    def get(self, unit_id: str) -> Unit | None:
        r = self.db.execute("SELECT * FROM units WHERE unit_id=?", (unit_id,)).fetchone()
        return self._row_to_unit(r) if r else None

    def get_with_sentences(self, unit_id: str) -> Unit | None:
        u = self.get(unit_id)
        return self.with_sentences(u) if u else None

    def units_of_source(self, source_id: str) -> list[Unit]:
        rows = self.db.execute(
            "SELECT * FROM units WHERE source_id=? ORDER BY row_id",
            (source_id,)).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def all_units(self) -> list[Unit]:
        rows = self.db.execute("SELECT * FROM units ORDER BY row_id").fetchall()
        return [self._row_to_unit(r) for r in rows]

    def sentences_of(self, unit_ids: Iterable[str]) -> list[tuple[str, int, int, str]]:
        ids = list(unit_ids)
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        return [tuple(r) for r in self.db.execute(
            f"SELECT unit_id, start, end, text FROM sentences WHERE unit_id IN ({q})",
            ids).fetchall()]

    def count(self) -> dict[str, int]:
        one = lambda t: self.db.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return {"units": one("units"), "sentences": one("sentences"),
                "edges": one("edges"), "vectors": one("vectors")}

    def facets(self) -> dict[str, set[str]]:
        """Distinct project / speaker values actually stored.

        `Engine.projects` and `Engine.roster` are built during ingest, so a freshly
        opened engine over an existing store reports 0 for both while the database
        holds the real values -- measured: `projects: 0, roster: 0` against 3 and 6
        in `runs/ui/ui.db`. Anything that *displays* corpus size must read the
        corpus, not the session.
        """
        out: dict[str, set[str]] = {}
        for col, key in (("project", "projects"), ("speaker", "speakers")):
            out[key] = {r[col] for r in self.db.execute(
                f"SELECT DISTINCT {col} FROM units WHERE {col} IS NOT NULL")}
        return out

    def edges_for(self, unit_id: str, direction: str = "both",
                  relation: str | None = None) -> list[Edge]:
        where, params = [], []
        if direction in ("from", "both"):
            where.append("from_id=?")
            params.append(unit_id)
        if direction in ("to", "both"):
            where.append("to_id=?")
            params.append(unit_id)
        sql = "SELECT * FROM edges WHERE (" + " OR ".join(where) + ")"
        if relation:
            sql += " AND relation=?"
            params.append(relation)
        return [Edge(from_id=r["from_id"], to_id=r["to_id"], relation=r["relation"],
                     score=r["score"], evidence_source_id=r["evidence_source_id"],
                     created_at=_dt(r["created_at"]), guard=r["guard"])
                for r in self.db.execute(sql, params).fetchall()]

    def history(self, subject_key: str, project: str | None = None) -> list[Unit]:
        sql = ("SELECT * FROM units WHERE json_extract(slots_json,'$.subject')=? ")
        params: list[Any] = [subject_key]
        if project:
            sql += "AND project=?"
            params.append(project)
        sql += "ORDER BY source_date, row_id"
        return [self._row_to_unit(r) for r in self.db.execute(sql, params).fetchall()]

    def current(self, subject_key: str, project: str | None = None) -> Unit | None:
        rows = [u for u in self.history(subject_key, project) if u.is_current]
        if not rows:
            return None
        rows.sort(key=lambda u: (u.source_date, u.unit_id))
        return rows[-1]

    def supersede_confidence(self, unit: "Unit | str") -> float:
        """Confidence of the edge that replaced this unit; 1.0 when nothing did.

        Accepts a Unit or an id: retrieval holds Units and passes them straight
        through, and binding a Unit into SQL raised ProgrammingError on the first
        run of the confidence-scaled demotion.
        """
        unit_id = unit.unit_id if isinstance(unit, Unit) else unit
        row = self.db.execute(
            "SELECT MAX(score) s FROM edges WHERE to_id=? AND relation='supersedes'",
            (unit_id,)).fetchone()
        return float(row["s"]) if row and row["s"] is not None else 1.0

    def supersede_chain(self, unit_id: str) -> list[str]:
        path, seen = [unit_id], {unit_id}
        cur = self.get(unit_id)
        while cur and cur.superseded_by:
            nxt = cur.superseded_by
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            cur = self.get(nxt)
        return path

    def rowids(self, unit_ids: Iterable[str]) -> list[int]:
        return [self._rowid_of[u] for u in unit_ids if u in self._rowid_of]

    def unit_ids_for(self, filters: Filters) -> list[str]:
        """SQL-side metadata filter, applied *before* scoring (6.7)."""
        where: list[str] = []
        params: list[Any] = []
        for col, vals in (("project", filters.projects), ("source_type", filters.source_types),
                          ("status", filters.statuses), ("type", filters.types),
                          ("speaker", filters.speakers)):
            if vals:
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                params += list(vals)
        if filters.date_from:
            where.append("source_date>=?")
            params.append(_d(filters.date_from))
        if filters.date_to:
            where.append("source_date<=?")
            params.append(_d(filters.date_to))
        if filters.current_only:
            where.append("valid_to IS NULL AND superseded_by IS NULL")
        if filters.subject_prefix:
            where.append("json_extract(slots_json,'$.subject') LIKE ?")
            params.append(filters.subject_prefix + "%")
        sql = "SELECT unit_id FROM units"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY row_id"
        return [r["unit_id"] for r in self.db.execute(sql, params).fetchall()]

    def neighbors(self, vec, k: int = 10, filters: Filters | None = None,
                  use: str = "base") -> list[tuple[str, float]]:
        import time

        t0 = time.perf_counter()
        self._ensure_index(use)
        q = np.asarray(vec, dtype="float32").reshape(1, -1)
        allow = None
        if filters is not None and not filters.is_empty():
            allow = set(self.rowids(self.unit_ids_for(filters)))
        out = self.faiss.search(q, k, allow_rowids=allow, use=use)
        self.last_search_ms = round((time.perf_counter() - t0) * 1000, 3)
        return out

    def vector_matrix(self, use: str = "base") -> tuple[list[str], np.ndarray]:
        rows = self.db.execute(
            "SELECT unit_id, base, refined, dim FROM vectors ORDER BY unit_id").fetchall()
        ids: list[str] = []
        mat: list[list[float]] = []
        for r in rows:
            blob = r["refined"] if (use == "refined" and r["refined"]) else r["base"]
            if blob is None:
                continue
            ids.append(r["unit_id"])
            mat.append(np.frombuffer(blob, dtype="float32").tolist())
        dim = len(mat[0]) if mat else 0
        return ids, np.asarray(mat, dtype="float32").reshape(len(mat), dim)

    # --- index lifecycle ---------------------------------------------------
    def _ensure_index(self, use: str = "base") -> None:
        if use in self._index_dirty or use not in self.faiss.indexes:
            with self._lock:
                # double-check under the lock: another thread may have rebuilt
                if use in self._index_dirty or use not in self.faiss.indexes:
                    self._rebuild_index_locked(use)

    def _rebuild_index_locked(self, use: str = "base") -> int:
        ids, mat = self.vector_matrix(use)
        self.faiss.rebuild(ids, mat, rowid_of=self._rowid_of, use=use)
        self._index_dirty.discard(use)
        if use == "base":
            self.fts.sync_from_sentences()
            self._index_dirty.discard("lexical")
        return len(ids)

    def rebuild_index(self, use: str = "base") -> int:
        """Deterministic function of the `vectors` table (2.4). Idempotent."""
        with self._lock:
            return self._rebuild_index_locked(use)

    def export_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w", encoding="utf-8") as fh:
            for u in self.all_units():
                fh.write(json.dumps(self.with_sentences(u).model_dump(mode="json"),
                                    ensure_ascii=False) + "\n")
                n += 1
        return n

    def with_sentences(self, unit: Unit) -> Unit:
        rows = self.db.execute(
            "SELECT start, end, text FROM sentences WHERE unit_id=? ORDER BY start",
            (unit.unit_id,)).fetchall()
        return unit.model_copy(update={"sentences": [
            Span(start=r["start"], end=r["end"], surface=r["text"]) for r in rows]})

    def close(self) -> None:
        self.faiss.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- internals ---------------------------------------------------------
    def _load_rowids(self) -> None:
        self._rowid_of = {r["unit_id"]: r["row_id"] for r in self.db.execute(
            "SELECT unit_id, row_id FROM units")}

    @staticmethod
    def _row_to_unit(r: sqlite3.Row) -> Unit:
        return Unit(
            unit_id=r["unit_id"], source_id=r["source_id"],
            source_type=r["source_type"], source_date=_day(r["source_date"]),
            ingested_at=_dt(r["ingested_at"]), text=r["text"], sentences=[],
            speaker=r["speaker"], project=r["project"], turn_index=r["turn_index"],
            kind=r["kind"], type=r["type"], status=r["status"],
            slots=Slots(**json.loads(r["slots_json"] or "{}")),
            valid_from=_day(r["valid_from"]), valid_to=_day(r["valid_to"]),
            superseded_by=r["superseded_by"],
            embed_model_version=r["embed_model_version"],
            gold_event_ids=json.loads(r["gold_event_ids_json"] or "[]"))
