"""Append-only v4 store: messages, chunks, units, typed links, state events.

SQLite holds the rows and the FTS5 index. The contract from ARCH_V4 §1/§5 is
enforced here, in one place:

* a message row is immutable — re-adding it with any different field raises;
* a chunk span is immutable — same rule;
* links and state events are append-only — a duplicate id with a different
  payload raises instead of being silently ignored;
* `current_state` never rewrites a row: it is a view whose rule is "latest
  live event, else the open proposal".

`origin` separates real and synthetic rows (§5 "v4 tables always separate real
and synthetic"), so a synthetic row can never be counted as a real result.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from engine.schemas_v4 import (
    LIVE_STATUSES, OPEN_STATUSES, ChunkRecord, LinkEvent, MessageRecord,
    SemanticUnit, StateEvent,
)

_MESSAGE_FIELDS = ("source_id", "project_id", "speaker", "timestamp", "text",
                   "source_type", "origin", "content_hash")
_CHUNK_FIELDS = ("message_id", "start", "end", "text", "sentence_index")


class V4Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        # Streamlit caches the store in one script thread and reruns the query
        # in another; the default check_same_thread=True turns that into
        # "SQLite objects created in a thread can only be used in that same
        # thread". Python's sqlite3 is built in serialized mode, so sharing the
        # connection across threads is safe here.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._schema()

    # --- schema ------------------------------------------------------------
    def _schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS messages(
          message_id TEXT PRIMARY KEY, source_id TEXT, project_id TEXT,
          speaker TEXT, timestamp TEXT, text TEXT, source_type TEXT,
          ingested_at TEXT, content_hash TEXT, origin TEXT DEFAULT 'real');
        CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY, message_id TEXT, start INTEGER, end INTEGER,
          text TEXT, sentence_index INTEGER);
        CREATE TABLE IF NOT EXISTS units(
          unit_id TEXT PRIMARY KEY, message_id TEXT, chunk_ids TEXT, text TEXT,
          kind TEXT, status TEXT, intent TEXT, entities TEXT, owner TEXT,
          deadline TEXT, slot TEXT, value TEXT, ev_start INTEGER, ev_end INTEGER,
          ev_text TEXT, confidence REAL, model_version TEXT,
          origin TEXT DEFAULT 'real', point TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS links(
          link_id TEXT PRIMARY KEY, from_unit_id TEXT, to_unit_id TEXT,
          relation TEXT, confidence REAL, evidence TEXT, model_version TEXT,
          created_at TEXT);
        CREATE TABLE IF NOT EXISTS state_events(
          state_id TEXT PRIMARY KEY, event_id TEXT, slot TEXT, value TEXT,
          status TEXT, evidence_unit TEXT, created_at TEXT,
          origin TEXT DEFAULT 'real');
        CREATE INDEX IF NOT EXISTS units_message ON units(message_id);
        CREATE INDEX IF NOT EXISTS links_from ON links(from_unit_id);
        CREATE INDEX IF NOT EXISTS links_to ON links(to_unit_id);
        CREATE INDEX IF NOT EXISTS state_event ON state_events(event_id, slot);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text, chunk_id UNINDEXED, message_id UNINDEXED, tokenize='unicode61');
        CREATE TRIGGER IF NOT EXISTS messages_no_update
          BEFORE UPDATE ON messages
          BEGIN SELECT RAISE(ABORT, 'messages are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS messages_no_delete
          BEFORE DELETE ON messages
          BEGIN SELECT RAISE(ABORT, 'messages are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS chunks_no_update
          BEFORE UPDATE ON chunks
          BEGIN SELECT RAISE(ABORT, 'chunks are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS units_no_update
          BEFORE UPDATE ON units
          BEGIN SELECT RAISE(ABORT, 'units are immutable'); END;
        """)
        self.db.commit()

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    # --- immutable writes --------------------------------------------------
    def add_message(self, m: MessageRecord) -> None:
        row = self.db.execute("SELECT * FROM messages WHERE message_id=?",
                              (m.message_id,)).fetchone()
        if row is not None:
            changed = [f for f in _MESSAGE_FIELDS
                       if f not in ("ingested_at",)
                       and str(row[f]) != str(getattr(m, f))
                       and not (f == "content_hash" and not getattr(m, f))]
            if changed:
                raise ValueError(
                    f"immutable message {m.message_id} changed: {changed}")
            return
        self.db.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)",
            (m.message_id, m.source_id, m.project_id, m.speaker, m.timestamp,
             m.text, m.source_type, m.ingested_at, m.content_hash, m.origin))
        self.db.commit()

    def add_chunk(self, c: ChunkRecord) -> None:
        row = self.db.execute("SELECT * FROM chunks WHERE chunk_id=?",
                              (c.chunk_id,)).fetchone()
        if row is not None:
            changed = [f for f in _CHUNK_FIELDS
                       if str(row[f]) != str(getattr(c, f))]
            if changed:
                raise ValueError(f"immutable chunk {c.chunk_id} changed: {changed}")
            return
        self.db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                        (c.chunk_id, c.message_id, c.start, c.end, c.text,
                         c.sentence_index))
        self.db.execute(
            "INSERT INTO chunks_fts(text, chunk_id, message_id) VALUES (?,?,?)",
            (c.text, c.chunk_id, c.message_id))
        self.db.commit()

    def add_unit(self, u: SemanticUnit) -> None:
        ev = u.evidence
        self.db.execute(
            """INSERT OR REPLACE INTO units
               (unit_id, message_id, chunk_ids, text, kind, status, intent,
                entities, owner, deadline, slot, value, ev_start, ev_end,
                ev_text, confidence, model_version, origin, point)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (u.unit_id, u.message_id, self._json(u.chunk_ids), u.text, u.kind,
             u.status, u.intent, self._json(u.entities), u.owner, u.deadline,
             u.slot, u.value,
             ev.start if ev else None, ev.end if ev else None,
             ev.text if ev else None, float(u.confidence), u.model_version,
             u.origin, u.point))
        self.db.commit()

    def add_link(self, e: LinkEvent) -> None:
        # created_at is a write timestamp, not payload: re-ingesting the same
        # message must be a no-op, not a rewrite conflict.
        self._append_only("links", "link_id",
                          (e.from_unit_id, e.to_unit_id, e.relation,
                           float(e.confidence), e.evidence, e.model_version),
                          e.link_id)
        self.db.execute(
            "INSERT OR IGNORE INTO links VALUES (?,?,?,?,?,?,?,?)",
            (e.link_id, e.from_unit_id, e.to_unit_id, e.relation,
             float(e.confidence), e.evidence, e.model_version, e.created_at))
        self.db.commit()

    def add_state(self, e: StateEvent) -> None:
        self._append_only("state_events", "state_id",
                          (e.event_id, e.slot, e.value, e.status,
                           e.evidence_unit), e.state_id)
        self.db.execute(
            "INSERT OR IGNORE INTO state_events VALUES (?,?,?,?,?,?,?,?)",
            (e.state_id, e.event_id, e.slot, e.value, e.status, e.evidence_unit,
             e.created_at, e.origin))
        self.db.commit()

    def _append_only(self, table: str, key: str, payload: tuple,
                     value: str | None = None) -> None:
        if value is None:
            value = payload
        row = self.db.execute(f"SELECT * FROM {table} WHERE {key}=?",
                              (value,)).fetchone()
        if row is None:
            return
        # idempotent re-ingest: the same logical row with a new created_at is
        # not a rewrite. Only the semantic columns decide.
        volatile = {"created_at", "ingested_at", key}
        cols = list(row.keys())[1:1 + len(payload)]
        for col, new in zip(cols, payload):
            if col in volatile:
                continue
            if str(row[col]) != str(new):
                raise ValueError(
                    f"append-only {table} row {value} would be rewritten")

    # --- reads -------------------------------------------------------------
    def message(self, message_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM messages WHERE message_id=?",
                              (message_id,)).fetchone()
        return dict(row) if row else None

    def all_messages(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM messages ORDER BY timestamp, rowid")]

    def chunk(self, chunk_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM chunks WHERE chunk_id=?",
                              (chunk_id,)).fetchone()
        return dict(row) if row else None

    def chunks_for_unit(self, unit_id: str) -> list[dict]:
        u = self.unit(unit_id)
        if not u:
            return []
        ids = u.get("chunk_ids") or []
        if isinstance(ids, str):
            ids = json.loads(ids)
        out = [self.chunk(c) for c in ids]
        return [c for c in out if c]

    def unit(self, unit_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM units WHERE unit_id=?",
                              (unit_id,)).fetchone()
        return self._unit_row(row) if row else None

    def all_units(self) -> list[dict]:
        return [self._unit_row(r) for r in self.db.execute(
            "SELECT * FROM units ORDER BY rowid")]

    def units_for_message(self, message_id: str) -> list[dict]:
        return [self._unit_row(r) for r in self.db.execute(
            "SELECT * FROM units WHERE message_id=? ORDER BY rowid", (message_id,))]

    def units_for_project(self, project_id: str, limit: int | None = None
                          ) -> list[dict]:
        sql = ("SELECT u.* FROM units u JOIN messages m "
               "ON u.message_id=m.message_id WHERE m.project_id=? "
               "ORDER BY m.timestamp, m.rowid, u.rowid")
        rows = self.db.execute(sql, (project_id,)).fetchall()
        out = [self._unit_row(r) for r in rows]
        return out[-limit:] if limit else out

    def unit_projects(self) -> dict[str, str]:
        """unit_id -> project_id, for scoping a search to one project."""
        rows = self.db.execute(
            "SELECT u.unit_id, m.project_id FROM units u JOIN messages m "
            "ON u.message_id=m.message_id")
        return {r["unit_id"]: r["project_id"] for r in rows}

    @staticmethod
    def _unit_row(row) -> dict:
        u = dict(row)
        for key in ("chunk_ids", "entities"):
            raw = u.get(key)
            u[key] = json.loads(raw) if isinstance(raw, str) and raw else []
        if u.get("ev_start") is not None:
            u["evidence"] = {"start": u["ev_start"], "end": u["ev_end"],
                             "text": u["ev_text"] or ""}
        else:
            u["evidence"] = None
        return u

    def links(self, relation: str | None = None) -> list[dict]:
        if relation:
            rows = self.db.execute(
                "SELECT * FROM links WHERE relation=? ORDER BY rowid", (relation,))
        else:
            rows = self.db.execute("SELECT * FROM links ORDER BY rowid")
        return [dict(r) for r in rows]

    def links_of(self, unit_id: str, relation: str | None = None,
                 direction: str = "both") -> list[dict]:
        sql = ["SELECT * FROM links WHERE "]
        if direction == "out":
            sql.append("from_unit_id=?")
            args: tuple = (unit_id,)
        elif direction == "in":
            sql.append("to_unit_id=?")
            args = (unit_id,)
        else:
            sql.append("(from_unit_id=? OR to_unit_id=?)")
            args = (unit_id, unit_id)
        if relation:
            sql.append(" AND relation=?")
            args = args + (relation,)
        return [dict(r) for r in self.db.execute("".join(sql) + " ORDER BY rowid",
                                                 args)]

    def links_between(self, a: str, b: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM links WHERE (from_unit_id=? AND to_unit_id=?) "
            "OR (from_unit_id=? AND to_unit_id=?)",
            (a, b, b, a))]

    def superseded_units(self) -> dict[str, float]:
        """unit_id -> best supersede confidence. These are the old values."""
        out: dict[str, float] = {}
        for r in self.links("supersedes"):
            out[r["from_unit_id"]] = max(out.get(r["from_unit_id"], 0.0),
                                         float(r["confidence"]))
        return out

    def contradicted_units(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.links("contradicts"):
            out[r["from_unit_id"]] = max(out.get(r["from_unit_id"], 0.0),
                                         float(r["confidence"]))
        return out

    def state_events(self, evidence_unit: str | None = None) -> list[dict]:
        if evidence_unit:
            rows = self.db.execute(
                "SELECT * FROM state_events WHERE evidence_unit=? ORDER BY rowid",
                (evidence_unit,))
        else:
            rows = self.db.execute("SELECT * FROM state_events ORDER BY rowid")
        return [dict(r) for r in rows]

    def history(self, event_id: str, slot: str | None = None) -> list[dict]:
        q = ("SELECT * FROM state_events WHERE event_id=? "
             + ("AND slot=? " if slot else "") + "ORDER BY created_at, rowid")
        args = (event_id, slot) if slot else (event_id,)
        return [dict(r) for r in self.db.execute(q, args)]

    def current_state(self, event_id: str, slot: str | None = None) -> dict | None:
        """§5 view: latest live event, else the open proposal, else None.

        A proposal that arrives after a confirmation is kept in `history` and in
        `open` but never becomes `current`.
        """
        rows = self.history(event_id, slot)
        live = [r for r in rows if r["status"] in LIVE_STATUSES]
        if live:
            cur = live[-1]
            cur["open"] = False
            cur["transition"] = self._transition(rows, cur)
            return cur
        open_rows = [r for r in rows if r["status"] in OPEN_STATUSES]
        if open_rows:
            cur = open_rows[-1]
            cur["open"] = True
            cur["transition"] = self._transition(rows, cur)
            return cur
        return None

    def current_for_unit(self, unit_id: str) -> dict | None:
        """The current state, only if this unit is its evidence.

        Returning the slot's current state for every unit on that slot attached
        one turn's value to unrelated turns (measured: 76 of 103 units), which
        is what made the answer cite the right slot and the wrong message.
        """
        u = self.unit(unit_id)
        if not u or not u.get("slot"):
            return None
        cur = self.current_state(event_id_for(u, self), u["slot"])
        if cur and cur.get("evidence_unit") == unit_id:
            return cur
        return None

    @staticmethod
    def _transition(rows: list[dict], cur: dict) -> dict | None:
        """The event that changed the state: previous live value -> current."""
        prior = [r for r in rows
                 if r["status"] in LIVE_STATUSES
                 and (r["created_at"], r["state_id"]) < (cur["created_at"],
                                                         cur["state_id"])]
        if not prior:
            return None
        prev = prior[-1]
        return {"from": prev["value"], "from_status": prev["status"],
                "to": cur["value"], "to_status": cur["status"],
                "evidence_unit": cur["evidence_unit"],
                "from_evidence_unit": prev["evidence_unit"]}

    # --- lexical -----------------------------------------------------------
    def bm25(self, query: str, k: int = 20) -> list[dict]:
        """FTS5 BM25 with an OR backfill.

        A 40-word ASR turn ANDs to zero matches, which silently removed the
        lexical layer from the fusion (measured: 0/40 candidate recall for
        BM25 alone). AND first keeps precision when it works; OR fills the
        list with ranked partial matches when it does not.
        """
        from engine.store.fts import fts_quote, fts_quote_or
        q = fts_quote(query)
        if not q:
            return []
        rows = self.db.execute(
            """SELECT chunk_id, message_id, text, bm25(chunks_fts) AS rank
               FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
            (q, k)).fetchall()
        out = [dict(r) for r in rows]
        if len(out) < k:
            seen = {r["chunk_id"] for r in out}
            qor = fts_quote_or(query)
            if qor:
                more = self.db.execute(
                    """SELECT chunk_id, message_id, text, bm25(chunks_fts) AS rank
                       FROM chunks_fts WHERE chunks_fts MATCH ?
                       ORDER BY rank LIMIT ?""", (qor, k * 4)).fetchall()
                for r in more:
                    if r["chunk_id"] in seen:
                        continue
                    seen.add(r["chunk_id"])
                    out.append(dict(r))
                    if len(out) >= k:
                        break
        return out

    def close(self) -> None:
        self.db.close()


def event_id_for(unit: dict, store: "V4Store") -> str:
    """Stable event identity: project + slot.

    The project comes from the immutable message, so two projects naming the
    same slot ("budget") never share a state row.
    """
    msg = store.message(unit.get("message_id", "")) or {}
    project = msg.get("project_id") or "default"
    point = unit.get("point") or ""
    return f"{project}:{unit.get('slot', '')}" + (f":{point}" if point else "")


__all__ = ["V4Store", "Evidence", "event_id_for"]
