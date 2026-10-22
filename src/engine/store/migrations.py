"""Schema creation for the SQLite store (BUILD_GUIDE 2.4).

Single migration object so `rebuild_index` and tests can reason about layout.
`units.unit_id` is TEXT PRIMARY KEY; InnoDB-style implicit rowid is the stable
integer key for FAISS — stable *because* units are never deleted (2.2).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = [
    """
    CREATE TABLE IF NOT EXISTS units (
        row_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_id           TEXT NOT NULL UNIQUE,
        source_id         TEXT NOT NULL,
        source_type       TEXT NOT NULL,
        source_date       TEXT NOT NULL,
        ingested_at       TEXT NOT NULL,
        text              TEXT NOT NULL,
        speaker           TEXT,
        project           TEXT,
        turn_index        INTEGER,
        kind              TEXT,
        type              TEXT,
        status            TEXT,
        slots_json        TEXT NOT NULL DEFAULT '{}',
        valid_from        TEXT,
        valid_to          TEXT,
        superseded_by     TEXT,
        embed_model_version TEXT,
        gold_event_ids_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentences (
        sent_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_id  TEXT NOT NULL REFERENCES units(unit_id),
        start    INTEGER NOT NULL,
        end      INTEGER NOT NULL,
        text     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id            TEXT NOT NULL REFERENCES units(unit_id),
        to_id              TEXT NOT NULL REFERENCES units(unit_id),
        relation           TEXT NOT NULL,
        score              REAL NOT NULL,
        evidence_source_id TEXT NOT NULL,
        created_at         TEXT NOT NULL,
        guard              TEXT,
        UNIQUE (from_id, to_id, relation)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vectors (
        unit_id       TEXT PRIMARY KEY REFERENCES units(unit_id),
        base          BLOB,
        refined       BLOB,
        model_version TEXT,
        dim           INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)
    """,
    # indices named in 2.4
    "CREATE INDEX IF NOT EXISTS idx_units_project_date ON units(project, source_date)",
    "CREATE INDEX IF NOT EXISTS idx_units_status ON units(status)",
    "CREATE INDEX IF NOT EXISTS idx_units_type ON units(type)",
    "CREATE INDEX IF NOT EXISTS idx_units_valid_to ON units(valid_to)",
    "CREATE INDEX IF NOT EXISTS idx_units_source ON units(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_sent_unit ON sentences(unit_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id, relation)",
    "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id, relation)",
    "CREATE INDEX IF NOT EXISTS idx_units_subject ON units(json_extract(slots_json, '$.subject'))",
]

# FTS5 over sentences (6.6). unicode61 with remove_diacritics=2 is the French
# analyser baseline; accent stripping is also ablated in bm25s.
FTS_DDL = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(
        text, unit_id UNINDEXED, sent_id UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
]
