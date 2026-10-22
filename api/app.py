"""FastAPI surface (BUILD_GUIDE 6.15): /ingest /query /ask /unit /history /offline /health

Every response is produced from a Pydantic contract, so the JSON schemas in
`schemas/*.json` describe this API exactly. Nothing here executes an external
action: `/ask` returns proposals with `needs_confirmation=true` (0.9).
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import date
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query as QueryArg
from pydantic import BaseModel, Field

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "models" / "hf"))

from engine.config import load_config, repo_root            # noqa: E402
from engine.engine import Engine                            # noqa: E402
from engine.agent.router import resolve_project_for_text    # noqa: E402
from engine.schemas import (ActionResult, DraftEmail, Filters, Hit, IngestReport,  # noqa: E402
                            Query, Source, Unit)

app = FastAPI(title="Semantic Agent Engine", version="0.1.0",
              description="Typed-relation memory: embeddings find what it is about, "
                          "the relation layer decides whether it is still true.")

_LOCK = threading.Lock()
_ENGINE: Engine | None = None


def engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        cfg = load_config()
        db = Path(cfg["store"]["path"])
        db.parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = Engine(cfg, path=db)
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.close()
    _ENGINE = None


class IngestBody(BaseModel):
    source: Source


class AskBody(BaseModel):
    text: str = Field(min_length=1)
    intent: str | None = None
    project: str | None = None
    mode: str = "current"


class QueryBody(BaseModel):
    text: str = Field(min_length=1)
    project: str | None = None
    mode: str = "current"
    k: int = Field(default=10, ge=1, le=50)
    filters: Filters | None = None


@app.get("/health")
def health() -> dict:
    e = engine()
    return {"status": "ok", **e.stats()}


@app.post("/ingest", response_model=IngestReport)
def ingest(body: IngestBody = Body(...)) -> IngestReport:
    with _LOCK:
        return engine().ingest(body.source)


@app.post("/query", response_model=list[Hit])
def query(body: QueryBody = Body(...)) -> list[Hit]:
    e = engine()
    read = e.read_text(body.text, project=body.project)
    # the same accent-folded project resolution the router uses (D-038): a raw
    # "projet Boreal" capture must not become a hard filter for a spelling the
    # store does not hold
    proj = (body.project or resolve_project_for_text(e, body.text)
            or read.unit.project)
    q = Query(text=body.text, project=proj,
              slots=read.unit.slots, mode=body.mode)  # type: ignore[arg-type]
    return e.retrieve(q, k=body.k,
                      filters=body.filters if body.filters else None)


@app.post("/ask", response_model=ActionResult)
def ask(body: AskBody = Body(...)) -> ActionResult:
    with _LOCK:
        res = engine().ask(body.text, intent=body.intent, project=body.project,
                           mode=body.mode)
    # belt and braces: an action leaving this API must still be a proposal
    if isinstance(res.payload, dict) and res.payload.get("executed"):
        raise HTTPException(500, "engine returned an executed action")
    return res


@app.get("/unit/{unit_id}", response_model=Unit)
def get_unit(unit_id: str) -> Unit:
    u = engine().store.get_with_sentences(unit_id)
    if u is None:
        raise HTTPException(404, f"no unit {unit_id}")
    return u


@app.get("/history")
def history(subject: str = QueryArg(...), project: str | None = None) -> list[Unit]:
    return engine().store.history(subject, project)


@app.get("/search")
def search(pattern: str, project: str | None = None, limit: int = 20) -> list[Unit]:
    rows = engine().store.db.execute(
        "SELECT unit_id FROM units WHERE text LIKE ? "
        + ("AND project=? " if project else "")
        + "ORDER BY source_date LIMIT ?",
        (f"%{pattern}%", *(([project] if project else [])), limit)).fetchall()
    out = [engine().store.get(r["unit_id"]) for r in rows]
    return [u for u in out if u is not None]


@app.post("/offline")
def offline() -> dict:
    with _LOCK:
        from engine.offline import run_offline

        return run_offline(engine())


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
