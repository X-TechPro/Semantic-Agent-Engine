#!/usr/bin/env python3
"""Local Flask dashboard for the French synthetic/custom meeting memory."""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

if sys.prefix == sys.base_prefix:
    print(
        "[warn] app.py is running outside .venv. Use ./run_app.sh or "
        ".venv/bin/python app.py; a system torchvision/torch mismatch otherwise "
        "surfaces as \"Could not import module 'XLMRobertaModel'\".",
        file=sys.stderr,
    )

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from memory_engine import MeetingMemoryEngine
from agent_engine import AgentEngine

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024
app.json.sort_keys = False

_engine: MeetingMemoryEngine | None = None
_engine_lock = threading.Lock()
_agent: AgentEngine | None = None
_agent_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def get_engine() -> MeetingMemoryEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                import os
                preference = os.environ.get("MEETING_DEVICE", "auto")
                _engine = MeetingMemoryEngine(device=preference)
    return _engine


def get_agent() -> AgentEngine:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = AgentEngine(get_engine())
    return _agent


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Fichier trop volumineux. Limite : 128 Mo."}), 413


@app.errorhandler(Exception)
def handle_error(error):
    code = 400
    if isinstance(error, KeyError):
        code = 404
    elif isinstance(error, RuntimeError) and "sentence-transformers" in str(error):
        code = 503
    app.logger.exception("API error")
    return jsonify({"error": str(error)}), code


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    engine = get_engine()
    return jsonify({
        "datasets": engine.dataset_summaries(),
        "active_dataset": "v10",
        "models": engine.model_status(),
        "baseline_metrics": engine.baseline_metrics(),
    })


@app.get("/api/meetings")
def api_meetings():
    engine = get_engine()
    slug = request.args.get("dataset", "synthetic")
    return jsonify({"meetings": engine.meeting_list(slug)})


@app.get("/api/meeting/<meeting_id>")
def api_meeting(meeting_id):
    engine = get_engine()
    slug = request.args.get("dataset", "synthetic")
    return jsonify(engine.get_meeting(slug, meeting_id))


@app.get("/api/graph")
def api_graph():
    from hybrid_retrieval import graph_payload
    engine = get_engine()
    slug = request.args.get("dataset", "synthetic")
    return jsonify(graph_payload(engine, slug, None))


@app.get("/api/graph/<meeting_id>")
def api_graph_meeting(meeting_id):
    from hybrid_retrieval import graph_payload
    engine = get_engine()
    slug = request.args.get("dataset", "synthetic")
    return jsonify(graph_payload(engine, slug, meeting_id))


@app.post("/api/search")
def api_search():
    payload = request.get_json(force=True)
    engine = get_engine()
    result = engine.search(
        slug=payload.get("dataset", "synthetic"),
        query=str(payload.get("query", "")),
        method=payload.get("method", "feature_aware"),
        top_k=int(payload.get("top_k", 10)),
        meeting_id=payload.get("meeting_id") or None,
    )
    return jsonify(result)


@app.post("/api/import")
def api_import():
    if "meetings_file" not in request.files:
        return jsonify({"error": "Sélectionnez un fichier JSON de réunions."}), 400
    meetings_file = request.files["meetings_file"]
    if not meetings_file.filename:
        return jsonify({"error": "Nom de fichier vide."}), 400
    query_file = request.files.get("queries_file")
    name = request.form.get("dataset_name", "Dataset personnalisé").strip() or "Dataset personnalisé"
    result = get_engine().import_dataset(
        meetings_file.filename,
        meetings_file.read(),
        name,
        query_file.read() if query_file and query_file.filename else None,
        query_file.filename if query_file and query_file.filename else None,
    )
    return jsonify(result)


def _run_benchmark(job_id: str, slug: str, split: str, methods: list[str]):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()
    try:
        result = get_engine().benchmark(slug, split=split, methods=methods)
        with _jobs_lock:
            _jobs[job_id].update({"status": "completed", "result": result, "finished_at": time.time()})
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({"status": "failed", "error": str(exc),
                                  "traceback": traceback.format_exc(), "finished_at": time.time()})


@app.post("/api/benchmark")
def api_benchmark():
    payload = request.get_json(force=True)
    slug = payload.get("dataset", "synthetic")
    split = payload.get("split", "test")
    methods = payload.get("methods") or ["bm25", "bge_only", "colbert", "rexu_v2_1",
                                         "rexu_v2_2", "graphrag_ft_router", "rexu_v3"]
    methods = [m for m in methods if m in {"bm25", "bge_only", "colbert", "rexu_v2_1",
                                         "rexu_v2_2", "graphrag_ft_router", "rexu_v3"}]
    if not methods:
        return jsonify({"error": "Aucune méthode valide sélectionnée."}), 400
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "dataset": slug, "split": split,
                         "methods": methods, "created_at": time.time()}
    thread = threading.Thread(target=_run_benchmark, args=(job_id, slug, split, methods), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/jobs/<job_id>")
def api_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job inconnu."}), 404
        return jsonify(job)


@app.get("/api/benchmarks")
def api_benchmarks():
    engine = get_engine()
    return jsonify({"baseline": engine.baseline_metrics()})


@app.get("/api/agent/health")
def api_agent_health():
    return jsonify(get_agent().health())


@app.post("/api/agent")
def api_agent():
    payload = request.get_json(force=True)
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "La question est vide."}), 400
    slug = payload.get("dataset", "synthetic")
    meeting_id = payload.get("meeting_id") or None
    top_k = max(1, min(int(payload.get("top_k", 8)), 20))
    use_llm = bool(payload.get("use_llm", True))
    history = payload.get("history") or []
    agent = get_agent()

    def generate():
        try:
            for event in agent.stream(query, slug=slug, meeting_id=meeting_id, top_k=top_k,
                                      use_llm=use_llm, history=history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    # Local-only by default. Do not bind to 0.0.0.0 unless you understand the exposure.
    print("\n  SEMANTIC MEETING MEMORY  ·  http://127.0.0.1:5000\n")
    print("  Device:", get_engine().device)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
