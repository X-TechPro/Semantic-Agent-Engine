"""Streamlit UI (BUILD_GUIDE 6.15) — evidence-first, with a debug panel.

Run: `make ui` (streamlit run ui/app.py). It opens one engine over the store
named in configs (default `configs/system_B.yaml`), lets you ingest the demo
fixture or your own JSON, ask in French, and inspect per-stage scores.

The point of this app is traceability, so nothing is rendered without its date,
speaker, status badge and unit id, and superseded evidence is shown as history.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from engine.config import load_config, repo_root          # noqa: E402
from engine.engine import Engine                          # noqa: E402
from engine.agent.router import resolve_project_for_text  # noqa: E402
from engine.schemas import Query, Source                  # noqa: E402

STATUS_COLOR = {
    "confirmed": "#1a7f37", "cancelled": "#b35900", "proposed": "#7a5cc4",
    "hypothesis": "#7a5cc4", "conditional": "#7a5cc4", "mention": "#57606a",
}
INTENTS = ["auto", "answer", "compare", "build_dossier", "draft_email",
           "create_task", "schedule_event", "clarify"]


@st.cache_resource(show_spinner="opening the engine…")
def get_engine(config_name: str, store_path: str, force: bool):
    cfg = load_config(config_name, overrides={"store": {"path": store_path}})
    eng = Engine(cfg, path=store_path)
    try:
        from engine.agent.router import Agent  # noqa: F401  (warm the import)
    except Exception:
        pass
    return eng


st.set_page_config(page_title="Semantic Agent Engine", layout="wide",
                   page_icon="◆")
st.title("◆ Typed-relation memory — Semantic Agent Engine")
st.caption("Embeddings find what it is about · the relation layer decides whether "
           "it is still true and who said it.")
st.markdown("""
<style>
div[data-testid="stMetric"] {background: rgba(127,127,127,.07);
  border-radius: 10px; padding: 8px 12px;}
.block-container {padding-top: 2.2rem;}
h1 {letter-spacing: .2px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Engine")
    # Menu built from the directory, and the value handed to `load_config` is the
    # **absolute** path. Two wrong forms were measured here before this one worked:
    # `configs/system_B.yaml` (the original literal) and `system_B` both raise
    # FileNotFoundError, because `load_config` resolves a relative name against
    # `configs/` without appending an extension -- so the former became
    # `configs/configs/system_B.yaml` and the latter a nameless `configs/system_B`.
    # The typed list also offered `configs/demo.yaml`, which does not exist at all.
    # Listing the files keeps the menu honest; the absolute path keeps it loadable.
    found = {p.stem: str(p) for p in sorted((repo_root() / "configs").glob("*.yaml"))}
    if not found:
        st.error(f"no *.yaml under {repo_root() / 'configs'} — nothing to open")
        st.stop()
    labels = list(found)
    config_label = st.selectbox("config", labels,
                                index=labels.index("system_B") if "system_B" in labels
                                else 0)
    config_name = found[config_label]
    store_path = st.text_input("store", str(repo_root() / "runs" / "ui" / "ui.db"))
    if st.button("re-open engine"):
        get_engine.clear()
    linker_on = "baseline_A" not in config_name
    st.write(("System B (typed relations on)" if linker_on
              else "Baseline A (plain hybrid RAG)"))

    st.subheader("Ingest")
    fixture = repo_root() / "tests" / "golden" / "demo_scenarios.json"
    if st.button("load the Atlas demo fixture"):
        data = json.loads(fixture.read_text(encoding="utf-8"))
        eng = get_engine(config_name, store_path, False)
        for raw in sorted(data["sources"], key=lambda r: r["source_date"]):
            raw = dict(raw)
            raw["source_date"] = date.fromisoformat(raw["source_date"])
            eng.ingest(Source(**raw))
        st.success("ingested")
    up = st.file_uploader("or upload a meeting JSON", type="json")
    if up is not None:
        raw = json.loads(up.getvalue().decode("utf-8"))
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        report = get_engine(config_name, store_path, False).ingest(Source(**raw))
        st.write(f"units {report.n_units} · stored {report.n_stored} · "
                 f"skipped {report.n_skipped} · edges {report.n_edges} · "
                 f"superseded {report.n_superseded}")

eng = get_engine(config_name, store_path, False)
stats = eng.stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("units", stats["units"])
c2.metric("edges", stats["edges"])
c3.metric("projects", stats["projects"])
c4.metric("people", stats["roster"])
c5.metric("embed", stats["embed_model_version"].split("@")[0])

left, right = st.columns([1.25, 1])
with left:
    text = st.text_input("Ask in French",
                         "Quelle décision a finalement été prise concernant le "
                         "budget du projet Atlas ?",
                         key="ask")
    intent = st.selectbox("intent", INTENTS)
    mode = st.select_slider("query mode", ["current", "history", "compare"],
                            value="current")
    run = st.button("Ask", type="primary") or text

    if run and text:
        res = eng.ask(text, intent=None if intent == "auto" else intent, mode=mode)
        payload = res.payload
        st.markdown(f"**operation:** `{res.operation}` · "
                    f"**intent:** `{res.intent}` · notes {res.notes}")
        if res.clarification:
            st.info(f"❓ {res.clarification.question}")
            for opt in res.clarification.options:
                st.write(f" · {opt}")
        elif res.operation == "answer":
            if payload.get("answer"):
                st.markdown(payload["answer"].replace("\n", "  \n"))
            else:
                st.warning(payload.get("no_answer_reason") or "no answer")
            for h in payload.get("superseded_history", [])[:5]:
                st.caption(f"🗄 history · {h['date']} · {h['text'][:90]}")
        elif res.operation == "build_dossier":
            st.markdown(payload.get("markdown", "")[:20000])
        elif res.operation == "draft_email":
            r = payload["recipient"]
            badge = {"resolved": "🟢", "email_missing": "🟡",
                     "ambiguous": "🟠", "not_found": "🔴"}.get(
                         r["resolution_status"], "⚪")
            st.markdown(f"**to** {badge} {r['name']} "
                        f"`{r['email'] or '— adresse inconnue, jamais inventée —'}`")
            st.caption(f"justification: {r['justification']}")
            st.markdown(f"**subject:** {payload['subject']}")
            st.text_area("body", payload["body"], height=220)
            st.error("⚠ needs_confirmation = true · executed = false — "
                     "rien n'est envoyé.")
            st.download_button("télécharger le JSON",
                               json.dumps(payload, ensure_ascii=False, indent=1),
                               file_name="draft_email.json")
        else:
            st.json(payload)

        if st.checkbox("debug: per-stage scores", value=False):
            q = eng.read_text(text)
            hits = eng.retrieve(Query(text=text,
                                      project=resolve_project_for_text(eng, text)
                                      or q.unit.project,
                                      slots=q.unit.slots, mode=mode))
            st.json({"read": {"kind": q.kind, "intent": q.intent,
                              "status": q.status,
                              "slots": q.unit.slots.model_dump()},
                     "hits": [{"rank": h.rank, "unit_id": h.unit.unit_id,
                               "score": h.score, "parts": h.score_parts,
                               "current": h.unit.is_current,
                               "path": h.path, "text": h.unit.text}
                              for h in hits[:10]]}, expanded=False)

with right:
    st.subheader("Evidence")
    q = eng.read_text(text) if text else None
    # the panel must show the SAME scoping the answer used: read_text leaves
    # project None, so without the router's resolution the raw hybrid list
    # (Atlas on top) sat next to a Boréal answer (measured 2026-10-22)
    proj = (resolve_project_for_text(eng, text) or (q.unit.project if q else None)) \
        if text else None
    hits = eng.retrieve(Query(text=text, project=proj,
                              slots=(q.unit.slots if q else None)), k=12) \
        if text else []
    if not hits:
        st.info("Rien dans l'index — chargez le fixture dans la barre latérale.")
    for i, h in enumerate(hits):
        u = h.unit
        col = STATUS_COLOR.get(u.status or "", "#57606a")
        head = f"{u.status} · {u.type} · {u.source_date} · {u.speaker or '—'}"
        if not u.is_current:
            head += " · ⚠ superseded"
        if i == 0:
            head = f"★ top answer · {head}"
        st.markdown(
            f"<div style='border-left:{'4px solid #1f6feb' if i == 0 else f'3px solid {col}'};"
            f"padding:6px 10px;margin-bottom:8px;"
            f"background:{'rgba(31,111,235,.10)' if i == 0 else 'rgba(127,127,127,.06)'}'>"
            f"<b style='color:{col}'>{head}</b><br>{u.text}"
            f"<br><code style='font-size:.75em'>{u.unit_id} · {u.source_id}"
            f"{' · ' + str(u.slots.subject) if u.slots.subject else ''}</code></div>",
            unsafe_allow_html=True)

    st.subheader("Timeline for a subject")
    subjects = sorted({u.slots.subject for u in eng.store.all_units()
                       if u.slots.subject})
    pick = st.selectbox("subject", subjects or ["—"])
    if pick and subjects:
        for u in eng.store.history(pick):
            tag = "en vigueur" if u.is_current else f"remplacé le {u.valid_to}"
            st.markdown(f"- `{u.source_date}` **{u.status}** · {u.text[:96]} "
                        f"· *{tag}*")

st.caption(f"config `{config_label}` · connectors are simulated: nothing is ever "
           f"sent, created or scheduled.")
