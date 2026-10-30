"""Streamlit demo for the Semantic Agent Engine.

Two surfaces, one page:

* **Ask** — the v3 memory: retrieved evidence with dates, speakers, status and
  unit ids, and an answer rendered from the *same* units the evidence shows.
* **Meetings (v4)** — the current pipeline on the five hand-written meetings:
  retrieval table, memory state, cited answer or refusal.

Run: `make ui`. Shareable state: `?q=...` on Ask, `?m=...&q=...` on Meetings.
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.agent.router import resolve_project_for_text          # noqa: E402
from engine.config import load_config, repo_root                   # noqa: E402
from engine.engine import Engine                                   # noqa: E402
from engine.schemas import Query, Source                           # noqa: E402

STATUS_CLASS = {
    "confirmed": "b-ok", "cancelled": "b-bad", "proposed": "b-soft",
    "hypothesis": "b-soft", "conditional": "b-soft", "mention": "b-old",
    "unknown": "b-old",
}
INTENTS = ["auto", "answer", "compare", "build_dossier", "draft_email",
           "create_task", "schedule_event", "clarify"]
DEFAULT_Q = "Quelle décision a finalement été prise concernant le budget du projet Atlas ?"

CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"] {
  visibility: hidden; height: 0;
}
.block-container {padding: 1.6rem 2rem 4rem; max-width: 1240px;}
h1, h2, h3 {letter-spacing: -.01em;}
.sae-hero {display:flex; align-items:baseline; gap:12px; margin-bottom:2px;}
.sae-hero h1 {font-size:1.55rem; margin:0;}
.sae-sub {color:#8b949e; font-size:.92rem; margin:2px 0 18px 0;}
.sae-card {border:1px solid rgba(127,127,127,.25); border-radius:14px;
  padding:14px 16px; margin-bottom:10px; background:rgba(127,127,127,.045);}
.sae-lead {font-size:1.05rem; line-height:1.55; font-weight:600;}
.sae-text {font-size:.93rem; line-height:1.5;}
.sae-meta {font-size:.76rem; color:#8b949e; margin-top:8px; display:flex;
  gap:10px; flex-wrap:wrap; align-items:center;}
.sae-meta code {font-size:.72rem; opacity:.85;}
.badge {padding:1px 9px; border-radius:999px; font-size:.71rem;
  border:1px solid currentColor; white-space:nowrap;}
.b-ok{color:#3fb950}.b-soft{color:#a371f7}.b-old{color:#8b949e}
.b-bad{color:#f85149}.b-info{color:#58a6ff}
.sae-q {color:#8b949e; font-size:.74rem; text-transform:uppercase;
  letter-spacing:.09em; margin-bottom:6px;}
.sae-rule {height:1px; background:rgba(127,127,127,.18); margin:14px 0;}
.rank {color:#58a6ff; font-weight:700; font-variant-numeric:tabular-nums;
  margin-right:8px;}
[data-testid="stMetricValue"] {font-size:1.3rem;}
[data-testid="stMetricLabel"] {font-size:.72rem; letter-spacing:.06em;
  text-transform:uppercase;}
</style>
"""


@st.cache_resource(show_spinner="opening the engine…")
def get_engine(config_name: str, store_path: str):
    cfg = load_config(config_name, overrides={"store": {"path": store_path}})
    return Engine(cfg, path=store_path)


@st.cache_resource(show_spinner="ingesting meeting and loading models…")
def get_v4(path_str: str, rerank: bool, device: str):
    from engine.cli_v4 import build

    return build(Path(path_str), rerank, device)


def esc(text: str | None) -> str:
    return html.escape(str(text or ""))


def badge(status: str | None) -> str:
    cls = STATUS_CLASS.get(status or "", "b-old")
    return f"<span class='badge {cls}'>{esc(status or '—')}</span>"


def fmt_date(value) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d/%m/%Y")
    except ValueError:
        return str(value or "—")


def strip_cite(text: str) -> str:
    return re.sub(r"\s*\[[\w:.-]+\]\s*$", "", text or "").strip()


def unit_map(eng: Engine) -> dict:
    return {u.unit_id: u for u in eng.store.all_units()}


def evidence_card(h, units: dict, top: bool) -> str:
    u = units.get(h.unit.unit_id)
    if u is None:
        return ""
    border = "#58a6ff" if top else "rgba(127,127,127,.28)"
    bg = "rgba(88,166,255,.08)" if top else "rgba(127,127,127,.03)"
    head = f"{badge(u.status)} <span class='badge b-info'>{esc(u.type)}</span>"
    if not u.is_current:
        head += " <span class='badge b-old'>remplacé</span>"
    return (
        f"<div class='sae-card' style='border-left:3px solid {border};"
        f"background:{bg}'>"
        f"<div class='sae-text'><span class='rank'>#{h.rank}</span>"
        f"{esc(u.text)}</div>"
        f"<div class='sae-meta'>{head}"
        f"<span>{fmt_date(u.source_date)}</span>"
        f"<span>{esc(u.speaker or '—')}</span>"
        f"<code>{esc(u.unit_id)}</code>"
        f"<span>score {h.score:.3f}</span></div></div>")


def answer_card(text: str, units: dict, unit_ids: list[str],
                lead_id: str | None = None) -> str:
    out = ["<div class='sae-card'>"]
    out.append(f"<div class='sae-lead'>{esc(text)}</div>")
    if unit_ids:
        u = units.get(lead_id or unit_ids[0])
        if u is not None:
            out.append(
                f"<div class='sae-meta'>{badge(u.status)}"
                f"<span>{fmt_date(u.source_date)}</span>"
                f"<span>{esc(u.speaker or '—')}</span>"
                f"<code>{esc(u.unit_id)}</code></div>")
    out.append("</div>")
    return "".join(out)


def section(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='sae-q'>{esc(title)}</div>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def sidebar() -> tuple[Engine, str]:
    st.sidebar.markdown("### ◆ Semantic Agent Engine")
    views = ["Ask", "Meetings (v4)", "Timeline", "Debug"]
    want = {"v4": "Meetings (v4)", "timeline": "Timeline",
            "debug": "Debug"}.get(str(st.query_params.get("view", "")), "Ask")
    nav = st.sidebar.radio("view", views, index=views.index(want),
                           label_visibility="collapsed")
    st.sidebar.markdown("<div class='sae-rule'></div>", unsafe_allow_html=True)

    found = {p.stem: str(p) for p in sorted((repo_root() / "configs").glob("*.yaml"))}
    if not found:
        st.error(f"no *.yaml under {repo_root() / 'configs'}")
        st.stop()
    labels = list(found)
    config_label = st.sidebar.selectbox(
        "config", labels,
        index=labels.index("system_B") if "system_B" in labels else 0)
    config_name = found[config_label]
    store_path = st.sidebar.text_input(
        "store", str(repo_root() / "runs" / "ui" / "ui.db"))
    if st.sidebar.button("re-open engine", use_container_width=True):
        get_engine.clear()
        st.rerun()
    st.sidebar.caption("System B: typed relations on" if "baseline_A" not in config_name
                       else "Baseline A: plain hybrid RAG")

    st.sidebar.markdown("<div class='sae-rule'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Ingest**")
    fixture = repo_root() / "tests" / "golden" / "demo_scenarios.json"
    if st.sidebar.button("load the Atlas fixture", use_container_width=True):
        data = json.loads(fixture.read_text(encoding="utf-8"))
        eng = get_engine(config_name, store_path)
        for raw in sorted(data["sources"], key=lambda r: r["source_date"]):
            raw = dict(raw)
            raw["source_date"] = date.fromisoformat(raw["source_date"])
            eng.ingest(Source(**raw))
        st.sidebar.success("ingested")
        st.rerun()
    up = st.sidebar.file_uploader("or upload a meeting JSON", type="json")
    if up is not None:
        raw = dict(json.loads(up.getvalue().decode("utf-8")))
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        report = get_engine(config_name, store_path).ingest(Source(**raw))
        st.sidebar.write(f"units {report.n_units} · stored {report.n_stored} · "
                         f"edges {report.n_edges} · superseded {report.n_superseded}")

    eng = get_engine(config_name, store_path)
    stats = eng.stats()
    st.sidebar.markdown("<div class='sae-rule'></div>", unsafe_allow_html=True)
    a, b, c = st.sidebar.columns(3)
    a.metric("units", stats["units"])
    b.metric("edges", stats["edges"])
    c.metric("projects", stats["projects"])
    st.sidebar.caption(f"embed {stats['embed_model_version'].split('@')[0]} · "
                       f"roster {stats['roster']} · config {config_label}")
    st.sidebar.markdown(
        "<div class='sae-rule'></div><div style='font-size:.75rem;color:#8b949e'>"
        "nothing is ever sent or created: every action is a draft.</div>",
        unsafe_allow_html=True)
    return eng, nav


def view_ask(eng: Engine) -> None:
    qparam = str(st.query_params.get("q", "") or "")
    units = unit_map(eng)
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.form("ask_v3", border=False):
            text = st.text_area("Question", qparam or DEFAULT_Q, height=84,
                                label_visibility="collapsed")
            c1, c2, c3 = st.columns([1, 1.4, .8])
            intent = c1.selectbox("intent", INTENTS)
            mode = c2.radio("mode", ["current", "history", "compare"],
                            horizontal=True)
            submitted = c3.form_submit_button("Ask", type="primary",
                                              use_container_width=True)
        if submitted or qparam:
            key = (text, intent, mode)
            if st.session_state.get("v3_key") != key:
                with st.spinner("searching memory…"):
                    st.session_state["v3_res"] = eng.ask(
                        text, intent=None if intent == "auto" else intent,
                        mode=mode)
                st.session_state["v3_key"] = key
            res = st.session_state["v3_res"]
            payload = res.payload

            st.markdown(f"<div class='sae-meta'>operation "
                        f"<code>{esc(res.operation)}</code> · intent "
                        f"<code>{esc(res.intent)}</code></div>",
                        unsafe_allow_html=True)
            if res.notes:
                st.caption(" · ".join(str(n) for n in res.notes))

            if res.clarification:
                st.info(f"❓ {res.clarification.question}")
                for opt in res.clarification.options:
                    st.markdown(f"- {opt}")
            elif res.operation == "answer":
                reason = payload.get("no_answer_reason")
                answer = payload.get("answer")
                claims = payload.get("claims") or []
                if answer and claims:
                    lead = strip_cite(claims[0]["text"])
                    ids = [c["unit_ids"][0] for c in claims if c.get("unit_ids")]
                    st.markdown(answer_card(lead, units, ids, ids[0] if ids else None),
                                unsafe_allow_html=True)
                    for c in claims[1:]:
                        cid = (c.get("unit_ids") or [None])[0]
                        u = units.get(cid) if cid else None
                        meta = (f"{badge(u.status)} <span>{fmt_date(u.source_date)}"
                                f"</span> <span>{esc(u.speaker or '—')}</span>"
                                f" <code>{esc(cid)}</code>" if u else "")
                        st.markdown(
                            f"<div class='sae-card'><div class='sae-text'>"
                            f"{esc(strip_cite(c['text']))}</div>"
                            f"<div class='sae-meta'>{meta}</div></div>",
                            unsafe_allow_html=True)
                    with st.expander("raw answer · citations"):
                        st.code(answer, language=None)
                elif answer:
                    st.markdown(f"<div class='sae-card'><div class='sae-lead'>"
                                f"{esc(answer)}</div></div>",
                                unsafe_allow_html=True)
                else:
                    st.warning(reason or "no answer")
                for h in payload.get("superseded_history", [])[:5]:
                    st.caption(f"history · {fmt_date(h['date'])} · {h['text'][:110]}")
            elif res.operation == "build_dossier":
                st.markdown(payload.get("markdown", "")[:20000])
            elif res.operation == "draft_email":
                r = payload["recipient"]
                mark = {"resolved": "🟢", "email_missing": "🟡",
                        "ambiguous": "🟠", "not_found": "🔴"}.get(
                            r["resolution_status"], "⚪")
                st.markdown(f"**to** {mark} {r['name']} "
                            f"`{r['email'] or '— adresse inconnue, jamais inventée —'}`")
                st.caption(f"justification: {r['justification']}")
                st.markdown(f"**subject:** {payload['subject']}")
                st.text_area("body", payload["body"], height=220)
                st.error("needs_confirmation = true · executed = false — rien n'est envoyé.")
                st.download_button(
                    "télécharger le JSON",
                    json.dumps(payload, ensure_ascii=False, indent=1),
                    file_name="draft_email.json")
            else:
                st.json(payload)
        else:
            st.caption("Ask a question, or open `?q=...` to share one. "
                       "Evidence is listed on the right, with the unit ids the "
                       "answer cites.")

    with right:
        section("Evidence", "same scope the answer used · rank · status · date · speaker")
        q = eng.read_text(text) if text else None
        proj = (resolve_project_for_text(eng, text) or (q.unit.project if q else None)) \
            if text else None
        hits = eng.retrieve(Query(text=text, project=proj,
                                  slots=(q.unit.slots if q else None)), k=12) \
            if text else []
        if not hits:
            st.info("Nothing in the index — load the fixture in the sidebar.")
        for h in hits:
            st.markdown(evidence_card(h, units, h.rank == 1), unsafe_allow_html=True)


def view_meetings() -> None:
    files = sorted((repo_root() / "data" / "mine").glob("*.meeting.json"))
    if not files:
        st.error("no meetings under data/mine")
        return
    mparam = str(st.query_params.get("m", "") or "")
    qparam = str(st.query_params.get("q", "") or "")
    names = [p.name for p in files]
    default = names.index(mparam) if mparam in names else 0

    c1, c2, c3 = st.columns([2, 2, 1])
    picked = c1.selectbox("meeting", names, index=default)
    rerank = c2.toggle("rerank (Qwen3-0.6B)", value=True,
                       help="uncheck for a faster, GPU-free run")
    k = c3.slider("k", 3, 15, 10)
    path = files[names.index(picked)]
    answerer = st.radio("answerer", ["hybrid", "llm", "rules"], horizontal=True,
                        help="hybrid: memory answer when it has proof, else the LLM")

    store, search, mid = get_v4(str(path), rerank, "cuda")
    with st.form("ask_v4", border=False):
        text = st.text_area("Question", qparam or "Quel budget a été validé ?",
                            height=80, label_visibility="collapsed")
        submitted = st.form_submit_button("Ask", type="primary")
    if not (submitted or qparam):
        st.caption("Five hand-written meetings. Pick one, ask, and the retrieval "
                   "table, the memory state and the proof are shown below.")
        return

    with st.spinner("retrieving and answering…"):
        hits = search.retrieve(text, k=k, project=mid)
        from engine.answer.v4 import answer as answer_rules

        ans = None
        if answerer in ("hybrid", "llm"):
            # the model classifies the request; the store answers a lookup
            from engine.answer.meta import lookup
            from engine.query.route import classify

            route = classify(text)
            if route:
                ans = lookup(store, route["object"], route["ordinal"])
        if ans is None:
            ans = answer_rules(text, hits, store=store)
            if answerer == "llm" or (answerer == "hybrid" and ans.unknown):
                try:
                    from engine.answer.llm import answer_llm

                    ans = answer_llm(text, hits, store=store)
                except Exception as exc:  # ollama down: keep the rules answer
                    st.warning(f"LLM unavailable ({exc}); showing the deterministic answer")

    l, r = st.columns([1.3, 1], gap="large")
    with l:
        section("Answer", f"{answerer} · unknown={ans.unknown} · verified={ans.verified}")
        tone = "b-old" if ans.unknown else ("b-ok" if ans.verified else "b-bad")
        st.markdown(
            f"<div class='sae-card'><div class='sae-lead'>{esc(ans.text)}</div>"
            f"<div class='sae-meta'><span class='badge {tone}'>"
            f"{'refused' if ans.unknown else ('verified' if ans.verified else 'unverified')}"
            f"</span><span>status {esc(ans.status)}</span>"
            f"<span>k={k} · {len(hits)} hits</span></div></div>",
            unsafe_allow_html=True)
        if ans.citations:
            section("Proof")
            for c in ans.citations:
                msg = store.message(c.message_id) or {}
                span = (f"{c.start}–{c.end}" if c.start is not None else "unit")
                st.markdown(
                    f"<div class='sae-card'><div class='sae-text'>{esc(c.text)}</div>"
                    f"<div class='sae-meta'><code>{esc(c.unit_id)}</code>"
                    f"<span>{esc(msg.get('speaker', '?'))}</span>"
                    f"<span>{fmt_date(msg.get('timestamp'))}</span>"
                    f"<span>chars {span}</span></div></div>",
                    unsafe_allow_html=True)

    with r:
        section("Memory state", "latest live event per key")
        rows = [r0 for r0 in store.state_events()
                if r0["status"] in {"confirmed", "cancelled", "completed"}]
        latest: dict[str, dict] = {}
        for r0 in rows:
            latest[r0["event_id"]] = r0
        if latest:
            st.dataframe(
                [{"key": v["event_id"], "value": v["value"], "status": v["status"],
                  "evidence": v["evidence_unit"], "at": fmt_date(v["created_at"])}
                 for v in latest.values()],
                hide_index=True, use_container_width=True)
        else:
            st.caption("no live state yet")

    tab_hits, tab_state, tab_prompt = st.tabs(
        ["Retrieval", "State events", "Prompt sent to the LLM"])
    with tab_hits:
        st.dataframe(
            [{"#": h.rank, "unit": h.unit_id,
              "slot": (store.unit(h.unit_id) or {}).get("slot"),
              "point": (store.unit(h.unit_id) or {}).get("point") or "—",
              "status": (store.unit(h.unit_id) or {}).get("status"),
              "text": ((store.unit(h.unit_id) or {}).get("text") or "")[:120]}
             for h in hits],
            hide_index=True, use_container_width=True)
    with tab_state:
        st.dataframe(store.state_events(), hide_index=True, use_container_width=True)
    with tab_prompt:
        from engine.answer.llm import build_prompt

        user, _rows = build_prompt(text, hits, store)
        st.code(user, language=None)


def view_timeline(eng: Engine) -> None:
    units = eng.store.all_units()
    subjects = sorted({u.slots.subject for u in units if u.slots.subject})
    if not subjects:
        st.info("No subjects yet — load the fixture.")
        return
    pick = st.selectbox("subject", subjects)
    st.markdown("<div class='sae-rule'></div>", unsafe_allow_html=True)
    for u in sorted(eng.store.history(pick), key=lambda x: x.source_date):
        tag = ("en vigueur" if u.is_current else f"remplacé le {u.valid_to}")
        cls = "b-ok" if u.is_current else "b-old"
        st.markdown(
            f"<div class='sae-card'><div class='sae-text'>{esc(u.text)}</div>"
            f"<div class='sae-meta'>{badge(u.status)}"
            f"<span>{fmt_date(u.source_date)}</span>"
            f"<span>{esc(u.speaker or '—')}</span>"
            f"<span class='badge {cls}'>{esc(tag)}</span>"
            f"<code>{esc(u.unit_id)}</code></div></div>",
            unsafe_allow_html=True)


def view_debug(eng: Engine) -> None:
    st.markdown("**stats**")
    st.json(eng.stats())
    text = st.text_input("question to trace", DEFAULT_Q)
    if text:
        q = eng.read_text(text)
        proj = resolve_project_for_text(eng, text) or q.unit.project
        hits = eng.retrieve(Query(text=text, project=proj, slots=q.unit.slots), k=10)
        st.json({"read": {"kind": q.kind, "intent": q.intent, "status": q.status,
                          "slots": q.unit.slots.model_dump()},
                 "hits": [{"rank": h.rank, "unit_id": h.unit.unit_id,
                           "score": h.score, "parts": h.score_parts,
                           "current": h.unit.is_current, "path": h.path,
                           "text": h.unit.text} for h in hits]}, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Semantic Agent Engine", page_icon="◆",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='sae-hero'><h1>◆ Semantic Agent Engine</h1></div>"
        "<div class='sae-sub'>A memory for meeting talk that knows what is still "
        "true — answers cite the sentence they came from.</div>",
        unsafe_allow_html=True)

    eng, nav = sidebar()

    if nav == "Ask":
        view_ask(eng)
    elif nav == "Meetings (v4)":
        view_meetings()
    elif nav == "Timeline":
        view_timeline(eng)
    else:
        view_debug(eng)


main()
