"""v6 semantic agent: typed memory -> hybrid retrieval -> tool routing -> local LLM.

The retrieval, intent, slots and tool decisions are deterministic and auditable;
the LLM only phrases the grounded answer from the top-k JSON chunks. Every run
returns a trace so the UI can show what the engine did before the generation.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date
from typing import Any, Iterator

import joblib
import numpy as np

import run_french_lab as lab
from hybrid_retrieval import ACTS, TOPICS, HybridRetriever, parse_query

ART = lab.ART
INTENTS = ["answer", "compare", "build_dossier", "draft_email", "create_task", "schedule_event", "clarify"]
HYPOTHETICAL = re.compile(r"\b(on pourrait|peut-être|peut etre|et si|si on|hypothèse|hypothese|on envisage)\b", re.I)
CITATION_RE = re.compile(r"\[(?:chunk_id\s*:\s*)?([A-Za-z0-9:_\-]+)\]", re.I)


def item(evidence: dict[str, Any]) -> dict[str, Any]:
    return {"statement": evidence["message"], "speaker": evidence["username"],
            "date": evidence["meeting_date"], "timestamp": evidence["timestamp"],
            "chunk_id": evidence["chunk_id"], "meeting_id": evidence["meeting_id"]}


def unit_status(evidence: dict[str, Any]) -> str:
    text = evidence["message"].casefold()
    if HYPOTHETICAL.search(text):
        return "hypothesis"
    if any(x in text for x in ("sous réserve", "sous reserve", "à condition", "a condition", "éventuellement", "eventuellement")):
        return "conditional"
    if evidence.get("act") == "decision" or any(x in text for x in ("décision:", "decision:", "c'est acté", "retenue est")):
        return "decided"
    if evidence.get("act") == "proposition" or any(x in text for x in ("je propose", "on devrait", "proposition")):
        return "proposed"
    if evidence.get("act") == "engagement" or "prends en charge" in text or "prend en charge" in text:
        return "committed"
    return "mention"


class AgentEngine:
    def __init__(self, memory, model: str | None = None, ollama_url: str | None = None):
        self.memory = memory
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
        self.ollama_url = (ollama_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.intent_vec = None
        self.intent_clf = None
        path = ART / "intent_classifier.joblib"
        if path.exists():
            try:
                self.intent_vec, self.intent_clf = joblib.load(path)
            except Exception:
                self.intent_vec = self.intent_clf = None

    # ---- model health ------------------------------------------------------
    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=4) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [m.get("name", "") for m in payload.get("models", [])]
            available = any(n == self.model or n.startswith(self.model + ":") for n in names)
            return {"status": "ready" if available else "model_missing", "model": self.model,
                    "ollama_url": self.ollama_url, "models": names}
        except Exception as exc:
            return {"status": "unavailable", "model": self.model, "ollama_url": self.ollama_url,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _llm(self, messages: list[dict[str, str]], max_tokens: int = 500) -> dict[str, Any]:
        started = time.perf_counter()
        body = json.dumps({"model": self.model, "stream": False, "think": False,
                           "messages": messages, "options": {"temperature": 0.1, "num_predict": max_tokens}}).encode("utf-8")
        request = urllib.request.Request(f"{self.ollama_url}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = (payload.get("message") or {}).get("content", "").strip()
            return {"status": "ok", "text": text, "model": self.model,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "eval_count": payload.get("eval_count"), "prompt_eval_count": payload.get("prompt_eval_count")}
        except Exception as exc:
            return {"status": "error", "text": "", "model": self.model,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}"}

    # ---- analysis ----------------------------------------------------------
    def _detect_project(self, query: str, ds) -> dict[str, Any] | None:
        def norm(text: str) -> set[str]:
            text = unicodedata.normalize("NFKD", text.casefold())
            text = "".join(c for c in text if not unicodedata.combining(c))
            return {t for t in re.findall(r"[a-z0-9]+", text)
                    if t not in {"de", "des", "du", "la", "le", "les", "un", "une", "projet", "reunion", "réunion"}}
        q = norm(query)
        best = None
        for mid, meeting in ds["meetings"].items():
            names = [meeting.get("project") or "", meeting.get("project_id") or "", meeting.get("title") or ""]
            for name in names:
                tokens = norm(name)
                if tokens and tokens <= q:
                    score = len(tokens)
                    if best is None or score > best["score"]:
                        best = {"project_id": meeting.get("project_id") or meeting.get("project"),
                                "project": meeting.get("project") or meeting.get("project_id"),
                                "meeting_id": mid, "name": name, "score": score}
        return best

    def _detect_speaker(self, query: str, ds) -> str | None:
        q = query.casefold()
        speakers = sorted({record["username"] for record in ds["records"].values()}, key=len, reverse=True)
        for speaker in speakers:
            if speaker and speaker.casefold() in q:
                return speaker
        return None

    def analyze(self, query: str, slug: str) -> dict[str, Any]:
        ds = self.memory._get_dataset(slug)
        plan = parse_query(query, {r["username"] for r in ds["records"].values()})
        project = self._detect_project(query, ds)
        speaker = self._detect_speaker(query, ds)
        topic = lab.detect_topic(query)
        intent = self._route_intent(query)
        hypothetical = bool(HYPOTHETICAL.search(query))
        if hypothetical and intent in ("create_task", "schedule_event", "draft_email"):
            intent = "answer"
        slots = {"topic": lab.TOPICS[topic]["label"] if topic else None, "topic_id": topic,
                 "project_id": project["project_id"] if project else None,
                 "project": project["project"] if project else None,
                 "meeting_id": project["meeting_id"] if project else None,
                 "speaker": speaker, "dates": plan.get("dates", []), "latest": plan.get("latest", False),
                 "earliest": plan.get("earliest", False), "count": plan.get("count", False)}
        return {"intent": intent, "slots": slots, "plan": plan, "hypothetical": hypothetical}

    def _route_intent(self, query: str) -> str:
        q = query.casefold()
        if any(x in q for x in ("est-ce que", "est ce que")) and any(x in q for x in ("valid", "proposition", "décision", "decision")):
            return "answer"
        action_cue = bool(re.search(r"\b(pr[ée]pare|r[ée]dige|[ée]cris|cr[ée]e|ajoute|propose|planifie|trouve|bloque|envoie|transf[èe]re|fais)\b", q))
        action_cue = action_cue or any(x in q for x in ("dossier", "mail", "email", "courriel", "tâche", "tache",
                                                         "créneau", "creneau", "rendez-vous", "calendrier"))
        if not action_cue:
            if any(x in q for x in ("compare", "comparer", "changé", "contredit")):
                return "compare"
            if any(x in q for x in ("bonne personne", "bon interlocuteur", "envoie ça", "transfère")):
                return "clarify"
            return "answer"
        if self.intent_vec is not None and self.intent_clf is not None:
            try:
                return lab.route_intent(query, self.intent_vec, self.intent_clf)
            except Exception:
                pass
        q = query.casefold()
        if any(x in q for x in ("dossier", "synthèse", "synthese", "historique")):
            return "build_dossier"
        if any(x in q for x in ("mail", "email", "courriel")):
            return "draft_email"
        if any(x in q for x in ("tâche", "tache", "to-do", "action à réaliser")):
            return "create_task"
        if any(x in q for x in ("créneau", "creneau", "planifie", "rendez-vous", "calendrier")):
            return "schedule_event"
        if any(x in q for x in ("compare", "comparer", "changé", "contredit")):
            return "compare"
        if any(x in q for x in ("bonne personne", "bon interlocuteur", "envoie ça", "transfère")):
            return "clarify"
        return "answer"

    # ---- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, slug: str, meeting_id: str | None, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        retriever = HybridRetriever.get(self.memory, slug)
        started = time.perf_counter()
        payload = retriever.search(query, top_k=max(top_k * 3, top_k), meeting_id=meeting_id, method="graphrag")
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        evidence = []
        ds = self.memory._get_dataset(slug)
        for row in payload["results"]:
            record = ds["records"][row["chunk_id"]]
            act = ACTS[int(np.argmax(record["act_probs"]))] if "act_probs" in record else None
            topic_idx = int(np.argmax(record["topic_probs"])) if "topic_probs" in record else -1
            topic = TOPICS[topic_idx] if 0 <= topic_idx < len(TOPICS) else None
            evidence.append({**row, "act": act, "topic": topic,
                             "project_id": record.get("project_id"),
                             "status": unit_status({**row, "act": act})})
        return evidence, {"elapsed_ms": elapsed, "plan": payload.get("plan"),
                          "candidate_count": payload.get("candidate_count"),
                          "method": payload.get("method")}

    def _pick_evidence(self, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> list[dict[str, Any]]:
        rows = evidence
        if slots.get("project_id"):
            filtered = [e for e in rows if e.get("project_id") == slots["project_id"]]
            if filtered:
                rows = filtered
        if slots.get("speaker"):
            filtered = [e for e in rows if e["username"] == slots["speaker"]]
            if filtered:
                rows = filtered
        return rows

    # ---- tools -------------------------------------------------------------
    def _dossier(self, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
        sections = {name: [] for name in ("contexte", "faits", "chronologie", "decisions",
                                          "points_ouverts", "responsables", "risques", "references")}
        for e in evidence:
            row = item(e)
            sections["chronologie"].append({**row, "status": e["status"]})
            act = e.get("act")
            if e.get("act") == "decision" or e.get("status") == "decided":
                sections["decisions"].append(row)
            elif e.get("act") == "engagement" or e.get("status") == "committed":
                sections["responsables"].append(row)
            elif act == "risque" or "risque" in e["message"].casefold():
                sections["risques"].append(row)
            elif act == "objection" or "réserve" in e["message"].casefold():
                sections["points_ouverts"].append(row)
            elif act in ("information", "proposition", "question", "reponse", "mise_a_jour"):
                sections["faits"].append(row)
        sections["faits"] = [row for row in sections["faits"]
                             if len(row["statement"]) >= 25 and "on s'arrête" not in row["statement"]]
        sections["contexte"] = sections["decisions"][:1] or sections["faits"][:1]
        sections["chronologie"].sort(key=lambda x: (x["date"], x["timestamp"]))
        for e in evidence:
            if e["chunk_id"] not in sections["references"]:
                sections["references"].append(e["chunk_id"])
        missing = [name for name in ("decisions", "responsables", "risques", "points_ouverts") if not sections[name]]
        return {"intent": "build_dossier", "tool_candidate": "dossier.build",
                "section_order": list(sections), "sections": sections, "missing_information": missing,
                "source_unit_ids": sections["references"]}

    def _recipient(self, query: str, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        explicit = slots.get("speaker")
        if explicit:
            row = next((e for e in evidence if e["username"] == explicit), None)
            if row:
                return {"name": explicit, "evidence_chunk_id": row["chunk_id"]}, row
        engagements = [e for e in evidence if e.get("act") == "engagement" or e.get("status") == "committed"]
        names = []
        for e in engagements:
            if e["username"] not in names:
                names.append(e["username"])
        if len(names) == 1:
            row = engagements[0]
            return {"name": names[0], "evidence_chunk_id": row["chunk_id"]}, row
        if len(names) > 1:
            def norm(text: str) -> set[str]:
                text = unicodedata.normalize("NFKD", text.casefold())
                text = "".join(c for c in text if not unicodedata.combining(c))
                return {t for t in re.findall(r"[a-z0-9]+", text)
                        if t not in {"le", "la", "les", "de", "des", "du", "un", "une", "pour", "et", "a", "à", "qui", "est"}}
            q = norm(query)
            scored = sorted(((len(q & norm(e["message"])), e) for e in engagements), key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                row = scored[0][1]
                return {"name": row["username"], "evidence_chunk_id": row["chunk_id"]}, row
            return {"ambiguous": names}, None
        return None, None

    def _email(self, query: str, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
        recipient, row = self._recipient(query, evidence, slots)
        if recipient is None or recipient.get("ambiguous"):
            return {"intent": "clarify", "question": "Plusieurs responsables possibles ou aucun responsable explicite. De qui parlez-vous ?",
                    "candidates": recipient.get("ambiguous", []) if recipient else [], "executed": False}
        topic = (slots.get("topic") or slots.get("project")
                 or (evidence[0]["meeting_title"] if evidence else "le sujet"))
        body = (f"Bonjour {recipient['name']},\n\nPourrais-tu nous transmettre la dernière version validée concernant {topic} ?\n"
                "Merci de préciser la date et la source de la version que tu considères comme courante.\n\nMerci.")
        return {"intent": "draft_email", "tool_candidate": "email.create_draft",
                "recipient": {"name": recipient["name"], "email": None,
                              "evidence_unit_ids": [recipient["evidence_chunk_id"]],
                              "resolution_status": "email_missing"},
                "subject": f"Suivi — {topic}", "body": body,
                "source_unit_ids": [e["chunk_id"] for e in evidence[:5]],
                "needs_confirmation": True, "executed": False}

    def _task(self, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
        row = next((e for e in evidence if e.get("act") == "engagement" or e.get("status") == "committed"), None)
        source = row or (evidence[0] if evidence else None)
        return {"intent": "create_task", "tool_candidate": "task.create_proposal",
                "task_proposal": source["message"] if source else None,
                "owner": source["username"] if source else None,
                "evidence_unit_ids": [source["chunk_id"]] if source else [],
                "source_unit_ids": [e["chunk_id"] for e in evidence[:5]],
                "needs_confirmation": True, "executed": False}

    def _event(self, query: str, evidence: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
        plan = parse_query(query, {e["username"] for e in evidence})
        return {"intent": "schedule_event", "tool_candidate": "calendar.create_proposal",
                "event_proposal": {"title": query, "date": plan.get("dates", [None])[0] if plan.get("dates") else None,
                                   "time": plan.get("times", [None])[0] if plan.get("times") else None},
                "source_unit_ids": [e["chunk_id"] for e in evidence[:5]],
                "needs_confirmation": True, "executed": False}

    def _compare(self, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        decisions = [e for e in evidence if e.get("act") == "decision" or e.get("status") == "decided"]
        decisions.sort(key=lambda e: (e["meeting_date"], e["timestamp"]))
        timeline = [item(e) for e in decisions]
        changes = []
        for old, new in zip(decisions, decisions[1:]):
            changes.append({"from": old["chunk_id"], "to": new["chunk_id"],
                            "relation": "supersedes" if new["meeting_date"] >= old["meeting_date"] else "contradicts",
                            "reason": "décision plus récente sur le même périmètre"})
        return {"intent": "compare", "tool_candidate": "memory.compare", "timeline": timeline, "changes": changes,
                "source_unit_ids": [e["chunk_id"] for e in decisions]}

    # ---- generation --------------------------------------------------------
    def _prompt(self, query: str, analysis: dict[str, Any], evidence: list[dict[str, Any]],
                action: dict[str, Any] | None, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        sources = []
        for e in evidence[:8]:
            sources.append({"chunk_id": e["chunk_id"], "meeting": e["meeting_title"],
                            "speaker": e["username"], "timestamp": f"{e['meeting_date']} {e['timestamp']}",
                            "act": e.get("act"), "status": e["status"], "source_json": e["source_json"]})
        system = (
            "Tu es le moteur de mémoire sémantique d'un agent de réunion. "
            "Réponds uniquement à partir des extraits JSON fournis. Cite chaque fait avec [chunk_id]. "
            "N'invente ni personne, ni date, ni décision, ni adresse e-mail. "
            "Si l'information manque, dis-le explicitement. "
            "Une proposition n'est pas une décision ; une phrase hypothétique ne déclenche aucune action. "
            "Les actions sont des propositions à confirmer, jamais exécutées."
        )
        user = {"query": query, "intent": analysis["intent"], "slots": analysis["slots"],
                "sources": sources, "tool_output": action}
        messages = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])})
        messages.append({"role": "user", "content": "Contexte JSON:\n" + json.dumps(user, ensure_ascii=False) +
                                                    "\n\nRéponds en français, en 3 à 8 phrases maximum, avec les citations [chunk_id]."})
        return messages

    def _fallback_answer(self, evidence: list[dict[str, Any]], action: dict[str, Any] | None) -> str:
        if not evidence:
            return "Je ne trouve pas d'information suffisante dans la mémoire pour répondre."
        parts = [f"[{e['chunk_id']}] {e['message']}" for e in evidence[:3]]
        prefix = "D'après la mémoire : "
        if action and action.get("intent") == "draft_email":
            prefix = "Brouillon proposé (non envoyé). "
        return prefix + " ".join(parts)

    def _citations(self, text: str, evidence: list[dict[str, Any]]) -> list[str]:
        found = CITATION_RE.findall(text or "")
        valid = [e["chunk_id"] for e in evidence]
        return sorted({c for c in found if c in valid})

    def stream(self, query: str, slug: str = "synthetic", meeting_id: str | None = None,
               top_k: int = 8, use_llm: bool = True,
               history: list[dict[str, str]] | None = None) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        analysis = self.analyze(query, slug)
        yield {"type": "step", "step": "analyze", "label": "Analyse d'intention et slots", "status": "done",
               "detail": {"intent": analysis["intent"], "slots": analysis["slots"], "hypothetical": analysis["hypothetical"]}}
        if analysis["intent"] == "clarify":
            yield {"type": "result", "payload": {"answer": "De quel projet ou de quel sujet parlez-vous ?",
                                                 "intent": "clarify", "analysis": analysis, "evidence": [],
                                                 "action": {"intent": "clarify", "question": "De quel projet ou de quel sujet parlez-vous ?",
                                                            "executed": False}, "citations": [],
                                                 "trace_ms": round((time.perf_counter() - started) * 1000, 1)}}
            return
        evidence, retrieval = self.retrieve(query, slug, meeting_id, top_k)
        evidence = self._pick_evidence(evidence, analysis["slots"])
        evidence_limit = len(evidence) if analysis["intent"] == "build_dossier" else top_k
        evidence = evidence[:evidence_limit]
        yield {"type": "step", "step": "retrieval", "label": "Recherche hybride + graphe", "status": "done",
               "detail": {"method": retrieval["method"], "candidates": retrieval["candidate_count"],
                          "elapsed_ms": retrieval["elapsed_ms"], "query_plan": retrieval["plan"],
                          "top": [{"chunk_id": e["chunk_id"], "speaker": e["username"], "status": e["status"],
                                   "score": round(e["score"], 4)} for e in evidence[:top_k]]}}
        action: dict[str, Any] | None = None
        if analysis["intent"] == "build_dossier":
            action = self._dossier(evidence, analysis["slots"])
        elif analysis["intent"] == "draft_email":
            action = self._email(query, evidence, analysis["slots"])
        elif analysis["intent"] == "create_task":
            action = self._task(evidence, analysis["slots"])
        elif analysis["intent"] == "schedule_event":
            action = self._event(query, evidence, analysis["slots"])
        elif analysis["intent"] == "compare":
            action = self._compare(evidence)
        if action is not None:
            yield {"type": "step", "step": "tool", "label": f"Outil {analysis['intent']}", "status": "done",
                   "detail": {"tool": action.get("tool_candidate") or action.get("intent"), "executed": False}}
        if action and action.get("intent") == "clarify":
            yield {"type": "result", "payload": {"answer": action["question"], "intent": "clarify",
                                                 "analysis": analysis, "evidence": evidence[:top_k],
                                                 "action": action, "citations": [],
                                                 "trace_ms": round((time.perf_counter() - started) * 1000, 1)}}
            return
        llm = {"status": "skipped"}
        if use_llm:
            yield {"type": "step", "step": "llm", "label": f"Génération {self.model}", "status": "running",
                   "detail": {"model": self.model}}
            llm = self._llm(self._prompt(query, analysis, evidence, action, history))
            yield {"type": "step", "step": "llm", "label": f"Génération {self.model}", "status": llm["status"],
                   "detail": {k: llm.get(k) for k in ("model", "elapsed_ms", "eval_count", "prompt_eval_count", "error") if llm.get(k) is not None}}
        answer = llm.get("text") or self._fallback_answer(evidence, action)
        payload = {"answer": answer, "intent": analysis["intent"], "analysis": analysis,
                   "evidence": evidence, "action": action,
                   "citations": self._citations(answer, evidence),
                   "llm": {k: llm.get(k) for k in ("status", "model", "elapsed_ms", "eval_count", "prompt_eval_count", "error") if llm.get(k) is not None},
                   "trace_ms": round((time.perf_counter() - started) * 1000, 1)}
        yield {"type": "result", "payload": payload}

    def run(self, query: str, slug: str = "synthetic", meeting_id: str | None = None,
            top_k: int = 8, use_llm: bool = True,
            history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        result = None
        steps = []
        for event in self.stream(query, slug, meeting_id, top_k, use_llm, history):
            if event["type"] == "step":
                steps.append(event)
            elif event["type"] == "result":
                result = event["payload"]
        payload = result or {"answer": "", "intent": "clarify", "evidence": [], "action": None,
                             "citations": [], "llm": {"status": "error"}}
        payload["steps"] = steps
        return payload
