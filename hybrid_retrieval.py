"""End-to-end hybrid meeting memory.

Three retrieval signals, one fused ranker:

1. ColBERT-style late interaction over multilingual BGE-M3 token embeddings
   (MaxSim). ColBERTv2's released checkpoint is English MS-MARCO; the
   multilingual equivalent is a token-level encoder over the same
   late-interaction scoring function.
2. GraphRAG over a tagged meeting graph (chunks, speakers, topics, acts and
   the persisted semantic relations). Propagation is a differentiable
   personalized-PageRank step whose edge weights are parameters.
3. A temporal subspace modelled after TMRL (arXiv 2601.05549): Fourier time
   features plus an explicit recency/interval operator. TMRL learns the
   subspace with LoRA on temporal contrastive data; here the subspace is
   fixed and the fusion weights are learned on the training split.

The query plan is lexical and deterministic (latest / author / act / date),
so out-of-distribution phrasing cannot be silently routed into the wrong
operation the way the old act-argmax router was.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import run_french_lab as lab

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ROOT / "data"
HYBRID_DIR = ART / "hybrid"
HYBRID_DIR.mkdir(parents=True, exist_ok=True)
RELATIONS_DB = ART / "french_meeting_memory_v2.sqlite3"

ACTS = list(lab.ACTS)
TOPICS = list(lab.TOPIC_KEYS)
ACT_KEYWORDS = {
    "decision": ("décision", "decision", "décidé", "decide", "décide", "acté", "validé",
                 "retenu", "tranché", "choisi", "résolu", "resolu", "agreed", "settled", "accord"),
    "risque": ("risque", "risques", "danger", "menace", "incertitude", "risk", "risky",
               "mal tourner", "problème", "probleme", "crainte"),
    "objection": ("objection", "réserve", "reserve", "conteste", "contesté", "désaccord",
                  "desaccord", "oppos", "inquiet", "inqui", "worried", "concern", "mis en garde",
                  "alerte", "averti", "signalé", "débat"),
    "engagement": ("engagement", "engage", "s'engage", "prend en charge", "pris en charge",
                   "responsable", "action item", "commitment", "will do", "owner", "doit"),
    "proposition": ("propose", "proposition", "suggère", "suggere", "suggest", "proposal"),
    "question": ("question", "demande", "ask", "clarif"),
}
FOCUS_KEYWORDS = {
    "decision": ("décision", "decision"),
    "engagement": ("prends en charge", "prend en charge", "s'engage", "engagé"),
    "objection": ("réserve", "objection", "conteste", "désaccord"),
    "risque": ("risque", "danger", "menace"),
}
STOPWORDS = set(
    "le la les un une des de du au aux et ou mais donc car que qui quoi quand comment "
    "quel quelle quels quelles est sont a ont ete été pour par sur dans avec sans en y il elle ils elles "
    "je tu nous vous on ce cette ces cet mon ma mes ton ta tes son sa ses notre votre leur leurs "
    "the a an of to in on for with and or but what which who whom when how is are was were be been "
    "did does do this that these those it its as at by from".split()
)


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------
def record_time(rec: dict[str, Any]) -> datetime:
    date = str(rec.get("meeting_date") or "1970-01-01")
    ts = str(rec.get("timestamp") or "00:00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date}T{ts}", fmt) if "T" not in fmt else datetime.strptime(f"{date} {ts}", fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(f"{date}T{ts}")
    except ValueError:
        m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", ts)
        if m:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            base = datetime.fromisoformat(date)
            return base.replace(hour=hh, minute=mm, second=ss)
    return datetime.fromisoformat(date)


def fourier_time(dt: datetime, dims: int = 32) -> np.ndarray:
    """Periodic time features; the compact temporal subspace from TMRL."""
    day_of_year = dt.timetuple().tm_yday
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    periods = np.array([3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0])
    values = np.array([seconds, seconds, day_of_year % 7 * 86400.0, day_of_year * 86400.0,
                       (dt.year - 2000) * 365 * 86400.0])
    feats = []
    for period, value in zip(periods, values):
        angle = 2 * math.pi * value / period
        feats.extend([math.sin(angle), math.cos(angle)])
    vec = np.asarray(feats, dtype=np.float32)
    if dims <= len(vec):
        vec = vec[:dims]
    else:
        vec = np.pad(vec, (0, dims - len(vec)))
    return vec / (np.linalg.norm(vec) + 1e-9)


# --------------------------------------------------------------------------
# query plan
# --------------------------------------------------------------------------
LATEST_RE = re.compile(
    r"\b(derni[eè]r\w*|derni[eè]res?|last|latest|final|most recent|en dernier|tout dernier|"
    r"cl[oô]t\w*|fin de (?:s[ée]ance|r[ée]union)|termin\w*|mot de la fin|"
    r"derni[eè]re intervention|dernier message)\b", re.I)
EARLIEST_RE = re.compile(r"\b(premier|premi[eè]re|first|au d[ée]but|d[ée]but|debut|ouvert|ouverte|ouvre)\b", re.I)
COUNT_RE = re.compile(r"\b(combien|nombre de|how many|count)\b", re.I)


FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}


def parse_dates(query: str) -> list[str]:
    dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    for day, month, year in re.findall(
            r"\b(\d{1,2})(?:er)?\s+([a-zéûôà]+)\s+(20\d{2})\b", query.casefold()):
        m = FRENCH_MONTHS.get(month)
        if m:
            dates.append(f"{year}-{m:02d}-{int(day):02d}")
    return list(dict.fromkeys(dates))


def parse_query(query: str, speakers: set[str]) -> dict[str, Any]:
    q = query.casefold()
    plan: dict[str, Any] = {"latest": bool(LATEST_RE.search(q)),
                            "earliest": bool(EARLIEST_RE.search(q)),
                            "count": bool(COUNT_RE.search(q)),
                            "acts": [], "speaker": None,
                            "dates": [], "times": [], "interval": None}
    if plan["latest"]:
        plan["earliest"] = False
    for act, words in ACT_KEYWORDS.items():
        if any(w in q for w in words):
            plan["acts"].append(act)
    for name in sorted(speakers, key=len, reverse=True):
        if name and name.casefold() in q:
            plan["speaker"] = name
            break
    plan["dates"] = parse_dates(query)
    plan["times"] = re.findall(r"\b((?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)\b", query)
    if "entre" in q and " et " in q:
        parts = q.split("entre", 1)[1].split(" et ", 1)
        plan["interval"] = tuple(p.strip() for p in parts)
    for marker in ("après", "apres", "after"):
        if marker in q:
            plan["after"] = q.split(marker, 1)[1].strip()
    for marker in ("avant", "before"):
        if marker in q:
            plan["before"] = q.split(marker, 1)[1].strip()
    act_words = {w for words in ACT_KEYWORDS.values() for w in words}
    time_words = {"dernier", "dernière", "derniere", "derniers", "dernières", "dernieres", "last",
                  "latest", "final", "récent", "recent", "tout", "toute", "avant", "après", "apres"}
    residual = []
    for token in re.findall(r"[a-zà-ÿ0-9]+", q):
        if token in STOPWORDS or token in act_words or token in time_words or len(token) < 3:
            continue
        if plan.get("speaker") and token in plan["speaker"].casefold():
            continue
        residual.append(token)
    plan["residual"] = residual
    return plan


def operator_category(plan: dict[str, Any]) -> str:
    acts = set(plan.get("acts", []))
    if plan.get("count"):
        return "count"
    if plan.get("earliest") and "decision" in acts:
        return "first_decision"
    if plan.get("latest") and "decision" in acts:
        return "latest_decision"
    if "objection" in acts:
        return "objection"
    if "engagement" in acts:
        return "commitment"
    if plan.get("latest"):
        return "latest_message"
    if plan.get("dates"):
        return "dated"
    if plan.get("speaker"):
        return "speaker"
    return "generic"


# --------------------------------------------------------------------------
# ColBERT-style multi-vector index
# --------------------------------------------------------------------------
class MultiVectorIndex:
    def __init__(self, records: dict[str, dict[str, Any]], slug: str, device: str = "cuda",
                 model_name: str = "BAAI/bge-m3", max_len: int = 128,
                 projection_path: Path | None = None):
        self.records = records
        self.ids = list(records)
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        self.slug = slug
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model_name = model_name
        self.max_len = max_len
        self.projection_path = projection_path
        self.projection = None
        self._model = None
        self._tokenizer = None
        self.tokens: torch.Tensor | None = None
        self.offsets: torch.Tensor | None = None
        self.doc_index: torch.Tensor | None = None
        suffix = f"-ft-{projection_path.stem}-api2" if projection_path else ""
        self._cache_path = HYBRID_DIR / f"{slug}-{model_name.replace('/', '__')}-L{max_len}{suffix}.pt"
        self._load_cache()

    def _load_cache(self):
        if not self._cache_path.exists():
            return
        try:
            payload = torch.load(self._cache_path, map_location="cpu", weights_only=False)
        except Exception:
            return
        if payload.get("ids") != self.ids or payload.get("model") != self.model_name:
            return
        self.tokens = payload["tokens"].to(torch.float16)
        self.offsets = payload["offsets"].to(torch.long)
        self.doc_index = torch.repeat_interleave(torch.arange(len(self.ids)), self.offsets[1:] - self.offsets[:-1])

    def _ensure_model(self):
        if self._model is not None:
            return
        os.environ.setdefault("HF_HOME", str(ART / "hf_cache"))
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(ART / "hf_cache"))
        self._model = AutoModel.from_pretrained(
            self.model_name, cache_dir=str(ART / "hf_cache"), dtype=torch.float16
        ).to(self.device).eval()
        if self.projection_path and self.projection is None:
            self.projection = torch.nn.Linear(1024, 1024).to(self.device)
            state = torch.load(self.projection_path, map_location="cpu", weights_only=False)["projection"]
            self.projection.load_state_dict(state)
            self.projection.eval()

    def build(self, batch_size: int = 16):
        if self.tokens is not None:
            return
        self._ensure_model()
        assert self._tokenizer is not None
        packed: list[torch.Tensor] = []
        offsets = [0]
        texts = [self.records[cid]["encoder_text"] for cid in self.ids]
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True, max_length=self.max_len,
                                      return_tensors="pt").to(self.device)
                hidden = self._model(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].bool()
                if self.projection is not None:
                    # Official BGE-M3 multi-vector path: colbert_linear on [:, 1:].
                    hidden = hidden[:, 1:]
                    mask = mask[:, 1:]
                    hidden = self.projection(hidden)
                hidden = F.normalize(hidden, dim=-1)
                for i in range(hidden.size(0)):
                    vec = hidden[i][mask[i]].to(torch.float16).cpu()
                    packed.append(vec)
                    offsets.append(offsets[-1] + vec.size(0))
                if start % (batch_size * 40) == 0:
                    print(f"  colbert index: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        self.tokens = torch.cat(packed, dim=0)
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        self.doc_index = torch.repeat_interleave(
            torch.arange(len(self.ids)), self.offsets[1:] - self.offsets[:-1]
        )
        torch.save({"ids": self.ids, "offsets": self.offsets, "tokens": self.tokens,
                    "model": self.model_name, "max_len": self.max_len}, self._cache_path)

    def _to_device(self):
        if self.tokens is None:
            self.build()
        if self.tokens.device != torch.device(self.device):
            self.tokens = self.tokens.to(self.device)
            self.doc_index = self.doc_index.to(self.device)

    def score(self, query: str) -> np.ndarray:
        self._to_device()
        self._ensure_model()
        assert self._tokenizer is not None and self.tokens is not None and self.doc_index is not None
        with torch.inference_mode():
            enc = self._tokenizer([query], truncation=True, max_length=64, return_tensors="pt").to(self.device)
            hidden = self._model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].bool()
            if self.projection is not None:
                # Official BGE-M3 multi-vector path: colbert_linear on [:, 1:].
                hidden = hidden[:, 1:]
                mask = mask[:, 1:]
                hidden = self.projection(hidden)
            q = F.normalize(hidden[0][mask[0]], dim=-1).to(torch.float16)
            sims = q @ self.tokens.T
            maxima = torch.full((q.size(0), len(self.ids)), -1e4, device=self.device,
                                dtype=torch.float16)
            maxima.scatter_reduce_(1, self.doc_index.unsqueeze(0).expand(q.size(0), -1), sims, reduce="amax",
                                   include_self=True)
            scores = maxima.sum(dim=0)
        return scores.float().cpu().numpy()


# --------------------------------------------------------------------------
# tagged graph + differentiable PPR
# --------------------------------------------------------------------------
class GraphMemory:
    def __init__(self, ds: dict[str, Any]):
        self.ds = ds
        self.records: dict[str, dict[str, Any]] = ds["records"]
        self.meetings: dict[str, dict[str, Any]] = ds["meetings"]
        self.ids = list(self.records)
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        totals = np.zeros(len(TOPICS), dtype=np.float64)
        for record in self.records.values():
            if "topic_probs" in record:
                totals += record["topic_probs"]
        transcribed = sum(1 for m in self.meetings.values() if m.get("style") == "transcribed")
        self.topics_meaningful = bool(
            totals.sum() > 0
            and totals.max() / totals.sum() < 0.65
            and transcribed * 2 <= len(self.meetings)
        )
        self._load_relations()
        self._detect_references()

    def _detect_references(self):
        """Cross-meeting references: explicit ISO dates and 'last session' phrases."""
        self.date_to_meeting = {m.get("meeting_date"): mid for mid, m in self.meetings.items() if m.get("meeting_date")}
        by_project: dict[str, list[str]] = {}
        for mid, m in self.meetings.items():
            by_project.setdefault(m.get("project_id") or m.get("project") or "project", []).append(mid)
        previous: dict[str, str] = {}
        for mids in by_project.values():
            mids.sort(key=lambda mid: (self.meetings[mid].get("meeting_date") or "", mid))
            for a, b in zip(mids, mids[1:]):
                previous[b] = a
        refs = []
        seen = set()
        for cid, r in self.records.items():
            text = r["message_text"].casefold()
            for value in set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)):
                target = self.date_to_meeting.get(value)
                if target and target != r["meeting_id"]:
                    key = (cid, target, "date")
                    if key not in seen:
                        seen.add(key)
                        refs.append({"source": cid, "source_meeting": r["meeting_id"],
                                     "target_meeting": target, "kind": "date", "phrase": value})
            if any(x in text for x in ("réunion précédente", "séance précédente", "la dernière fois")):
                target = previous.get(r["meeting_id"])
                if target:
                    key = (cid, target, "previous")
                    if key not in seen:
                        seen.add(key)
                        refs.append({"source": cid, "source_meeting": r["meeting_id"],
                                     "target_meeting": target, "kind": "previous", "phrase": "réunion précédente"})
        self.references = refs

    def _load_relations(self):
        self.relations: list[tuple[str, str, str, float]] = []
        if not RELATIONS_DB.exists():
            return
        try:
            conn = sqlite3.connect(RELATIONS_DB)
            rows = conn.execute("SELECT source_chunk,target_chunk,relation,confidence FROM semantic_relations").fetchall()
            conn.close()
        except sqlite3.Error:
            return
        for src, dst, rel, conf in rows:
            if src in self.pos and dst in self.pos:
                self.relations.append((src, dst, rel, float(conf)))
        # A dataset may ship its own typed graph (v8 multi-hop seed).
        for edge in self.ds.get("relations", []):
            src, dst = edge.get("source"), edge.get("target")
            if src in self.pos and dst in self.pos:
                self.relations.append((src, dst, edge.get("relation", "related"),
                                       float(edge.get("confidence", 1.0))))

    def build_torch(self, device: str = "cpu"):
        n = len(self.ids)
        src, dst, weights = [], [], []
        for a, b, rel, conf in self.relations:
            i, j = self.pos[a], self.pos[b]
            src.extend([i, j]); dst.extend([j, i]); weights.extend([conf, conf])
        by_meeting: dict[str, list[str]] = {}
        for cid in self.ids:
            by_meeting.setdefault(self.records[cid]["meeting_id"], []).append(cid)
        for cids in by_meeting.values():
            cids.sort(key=lambda c: (self.records[c]["message_id"], self.records[c]["timestamp"]))
            for a, b in zip(cids, cids[1:]):
                i, j = self.pos[a], self.pos[b]
                src.extend([i, j]); dst.extend([j, i]); weights.extend([0.15, 0.15])
            by_speaker: dict[str, str] = {}
            for cid in cids:
                speaker = self.records[cid]["username"]
                if speaker in by_speaker:
                    i, j = self.pos[by_speaker[speaker]], self.pos[cid]
                    src.extend([i, j]); dst.extend([j, i]); weights.extend([0.10, 0.10])
                by_speaker[speaker] = cid
        self.edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        self.edge_weight = torch.tensor(weights, dtype=torch.float32, device=device, requires_grad=True)
        self.n = n
        self.device = device

    def ppr(self, seeds: np.ndarray, alpha: float = 0.15, steps: int = 2) -> np.ndarray:
        if not hasattr(self, "edge_index"):
            self.build_torch(device="cuda" if torch.cuda.is_available() else "cpu")
        device = self.edge_index.device
        s = torch.as_tensor(seeds, dtype=torch.float32, device=device)
        s = s / (s.sum() + 1e-9)
        w = torch.clamp(self.edge_weight, min=1e-4)
        col = self.edge_index[0]
        deg = torch.zeros(self.n, device=device).scatter_add_(0, col, w)
        norm = w / (deg[col] + 1e-9)
        adj = torch.sparse_coo_tensor(self.edge_index, norm, (self.n, self.n), device=device)
        h = s
        for _ in range(steps):
            h = (1 - alpha) * s + alpha * torch.sparse.mm(adj, h.unsqueeze(1)).squeeze(1)
        return h.detach().cpu().numpy()

    # ---- payloads for the UI ------------------------------------------------
    def _chunk_node(self, cid: str) -> dict[str, Any]:
        r = self.records[cid]
        act = ACTS[int(np.argmax(r["act_probs"]))] if "act_probs" in r else ""
        topic_idx = int(np.argmax(r["topic_probs"])) if "topic_probs" in r else -1
        topic = TOPICS[topic_idx] if self.topics_meaningful and 0 <= topic_idx < len(TOPICS) else ""
        return {"id": cid, "label": f"#{r['message_id']}", "group": "chunk",
                "tags": ["chunk", act, topic] if topic else ["chunk", act],
                "title": f"{r['username']} · {r['timestamp']}\n{r['message_text'][:220]}",
                "speaker": r["username"], "time": f"{r['meeting_date']} {r['timestamp']}",
                "act": act, "topic": topic, "text": r["message_text"],
                "meeting_id": r["meeting_id"], "value": 1 + float(r["act_probs"][ACTS.index(act)]) if act else 1}

    def payload_global(self) -> dict[str, Any]:
        """Project constellation: projects in a grid, their meetings around them,
        and each project's dominant theme above it. Fixed positions keep the
        whole-memory view readable without a force hairball."""
        nodes, edges = [], []
        groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for mid, m in self.meetings.items():
            pid = m.get("project_id") or m.get("project") or "project"
            groups.setdefault(pid, []).append((mid, m))
        project_names = {pid: (items[0][1].get("project") or pid) for pid, items in groups.items()}
        cols = 4 if len(groups) > 12 else 3
        x_spacing, y_spacing = (420, 300) if cols == 4 else (470, 320)
        for gi, (pid, items) in enumerate(groups.items()):
            gx = (gi % cols) * x_spacing
            gy = (gi // cols) * y_spacing
            nodes.append({"id": f"project:{pid}", "label": project_names[pid], "group": "project",
                          "tags": ["project"], "title": project_names[pid], "value": len(items),
                          "x": gx, "y": gy})
            counts = np.zeros(len(TOPICS), dtype=np.float32)
            count = len(items)
            if count <= 4:
                offsets = [(170, -70), (170, 0), (170, 70), (170, 140)][:count]
            else:
                # Large single-project corpora (v8/v9: 60 meetings) need a grid;
                # the old 4 offsets stacked every meeting on top of each other.
                cols = int(np.ceil(np.sqrt(count)))
                spacing = 110
                # Leave a top margin so the first row is not clipped after fit().
                offsets = [((k % cols) * spacing, 100 + (k // cols) * spacing)
                           for k in range(count)]
            for k, (mid, m) in enumerate(items):
                ox, oy = offsets[k % len(offsets)]
                short = m["title"]
                for prefix in (f"{project_names[pid]} · ", f"{project_names[pid]} — ",
                               f"Réunion de suivi {project_names[pid]} — "):
                    if short.startswith(prefix):
                        short = short[len(prefix):]
                        break
                # A 60-meeting single-project graph cannot draw 60 full titles;
                # use compact indexes and keep the full title in the inspector.
                label = f"{k + 1:02d}" if count > 10 else short
                nodes.append({"id": f"meeting:{mid}", "label": label, "group": "meeting",
                              "tags": ["meeting", m.get("style", "")],
                              "title": f"{m['title']}\n{m['meeting_date']} · {m['message_count']} messages",
                              "meeting_id": mid, "value": max(1, m["message_count"]),
                              "x": gx + ox, "y": gy + oy})
                edges.append({"from": f"project:{pid}", "to": f"meeting:{mid}",
                              "tags": ["contains"], "width": 1})
                for cid in m["messages"]:
                    r = self.records[cid]
                    if "topic_probs" in r:
                        counts += r["topic_probs"]
            if self.topics_meaningful and counts.size and counts.max() > 0:
                idx = int(np.argmax(counts))
                if idx < len(TOPICS):
                    theme = TOPICS[idx]
                    nodes.append({"id": f"theme:{pid}:{theme}", "label": theme, "group": "topic",
                                  "tags": ["topic", theme], "title": f"{theme} · {project_names[pid]}",
                                  "value": float(counts[idx]),
                                  "x": gx, "y": gy - 120})
                    edges.append({"from": f"project:{pid}", "to": f"theme:{pid}:{theme}",
                                  "tags": ["theme"], "width": 1, "weight": float(counts[idx])})
        seen_refs = set()
        for ref in self.references:
            key = (ref["source_meeting"], ref["target_meeting"])
            if key in seen_refs:
                continue
            seen_refs.add(key)
            edges.append({"from": f"meeting:{ref['source_meeting']}", "to": f"meeting:{ref['target_meeting']}",
                          "label": "cite", "tags": ["references"], "width": 2})
        legend = [{"group": "project", "label": "Projet"}, {"group": "meeting", "label": "Réunion"}]
        if self.topics_meaningful:
            legend.append({"group": "topic", "label": "Thème dominant"})
        return {"scope": "global", "nodes": nodes, "edges": edges, "legend": legend,
                "stats": {"nodes": len(nodes), "edges": len(edges),
                          "meetings": len(self.meetings),
                          "projects": len(groups)}}

    def payload_meeting(self, meeting_id: str) -> dict[str, Any]:
        m = self.meetings[meeting_id]
        nodes, edges = [], []
        speakers, acts, topics = set(), set(), set()
        topics_list = TOPICS
        for cid in m["messages"]:
            node = self._chunk_node(cid)
            nodes.append(node)
            speakers.add(node["speaker"])
            if node["act"]:
                acts.add(node["act"])
            if node["topic"]:
                topics.add(node["topic"])
        for speaker in sorted(speakers):
            nodes.append({"id": f"speaker:{speaker}", "label": speaker, "group": "speaker",
                          "tags": ["speaker"], "title": speaker, "value": 1})
        for t in sorted(topics):
            nodes.append({"id": f"topic:{t}", "label": t, "group": "topic", "tags": ["topic"], "title": t, "value": 1})
        for a in sorted(acts):
            nodes.append({"id": f"act:{a}", "label": a, "group": "act", "tags": ["act"], "title": a, "value": 1})
        for cid in m["messages"]:
            r = self.records[cid]
            edges.append({"from": f"speaker:{r['username']}", "to": cid, "tags": ["said"], "width": 1})
            if "act_probs" in r:
                for idx, prob in enumerate(r["act_probs"]):
                    if prob > 0.25:
                        edges.append({"from": cid, "to": f"act:{ACTS[idx]}", "tags": ["act"],
                                      "width": 1, "weight": float(prob)})
            if "topic_probs" in r and topics_list and self.topics_meaningful:
                for idx, prob in enumerate(r["topic_probs"]):
                    if prob > 0.20 and idx < len(topics_list):
                        edges.append({"from": cid, "to": f"topic:{topics_list[idx]}", "tags": ["topic"],
                                      "width": 1, "weight": float(prob)})
        ordered = sorted(m["messages"], key=lambda c: (self.records[c]["message_id"], self.records[c]["timestamp"]))
        for a, b in zip(ordered, ordered[1:]):
            edges.append({"from": a, "to": b, "tags": ["precedes"], "width": 1})
        for a, b, rel, conf in self.relations:
            if self.records[a]["meeting_id"] == meeting_id:
                edges.append({"from": a, "to": b, "label": rel, "tags": ["relation", rel],
                              "width": max(1, int(round(conf * 3))), "weight": conf, "relation": rel})
        if not any("relation" in edge["tags"] for edge in edges):
            from relation_engine import predict_relation
            cids = m["messages"]
            decisions = [c for c in cids if "décision:" in self.records[c]["message_text"].casefold()
                         or "decision:" in self.records[c]["message_text"].casefold()]
            commitments = [c for c in cids if "prends en charge" in self.records[c]["message_text"].casefold()
                           or "prends prends en charge" in self.records[c]["message_text"].casefold()]
            objections = [c for c in cids if "réserve" in self.records[c]["message_text"].casefold()
                          or "reserve" in self.records[c]["message_text"].casefold()]
            for decision in decisions:
                for commitment in commitments:
                    predicted = predict_relation(self.records[decision], self.records[commitment])
                    if predicted["relation"] == "creates_action":
                        edges.append({"from": decision, "to": commitment, "label": "creates_action",
                                      "tags": ["relation", "creates_action"], "width": 2,
                                      "weight": predicted["confidence"], "relation": "creates_action"})
                for objection in objections:
                    predicted = predict_relation(self.records[objection], self.records[decision])
                    if predicted["relation"] == "contradicts":
                        edges.append({"from": objection, "to": decision, "label": "contradicts",
                                      "tags": ["relation", "contradicts"], "width": 2,
                                      "weight": predicted["confidence"], "relation": "contradicts"})
        target_meetings = {ref["target_meeting"] for ref in self.references
                           if ref["source_meeting"] == meeting_id}
        known = {n["id"] for n in nodes}
        for target in sorted(target_meetings):
            node_id = f"meeting:{target}"
            if node_id not in known:
                tm = self.meetings[target]
                nodes.append({"id": node_id, "label": tm["title"], "group": "meeting",
                              "tags": ["meeting", "referenced"], "title": f"{tm['title']}\n{tm['meeting_date']}",
                              "meeting_id": target, "value": max(1, tm.get("message_count", 1))})
                known.add(node_id)
        for ref in self.references:
            if ref["source_meeting"] == meeting_id:
                edges.append({"from": ref["source"], "to": f"meeting:{ref['target_meeting']}",
                              "label": "cite", "tags": ["references"], "width": 2})
        for edge in edges:
            for endpoint in (edge["from"], edge["to"]):
                if endpoint in known:
                    continue
                prefix, _, name = endpoint.partition(":")
                if prefix == "act":
                    nodes.append({"id": endpoint, "label": name, "group": "act", "tags": ["act"],
                                  "title": name, "value": 1})
                elif prefix == "topic":
                    nodes.append({"id": endpoint, "label": name, "group": "topic", "tags": ["topic"],
                                  "title": name, "value": 1})
                else:
                    continue
                known.add(endpoint)
        legend = [{"group": "chunk", "label": "Message"}, {"group": "speaker", "label": "Intervenant"}]
        if self.topics_meaningful:
            legend.append({"group": "topic", "label": "Thème"})
        legend.append({"group": "act", "label": "Acte"})
        return {"scope": "meeting", "meeting_id": meeting_id, "meeting": {
                    "title": m["title"], "date": m["meeting_date"], "style": m.get("style")},
                "nodes": nodes, "edges": edges,
                "legend": legend,
                "stats": {"nodes": len(nodes), "edges": len(edges), "messages": len(m["messages"]),
                          "speakers": len(speakers), "topics": len(topics), "acts": len(acts),
                          "relations": sum(1 for e in edges if "relation" in e["tags"])}}


# --------------------------------------------------------------------------
# hybrid retriever
# --------------------------------------------------------------------------
class HybridRetriever:
    _instances: dict[tuple[str, str], "HybridRetriever"] = {}

    def __init__(self, engine, slug: str = "synthetic"):
        self.engine = engine
        self.slug = slug
        self.ds = engine._get_dataset(slug)
        self.records = self.ds["records"]
        self.ids = list(self.records)
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        self.colbert = MultiVectorIndex(self.records, slug, device=engine.device)
        projection = ART / "bge_colbert_projection_mixed.pt"
        if not projection.exists():
            projection = ART / "bge_colbert_projection.pt"
        self.colbert_ft = (MultiVectorIndex(self.records, slug, device=engine.device,
                                           projection_path=projection)
                           if projection.exists() else self.colbert)
        self._colbert_v2 = None
        self._colbert_ft_v2 = None
        self.bge_vecs: torch.Tensor | None = None
        self.graph = GraphMemory(self.ds)
        speakers = {r["username"] for r in self.records.values()}
        self.speakers = speakers
        self.times = np.array([record_time(self.records[c]).timestamp() for c in self.ids], dtype=np.float64)
        self.time_vecs = np.stack([fourier_time(record_time(self.records[c])) for c in self.ids])
        self._fusion_path = HYBRID_DIR / f"fusion-{slug}.json"
        self.weights = torch.tensor([0.40, 0.15, 0.10, 0.10, 0.05, 0.20], dtype=torch.float32)
        if self._fusion_path.exists():
            try:
                payload = json.loads(self._fusion_path.read_text())
                self.weights = torch.tensor(payload["weights"], dtype=torch.float32)
            except Exception:
                pass
        self.category_weights: dict[str, torch.Tensor] = {}
        self.tuned_default: torch.Tensor | None = None
        self._load_category_fusion()

    @classmethod
    def get(cls, engine, slug: str) -> "HybridRetriever":
        key = (id(engine), slug)
        if key not in cls._instances:
            cls._instances[key] = cls(engine, slug)
        return cls._instances[key]

    def _load_category_fusion(self):
        paths = [
            DATA / "fusion_categories.json",
            HYBRID_DIR / f"fusion-categories-{self.slug}.json",
            HYBRID_DIR / "fusion-categories.json",
        ]
        for path in paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if payload.get("default"):
                self.tuned_default = torch.tensor(payload["default"], dtype=torch.float32)
            self.category_weights = {
                cat: torch.tensor(weights, dtype=torch.float32)
                for cat, weights in (payload.get("categories") or {}).items() if weights
            }
            return

    def _weights_for(self, category: str | None, tuned: bool = False) -> np.ndarray:
        if tuned and category in self.category_weights:
            weights = self.category_weights[category]
        else:
            weights = self.weights
        return weights.detach().cpu().numpy()

    # ---- signals ------------------------------------------------------------
    def late_colbert(self):
        if self._colbert_v2 is None:
            from colbert_v2 import LateChunkedColBERTv2
            self._colbert_v2 = LateChunkedColBERTv2(self.records, self.ds["meetings"], self.slug,
                                                    device=self.engine.device)
        return self._colbert_v2

    def colbert_ft_v2(self):
        if self._colbert_ft_v2 is None:
            from bge_colbert_v2 import CompressedColBERTIndex
            self.colbert_ft.build()
            self._colbert_ft_v2 = CompressedColBERTIndex(self.colbert_ft, device=self.engine.device)
        return self._colbert_ft_v2

    def _load_bge_dense(self):
        if self.bge_vecs is not None:
            return
        path = HYBRID_DIR / f"{self.slug}-bge-m3-dense.pt"
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("ids") == self.ids:
                    self.bge_vecs = payload["vecs"].to(torch.float16)
                    return
            except Exception:
                pass
        self.colbert._ensure_model()
        tokenizer, model = self.colbert._tokenizer, self.colbert._model
        device = self.colbert.device
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(self.ids), 64):
                batch = [self.records[cid]["encoder_text"] for cid in self.ids[start:start + 64]]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=128,
                                return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state.float()
                vectors.append(F.normalize(hidden[:, 0], dim=-1).cpu())
        self.bge_vecs = torch.cat(vectors, dim=0).to(torch.float16)
        torch.save({"ids": self.ids, "vecs": self.bge_vecs}, path)

    def bge_dense_score(self, query: str) -> np.ndarray:
        self._load_bge_dense()
        self.colbert._ensure_model()
        tokenizer, model = self.colbert._tokenizer, self.colbert._model
        device = self.colbert.device
        with torch.inference_mode():
            enc = tokenizer([query], truncation=True, max_length=64, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state.float()
            q = F.normalize(hidden[0, 0], dim=-1).cpu().to(torch.float16)
        assert self.bge_vecs is not None
        return (self.bge_vecs @ q).float().numpy()

    def _rank_pct(self, values: np.ndarray) -> np.ndarray:
        if len(values) <= 1:
            return np.ones_like(values, dtype=np.float32)
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return (order.astype(np.float32) / (len(values) - 1))

    def _temporal(self, plan: dict[str, Any], candidate_ids: list[str]) -> tuple[np.ndarray, float | None]:
        idx = [self.id_to_pos[c] for c in candidate_ids]
        times = self.times[idx]
        if plan.get("latest"):
            rank = np.argsort(np.argsort(times, kind="stable"), kind="stable").astype(np.float32)
            recency = rank / max(1, len(rank) - 1)
            anchor = times.max()
        elif plan.get("earliest"):
            rank = np.argsort(np.argsort(times, kind="stable"), kind="stable").astype(np.float32)
            recency = rank / max(1, len(rank) - 1)
            anchor = times.min()
        elif plan.get("dates"):
            anchor = datetime.fromisoformat(plan["dates"][0]).timestamp()
            recency = np.exp(-np.abs(times - anchor) / 86400.0)
        elif plan.get("times"):
            hh, mm = plan["times"][0].split(":")[:2]
            base = datetime.fromisoformat(self.records[candidate_ids[0]]["meeting_date"])
            anchor = base.replace(hour=int(hh), minute=int(mm)).timestamp()
            recency = np.exp(-np.abs(times - anchor) / 3600.0)
        else:
            return np.zeros(len(candidate_ids), dtype=np.float32), None
        anchor_dt = datetime.fromtimestamp(anchor)
        qvec = fourier_time(anchor_dt)
        subspace = self.time_vecs[idx] @ qvec
        subspace = (subspace - subspace.min()) / (np.ptp(subspace) + 1e-9)
        return (0.65 * recency + 0.35 * subspace).astype(np.float32), anchor

    def _act_scores(self, plan: dict[str, Any], candidate_ids: list[str]) -> np.ndarray:
        if not plan["acts"]:
            return np.zeros(len(candidate_ids), dtype=np.float32)
        wanted = set(plan["acts"])
        out = np.zeros(len(candidate_ids), dtype=np.float32)
        for i, cid in enumerate(candidate_ids):
            record = self.records[cid]
            acts = record.get("act_probs")
            text = record["message_text"].casefold()
            lexical = any(word in text for act in wanted for word in FOCUS_KEYWORDS.get(act, ()))
            if acts is not None:
                predicted = ACTS[int(np.argmax(acts))]
                if predicted in wanted:
                    out[i] = 1.0
                out[i] += 0.25 * sum(float(acts[ACTS.index(a)]) for a in wanted)
            if lexical:
                out[i] += 0.6
        return np.clip(out, 0, 1.25).astype(np.float32)

    def signals(self, query: str, meeting_id: str | None = None, late: bool = False,
                dense_bge: bool = False) -> dict[str, Any]:
        plan = parse_query(query, self.speakers)
        candidates = self.ds["by_meeting"].get(meeting_id, []) if meeting_id else list(self.ids)
        if late:
            v2 = self.late_colbert().score(query)
            col_full = v2["maxsim"]
        else:
            col_full = self.colbert.score(query)
        bm_full = self.ds["bm25"].scores(query, self.ids)
        qz = self.engine._encode_custom([query])[0][0]
        custom_full = np.asarray([float(np.dot(qz, self.records[c]["vector"])) for c in self.ids], dtype=np.float32)
        bge_full = self.bge_dense_score(query) if dense_bge else None
        seed = 0.65 * self._rank_pct(col_full) + 0.35 * self._rank_pct(bm_full)
        seed = np.where(seed >= np.partition(seed, -30)[-30], seed, 0.0)
        graph_full = self.graph.ppr(seed)
        idx = [self.id_to_pos[c] for c in candidates]
        signals = {
            "colbert": col_full[idx],
            "bm25": bm_full[idx],
            "graph": graph_full[idx],
            "custom": custom_full[idx],
        }
        if late:
            signals["maxsim_v2"] = v2["maxsim"][idx]
            signals["late_vec"] = v2["late_vec"][idx]
        if bge_full is not None:
            signals["bge_dense"] = bge_full[idx]
        signals["temporal"], anchor = self._temporal(plan, candidates)
        signals["act"] = self._act_scores(plan, candidates)
        date_match = np.zeros(len(candidates), dtype=np.float32)
        if plan.get("dates"):
            wanted_dates = set(plan["dates"])
            date_match = np.array([1.0 if self.records[c]["meeting_date"] in wanted_dates else 0.0
                                   for c in candidates], dtype=np.float32)
        signals["date"] = date_match
        speaker_match = np.zeros(len(candidates), dtype=np.float32)
        if plan["speaker"]:
            speaker_match = np.array([1.0 if self.records[c]["username"] == plan["speaker"] else 0.0
                                      for c in candidates], dtype=np.float32)
        signals["speaker"] = speaker_match
        return {"plan": plan, "candidates": candidates, "signals": signals, "anchor": anchor}

    def fuse(self, sig: dict[str, np.ndarray], plan: dict[str, Any],
             dense_key: str = "custom", category: str | None = None,
             soft: bool = False) -> np.ndarray:
        col = self._rank_pct(sig["colbert"])
        bm = self._rank_pct(sig["bm25"])
        graph = self._rank_pct(sig["graph"])
        custom = self._rank_pct(sig[dense_key])
        temporal = sig["temporal"]
        act = sig["act"]
        w = self._weights_for(category, tuned=soft)
        allowed = np.ones_like(temporal, dtype=bool)
        if soft:
            score = (w[0] * col + w[1] * bm + w[2] * graph + w[3] * temporal
                     + w[4] * act + w[5] * custom)
            if plan.get("latest") and not plan["acts"] and not plan.get("speaker"):
                return temporal
            focus = np.clip(act, 0.0, 1.0)
            if plan.get("latest") and plan["acts"]:
                score = score + 0.35 * temporal + 0.25 * focus
            elif plan.get("earliest") and plan["acts"]:
                score = score - 0.35 * temporal + 0.25 * focus
            elif plan.get("count") and "decision" in plan["acts"]:
                score = score + 0.25 * focus
            elif plan.get("dates") and "decision" in plan["acts"] and not plan.get("latest"):
                score = score + 0.15 * sig["date"] + 0.25 * focus
            elif plan.get("earliest"):
                score = score - 0.35 * temporal
            elif plan.get("latest"):
                score = score + 0.20 * temporal
            if plan.get("speaker"):
                score = score + 0.35 * sig["speaker"]
            if plan.get("dates") and not plan.get("latest"):
                score = score + 0.15 * sig["date"]
            return score
        if plan.get("speaker"):
            allowed &= sig["speaker"] > 0
        if plan.get("dates") and not plan.get("latest") and sig["date"].any():
            allowed &= sig["date"] > 0
        if plan.get("latest") and not plan["acts"] and not plan.get("speaker"):
            return temporal
        if plan.get("latest") and plan["acts"]:
            focused = allowed & (sig["act"] > 0.5)
            if not focused.any():
                focused = allowed
            if len(plan.get("residual", [])) >= 2:
                return np.where(focused, 0.45 * col + 0.20 * custom + 0.20 * temporal + 0.15 * bm, -5.0)
            return np.where(focused, 0.55 * temporal + 0.20 * col + 0.15 * custom + 0.10 * bm, -5.0)
        if plan.get("count") and "decision" in plan["acts"]:
            focused = allowed & (sig["act"] > 0.5)
            if focused.any():
                return np.where(focused, 0.45 * col + 0.25 * bm + 0.20 * custom + 0.10 * graph, -5.0)
        if plan.get("dates") and "decision" in plan["acts"] and not plan.get("latest"):
            focused = allowed & (sig["act"] > 0.5)
            if focused.any():
                return np.where(focused, 0.50 * col + 0.20 * bm + 0.20 * custom + 0.10 * temporal, -5.0)
        if plan.get("earliest") and plan["acts"]:
            focused = allowed & (sig["act"] > 0.5)
            if focused.any():
                return np.where(focused, -0.55 * temporal + 0.25 * col + 0.12 * custom + 0.08 * bm, -5.0)
        score = (w[0] * col + w[1] * bm + w[2] * graph + w[3] * temporal
                 + w[4] * act + w[5] * custom)
        if plan.get("dates") and not plan.get("latest") and sig["date"].any():
            score = np.where(allowed, score + 0.15 * sig["date"], -5.0)
        if plan.get("speaker"):
            score = np.where(allowed, score + 0.35, -5.0)
        if plan.get("latest"):
            score = score + 0.20 * temporal
        return score

    def method_scores(self, query: str, meeting_id: str | None = None) -> dict[str, Any]:
        """Return v1 and v2 scores over the same candidates in one retrieval pass."""
        payload = self.signals(query, meeting_id, late=False, dense_bge=True)
        late_payload = self.signals(query, meeting_id, late=True)
        sig = payload["signals"]
        return {
            "candidates": payload["candidates"],
            "plan": payload["plan"],
            "colbert": sig["colbert"],
            "colbert_v2": late_payload["signals"]["colbert"],
            "graphrag": self.fuse(sig, payload["plan"], "custom"),
            "graphrag_bge": self.fuse(sig, payload["plan"], "bge_dense"),
            "graphrag_auto": self.fuse(sig, payload["plan"],
                                       "bge_dense" if not self.graph.topics_meaningful else "custom"),
            "graphrag_tuned": self.fuse(
                sig, payload["plan"],
                "bge_dense" if not self.graph.topics_meaningful else "custom",
                operator_category(payload["plan"]), soft=True),
            "graphrag_soft": self.fuse(
                sig, payload["plan"],
                "bge_dense" if not self.graph.topics_meaningful else "custom",
                None, soft=True),
            "graphrag_cat": self.fuse(
                sig, payload["plan"],
                "bge_dense" if not self.graph.topics_meaningful else "custom",
                operator_category(payload["plan"]), soft=False),
            "graphrag_guarded": self._guarded(
                sig, payload["plan"],
                "bge_dense" if not self.graph.topics_meaningful else "custom"),
            "graphrag_best": (self._guarded(sig, payload["plan"], "custom")
                              if self.graph.topics_meaningful
                              else self.fuse(sig, payload["plan"], "bge_dense")),
            "graphrag_v2": self.fuse(late_payload["signals"], late_payload["plan"]),
        }

    def _guarded(self, sig: dict[str, np.ndarray], plan: dict[str, Any],
                 dense_key: str, margin: float = 0.05) -> np.ndarray:
        base = self.fuse(sig, plan, dense_key)
        if not self.category_weights and self.tuned_default is None:
            return base
        tuned = self.fuse(sig, plan, dense_key, operator_category(plan), soft=True)
        top = np.sort(base)
        if len(top) >= 2:
            gap = (top[-1] - top[-2]) / (abs(top[-1]) + 1e-9)
            if gap > margin:
                return base
        return tuned

    def _feature_matrix(self, sig: dict[str, np.ndarray]) -> torch.Tensor:
        return torch.stack([
            torch.as_tensor(self._rank_pct(sig["colbert"])),
            torch.as_tensor(self._rank_pct(sig["bm25"])),
            torch.as_tensor(self._rank_pct(sig["graph"])),
            torch.as_tensor(sig["temporal"]),
            torch.as_tensor(sig["act"]),
            torch.as_tensor(self._rank_pct(sig["custom"])),
        ], dim=1)

    def train_category_fusion(self, train_queries: list[dict], val_queries: list[dict] | None = None,
                              epochs: int = 150, lr: float = 0.05) -> dict[str, Any]:
        def collect(queries):
            rows = []
            for q in queries:
                target = q.get("target_chunk_id")
                if not target:
                    continue
                payload = self.signals(q["query"], q.get("meeting_id"), dense_bge=False)
                candidates = payload["candidates"]
                if target not in candidates:
                    continue
                rows.append((operator_category(payload["plan"]),
                             self._feature_matrix(payload["signals"]), candidates.index(target)))
            return rows

        train_rows = collect(train_queries)
        val_rows = collect(val_queries or [])

        def constrain(weights: torch.Tensor) -> torch.Tensor:
            weights = weights.clamp(0.0, 0.6)
            return weights / weights.sum().clamp_min(1e-6)

        def fit(rows, init):
            weights = torch.nn.Parameter(init.clone())
            opt = torch.optim.AdamW([weights], lr=lr)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                losses = [F.cross_entropy((feat @ weights).unsqueeze(0), torch.tensor([target]))
                          for _, feat, target in rows]
                loss = torch.stack(losses).mean() + 1e-2 * ((weights - init) ** 2).sum()
                loss.backward()
                opt.step()
            return constrain(weights.detach())

        def mrr(rows, weights):
            values = []
            for _, feat, target in rows:
                scores = feat @ weights
                rank = int((scores > scores[target]).sum()) + 1
                values.append(1.0 / rank if rank <= 10 else 0.0)
            return float(np.mean(values)) if values else 0.0

        initial = torch.tensor(self.weights, dtype=torch.float32)
        global_weights = fit(train_rows, initial) if train_rows else initial
        default_val, global_val = mrr(val_rows, initial), mrr(val_rows, global_weights)
        chosen_global = global_weights if global_val >= default_val else initial

        by_category = defaultdict(list)
        for row in train_rows:
            by_category[row[0]].append(row)
        val_by_category = defaultdict(list)
        for row in val_rows:
            val_by_category[row[0]].append(row)

        categories: dict[str, torch.Tensor] = {}
        validation = {}
        for category, rows in by_category.items():
            if len(rows) < 30 or category in ("generic", "count", "first_decision", "dated", "speaker"):
                continue
            candidate = fit(rows, initial)
            val_rows_cat = val_by_category.get(category, [])
            if len(val_rows_cat) >= 10:
                base = mrr(val_rows_cat, initial)
                improved = mrr(val_rows_cat, candidate)
                validation[category] = {"base": base, "candidate": improved, "n": len(rows)}
                if improved >= base:
                    categories[category] = candidate
            # No validation slice: keep hand weights rather than guess.

        self.tuned_default = chosen_global
        self.category_weights = categories
        payload = {
            "default": self.tuned_default.tolist(),
            "categories": {cat: weights.tolist() for cat, weights in categories.items()},
            "validation": {"default": default_val, "global": global_val, **validation},
            "train_queries": len(train_rows), "val_queries": len(val_rows),
        }
        (HYBRID_DIR / "fusion-categories.json").write_text(json.dumps(payload, indent=2))
        return payload

    def search(self, query: str, top_k: int = 10, meeting_id: str | None = None,
               method: str = "graphrag", extra_scores: np.ndarray | None = None,
               extra_weight: float = 0.3) -> dict[str, Any]:
        late = method in ("colbert_v2", "graphrag_v2")
        auto_bge = method in ("graphrag_auto", "graphrag_tuned", "graphrag_soft",
                              "graphrag_cat", "graphrag_guarded", "graphrag_best",
                              "graphrag_hybrid", "graphrag_ft_router") and not self.graph.topics_meaningful
        use_bge = method == "graphrag_bge" or auto_bge
        candidates = self.ds["by_meeting"].get(meeting_id, []) if meeting_id else list(self.ids)
        if method in ("rexu_v2_1", "rexu_v2_2",
                      "colbert_ft", "colbert_ft_v2"):
            plan, sig = {}, {}
        else:
            plan = parse_query(query, self.speakers)
            payload = self.signals(query, meeting_id, late=late, dense_bge=use_bge)
            plan, candidates, sig = payload["plan"], payload["candidates"], payload["signals"]
        if method == "bge_only":
            full = self.bge_dense_score(query)
            score = np.asarray([full[self.id_to_pos[c]] for c in candidates], dtype=np.float32)
        elif method in ("rexu_v2_1", "colbert_ft"):
            full = self.colbert_ft.score(query)
            score = np.asarray([full[self.id_to_pos[c]] for c in candidates], dtype=np.float32)
        elif method in ("rexu_v2_2", "colbert_ft_v2"):
            full = self.colbert_ft_v2().score(query)
            score = np.asarray([full[self.id_to_pos[c]] for c in candidates], dtype=np.float32)
        elif method in ("colbert", "colbert_v2"):
            score = sig["colbert"]
        elif method == "graphrag_tuned":
            score = self.fuse(sig, plan, "bge_dense" if use_bge else "custom",
                              operator_category(plan), soft=True)
        elif method == "graphrag_soft":
            score = self.fuse(sig, plan, "bge_dense" if use_bge else "custom", None, soft=True)
        elif method == "graphrag_cat":
            score = self.fuse(sig, plan, "bge_dense" if use_bge else "custom",
                              operator_category(plan), soft=False)
        elif method == "graphrag_guarded":
            score = self._guarded(sig, plan, "bge_dense" if use_bge else "custom")
        elif method == "graphrag_best":
            if self.graph.topics_meaningful:
                score = self._guarded(sig, plan, "custom")
            else:
                score = self.fuse(sig, plan, "bge_dense")
        elif method in ("graphrag_hybrid", "graphrag_ft_router"):
            if extra_scores is None:
                raise ValueError(f"{method} requires extra_scores")
            category = operator_category(plan)
            graph_route = (method == "graphrag_hybrid"
                           or category == "latest_message"
                           or (category == "latest_decision" and meeting_id is None))
            if graph_route:
                base = self.fuse(sig, plan, "bge_dense" if use_bge else "custom",
                                 category, soft=True)
                idx = [self.id_to_pos[c] for c in candidates]
                extra = np.asarray(extra_scores, dtype=np.float32)[idx]
                score = (1.0 - extra_weight) * self._rank_pct(extra) + extra_weight * self._rank_pct(base)
            else:
                full = self.colbert_ft.score(query)
                score = np.asarray([full[self.id_to_pos[c]] for c in candidates], dtype=np.float32)
        else:
            score = self.fuse(sig, plan, "bge_dense" if use_bge else "custom")
        order = np.argsort(-score)[:top_k]
        results = []
        for rank, i in enumerate(order, 1):
            if score[i] <= -4:
                continue
            cid = candidates[i]
            r = self.records[cid]
            results.append({
                "rank": rank, "chunk_id": cid, "score": float(score[i]),
                "meeting_id": r["meeting_id"], "meeting_title": r["meeting_title"],
                "meeting_date": r["meeting_date"], "username": r["username"],
                "timestamp": r["timestamp"], "message": r["message_text"],
                "source_json": r["raw"],
                "signals": {k: float(sig[k][i]) for k in sig},
            })
        explanations = {
            "bge_only": "BGE-M3 dense single-vector (CLS cosine)",
            "colbert": "ColBERT-style late interaction (BGE-M3 MaxSim)",
            "colbert_ft": "fine-tuned BGE-M3 mixed-style ColBERT projection",
            "colbert_ft_v2": "fine-tuned BGE-M3 ColBERT + ColBERTv2 2048-centroid compression",
            "rexu_v2_1": "Rexu v2.1 · BGE-M3 FT + ColBERT v1",
            "rexu_v2_2": "Rexu v2.2 · BGE-M3 FT + ColBERTv2",
            "colbert_v2": "ColBERTv2 residual codes + late-chunked BGE-M3 MaxSim",
            "graphrag": "ColBERT-style MaxSim + BM25 + differentiable GraphRAG PPR + temporal subspace",
            "graphrag_bge": "ColBERT-style MaxSim + BM25 + graph PPR + temporal + BGE-M3 dense",
            "graphrag_auto": "GraphRAG; BGE-M3 dense when the custom feature heads are not corpus-meaningful",
            "graphrag_tuned": "GraphRAG with per-operator learned fusion weights and soft act/date plan",
            "graphrag_soft": "GraphRAG with hand weights and a soft act/date plan",
            "graphrag_cat": "GraphRAG with per-operator learned weights and the hard act/date plan",
            "graphrag_guarded": "GraphRAG; keep v1 when its top-1 margin is clear, else tuned fusion",
            "graphrag_best": "guarded fusion on in-domain corpora, auto dense on transcribed corpora",
            "graphrag_hybrid": "0.7 * fine-tuned E5 rank + 0.3 * learned GraphRAG rank",
            "graphrag_ft_router": "fine-tuned BGE ColBERT; GraphRAG only for latest/global temporal queries",
            "graphrag_v2": "late-chunked ColBERTv2 + BM25 + differentiable GraphRAG PPR + temporal subspace",
        }
        explanation = explanations.get(method, explanations["graphrag"])
        return {"query": query, "method": method, "explanation": explanation,
                "candidate_count": len(candidates), "plan": plan,
                "results": results}

    # ---- learned, differentiable fusion -------------------------------------
    def train_fusion(self, queries: list[dict[str, Any]], epochs: int = 120, lr: float = 0.05) -> dict[str, Any]:
        rows = [q for q in queries if q.get("split") == "train"]
        features, targets = [], []
        for q in rows:
            payload = self.signals(q["query"], q.get("meeting_id"))
            sig, candidates = payload["signals"], payload["candidates"]
            if q["target_chunk_id"] not in candidates:
                continue
            features.append(torch.stack([
                torch.as_tensor(self._rank_pct(sig["colbert"])),
                torch.as_tensor(self._rank_pct(sig["bm25"])),
                torch.as_tensor(self._rank_pct(sig["graph"])),
                torch.as_tensor(sig["temporal"]),
                torch.as_tensor(sig["act"]),
                torch.as_tensor(self._rank_pct(sig["custom"])),
            ], dim=1))
            targets.append(candidates.index(q["target_chunk_id"]))
        if not features:
            raise RuntimeError("No train queries with signals were available")
        weights = torch.nn.Parameter(torch.tensor([0.40, 0.15, 0.10, 0.10, 0.05, 0.20], dtype=torch.float32))
        opt = torch.optim.AdamW([weights], lr=lr)
        history = []
        for epoch in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            losses = [F.cross_entropy((f @ weights).unsqueeze(0), torch.tensor([t]))
                      for f, t in zip(features, targets)]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weights], 1.0)
            opt.step()
            if epoch % 20 == 0 or epoch == 1:
                history.append({"epoch": epoch, "loss": float(loss.detach())})
        self.weights = weights.detach().clone()
        self._fusion_path.write_text(json.dumps({"weights": self.weights.tolist(), "history": history}, indent=2))
        return {"weights": self.weights.tolist(), "history": history, "queries": len(features)}


def graph_payload(engine, slug: str, meeting_id: str | None = None) -> dict[str, Any]:
    ds = engine._get_dataset(slug)
    graph = GraphMemory(ds)
    return graph.payload_meeting(meeting_id) if meeting_id else graph.payload_global()


def get_retriever(engine, slug: str) -> HybridRetriever:
    return HybridRetriever.get(engine, slug)
