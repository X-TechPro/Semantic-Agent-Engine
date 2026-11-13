from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import hashlib
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import run_french_lab as lab

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
DATASETS = DATA / "user_datasets"
DATASETS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(ART / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_SPECS = {
    "french_me5": {
        "name": "antoinelouis/french-me5-small",
        "query_prefix": "",
        "passage_prefix": "",
        "description": "French-specialized E5-small encoder",
    },
    "multilingual_e5": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "description": "Multilingual E5-small encoder",
    },
}


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def normalize_l2(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def choose_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé, mais torch.cuda.is_available() est False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class MeetingMemoryEngine:
    """Loads the custom French encoder and serves immutable meeting-memory retrieval."""

    def __init__(self, device: str = "auto", batch_size: int = 128):
        self.device = choose_device(device)
        self.batch_size = max(8, int(batch_size))
        torch.set_num_threads(min(8, torch.get_num_threads()))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.ckpt_path = ART / "french_tiny_encoder.pt"
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.vocab = lab.Vocab([])
        self.vocab.itos = ckpt["vocab"]
        self.vocab.stoi = {token: i for i, token in enumerate(self.vocab.itos)}
        self.model = lab.TinySemanticEncoder(len(self.vocab.itos), dim=int(ckpt.get("dim", 64)))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.custom_model_info = {
            "name": "custom-french-tiny",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "dimension": int(ckpt.get("dim", 64)),
            "vocabulary": len(self.vocab.itos),
            "epochs": ckpt.get("epochs"),
            "best_val_mrr": ckpt.get("best_val_mrr"),
        }
        self._datasets: dict[str, dict[str, Any]] = {}
        self._pretrained: dict[str, Any] = {}
        self._active_pretrained_specs: dict[str, dict[str, Any]] = {}
        self._pretrained_vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._pretrained_status: dict[str, dict[str, Any]] = {}
        self._pretrained_signature: dict[str, tuple[int, int] | None] = {}
        self._lock = threading.RLock()
        self._load_synthetic()
        self._load_builtin_extra()
        self._load_transcribed()
        self._load_custom_from_disk()

    def _load_custom_from_disk(self):
        """Rehydrate previously imported datasets from their manifest and isolated SQLite memory."""
        for folder in sorted(DATASETS.iterdir()):
            manifest_path = folder / "manifest.json"
            db_path = folder / "memory.sqlite3"
            if not manifest_path.exists() or not db_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                meetings = manifest.get("meetings", {})
                for mid, meeting in meetings.items():
                    meeting["messages"] = []
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                records = {}
                rows = conn.execute("""
                    SELECT m.chunk_id,m.meeting_id,m.immutable_json,m.message_id,
                           e.embedding,e.topic_probs,e.act_probs
                    FROM messages m JOIN message_embeddings e USING(chunk_id)
                    ORDER BY m.meeting_id,m.message_id
                """).fetchall()
                for row in rows:
                    mid = row["meeting_id"]
                    meeting = meetings.get(mid)
                    if meeting is None:
                        continue
                    raw = json.loads(row["immutable_json"])
                    rec = self._record(mid, raw, meeting, meeting.get("split", "custom"))
                    rec["chunk_id"] = row["chunk_id"]
                    rec["raw_json"] = row["immutable_json"]
                    rec["message_id"] = int(row["message_id"])
                    rec["vector"] = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                    rec["topic_probs"] = np.asarray(json.loads(row["topic_probs"]), dtype=np.float32)
                    rec["act_probs"] = np.asarray(json.loads(row["act_probs"]), dtype=np.float32)
                    records[rec["chunk_id"]] = rec
                    meeting["messages"].append(rec["chunk_id"])
                for row in conn.execute("SELECT meeting_id,summary FROM meetings"):
                    if row["meeting_id"] in meetings:
                        meetings[row["meeting_id"]]["summary"] = row["summary"] or ""
                conn.close()
                query_file = manifest.get("query_filename")
                query_path = folder / query_file if query_file else None
                slug = manifest.get("slug", folder.name)
                self._finalize_dataset(slug, manifest.get("name", slug), meetings, records,
                                       db_path, query_path, is_builtin=False)
            except Exception as exc:
                # A damaged custom dataset must not prevent the reference memory from starting.
                print(f"[warn] Could not load custom dataset {folder.name}: {type(exc).__name__}: {exc}")

    def _encode_custom(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zs, tps, aps = [], [], []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                ids, mask = self.vocab.batch(batch, self.device)
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        z, topic_logits, act_logits = self.model(ids, mask)
                else:
                    z, topic_logits, act_logits = self.model(ids, mask)
                zs.append(z.float().cpu().numpy())
                tps.append(torch.softmax(topic_logits.float(), -1).cpu().numpy())
                aps.append(torch.softmax(act_logits.float(), -1).cpu().numpy())
        if not zs:
            dim = int(self.custom_model_info["dimension"])
            return np.zeros((0, dim), np.float32), np.zeros((0, len(lab.TOPIC_KEYS)), np.float32), np.zeros((0, len(lab.ACTS)), np.float32)
        return normalize_l2(np.concatenate(zs)), np.concatenate(tps), np.concatenate(aps)

    def _load_synthetic(self):
        payload_path = DATA / "french_synthetic_meetings.json"
        db_path = ART / "french_meeting_memory_v2.sqlite3"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        meetings: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        for split, rows in payload.items():
            for meeting in rows:
                mid = meeting["meeting_id"]
                meetings[mid] = {
                    "meeting_id": mid,
                    "project_id": meeting.get("project_id", ""),
                    "project": meeting.get("project", ""),
                    "title": meeting.get("title", mid),
                    "meeting_date": meeting.get("meeting_date", ""),
                    "style": meeting.get("style", "formelle"),
                    "split": split,
                    "attendees": meeting.get("attendees", []),
                    "summary": "",
                    "messages": [],
                }
                for msg in meeting["messages"]:
                    cid = f"{mid}:{int(msg['message_id']):03d}"
                    row = conn.execute("SELECT embedding,topic_probs,act_probs FROM message_embeddings WHERE chunk_id=?", (cid,)).fetchone()
                    if row is None:
                        conn.close()
                        raise RuntimeError(f"Embedding manquant pour {cid} dans la mémoire synthétique")
                    records[cid] = self._record(mid, msg, meetings[mid], split)
                    records[cid]["vector"] = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                    records[cid]["topic_probs"] = np.asarray(json.loads(row["topic_probs"]), dtype=np.float32)
                    records[cid]["act_probs"] = np.asarray(json.loads(row["act_probs"]), dtype=np.float32)
                    meetings[mid]["messages"].append(cid)
                summary_row = conn.execute("SELECT summary FROM meetings WHERE meeting_id=?", (mid,)).fetchone()
                if summary_row:
                    meetings[mid]["summary"] = summary_row["summary"] or ""
        conn.close()
        self._finalize_dataset("synthetic", "Unified v7 · français", meetings, records, None, DATA / "french_retrieval_queries.json", is_builtin=True)

    def _load_builtin_extra(self):
        """Load the hand-written v10 multi-hop corpus as a built-in memory."""
        for slug, name in (("v10", "v10 · Les Rivières de Sel (5 réunions)"),
                           ("qmsum", "QMSum · test (35 réunions, anglais)")):
            meeting_path = DATA / f"{slug}_meetings.json"
            query_path = DATA / f"{slug}_queries.json"
            if not meeting_path.exists() or not query_path.exists():
                continue
            payload = json.loads(meeting_path.read_text(encoding="utf-8"))
            meetings, records, texts = {}, {}, []
            for meeting in payload["meetings"]:
                mid = meeting["meeting_id"]
                meetings[mid] = {
                    "meeting_id": mid,
                    "project_id": meeting.get("project_id", slug),
                    "project": meeting.get("project", "Les Rivières de Sel" if slug == "v10" else slug),
                    "title": meeting.get("title", mid),
                    "meeting_date": meeting.get("meeting_date", meeting.get("date", "")),
                    "style": meeting.get("style", slug),
                    "split": "train",
                    "attendees": meeting.get("attendees", []),
                    "summary": "",
                    "messages": [],
                }
                for msg in meeting["messages"]:
                    rec = self._record(mid, msg, meetings[mid], None)
                    rec["chat_style"] = f"{slug}-synthetic"
                    records[rec["chunk_id"]] = rec
                    texts.append(rec["encoder_text"])
            if not records:
                continue
            vectors, topic_probs, act_probs = self._encode_custom(texts)
            for i, cid in enumerate(records):
                records[cid]["vector"] = vectors[i]
                records[cid]["topic_probs"] = topic_probs[i]
                records[cid]["act_probs"] = act_probs[i]
            self._finalize_dataset(slug, name, meetings, records, None, query_path, is_builtin=True)

    def _load_transcribed(self):
        """Built-in messy-transcription corpus (30 meetings, 10 unrelated projects)."""
        payload_path = DATA / "transcribed_meetings.json"
        query_path = DATA / "transcribed_queries.json"
        if not payload_path.exists():
            return
        rows = json.loads(payload_path.read_text(encoding="utf-8"))
        meetings: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        texts: list[str] = []
        for meeting in rows:
            mid = str(meeting["meeting_id"])
            meetings[mid] = {
                "meeting_id": mid,
                "project_id": meeting.get("project_id", ""),
                "project": meeting.get("project", ""),
                "title": meeting.get("title", mid),
                "meeting_date": meeting.get("meeting_date", ""),
                "style": meeting.get("style", "transcribed"),
                "split": "test",
                "attendees": meeting.get("attendees", []),
                "summary": "",
                "messages": [],
            }
            for msg in meeting["messages"]:
                cid = f"{mid}:{int(msg['message_id']):03d}"
                record = self._record(mid, msg, meetings[mid], "test")
                records[cid] = record
                texts.append(record["encoder_text"])
        if not records:
            return
        vectors, topic_probs, act_probs = self._encode_custom(texts)
        for i, cid in enumerate(records):
            records[cid]["vector"] = vectors[i]
            records[cid]["topic_probs"] = topic_probs[i]
            records[cid]["act_probs"] = act_probs[i]
        self._finalize_dataset("transcribed", "Transcribed · 30 réunions", meetings, records,
                               None, query_path, is_builtin=True)

    @staticmethod
    def _record(mid: str, raw: dict[str, Any], meeting: dict[str, Any], split: str | None):
        username = str(raw.get("username", raw.get("speaker", raw.get("author", "Inconnu"))))
        message_id = raw.get("message_id", raw.get("id", len(meeting.get("messages", [])) + 1))
        try:
            message_id = int(message_id)
        except Exception:
            message_id = len(meeting.get("messages", [])) + 1
        timestamp = str(raw.get("timestamp", raw.get("time", "00:00:00")))
        text = str(raw.get("message", raw.get("text", raw.get("content", ""))))
        title = str(meeting.get("title", mid))
        date = str(meeting.get("meeting_date", meeting.get("date", "1970-01-01")))
        cid = f"{mid}:{int(message_id):03d}"
        return {
            "chunk_id": cid,
            "meeting_id": mid,
            "project_id": str(meeting.get("project_id", meeting.get("project", "custom"))),
            "split": split or meeting.get("split", "custom"),
            "style": meeting.get("style", "custom"),
            "meeting_title": title,
            "meeting_date": date,
            "raw": dict(raw),
            "raw_json": canonical(raw),
            "encoder_text": f"Réunion : {title}. Intervenant : {username}. Intervention : {text}",
            "username": username,
            "timestamp": timestamp,
            "message_text": text,
            "message_id": int(message_id),
        }

    def _finalize_dataset(self, slug, name, meetings, records, db_path, query_path, is_builtin=False):
        by_meeting = defaultdict(list)
        for cid, row in records.items():
            by_meeting[row["meeting_id"]].append(cid)
        for mid in by_meeting:
            by_meeting[mid].sort(key=lambda c: (records[c]["message_id"], records[c]["timestamp"]))
        for mid, meeting in meetings.items():
            meeting["messages"] = by_meeting[mid]
            meeting["message_count"] = len(by_meeting[mid])
            if not meeting.get("attendees"):
                meeting["attendees"] = sorted({records[c]["username"] for c in by_meeting[mid]})
        bm25 = lab.BM25(records)
        query_rows = []
        if query_path and Path(query_path).exists():
            raw_queries = json.loads(Path(query_path).read_text(encoding="utf-8"))
            query_rows = self._parse_queries(raw_queries, list(meetings.values()))
        self._datasets[slug] = {
            "slug": slug,
            "name": name,
            "meetings": meetings,
            "records": records,
            "by_meeting": by_meeting,
            "bm25": bm25,
            "queries": query_rows,
            "db_path": str(db_path) if db_path else None,
            "builtin": is_builtin,
            "created_at": time.time(),
            "pretrained_cache": {},
        }

    def dataset_summaries(self):
        with self._lock:
            result = []
            for slug, ds in self._datasets.items():
                rows = list(ds["meetings"].values())
                result.append({"slug": slug, "name": ds["name"], "meeting_count": len(rows),
                               "message_count": len(ds["records"]), "builtin": ds["builtin"],
                               "has_queries": bool(ds["queries"]), "style_counts": {
                                   "formelle": sum(m.get("style") == "formelle" for m in rows),
                                   "messy": sum(m.get("style") == "messy" for m in rows),
                                   "transcribed": sum(m.get("style") == "transcribed" for m in rows)}})
            return result

    def meeting_list(self, slug: str):
        ds = self._get_dataset(slug)
        out = []
        for m in ds["meetings"].values():
            speakers = sorted({ds["records"][c]["username"] for c in m["messages"]})
            out.append({k: m.get(k) for k in ("meeting_id", "project", "project_id", "title", "meeting_date", "style", "split", "summary", "message_count")})
            out[-1]["speakers"] = speakers
        out.sort(key=lambda x: (x.get("meeting_date") or "", x["meeting_id"]))
        return out

    def get_meeting(self, slug: str, meeting_id: str):
        ds = self._get_dataset(slug)
        m = ds["meetings"].get(meeting_id)
        if not m:
            raise KeyError(f"Réunion inconnue: {meeting_id}")
        return {"meeting": {k: v for k, v in m.items() if k != "messages"},
                "messages": [{"chunk_id": cid, "source_json": ds["records"][cid]["raw"],
                              "username": ds["records"][cid]["username"],
                              "timestamp": ds["records"][cid]["timestamp"],
                              "message": ds["records"][cid]["message_text"]} for cid in m["messages"]]}

    def _get_dataset(self, slug: str | None):
        slug = slug or "synthetic"
        if slug not in self._datasets:
            raise KeyError(f"Dataset inconnu: {slug}")
        return self._datasets[slug]

    @staticmethod
    def _comparison_signature() -> tuple[int, int] | None:
        path = ART / "pretrained_comparison.json"
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _drop_pretrained(self, key: str):
        self._pretrained.pop(key, None)
        self._active_pretrained_specs.pop(key, None)
        self._pretrained_status.pop(key, None)
        self._pretrained_signature.pop(key, None)
        for cache_key in [k for k in self._pretrained_vectors if k[1] == key]:
            self._pretrained_vectors.pop(cache_key, None)
        for ds in self._datasets.values():
            ds.get("pretrained_cache", {}).pop(key, None)

    def _load_pretrained(self, key: str):
        # A benchmark may produce a fine-tuned checkpoint after this process started;
        # reload when the comparison report changes instead of serving stale weights.
        signature = self._comparison_signature()
        if key in self._pretrained:
            if self._pretrained_signature.get(key) == signature:
                return self._pretrained[key]
            self._drop_pretrained(key)
        if key not in MODEL_SPECS:
            raise KeyError(key)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._pretrained_status[key] = {"available": False, "error": f"sentence-transformers indisponible: {exc}"}
            raise RuntimeError(self._pretrained_status[key]["error"])
        base_spec = MODEL_SPECS[key]
        spec = dict(base_spec)
        # Reuse validation-selected prefix mode and an accepted fine-tuned model, if the
        # separate supervised comparison script has produced one.
        comparison_path = ART / "pretrained_comparison.json"
        saved = {}
        if comparison_path.exists():
            try:
                comp = json.loads(comparison_path.read_text(encoding="utf-8"))
                saved = comp.get("models", {}).get(base_spec["name"], {})
            except Exception:
                saved = {}
        prefix_mode = saved.get("selected_prefix_mode")
        if key == "multilingual_e5":
            prefixes = {"e5_prefix": ("query: ", "passage: ")}
            prefix_mode = prefix_mode or "e5_prefix"
        else:
            prefixes = {"plain": ("", ""), "e5_prefix": ("query: ", "passage: ")}
            prefix_mode = prefix_mode if prefix_mode in prefixes else "plain"
        qp, pp = prefixes.get(prefix_mode, ("", ""))
        spec["query_prefix"], spec["passage_prefix"], spec["prefix_mode"] = qp, pp, prefix_mode
        local_saved = None
        fine = saved.get("fine_tuning", {})
        if fine.get("accepted") and fine.get("saved_model"):
            candidate = ROOT / fine["saved_model"]
            if candidate.exists():
                local_saved = candidate
        model_path = str(local_saved) if local_saved else base_spec["name"]
        model = SentenceTransformer(model_path, device=self.device,
                                    cache_folder=str(ART / "hf_cache") if not local_saved else None)
        model.max_seq_length = 128
        if self.device == "cuda":
            model.to(self.device)
            # 8 GB RTX 5060: fp16 inference halves activation/weight bandwidth.
            try:
                model.half()
            except Exception:
                pass
        spec["loaded_path"] = model_path
        self._active_pretrained_specs[key] = spec
        self._pretrained[key] = model
        self._pretrained_status[key] = {"available": True, "name": base_spec["name"], "device": self.device,
                                        "loaded_path": model_path, "prefix_mode": prefix_mode,
                                        "fine_tuned": bool(local_saved)}
        self._pretrained_signature[key] = signature
        return model

    def model_status(self):
        status = {k: self._pretrained_status.get(k, {"available": None, "loaded": k in self._pretrained, "name": v["name"]})
                  for k, v in MODEL_SPECS.items()}
        return {"device": self.device, "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "custom_model": self.custom_model_info, "pretrained": status}

    def _pretrained_vectors_for(self, slug: str, key: str, records: dict[str, dict[str, Any]]):
        cache_key = (slug, key)
        if cache_key in self._pretrained_vectors:
            return self._pretrained_vectors[cache_key]
        model = self._load_pretrained(key)
        spec = self._active_pretrained_specs.get(key, MODEL_SPECS[key])
        ids = list(records)
        texts = [spec["passage_prefix"] + records[c]["encoder_text"] for c in ids]
        vecs = model.encode(texts, batch_size=min(64, self.batch_size) if self.device == "cuda" else min(32, self.batch_size),
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False, device=self.device)
        mapped = {cid: np.asarray(vecs[i], dtype=np.float32) for i, cid in enumerate(ids)}
        self._pretrained_vectors[cache_key] = mapped
        self._datasets[slug]["pretrained_cache"][key] = mapped
        return mapped

    def _query_pretrained(self, key: str, query: str):
        model = self._load_pretrained(key)
        spec = self._active_pretrained_specs.get(key, MODEL_SPECS[key])
        return np.asarray(model.encode([spec["query_prefix"] + query], convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False,
                                      device=self.device)[0], dtype=np.float32)

    @staticmethod
    def _temporal_score(rows, candidates):
        from datetime import datetime
        vals = []
        for cid in candidates:
            r = rows[cid]
            date_value = str(r.get("meeting_date") or "1970-01-01")
            time_value = str(r.get("timestamp") or "00:00:00")
            parsed = None
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
                try:
                    parsed = datetime.strptime(date_value + "T" + time_value, fmt)
                    break
                except Exception:
                    pass
            if parsed is None:
                try:
                    parsed = datetime.fromisoformat(date_value + "T" + time_value)
                except Exception:
                    # Last-resort ordering: parse a clock value, then use message ID.
                    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", time_value)
                    if match:
                        hh, mm, ss = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
                        vals.append(float(hh * 3600 + mm * 60 + ss + int(r.get("message_id", 0)) * 1e-4))
                    else:
                        vals.append(float(r.get("message_id", 0)))
                    continue
            vals.append(parsed.timestamp())
        a = np.asarray(vals, dtype=np.float64)
        span = np.ptp(a)
        return ((a - a.min()) / (span + 1e-9)).astype(np.float32)

    def _feature_aware(self, ds, query: str, candidates: list[str], meeting_scope: str | None, query_features=None):
        if query_features is None:
            qz, qt, qa = self._encode_custom([query])
            qz, qt, qa = qz[0], qt[0], qa[0]
        else:
            qz, qt, qa = query_features
        rows = ds["records"]
        qfake = {"query_id": "app-query", "meeting_id": meeting_scope or "__all__", "query": query}
        dense = np.asarray([float(np.dot(qz, rows[c]["vector"])) for c in candidates], dtype=np.float32)
        bm = ds["bm25"].scores(query, candidates)
        topic = np.asarray([float(np.dot(qt, rows[c]["topic_probs"])) for c in candidates], dtype=np.float32)
        act = np.asarray([float(np.dot(qa, rows[c]["act_probs"])) for c in candidates], dtype=np.float32)
        temporal = self._temporal_score(rows, candidates)
        components = (candidates, topic, act, temporal, lab._rank_percentile(dense),
                      lab._rank_percentile(bm), dense, bm)
        mode = lab.infer_retrieval_mode(query, qt, qa)
        qfake["retrieval_mode"] = mode
        policy = json.loads((ART / "french_results.json").read_text(encoding="utf-8"))["feature_aware_retrieval"]["policy"]
        cand, scores = lab._policy_score(qfake, components, policy)
        if mode == "latest_message":
            # Chronology is structured metadata, not something the embedding should have to memorize.
            scores = temporal
        order = np.argsort(-scores)
        return [{"chunk_id": cand[i], "score": float(scores[i]), "source_json": rows[cand[i]]["raw"],
                 "meeting_id": rows[cand[i]]["meeting_id"], "meeting_title": rows[cand[i]]["meeting_title"],
                 "username": rows[cand[i]]["username"], "timestamp": rows[cand[i]]["timestamp"]} for i in order]

    def search(self, slug: str, query: str, method: str = "feature_aware", top_k: int = 10,
               meeting_id: str | None = None):
        if not query.strip():
            raise ValueError("La requête est vide.")
        ds = self._get_dataset(slug)
        rows = ds["records"]
        if meeting_id and meeting_id not in ds["meetings"]:
            raise KeyError(f"Réunion inconnue: {meeting_id}")
        candidates = ds["by_meeting"].get(meeting_id, []) if meeting_id else list(rows)
        top_k = max(1, min(int(top_k), 50))
        if method == "compare":
            methods = ["bm25", "dense", "feature_aware", "french_me5", "multilingual_e5"]
            result = {}
            for m in methods:
                try:
                    result[m] = self.search(slug, query, m, top_k, meeting_id)
                except Exception as exc:
                    result[m] = {"available": False, "error": str(exc), "results": []}
            return {"query": query, "method": "compare", "results_by_method": result}
        if method == "bm25":
            scores = ds["bm25"].scores(query, candidates)
            order = np.argsort(-scores)[:top_k]
            ranked = [(candidates[i], float(scores[i])) for i in order]
            explanation = "BM25 · matching lexical et fréquence des termes"
        elif method == "dense":
            qv, _, _ = self._encode_custom([query])
            scores = np.asarray([float(np.dot(qv[0], rows[c]["vector"])) for c in candidates], dtype=np.float32)
            order = np.argsort(-scores)[:top_k]
            ranked = [(candidates[i], float(scores[i])) for i in order]
            explanation = "Custom French Transformer · similarité cosinus"
        elif method == "feature_aware":
            full = self._feature_aware(ds, query, candidates, meeting_id)
            ranked = [(x["chunk_id"], x["score"]) for x in full[:top_k]]
            explanation = "Custom embeddings + topic/act heads + temporal policy"
        elif method in ("graphrag_hybrid", "graphrag_ft_router"):
            from hybrid_retrieval import get_retriever
            retriever = get_retriever(self, slug)
            qv = self._query_pretrained("multilingual_e5", query)
            pvec = self._pretrained_vectors_for(slug, "multilingual_e5", rows)
            extra = np.asarray([float(np.dot(qv, pvec[c])) for c in retriever.ids], dtype=np.float32)
            return retriever.search(query, top_k=top_k, meeting_id=meeting_id,
                                    method=method, extra_scores=extra)
        elif method in ("rexu_v3"):
            from pplx_rexu import get_rexu_v3
            return get_rexu_v3(self, slug).search(query, top_k=top_k, meeting_id=meeting_id, method=method)
        elif method in ("bge_only", "colbert", "colbert_ft", "colbert_ft_v2",
                        "rexu_v2_1", "rexu_v2_2", "graphrag",
                        "graphrag_bge", "graphrag_auto", "graphrag_tuned", "graphrag_best",
                        "colbert_v2", "graphrag_v2", "graphrag_ft_router"):
            from hybrid_retrieval import get_retriever
            return get_retriever(self, slug).search(query, top_k=top_k, meeting_id=meeting_id, method=method)
        elif method in MODEL_SPECS:
            pvec = self._pretrained_vectors_for(slug, method, rows)
            qvec = self._query_pretrained(method, query)
            scores = np.asarray([float(np.dot(qvec, pvec[c])) for c in candidates], dtype=np.float32)
            order = np.argsort(-scores)[:top_k]
            ranked = [(candidates[i], float(scores[i])) for i in order]
            explanation = MODEL_SPECS[method]["name"] + " · similarité cosinus"
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        results = []
        for rank, (cid, score) in enumerate(ranked, 1):
            r = rows[cid]
            results.append({"rank": rank, "chunk_id": cid, "score": score,
                            "meeting_id": r["meeting_id"], "meeting_title": r["meeting_title"],
                            "meeting_date": r["meeting_date"], "username": r["username"],
                            "timestamp": r["timestamp"], "message": r["message_text"],
                            "source_json": r["raw"]})
        return {"query": query, "method": method, "explanation": explanation,
                "candidate_count": len(candidates), "results": results}

    def import_dataset(self, filename: str, raw_bytes: bytes, display_name: str,
                       query_bytes: bytes | None = None, query_filename: str | None = None):
        decoded = raw_bytes.decode("utf-8-sig")
        if str(filename).lower().endswith(".jsonl"):
            objects = [json.loads(line) for line in decoded.splitlines() if line.strip()]
            grouped = defaultdict(list)
            metadata = {}
            for i, row in enumerate(objects, 1):
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("meeting_id", row.get("meeting", "custom-meeting-001")))
                metadata.setdefault(mid, {"meeting_id": mid, "title": row.get("meeting_title", mid),
                                          "meeting_date": row.get("meeting_date", "1970-01-01"),
                                          "style": row.get("style", "custom"), "messages": []})
                grouped[mid].append(row)
            obj = []
            for mid, msgs in grouped.items():
                # Keep every JSONL line exactly as uploaded; meeting metadata fields remain part of its source chunk.
                meeting = metadata[mid]
                meeting["messages"] = [dict(row) for row in msgs]
                obj.append(meeting)
        else:
            obj = json.loads(decoded)
        meetings = self._parse_meetings(obj)
        if not meetings:
            raise ValueError("Aucune réunion valide trouvée dans le JSON.")
        slug = f"custom-{uuid.uuid4().hex[:8]}"
        ds_dir = DATASETS / slug
        ds_dir.mkdir(parents=True, exist_ok=False)
        source_path = ds_dir / (Path(filename or "meetings.json").name)
        source_path.write_bytes(raw_bytes)
        query_path = None
        query_rows = []
        if query_bytes:
            query_obj = json.loads(query_bytes.decode("utf-8-sig"))
            query_rows = self._parse_queries(query_obj, meetings)
            query_path = ds_dir / (Path(query_filename or "queries.json").name)
            query_path.write_bytes(query_bytes)
        records: dict[str, dict[str, Any]] = {}
        metad: dict[str, dict[str, Any]] = {}
        texts = []
        cids = []
        for i, meeting in enumerate(meetings):
            mid = str(meeting["meeting_id"])
            meeting["messages"] = []
            metad[mid] = meeting
            for j, raw in enumerate(meeting["_source_messages"], 1):
                raw_id = raw.get("message_id", raw.get("id", j))
                try:
                    raw_id = int(raw_id)
                except Exception:
                    raw_id = j
                if raw_id < 0:
                    raw_id = j
                # chunk IDs are external references; original source JSON remains untouched.
                cid = f"{mid}:{raw_id:03d}"
                if cid in records:
                    cid = f"{mid}:{j:03d}"
                r = self._record(mid, raw, meeting, meeting.get("split", "custom"))
                r["chunk_id"] = cid
                r["raw_json"] = canonical(raw)
                records[cid] = r
                cids.append(cid)
                texts.append(r["encoder_text"])
                meeting["messages"].append(cid)
        vectors, topic_probs, act_probs = self._encode_custom(texts)
        for i, cid in enumerate(cids):
            records[cid]["vector"] = vectors[i]
            records[cid]["topic_probs"] = topic_probs[i]
            records[cid]["act_probs"] = act_probs[i]
        # Derive feature nodes from predictions, not synthetic annotations.
        conn_path = ds_dir / "memory.sqlite3"
        self._persist_custom_db(conn_path, metad, records)
        self._finalize_dataset(slug, display_name or slug, metad, records, conn_path, query_path)
        self._datasets[slug]["queries"] = query_rows
        manifest_meetings = {}
        for mid, meeting in metad.items():
            manifest_meetings[mid] = {k: v for k, v in meeting.items() if k != "_source_messages"}
        manifest = {
            "slug": slug, "name": display_name or slug,
            "source_filename": source_path.name,
            "query_filename": query_path.name if query_path else None,
            "meetings": manifest_meetings,
        }
        (ds_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"slug": slug, "name": display_name or slug, "meeting_count": len(metad),
                "message_count": len(records), "query_count": len(query_rows), "database": str(conn_path),
                "notes": ["Les chunks source sont conservés tels quels.",
                          "Les embeddings custom sont calculés avec le modèle entraîné sur les réunions synthétiques.",
                          "La qualité sur vos données est à mesurer : les labels/activations hors distribution peuvent être faibles."]}

    @staticmethod
    def _parse_meetings(obj):
        if isinstance(obj, dict) and "meetings" in obj:
            rows = obj["meetings"]
        elif isinstance(obj, dict) and "messages" in obj:
            rows = [obj]
        elif isinstance(obj, dict) and any(k in obj for k in ("train", "val", "test")):
            rows = []
            for split in ("train", "val", "test"):
                for meeting in obj.get(split, []):
                    item = dict(meeting)
                    item.setdefault("split", split)
                    rows.append(item)
        elif isinstance(obj, list) and obj and all(isinstance(x, dict) and "messages" in x for x in obj):
            rows = obj
        elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            rows = [{"meeting_id": "custom-meeting-001", "title": "Réunion importée", "messages": obj}]
        else:
            raise ValueError("Format attendu : liste de réunions, {meetings:[...]}, objet réunion unique, split train/val/test, ou liste de chunks.")
        cleaned = []
        used = set()
        for i, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                continue
            meeting = {k: row.get(k) for k in ("meeting_id", "project_id", "project", "title", "meeting_date", "date", "style", "split", "attendees") if row.get(k) is not None}
            mid = str(meeting.get("meeting_id") or f"custom-meeting-{i:03d}")
            if mid in used:
                mid = f"{mid}-{i:03d}"
            used.add(mid)
            meeting["meeting_id"] = mid
            meeting.setdefault("title", mid)
            meeting.setdefault("meeting_date", meeting.get("date", "1970-01-01"))
            meeting.setdefault("style", "custom")
            meeting["_source_messages"] = []
            for j, msg in enumerate(row["messages"], 1):
                if not isinstance(msg, dict):
                    continue
                text = msg.get("message", msg.get("text", msg.get("content")))
                if text is None:
                    continue
                # Do not rewrite user's object: aliases are read for indexing only.
                meeting["_source_messages"].append(dict(msg))
            if meeting["_source_messages"]:
                cleaned.append(meeting)
        return cleaned

    @staticmethod
    def _parse_queries(obj, meetings):
        rows = obj.get("queries", []) if isinstance(obj, dict) else obj
        if not isinstance(rows, list):
            raise ValueError("Le fichier de requêtes doit être une liste ou {queries:[...]}.")
        valid_meetings = {m["meeting_id"] for m in meetings}
        out = []
        for i, q in enumerate(rows, 1):
            if not isinstance(q, dict) or not q.get("query"):
                continue
            item = dict(q)
            item.setdefault("query_id", f"custom-query-{i:05d}")
            mid = item.get("meeting_id")
            if mid and mid not in valid_meetings:
                continue
            targets = item.get("target_chunk_ids") or []
            if isinstance(targets, str):
                targets = [targets]
            if not targets and item.get("target_chunk_id"):
                targets = [item["target_chunk_id"]]
            if not targets and mid and item.get("target_message_id") is not None:
                targets = [f"{mid}:{int(item['target_message_id']):03d}"]
            normalized = []
            for target in targets:
                if target is None:
                    continue
                target = str(target)
                if mid and ":" not in target:
                    try:
                        target = f"{mid}:{int(target):03d}"
                    except Exception:
                        pass
                normalized.append(target)
            if normalized:
                item["target_chunk_ids"] = normalized
                item.setdefault("target_chunk_id", normalized[0])
            out.append(item)
        return out

    def _persist_custom_db(self, db_path: Path, meetings, records):
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
        CREATE TABLE meetings(meeting_id TEXT PRIMARY KEY, project_id TEXT, project TEXT, title TEXT, meeting_date TEXT, style TEXT, summary TEXT, summary_embedding TEXT);
        CREATE TABLE messages(chunk_id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL, message_id INTEGER, timestamp TEXT, username TEXT, immutable_json TEXT NOT NULL, sha256 TEXT NOT NULL, FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id));
        CREATE TABLE message_embeddings(chunk_id TEXT PRIMARY KEY, model_name TEXT, embedding TEXT, topic_probs TEXT, act_probs TEXT, FOREIGN KEY(chunk_id) REFERENCES messages(chunk_id));
        CREATE TABLE feature_nodes(feature_id TEXT PRIMARY KEY, meeting_id TEXT, feature TEXT, kind TEXT, count INTEGER, importance REAL, chunk_ids TEXT, embedding TEXT);
        CREATE TABLE feature_edges(meeting_id TEXT, source TEXT, target TEXT, weight INTEGER, PRIMARY KEY(meeting_id,source,target));
        CREATE TRIGGER immutable_source_message_update BEFORE UPDATE ON messages BEGIN SELECT RAISE(ABORT,'immutable source JSON'); END;
        CREATE TRIGGER immutable_source_message_delete BEFORE DELETE ON messages BEGIN SELECT RAISE(ABORT,'immutable source JSON'); END;
        """)
        for mid, meeting in meetings.items():
            mids = meeting["messages"]
            msg_rows = [records[c] for c in mids]
            topic_probs = np.stack([r["topic_probs"] for r in msg_rows])
            act_probs = np.stack([r["act_probs"] for r in msg_rows])
            vecs = {r["chunk_id"]: r["vector"] for r in msg_rows}
            lab_meeting = {"meeting_id": mid, "title": meeting["title"], "meeting_date": meeting["meeting_date"]}
            if len(mids) >= 2:
                nodes, edges = lab.extract_features(lab_meeting, mids, topic_probs, act_probs, vecs)
            else:
                nodes, edges = [], []
            summary_records = {}
            for r in msg_rows:
                summary_row = dict(r)
                summary_row["raw"] = {"message": r["message_text"], "timestamp": r["timestamp"],
                                      "username": r["username"], "message_id": r["message_id"]}
                summary_records[r["chunk_id"]] = summary_row
            summary = lab.make_summary(lab_meeting, summary_records, topic_probs, act_probs)
            meeting["summary"] = summary
            summary_vec = normalize_l2(np.mean(np.stack([r["vector"] for r in msg_rows]), axis=0)[None, :])[0]
            conn.execute("INSERT INTO meetings VALUES(?,?,?,?,?,?,?,?)", (
                mid, meeting.get("project_id", ""), meeting.get("project", ""), meeting["title"], meeting["meeting_date"],
                meeting.get("style", "custom"), summary, json.dumps(summary_vec.tolist())))
            for r in msg_rows:
                raw = r["raw"]
                raw_id = r["message_id"]
                conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)", (
                    r["chunk_id"], mid, raw_id, r["timestamp"], r["username"], r["raw_json"], hashlib.sha256(r["raw_json"].encode()).hexdigest()))
                conn.execute("INSERT INTO message_embeddings VALUES(?,?,?,?,?)", (
                    r["chunk_id"], "custom-french-tiny", json.dumps(r["vector"].tolist()),
                    json.dumps(r["topic_probs"].tolist()), json.dumps(r["act_probs"].tolist())))
            for node in nodes:
                fid = f"{mid}:{node['feature']}"
                conn.execute("INSERT INTO feature_nodes VALUES(?,?,?,?,?,?,?,?)", (
                    fid, mid, node["feature"], node["kind"], node["count"], node["importance"],
                    json.dumps(node["chunk_ids"]), json.dumps(node["embedding"])))
            for edge in edges:
                conn.execute("INSERT INTO feature_edges VALUES(?,?,?,?)", (mid, edge["source"], edge["target"], edge["weight"]))
        conn.commit()
        conn.close()

    def _rank_for_benchmark(self, ds, query: str, method: str, meeting_id: str | None, qrow: dict,
                            query_vector=None, query_features=None):
        """Return a full candidate ordering; query encodings are batched in benchmark runs."""
        candidates = ds["by_meeting"].get(meeting_id, []) if meeting_id else list(ds["records"])
        rows = ds["records"]
        if method == "bm25":
            return [candidates[i] for i in np.argsort(-ds["bm25"].scores(query, candidates))]
        if method == "dense":
            if query_vector is None:
                query_vector = self._encode_custom([query])[0][0]
            scores = np.asarray([np.dot(query_vector, rows[c]["vector"]) for c in candidates])
            return [candidates[i] for i in np.argsort(-scores)]
        if method == "feature_aware":
            return [x["chunk_id"] for x in self._feature_aware(ds, query, candidates, meeting_id, query_features)]
        if method in MODEL_SPECS:
            pvec = self._pretrained_vectors_for(ds["slug"], method, rows)
            if query_vector is None:
                query_vector = self._query_pretrained(method, query)
            scores = np.asarray([np.dot(query_vector, pvec[c]) for c in candidates])
            return [candidates[i] for i in np.argsort(-scores)]
        if method in ("graphrag_hybrid", "graphrag_ft_router"):
            res = self.search(ds["slug"], query, method, top_k=min(10, len(candidates)),
                              meeting_id=meeting_id)
            return [x["chunk_id"] for x in res["results"]]
        if method in ("rexu_v3"):
            from pplx_rexu import get_rexu_v3
            res = get_rexu_v3(self, ds["slug"]).search(
                query, top_k=min(10, len(candidates)), meeting_id=meeting_id, method=method)
            return [x["chunk_id"] for x in res["results"]]
        if method in ("bge_only", "colbert", "colbert_ft", "colbert_ft_v2",
                      "rexu_v2_1", "rexu_v2_2", "graphrag",
                      "graphrag_bge", "graphrag_auto", "graphrag_tuned", "graphrag_best",
                      "colbert_v2", "graphrag_v2"):
            from hybrid_retrieval import get_retriever
            res = get_retriever(self, ds["slug"]).search(
                query, top_k=min(10, len(candidates)), meeting_id=meeting_id, method=method)
            return [x["chunk_id"] for x in res["results"]]
        raise ValueError(f"Unknown method: {method}")

    def benchmark(self, slug: str, split: str = "test", methods: list[str] | None = None, top_k: int = 10):
        ds = self._get_dataset(slug)
        queries = [q for q in ds["queries"] if q.get("split", "test") == split]
        if not queries:
            return {"status": "NO_LABELLED_QUERIES", "message": "Aucun couple query -> target_chunk_id pour ce split. L'import permet le search live, mais pas le calcul de Recall/MRR.", "query_count": 0, "metrics": {}}
        methods = methods or ["bm25", "bge_only", "colbert", "rexu_v2_1", "rexu_v2_2",
                              "graphrag_ft_router", "rexu_v3"]
        results = {}
        # Encode benchmark queries in batches for GPU throughput; avoid one tiny kernel launch per query.
        query_feature_cache = None
        custom_query_cache = None
        if any(m in methods for m in ("dense", "feature_aware")):
            qrows = [q for q in queries]
            qz, qtp, qap = self._encode_custom([q["query"] for q in qrows])
            custom_query_cache = {q["query_id"]: qz[i] for i, q in enumerate(qrows)}
            query_feature_cache = {q["query_id"]: (qz[i], qtp[i], qap[i]) for i, q in enumerate(qrows)}
        pretrained_query_cache = {}
        pretrained_init_errors = {}
        for m in methods:
            if m in MODEL_SPECS:
                try:
                    model = self._load_pretrained(m)
                    # Build passage vectors once and keep them cached per dataset/model.
                    self._pretrained_vectors_for(ds["slug"], m, ds["records"])
                    spec = self._active_pretrained_specs.get(m, MODEL_SPECS[m])
                    qtexts = [spec["query_prefix"] + q["query"] for q in queries]
                    qvectors = model.encode(qtexts, batch_size=min(64, self.batch_size) if self.device == "cuda" else 32,
                                            convert_to_numpy=True, normalize_embeddings=True,
                                            show_progress_bar=True, device=self.device)
                    pretrained_query_cache[m] = {q["query_id"]: np.asarray(qvectors[i], dtype=np.float32)
                                                  for i, q in enumerate(queries)}
                except Exception as exc:
                    pretrained_init_errors[m] = f"{type(exc).__name__}: {exc}"
        for method in methods:
            if method in pretrained_init_errors:
                results[method] = {"status": "UNAVAILABLE", "error": pretrained_init_errors[method],
                                   "evaluated_queries": 0, "skipped_queries": len(queries)}
                continue
            hits = {k: [] for k in (1, 3, 5, 10)}
            any_hits = {k: [] for k in (1, 5)}
            rr, ndcg = [], []
            evaluated, skipped, errors = 0, 0, []
            rankings = []
            for q in queries:
                mid = q.get("meeting_id")
                raw_targets = q.get("target_chunk_ids") or (
                    [q.get("target_chunk_id")] if q.get("target_chunk_id") else [])
                targets = []
                for target in raw_targets:
                    if not target:
                        continue
                    target = str(target)
                    if target not in ds["records"] and mid:
                        try:
                            message_id = int(target.rsplit(":", 1)[-1])
                            target = next((cid for cid in ds["by_meeting"].get(mid, [])
                                           if ds["records"][cid]["message_id"] == message_id), target)
                        except Exception:
                            pass
                    if target in ds["records"]:
                        targets.append(target)
                if not targets and q.get("target_json") is not None:
                    target_hash = canonical(q["target_json"])
                    target = next((cid for cid, r in ds["records"].items()
                                   if r["raw_json"] == target_hash and (not mid or r["meeting_id"] == mid)), None)
                    if target:
                        targets = [target]
                targets = list(dict.fromkeys(targets))
                if not targets:
                    skipped += 1
                    continue
                try:
                    qvec = (custom_query_cache or {}).get(q["query_id"]) if method == "dense" else pretrained_query_cache.get(method, {}).get(q["query_id"])
                    qfeatures = (query_feature_cache or {}).get(q["query_id"]) if method == "feature_aware" else None
                    ranked = self._rank_for_benchmark(ds, q["query"], method, mid, q, qvec, qfeatures)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                matched = [ranked.index(target) + 1 for target in targets if target in ranked]
                rank = min(matched) if matched else 9999
                for k in hits:
                    hits[k].append(len([r for r in matched if r <= k]) / len(targets))
                for k in any_hits:
                    any_hits[k].append(1.0 if any(r <= k for r in matched) else 0.0)
                rr.append(1.0 / rank if rank <= 10 else 0.0)
                ndcg.append(1.0 / math.log2(rank + 1) if rank <= 5 else 0.0)
                evaluated += 1
                if q.get("split", "test") == "test":
                    rankings.append({"query_id": q.get("query_id"), "query": q["query"], "meeting_id": mid,
                                     "target_chunk_id": targets[0], "target_chunk_ids": targets,
                                     "rank": rank, "matched_targets": matched, "top10": ranked[:10]})
            if evaluated:
                metric = {f"Recall@{k}": float(np.mean(v)) for k, v in hits.items()}
                metric.update({"Any Recall@1": float(np.mean(any_hits[1])),
                               "Any Recall@5": float(np.mean(any_hits[5])),
                               "MRR@10": float(np.mean(rr)), "nDCG@5": float(np.mean(ndcg)),
                               "evaluated_queries": evaluated, "skipped_queries": skipped})
                results[method] = {"metrics": metric, "rankings": rankings, "errors": errors[:3]}
            else:
                results[method] = {"status": "UNAVAILABLE_OR_NO_MATCHING_TARGETS", "evaluated_queries": 0,
                                   "skipped_queries": skipped, "errors": errors[:5]}
        return {"status": "OK", "dataset": slug, "split": split, "query_count": len(queries), "methods": results}

    def baseline_metrics(self):
        r = json.loads((ART / "french_results.json").read_text(encoding="utf-8"))
        return {"test": r.get("retrieval_metrics", {}).get("test", {}),
                "feature_aware": r.get("feature_aware_retrieval", {}).get("test", {}).get("overall", {}),
                "by_style": r.get("by_style", {}).get("test", {}),
                "pretrained": r.get("pretrained_comparison", {})}
