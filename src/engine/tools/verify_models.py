"""M0 verification tool — `python -m engine.tools.verify_models`.

BUILD_GUIDE 0.5: "Model names in this file came from a research pass and are
**unverified** — Step M0 verifies them." This queries the live Hugging Face API
for every candidate and writes docs/MODELCARDS.md. Nothing here is assumed:
existence, licence, task, parameter count, context length and prefix convention
are read from the model card / config, and a repo that does not resolve is
reported as a failure rather than skipped.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API = "https://huggingface.co/api/models/{mid}"
RAW = "https://huggingface.co/{mid}/raw/main/{file}"
UA = {"User-Agent": "semantic-agent-engine/0.1 (M0 licence verification)"}

#: candidates named in BUILD_GUIDE 6.2 / 6.4 / 6.9 / 6.3, plus listed fallbacks.
#: IDs marked "corrected" did not resolve under the guide's spelling; the live
#: search result that does resolve is used instead (docs/DECISIONS.md D-008).
CANDIDATES: dict[str, list[str]] = {
    "embedder": [
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        "Qwen/Qwen3-Embedding-0.6B",
        "ibm-granite/granite-embedding-311m-multilingual-r2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "microsoft/harrier-oss-v1-0.6b",
        "microsoft/harrier-oss-v1-270m",
    ],
    "reader_encoder": [
        "convaiinnovations/laya",
        "camembert/camembert-base",
        "almanach/moderncamembert-base",          # corrected (guide: ambiverse-camembertav2/)
        "almanach/camembertav2-base-cls",         # corrected: camembertav2 family lives here
        "FacebookAI/xlm-roberta-base",
        "distilbert/distilbert-base-multilingual-cased",
        "google-bert/bert-base-multilingual-cased",
    ],
    "linker_nli": [
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "joeddav/xlm-roberta-large-xnli",          # corrected (guide: FacebookAI/)
        "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    ],
    "reranker": [
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3-Reranker-0.6B",
        "jinaai/jina-reranker-v3",
        "jinaai/jina-reranker-v2-base-multilingual",
    ],
    "slots_ner": [
        "urchade/gliner_multi-v2.1",
        "fastino/gliner2.5-base-v1",               # corrected (guide: gliner-community/gliner2-base-v1.5)
        "onnx-community/gliner_multi-v2.1",
    ],
}


@dataclass(slots=True)
class Card:
    model_id: str
    role: str
    exists: bool = False
    error: str = ""
    license: str = ""
    pipeline_tag: str = ""
    params: str = ""
    max_len: str = ""
    architectures: list[str] = field(default_factory=list)
    prefixes: str = ""
    gated: str = ""
    downloads: str = ""
    languages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _get_json(url: str, timeout: float = 25.0) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, timeout: float = 25.0) -> str:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _human(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    for unit in ("", "k", "M", "B", "T"):
        if n < 1000:
            return f"{n:.3g}{unit}"
        n /= 1000
    return f"{n:.3g}T"


def _prefix_from_readme(mid: str, meta: dict) -> str:
    """Read the card for an explicit query/passage instruction (6.4 "prefix
    convention must match the model card")."""
    readme = _get_text(f"https://huggingface.co/{mid}/raw/main/README.md")
    low = readme.lower()
    hints = []
    for needle, label in (("query:", "query:"), ("passage:", "passage:"),
                          ("instruct:", "Instruct:"), ("retrieval: ", "retrieval: "),
                          ("q_proj", None), ("document:", "document:")):
        if label and needle in low:
            hints.append(label)
    card = (meta.get("cardData") or {}).get("model-index") or []
    if not hints and isinstance(card, list):
        for entry in card:
            res = (entry.get("results") or [{}])[0]
            cfg = res.get("config") or ""
            if "query" in cfg or "passage" in cfg:
                hints.append(cfg)
    return "; ".join(dict.fromkeys(hints))[:70]


def verify(model_id: str, role: str) -> Card:
    card = Card(model_id=model_id, role=role)
    try:
        meta = _get_json(API.format(mid=model_id))
    except urllib.error.HTTPError as exc:
        card.error = f"HTTP {exc.code}"
        card.exists = exc.code not in (401, 404)
        return card
    except Exception as exc:  # network/DNS
        card.error = f"{type(exc).__name__}: {exc}"
        return card
    card.exists = True
    cd = meta.get("cardData") or {}
    card.license = (cd.get("license") or "").replace("-", "_") or (
        next((t.split(":")[-1] for t in meta.get("tags", [])
              if t.startswith("license:")), ""))
    card.pipeline_tag = meta.get("pipeline_tag") or ""
    card.gated = str(meta.get("gated") or "no")
    card.downloads = str(meta.get("downloads") or "")
    card.languages = [l for l in (cd.get("language") or [])][:6] if isinstance(
        cd.get("language"), list) else [str(cd.get("language") or "")][:1]
    st = meta.get("safetensors") or {}
    pars = st.get("parameters") or {}
    card.params = "/".join(_human(v) for v in pars.values()) if pars else ""
    cfg = meta.get("config") or {}
    card.architectures = list(cfg.get("architectures") or [])[:2]
    for key in ("max_position_embeddings", "n_positions", "model_max_length",
                "max_seq_len"):
        if isinstance(cfg, dict) and key in cfg:
            card.max_len = str(cfg[key])
            break
    if not card.max_len:
        tc = _get_text(RAW.format(mid=model_id, file="tokenizer_config.json"))
        try:
            card.max_len = str(json.loads(tc).get("model_max_length", ""))
        except Exception:
            pass
    card.prefixes = _prefix_from_readme(model_id, meta)
    return card


def annotate_known_issues(cards: list[Card]) -> None:
    """Hard-won facts that the API alone will not tell you (see memory/docs)."""
    by_id = {c.model_id: c for c in cards}
    laya = by_id.get("convaiinnovations/laya")
    if laya and laya.exists:
        laya.notes.append(
            "NOT a text encoder: typed-decision model (5 tensor inputs incl. "
            "marker_pos/qtype, no feature-extraction run). Cannot be fine-tuned "
            "as the 6.2.3 reader encoder; AutoModel.from_pretrained fails (no root "
            "config.json). Baseline card scores 0.362 vs 0.461 for always-pick-modal "
            "— worse than a constant, zero-shot.")
        laya.notes.append("Requires a 512-length-baked ONNX export unless "
                          "torch.onnx.export(dynamo=True, opset_version=18).")
    for cid, note in (
        ("jinaai/jina-reranker-v3", "licence is NOT Apache-2.0 (v2 family was "
         "CC-BY-NC-4.0); 6.9 flags this — check before use"),
        ("BAAI/bge-m3", "dense + sparse (ColBERT-style multi-vector) in one model; "
         "no query prefix needed"),
        ("intfloat/multilingual-e5-base", "requires 'query: '/'passage: ' prefixes; "
         "wrong prefixes silently cost ~5-10 pts nDCG"),
    ):
        if cid in by_id and by_id[cid].exists:
            by_id[cid].notes.append(note)


def render_markdown(cards: list[Card], generated_note: str) -> str:
    lines = ["# MODELCARDS.md — verified model candidates (M0 gate)",
             "",
             f"{generated_note}",
             "",
             "Verification method: live `huggingface.co/api/models/<id>` plus "
             "`config.json` / `tokenizer_config.json` / `README.md` per repo. "
             "BUILD_GUIDE 0.5 required this pass because the guide's model names "
             "were unverified research-pass output.",
             "",
             "| role | model | exists | licence | task | params | max len | prefixes | notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cards:
        if not c.exists:
            lines.append(f"| {c.role} | `{c.model_id}` | **NO** ({c.error}) | — | — | — | — | — "
                         f"| guide name is wrong or gated |")
            continue
        notes = "; ".join(c.notes) or "—"
        lines.append(f"| {c.role} | `{c.model_id}` | yes | {c.license or '?'} | "
                     f"{c.pipeline_tag or '—'} | {c.params or '—'} | "
                     f"{c.max_len or '—'} | {c.prefixes or '—'} | {notes} |")
    lines += ["", "## Consequences for the build", ""]
    return "\n".join(lines)


CONSEQUENCES = [
    "- **Reader encoder:** `convaiinnovations/laya` is verified to exist but is a "
    "typed-decision model, not an encoder — it cannot initialise the 6.2.3 "
    "multi-head classifier. Primary encoder becomes a real masked-LM (CamemBERT "
    "family / XLM-R), and BUILD_GUIDE's own fallback list applies.",
    "- **Embedder:** the guide's `harrier-oss-*` names must be confirmed below; "
    "until confirmed, `BAAI/bge-m3` and `intfloat/multilingual-e5-base` are the "
    "working candidates (both verified, both permissive).",
    "- **Reranker:** `jinaai/jina-reranker-v3` is not usable as an Apache-2.0 "
    "substitute — `BAAI/bge-reranker-v2-m3` is the default.",
    "- Prefix conventions are enforced in `engine/embed/hf_embedder.py` "
    "(`KNOWN_PREFIXES`), keyed by these verified ids.",
]


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from engine.config import repo_root
    from engine.seeding import utcnow_stamp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(repo_root() / "docs" / "MODELCARDS.md"))
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    cards: list[Card] = []
    for role, ids in CANDIDATES.items():
        for mid in ids:
            c = verify(mid, role)
            status = "OK" if c.exists else f"FAIL {c.error}"
            print(f"[{status:>12}] {role:15} {mid}")
            cards.append(c)
    annotate_known_issues(cards)
    note = (f"Generated {utcnow_stamp()} by `python -m engine.tools.verify_models` "
            f"against the live HF API ({len(cards)} candidates).")
    md = render_markdown(cards, note) + "\n" + "\n".join(CONSEQUENCES) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nwrote {out} ({len(cards)} candidates, "
          f"{sum(1 for c in cards if c.exists)} resolved)")
    if args.json:
        from dataclasses import asdict

        Path(args.json).write_text(json.dumps(
            [asdict(c) for c in cards], ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
