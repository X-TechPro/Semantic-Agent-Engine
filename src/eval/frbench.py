"""Real-French evaluation (7.2-7.4): the current architecture against external gold.

Every headline number in this repo is internal validity — the gold labels and the
sentences both come from `gen/`. This module is the complement the guide demands:
three components scored against corpora whose annotations I did not author.

* `xnli`     — the linker's relation discrimination on real French sentences.
  XNLI's own README pins 0=entailment, 1=neutral, 2=contradiction; mapped to
  supports/unrelated/contradicts. The trained `PairClassifier` and the rules linker
  see pairs whose **metadata is held constant** (same project, dates one day apart,
  type/status identical), so every feature that varies is text-derived. A chance
  AUC says the shipped linker never learned text-only contradiction detection —
  which is what 7.2 predicted ("a vector proximity is not logical compatibility").
* `massive`  — the reader's intent head on 2,974 real French requests. Mapping is a
  rule over MASSIVE's own label names, listed in `MAP_RULES`; everything that does
  not map is an out-of-scope rejection test (expected: `none` or `clarify`), not a
  wrong answer. Val exists on disk for tuning; this scores **test only** (9.5).
* `miracl`   — the retrieval stack (BM25 / e5 dense / RRF hybrid / +cross-encoder)
  on French dev queries with human qrels, over the first 5,000 passages of corpus
  shard 0. Queries whose gold is outside the shard are excluded and counted —
  reporting their recall would demand the system find an unpresent document.

* `summre`   — the temporal layer itself (B-13): the 28 hand-labelled `supersedes`
  pairs in `data/gold/summre_temporal_test.json`, scored baseline-A vs System-B.
  Labels were frozen before any score existed and the split is TEST-ONLY (§9.5),
  so this function refuses to run twice against the same artefact directory.

Heavy imports (torch, store, pipeline) stay inside the functions so the pure
aggregation helpers below are unit-testable offline. Artefact directories are named
`<component>_<embedtag>_<limit>` so two arms can never overwrite each other (the
tripwire/split-manifest identity rule).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from engine.read.massive import (EXCLUDED_FROM_ANSWER, MAP_RULES, load_massive,
                                 map_massive_intent)

# --------------------------------------------------------------- XNLI (linker)

#: verified against data/external/facebook__xnli/README.md, not recalled
LABEL2REL = {0: "supports", 1: "unrelated", 2: "contradicts"}

PAIR_DATES = (date(2026, 1, 1), date(2026, 1, 2))  # premise older, hypothesis newer


def build_pair_units(premise: str, hypothesis: str) -> tuple[Any, Any]:
    """Units with constant metadata; only text differs (measured by the AUC)."""
    from engine.schemas import Slots, Span, Unit, make_unit_id, utcnow

    def u(text: str, d: date, seq: int) -> Unit:
        return Unit(
            unit_id=make_unit_id("xnli-fr", seq, 0, len(text), None),
            source_id="xnli-fr", source_type="note", source_date=d,
            ingested_at=utcnow(), text=text,
            sentences=[Span.whole(text)],
            project="xnli-fr", kind="info", type="fact", status="confirmed",
            slots=Slots())
    return u(premise, PAIR_DATES[0], 0), u(hypothesis, PAIR_DATES[1], 1)


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with 0.5 credit for ties. P(positive ranks above negative)."""
    if not pos or not neg:
        raise ValueError("auc needs both classes non-empty")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def xnli_report(scored: list[tuple[float, int]]) -> dict[str, Any]:
    """scored = [(classifier score, raw XNLI label)]. AUCs are the verdict."""
    by = defaultdict(list)
    for s, lab in scored:
        by[lab].append(s)
    if len(by) != 3:
        raise ValueError(f"expected 3 label classes, got {sorted(by)}")
    contra, neutral, entail = by[2], by[1], by[0]
    pooled_other = neutral + entail
    return {
        "n_pairs": len(scored), "class_counts": {LABEL2REL[k]: len(v) for k, v in sorted(by.items())},
        "mean_score_by_gold": {LABEL2REL[k]: round(statistics.mean(v), 4) for k, v in sorted(by.items())},
        "auc_contradiction_vs_neutral": auc(contra, neutral),
        "auc_contradiction_vs_entailment": auc(contra, entail),
        "auc_contradiction_vs_all_others": auc(contra, pooled_other),
        "auc_entailment_vs_neutral": auc(entail, neutral),
        "chance": 0.5,
        "reading": ("the classifier answers 'does this supersede'; the thesis claim under test is "
                    "whether it can separate a contradiction from a restatement and an unrelated "
                    "pair on real French, using text alone (metadata is constant by construction)"),
    }


# ------------------------------------------------------------- MASSIVE (reader)

# The mapping table and loader now live in `engine.read.massive` (imported at
# the top of this file). The reader warm-up trains on this vocabulary, and an
# eval module owning labels the engine consumes is the import direction
# reversed -- copying the table instead of importing it would be the
# reader-gold duplication failure in miniature.


def massive_report(rows: list[tuple[str | None, str, bool]]) -> dict[str, Any]:
    """rows = [(our_gold_or_None, predicted_intent, abstained)].

    Every rate keeps a measured 0.0: all comparisons are `is not None` / explicit
    fractions, never truthiness -- the trap that turned real zeros into "absent"
    three times in this project.
    """
    pred_counts = Counter(p for _, p, _ in rows)
    in_rows = [(g, p) for g, p, _ in rows if g is not None]
    oov = [(g, p) for g, p, _ in rows if g is None]
    correct = sum(1 for g, p in in_rows if g == p)
    # macro F1 over the gold classes that appear
    classes = sorted({g for g, _ in in_rows})
    f1: dict[str, float] = {}
    for c in classes:
        tp = sum(1 for g, p in in_rows if g == c and p == c)
        fp = sum(1 for g, p in in_rows if g != c and p == c)
        fn = sum(1 for g, p in in_rows if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1[c] = (round(2 * prec * rec / (prec + rec), 4)
                 if prec is not None and rec is not None and (prec + rec) else 0.0)
    oov_reject = sum(1 for _, p in oov if p in ("none", "clarify"))
    return {
        "n_rows": len(rows), "n_in_scope": len(in_rows), "n_out_of_scope": len(oov),
        "intent_accuracy_in_scope": (round(correct / len(in_rows), 4)
                                     if in_rows else None),
        "per_class_f1": f1,
        "macro_f1_in_scope": (round(sum(f1.values()) / len(f1), 4) if f1 else None),
        "oov_rejection_rate": (round(oov_reject / len(oov), 4) if oov else None),
        "oov_leakage": Counter(p for _, p in oov if p not in ("none", "clarify")).most_common(8),
        "pred_distribution": pred_counts.most_common(),
        "abstained_rate": round(sum(1 for *_, a in rows if a) / len(rows), 4) if rows else None,
        "mapping_rule": "label contains qa_/_query/transport_traffic -> answer; "
                        "calendar_set|alarm_set -> schedule_event; lists_createoradd|"
                        "takeaway_order -> create_task; email_sendemail -> draft_email; "
                        "else out-of-scope (must abstain to none/clarify)",
    }


# ------------------------------------------------------------- MIRACL (retrieval)

def parse_topics(lines: list[str]) -> dict[str, str]:
    out = {}
    for l in lines:
        if not l.strip():
            continue
        qid, text = l.rstrip("\n").split("\t", 1)
        out[qid] = text
    return out


def parse_qrels(lines: list[str]) -> dict[str, set[str]]:
    """Gold = relevance >= 1. The file also ships labelled negatives (rel 0), which
    are distractors for ranking, NOT gold -- counting them would fake a huge pool
    coverage (measured: 3429 rows contain many zeros)."""
    gold: dict[str, set[str]] = defaultdict(set)
    for l in lines:
        if not l.strip():
            continue
        qid, _q0, docid, rel = l.split("\t")
        if int(rel) >= 1:
            gold[qid].add(docid)
    return dict(gold)


def read_passages(path: Path, limit: int) -> list[dict[str, str]]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(out) >= limit:
                break
            r = json.loads(line)
            out.append({"docid": str(r["docid"]), "text": r["text"]})
    return out


def read_passages_pool(corpus_dir: Path, keep: set[str]) -> list[dict[str, str]]:
    """Streaming fetch of every passage whose docid is in `keep`, across all shards.

    Why a pool and not a head slice: indexing docs 0..4999 of shard 0 left 340 of
    343 dev queries with no gold document on disk at all (`n_scoreable: 3` measured)
    -- a retrieval verdict on 3 queries is not a verdict. The qrels file names every
    candidate the annotators actually judged (gold AND the official BM25 hard
    negatives), so the pool is those docs: full-corpus recall@10 numbers this is
    NOT, but it is the standard pooled-qrels protocol, the distractors are hard by
    construction, and every arm sees the same pool.
    """
    out, found = [], set()
    key = '{"docid": "'
    n = len(key)
    for shard in sorted(corpus_dir.glob("docs-*.jsonl.gz")):
        if not shard.name.endswith(".jsonl.gz"):
            continue
        with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith(key):
                    continue
                # slice the docid out before parsing: 28 shards x ~120k lines of
                # json.loads would cost ~15 min to keep ~3,300 documents
                try:
                    d = line[n:line.index('"', n)]
                except ValueError:
                    continue
                if d in keep and d not in found:
                    r = json.loads(line)
                    out.append({"docid": d, "text": r["text"]})
                    found.add(d)
        if found == keep:
            break
    missing = keep - found
    if missing:
        raise RuntimeError(f"{len(missing)} qrels docids absent from corpus shards "
                           f"(e.g. {sorted(missing)[:3]}) -- shards incomplete?")
    return out


def qrels_candidate_ids(qrels_file: Path) -> set[str]:
    """Every docid named by the qrels file, gold and labelled negatives alike."""
    ids: set[str] = set()
    for l in qrels_file.read_text().splitlines():
        if l.strip():
            ids.add(l.split("\t")[2])
    return ids


def miracl_metrics(gold: dict[str, set[str]], ranked_by_q: dict[str, list[str]],
                   ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    scored = [q for q in gold if gold[q]]
    acc: dict[str, list[float]] = defaultdict(list)
    for q in scored:
        g = gold[q]
        top = ranked_by_q.get(q, [])
        for k in ks:
            hit = len(g & set(top[:k])) / len(g)
            acc[f"recall@{k}"].append(hit)
        rr = 0.0
        for i, d in enumerate(top[:10], start=1):
            if d in g:
                rr = 1.0 / i
                break
        acc["mrr@10"].append(rr)
        # binary relevance: gain 1 for each gold doc; discounts 1/log2(i+1)
        from math import log2
        dcg = sum(1.0 / log2(i + 1) for i, d in enumerate(top[:10], start=1) if d in g)
        idcg = sum(1.0 / log2(i + 1) for i in range(1, min(len(g), 10) + 1))
        acc["ndcg@10"].append(dcg / idcg if idcg else 0.0)
    return {k: round(statistics.mean(v), 4) for k, v in sorted(acc.items())}


# ------------------------------------------------------------------ NLI head

def eval_xnli_with_head(split: str = "test") -> dict[str, Any]:
    """Score the trained NLI head on a split; `test` may be read exactly once.

    Selection happened on validation (9.5), so a test number is a one-event:
    refusing a second read is what keeps "the one honest test number" from
    turning into a tunable one.
    """
    import torch

    from engine.link import nli_finetune as N

    head = N.load_head()
    if head is None:
        raise RuntimeError("models/linker_nli missing: run `make nli`")
    d = run_dir_for("xnli-nli", "camembert-head", None, split)
    if split == "test" and (d / "result.json").exists():
        raise RuntimeError(
            f"test split already read once ({d}); refusing to re-read it")
    model, tok, meta = head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = N.evaluate(model, tok, N.load_fr(split), device,
                    maxlen=int(meta["hparams"]["maxlen"]))
    return {"corpus": f"facebook/xnli fr {split} (pinned revision)",
            "head": {"base": meta["base_model"], "revision": meta["data"]["revision"],
                     "n_train_rows": meta["data"]["n_train_rows"],
                     "val_macro_f1_at_selection": meta["best_val"]},
            "chance_accuracy": 0.3333,
            "test": ev}


# ------------------------------------------------------------------ rerank probe

def eval_xnli_with_reranker(split: str = "validation", limit: int | None = 2490
                            ) -> dict[str, Any]:
    """Can the cross-encoder the repo ALREADY ships separate the three XNLI classes?

    This is the instrument test that separates two very different conclusions from
    the pair-classifier null. bge-reranker-v2-m3 scores query-passage RELEVANCE, not
    NLI, so a high AUC here means the text signal is reachable by a component already
    installed and the choke is WIRING (the linker never consults a text scorer when
    slots are absent); a chance AUC means the signal is not in the shipped models at
    all and the guide's "XNLI-fr as starting point" fine-tune step is required.

    Tuning runs on VALIDATION, never test (9.5).
    """
    from engine.config import load_config
    from engine.retrieve.rerank import build_reranker

    cfg = load_config(overrides={"retrieve": {"rerank": {"backend": "cross_encoder"}}})
    reranker = build_reranker(cfg)
    if reranker is None:
        raise RuntimeError("cross_encoder reranker unavailable (model not cached?)")
    rows = load_xnli(split)[:limit]
    scored = []
    for premise, hypothesis, lab in rows:
        _old, new = build_pair_units(premise, hypothesis)
        ranked = reranker.rerank(premise, [new])
        scored.append((float(ranked[0][1]), lab))
    out = {"corpus": f"facebook/xnli fr {split} (pinned revision)",
           "reranker": reranker.name, "n_scored": len(scored),
           "aucs": xnli_report(scored)}
    return out


# ------------------------------------------------------------------ aggregation

def embed_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def run_dir_for(component: str, tag: str, limit: int | None, split: str) -> Path:
    """Identity in the name: component, embedder/reader tag, split, slice size.
    Two arms must never share a directory (the tripwire-overwrote-the-headline rule)."""
    lim = f"n{limit}" if limit else "full"
    return repo_root() / "runs" / "frbench" / f"{component}_{embed_tag(tag)}_{split}_{lim}"


def eval_linker_xnli(split: str = "test") -> dict[str, Any]:
    from engine.link.model import PairClassifier, build_features
    from engine.link.rules import propose as rules_propose
    from engine.config import MODELS_DIR

    clf = PairClassifier.load(Path(MODELS_DIR) / "linker_lr.json")
    if clf is None:
        raise RuntimeError("models/linker_lr.json missing: run `python -m engine.link.train`")
    rows = load_xnli(split)
    scored, jacc, rules_dist = [], [], Counter()
    for premise, hypothesis, lab in rows:
        old, new = build_pair_units(premise, hypothesis)
        scored.append((clf.score(new, old), lab))
        w1, w2 = set(_words(premise)), set(_words(hypothesis))
        jacc.append((len(w1 & w2) / max(1, len(w1 | w2)), lab))
        rules_dist[LABEL2REL[lab] + "<-" + str(rules_propose(new, old))] += 1
    out = {"corpus": f"facebook/xnli fr {split} (pinned revision, see PROVENANCE.json)",
           "classifier": {"threshold": clf.threshold, "trained_on": clf.meta.get("trained_on")},
           "auc_baselines": {"chance": 0.5,
                             "lexical_jaccard": xnli_report(jacc)},
           "pairclassifier": xnli_report(scored),
           "rules_linker_propose_distribution": dict(rules_dist)}
    return out


def _words(text: str) -> set[str]:
    import unicodedata
    out = set()
    for w in re.findall(r"[^\W\d_]+", (text or "").lower()):
        w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
        if len(w) > 3:
            out.add(w)
    return out


def load_xnli(split: str = "test") -> list[tuple[str, str, int]]:
    import pyarrow.parquet as pq
    root = repo_root() / "data/external/facebook__xnli"
    prov = json.loads((root / "PROVENANCE.json").read_text())
    f = next(x for x in prov["files"] if x.startswith(f"fr/{split}-"))
    t = pq.read_table(root / f)
    cols = t.column_names
    assert {"premise", "hypothesis", "label"} <= set(cols), cols
    return [(t.column("premise")[i].as_py(), t.column("hypothesis")[i].as_py(),
             int(t.column("label")[i].as_py())) for i in range(t.num_rows)]


def eval_reader_massive(split: str = "test", device: str = "cuda",
                        model_dir: str | None = None) -> dict[str, Any]:
    from engine.read.infer import build_reader
    from engine.schemas import Unit, Slots, Span, make_unit_id, utcnow

    override = {"backend": "model"}
    if model_dir:
        override["model_dir"] = model_dir
        # 9.5 one-read discipline on the warm checkpoint's test number: the
        # artefact directory is predictable from the read-back a loadable dir
        # WILL produce, so a second read is refused -- the guard
        # `eval_xnli_with_head` already applies to the NLI head.
        d = run_dir_for("massive", embed_tag(f"model({Path(model_dir).name})")[:32],
                        None, split)
        if split == "test" and (d / "result.json").exists():
            raise RuntimeError(f"warm-reader test already read once ({d}); "
                               "refusing to re-read it")
    cfg = load_config(overrides={"reader": override})
    # build_reader returns (reader, backend_actually_in_use) -- the label is the
    # builder's own fallback report, so a missing checkpoint can never be
    # published as a "model" run (D-011).
    reader, ran_label = build_reader(cfg, device=device)
    rows = load_massive(split)
    scored: list[tuple[str | None, str, bool]] = []
    kind_query = 0
    metas = []
    for r in rows:
        text = r["text"]
        u = Unit(unit_id=make_unit_id("massive-fr", int(r["id"]), 0, len(text), None),
                 source_id="massive-fr", source_type="note",
                 source_date=date(2026, 1, 1), ingested_at=utcnow(), text=text,
                 sentences=[Span.whole(text)], project="massive-fr", slots=Slots())
        rd = reader.read(u)
        scored.append((map_massive_intent(r["label"]), rd.intent, bool(rd.abstained)))
        if rd.kind == "query":
            kind_query += 1
        metas.append(reader.last_confidence.get("intent") if hasattr(reader, "last_confidence") else None)
    confs = [c for c in metas if c is not None]
    out = {"corpus": "mteb/amazon_massive_intent fr (pinned revision)",
           # what the builder actually produced, fallback included -- not what the
           # YAML asked for (D-011)
           "reader_backend": ran_label,
           "reader_class": type(reader).__name__,
           "requested_model_dir": model_dir,
           "kind_query_rate": round(kind_query / len(rows), 4),
           "mean_intent_confidence": round(statistics.mean(confs), 4) if confs else None,
           **massive_report(scored)}
    return out


def eval_retrieval_miracl(limit: int = 5000, device: str = "cuda",
                         with_rerank: bool = True, pool: str = "qrels",
                         concept_ckpt: str | None = None) -> dict[str, Any]:
    from engine.embed.base import build_embedder
    from engine.retrieve.bm25 import build_lexical
    from engine.retrieve.dense import DenseRetriever
    from engine.retrieve.fusion import rrf
    from engine.retrieve.rerank import build_reranker
    from engine.schemas import Slots, Span, Unit, make_unit_id, utcnow
    from engine.store.sqlite_store import SQLiteStore

    root = repo_root()
    cfg = load_config(overrides={"embed": {"backend": "st",
                                           "model_id": "intfloat/multilingual-e5-base"},
                                 "device": device,
                                 "retrieve": {"rerank": {"backend": "cross_encoder"}}})
    embedder = build_embedder(cfg)
    mroot = root / "data/external/miracl__miracl"
    croot = root / "data/external/miracl__miracl-corpus"
    topics = parse_topics((mroot / "miracl-v1.0-fr/topics/topics.miracl-v1.0-fr-dev.tsv")
                          .read_text().splitlines())
    qrels_file = mroot / "miracl-v1.0-fr/qrels/qrels.miracl-v1.0-fr-dev.tsv"
    qrels = parse_qrels(qrels_file.read_text().splitlines())
    corpus_dir = croot / "miracl-corpus-v1.0-fr"
    if pool == "qrels":
        # see read_passages_pool: the slice mode left 340/343 queries unjudgeable
        passages = read_passages_pool(corpus_dir, qrels_candidate_ids(qrels_file))
        slice_desc = f"qrels-pooled (all {len(passages)} judged candidates, 28 shards)"
        tag = embed_tag(embedder.version) + "-pool"
    else:
        passages = read_passages(corpus_dir / "docs-0.jsonl.gz", limit)
        slice_desc = f"shard-0 head slice (limit {limit})"
        tag = embed_tag(embedder.version)

    out_dir = run_dir_for("miracl", tag, None if pool == "qrels" else limit, "dev")
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "store.db"
    for stale in out_dir.glob("store.db*"):
        stale.unlink()
    store = SQLiteStore(db, embed_version=embedder.version)
    units = []
    for i, p in enumerate(passages):
        u = Unit(unit_id=f"m-{p['docid']}", source_id="miracl-fr-shard0",
                 source_type="note", source_date=date(2026, 1, 1),
                 ingested_at=utcnow(), text=p["text"], sentences=[Span.whole(p["text"])],
                 project="miracl-fr", slots=Slots())
        units.append(u)
    store.add_units(units)
    vecs = embedder.embed([u.text for u in units], role="doc")
    store.add_vectors([u.unit_id for u in units], vecs, kind="base")
    store.rebuild_index()
    lexical = build_lexical(store, cfg)
    dense = DenseRetriever(store, embedder, top_k=100)

    pool_ids = {u.unit_id.removeprefix("m-") for u in units}
    gold_in_pool = {q: {d for d in ds if d in pool_ids} for q, ds in qrels.items()}
    gold = {q: (gold_in_pool[q] if q in topics else set()) for q in topics}
    scoreable = {q: g for q, g in gold.items() if g}
    skipped = len(topics) - len(scoreable)

    K = cfg["retrieve"]
    arms: dict[str, dict[str, list[str]]] = {"dense": {}, "bm25": {}, "hybrid": {}}
    reranker = build_reranker(cfg) if with_rerank else None
    if reranker is not None:
        arms["hybrid+rerank"] = {}
    # 6.13's comparison on external data: refine THIS store's vectors with a
    # concept checkpoint trained in the same embed space (`load`'s version guard
    # refuses otherwise), then score the same queries over `vectors.refined`.
    # The query side stays frozen-base by design -- that is the anchoring
    # contract's whole point -- so a refined arm that wins did it by moving the
    # stored geometry a bounded amount, not by becoming a different space.
    concept_info: dict[str, Any] | None = None
    dense_refined: DenseRetriever | None = None
    if concept_ckpt:
        from engine.concept.refine_job import refine
        from engine.concept.train import load as load_concept
        cmodel, _cmeta = load_concept(Path(concept_ckpt),
                                      expect_embed_version=store.embed_version)
        rpt = refine(store, cmodel, k=int(cfg.get_path("concept.k_neighbors", 16)),
                     batch_size=32, device=device, writes=True,
                     max_drift=float(cfg.get_path("concept.max_drift", 0.25)))
        concept_info = {kk: rpt.get(kk) for kk in
                        ("anchored", "cosine_to_base_mean", "drift_p95",
                         "max_drift_allowed", "n_refined", "status")}
        if rpt.get("rebuilt_index"):
            dense_refined = DenseRetriever(store, embedder, top_k=100, use="refined")
            arms["dense_refined"] = {}
            arms["hybrid_refined"] = {}
    by_id = {u.unit_id: u for u in units}
    for q, text in topics.items():
        d_ids = [d.unit_id for d in dense.search(text, k=K["top_k_dense"])]
        l_ids = [r.unit_id for r in lexical.search(text, k=K["top_k_lexical"])]
        arms["dense"][q] = d_ids
        arms["bm25"][q] = l_ids
        fused = rrf({"dense": d_ids, "bm25": l_ids}, k=K["rrf_k"],
                    weights=dict(K["weights"]))
        fused_ids = [f.unit_id for f in fused][:K["final_k"]]
        arms["hybrid"][q] = fused_ids
        if reranker is not None:
            cand = [by_id[i] for i in fused_ids if i in by_id]
            arms["hybrid+rerank"][q] = [uid for uid, _ in reranker.rerank(text, cand)]
        if dense_refined is not None:
            rd_ids = [d.unit_id for d in dense_refined.search(text,
                                                              k=K["top_k_dense"])]
            arms["dense_refined"][q] = rd_ids
            fr = rrf({"dense": rd_ids, "bm25": l_ids}, k=K["rrf_k"],
                     weights=dict(K["weights"]))
            arms["hybrid_refined"][q] = [f.unit_id for f in fr][:K["final_k"]]

    gold_map = {q: {"m-" + d for d in g} for q, g in scoreable.items()}
    result = {"corpus": f"miracl/miracl dev fr + {slice_desc}",
              "embed_version": embedder.version, "n_passages": len(units),
              "n_queries": len(topics), "n_scoreable": len(scoreable),
              "n_skipped_gold_outside_shard": skipped,
              "pool_mode": pool, "concept_refine": concept_info,
              "arms": {arm: miracl_metrics(gold_map, ranked)
                       for arm, ranked in arms.items()}}
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    store.close()
    return result


# ------------------------------------------------------- SUMM-RE temporal (B-13)

def summre_pair_scores(rows: list[dict]) -> dict[str, Any]:
    """rows: [{rank_old, rank_new}] — first position in top-k, None = absent.

    A pure function so the win semantics are unit-testable offline: the newer
    claim "wins" when it appears above the superseded one, and a superseded
    unit pushed OUT of the pool counts as a win (that is exactly what validity
    demotion is for, and temporal@1's established reading). Rates are rates —
    0.0 over n>0 is a measurement and stays 0.0 (the falsy-zero trap this
    project has paid for three times); None only when there is no denominator.
    """
    n = len(rows)
    wins = stale = both = 0
    rrs = []
    for r in rows:
        ro, rn = r.get("rank_old"), r.get("rank_new")
        if rn is not None:
            wins += int(ro is None or rn < ro)
            rrs.append(1.0 / (rn + 1))
        if ro is not None and (rn is None or ro < rn):
            stale += 1
        if ro is not None and rn is not None:
            both += 1
    return {"n_pairs": n,
            "new_above_old": round(wins / n, 4) if n else None,
            "stale_above_new": round(stale / n, 4) if n else None,
            "mrr_new_at_k": round(sum(rrs) / n, 4) if n else None,
            "both_in_pool": both}


def build_meeting_source(mid: str, segments: list[dict],
                        on_date: date = date(2021, 10, 1)):
    """One Source, one Turn per filtered segment — `turn_index` == the gold idx.

    The alignment the whole arm rests on: gold labels point at positions in
    `meetings.json`'s filtered list and units carry `turn_index` back, so a
    pair is scoreable only if both sides produced units — measured, counted,
    never assumed. Timestamps are meeting-relative so ingest chronology
    matches the gold's old/new direction.
    """
    from engine.schemas import Source, Turn

    base = datetime(2021, 10, 5, 9, 0, tzinfo=timezone.utc)
    turns = [Turn(speaker=f"s{s['speaker']}", text=s["text"],
                  timestamp=base + timedelta(minutes=float(s["start_min"])),
                  turn_index=i)
             for i, s in enumerate(segments)]
    return Source(source_id=f"summre:{mid}", source_type="meeting",
                  source_date=on_date, title=f"SUMM-RE {mid} (test split)",
                  project=mid, language="fr", turns=turns)


def _rank_in(hits: list[str], uid: str | None) -> int | None:
    if uid is None:
        return None
    return hits.index(uid) if uid in hits else None


def eval_summre_temporal(device: str = "cuda", top_k: int = 30) -> dict[str, Any]:
    """The headline layer, externally: hand-labelled supersedes pairs, A vs B.

    Query = the SUPERSEDED claim's own text, mode=current. Baseline A (no
    linker, no scoping — row 4) has no reason to prefer the newer unit; System
    B earns the relation share of the headline exactly when demotion plus
    chain-following surfaces the newer claim first. Same corpus, same e5,
    same budgets: ONE embedder instance injected into both arms (the 8 GB
    fairness rule), the only difference being `linker_enabled`/`scoping`.

    Labels were frozen before any score existed, the set is TEST-ONLY (§9.5),
    and this function refuses to run a second time against its own artefact —
    one read is a confirmation, a second is a tuning fork.
    """
    from collections import Counter

    from engine.embed.base import build_embedder
    from engine.engine import Engine
    from engine.schemas import Query

    root = repo_root()
    gold = json.loads((root / "data" / "gold" / "summre_temporal_test.json")
                      .read_text(encoding="utf-8"))
    meetings = json.loads(
        (root / "data" / "external" / "summre_test" / "meetings.json")
        .read_text(encoding="utf-8"))

    cfg = load_config(overrides={"embed": {"backend": "sentence_transformers",
                                           "model_id": "intfloat/multilingual-e5-base",
                                           "device": device}})
    embedder = build_embedder(cfg)          # shared by both arms — one instance
    out_dir = run_dir_for("summre", embed_tag(embedder.version), None, "test")
    if (out_dir / "result.json").exists():
        raise RuntimeError(
            f"the SUMM-RE test read is already spent ({out_dir}); this set is "
            "test-only and a second read would turn one confirmation into a "
            "tunable two (9.5)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("store_*.db*"):
        stale.unlink()

    def pairs_of(label: str):
        return [(mid, th["topic"], r)
                for mid, mm in gold["meetings"].items()
                for th in mm["threads"] for r in th["relations"]
                if r["relation"] == label]

    sup = pairs_of("supersedes")
    con = pairs_of("contradicts")
    supo = pairs_of("supports")
    mid_in = sorted({m for m, _t, _r in sup})

    arms = {"baseline_A": (False, False), "system_B": (True, True)}
    engines = {name: Engine(cfg, path=out_dir / f"store_{name}.db",
                            linker_enabled=link_on, embedder=embedder)
               for name, (link_on, _s) in arms.items()}
    srcs = {mid: build_meeting_source(mid, meetings[mid]) for mid in mid_in}
    for eng in engines.values():
        for src in srcs.values():
            eng.ingest(src)

    unit_of: dict[str, dict[int, str]] = {}
    units_by_meeting: dict[str, list] = {}
    for mid in mid_in:
        us = engines["system_B"].store.units_of_source(f"summre:{mid}")
        units_by_meeting[mid] = us
        per: dict[int, str] = {}
        for u in us:
            if u.turn_index is not None:
                per.setdefault(int(u.turn_index), u.unit_id)
        unit_of[mid] = per

    rows: dict[str, list[dict]] = {name: [] for name in arms}
    for mid, _topic, r in sup:
        old_id = unit_of[mid].get(r["old"])
        new_id = unit_of[mid].get(r["new"])
        qtext = meetings[mid][r["old"]]["text"]
        for name, (link_on, scope_on) in arms.items():
            hits = [h.unit.unit_id for h in
                    engines[name].retrieve(Query(text=qtext, project=mid,
                                                 mode="current"),
                                           linker_enabled=link_on,
                                           scoping=scope_on)][:top_k]
            rows[name].append({"rank_old": _rank_in(hits, old_id),
                               "rank_new": _rank_in(hits, new_id)})

    bunits = {u.unit_id: u for us in units_by_meeting.values() for u in us}
    marked = direct = 0
    per_pair = []
    for (mid, topic, r), row_a, row_b in zip(sup, rows["baseline_A"],
                                             rows["system_B"]):
        old_id = unit_of[mid].get(r["old"])
        new_id = unit_of[mid].get(r["new"])
        ou = bunits.get(old_id)
        m = bool(ou and ou.superseded_by)
        d = bool(ou and ou.superseded_by == new_id)
        marked += int(m)
        direct += int(d)
        per_pair.append({"meeting": mid, "topic": topic,
                         "old_idx": r["old"], "new_idx": r["new"],
                         "A": row_a, "B": row_b,
                         "B_old_marked_superseded": m, "B_chain_direct": d})

    store_b = engines["system_B"].store

    def edge_found(label: str, plist) -> dict:
        n = hit = 0
        for mid, _topic, r in plist:
            a = unit_of[mid].get(r["old"])
            b = unit_of[mid].get(r["new"])
            if not (a and b):
                continue
            n += 1
            rels = ({e.relation for e in store_b.edges_for(a, direction="from")}
                    | {e.relation for e in store_b.edges_for(b,
                                                             direction="from")})
            hit += int(label in rels)
        return {"found": hit, "n": n,
                "rate": round(hit / n, 4) if n else None}

    status_mix = Counter(str(u.status) for us in units_by_meeting.values()
                         for u in us)
    result = {
        "corpus": ("linagora/SUMM-RE rev "
                   "47cb4ec3437ef5497b6092b8c6d1e0b5b1d86dc0 test, "
                   "hand-labelled temporal pairs (test-only, read once)"),
        "labels_file": "data/gold/summre_temporal_test.json",
        "n_supersedes_pairs": len(sup),
        "n_contradicts_pairs": len(con), "n_supports_pairs": len(supo),
        "top_k": top_k,
        "embed_version": embedder.version,
        "reader_backend": getattr(engines["system_B"], "reader_backend",
                                  "unknown"),
        "linker_backend": getattr(engines["system_B"].ingest_pipeline,
                                  "linker_backend", "unknown"),
        "arms": {name: summre_pair_scores(rows[name]) for name in arms},
        "mechanism_B": {"marked": marked, "direct": direct, "n_pairs": len(sup),
                        "marked_superseded_rate": round(marked / len(sup), 4)
                        if sup else None,
                        "direct_chain_rate": round(direct / len(sup), 4)
                        if sup else None,
                        "contradicts": edge_found("contradicts", con),
                        "supports": edge_found("supports", supo)},
        "instrument": {
            "status_mix_B": dict(sorted(status_mix.items())),
            "segments_vs_units": {mid: [len(meetings[mid]),
                                        len(units_by_meeting[mid])]
                                  for mid in mid_in},
            "turns_without_unit": sum(
                len(meetings[mid]) - len(unit_of[mid]) for mid in mid_in),
            "note": ("SUMM-RE ships no status annotations; the reader must "
                     "infer them from raw ASR text. status_mix_B bounds what the "
                     "relation layer could authorise on this corpus — read it "
                     "BEFORE any ranking delta (measure before diagnosing)."),
        },
        "pairs": per_pair,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for eng in engines.values():
        eng.close()
    import gc
    gc.collect()
    return result


# ----------------------------------------------------------------------- CLI

def run(component: str, *, split: str = "test", limit: int = 5000,
        device: str = "cuda", rerank: bool = True, pool: str = "qrels",
        concept: str | None = None,
        reader_model: str | None = None) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if component == "xnli":
        res = {"component": component, "result": eval_linker_xnli(split)}
        tag = "pairclassifier"
    elif component == "xnli-rerank":
        sp = split if split != "test" else "validation"   # tuning split guard, 9.5
        res = {"component": component,
               "result": eval_xnli_with_reranker(sp, limit=None)}
        tag = "crossencoder"
    elif component == "xnli-nli":
        res = {"component": component, "result": eval_xnli_with_head(split)}
        tag = "camembert-head"
    elif component == "massive":
        res = {"component": component,
               "result": eval_reader_massive(split, device, model_dir=reader_model)}
        # The artefact directory is named by the backend that ACTUALLY ran. The
        # first massive run silently got the rules reader (CUDA OOM: the MIRACL arm
        # held 7.03 GiB), and a static tag would have let the model run overwrite
        # that evidence -- two different experiments sharing one artefact path is
        # the bench-identity bug this project already paid for once.
        tag = embed_tag(str(res["result"].get("reader_backend", "unknown"))[:32])
    elif component == "summre":
        # B-13: writes its own artefact (it knows the embed version) and refuses
        # a second read — this split is test-only and already spent by design.
        r = eval_summre_temporal(device=device)
        return {"component": component, "result": r}
    elif component == "miracl":
        # eval_retrieval_miracl writes its own artefact (it knows the embed version)
        r = eval_retrieval_miracl(limit=limit, device=device, with_rerank=rerank,
                                  pool=pool, concept_ckpt=concept or None)
        return {"component": component, "result": r}
    else:
        raise ValueError(f"unknown component {component!r}")
    d = run_dir_for(component, tag, None, split)
    d.mkdir(parents=True, exist_ok=True)
    res["fetched_utc"] = stamp
    (d / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", default="all",
                    choices=["all", "xnli", "xnli-rerank", "xnli-nli", "massive",
                             "summre", "miracl"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--pool", default="qrels", choices=["qrels", "slice0"],
                    help="miracl candidate pool: the judged set (default) or the "
                         "invalid 5k-head slice kept for the record")
    ap.add_argument("--concept", default="",
                    help="path to a concept checkpoint trained in THIS embed space "
                         "to add the dense_refined/hybrid_refined arms (6.13)")
    ap.add_argument("--reader-model", default="",
                    help="massive arm only: trained-reader dir to score, e.g. "
                         "models/reader_warm; the artefact names the backend that "
                         "ACTUALLY ran, fallback included")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args(argv)
    # "all" stays the TUNING sweep: summre is deliberately absent — its labels
    # are test-only with one read, so `make frbench` must never be able to
    # re-spend it by forgetting a flag (9.5; the guard would raise, but a
    # sweep that can only ever half-run is the wrong shape too).
    comps = (["xnli", "massive", "miracl"] if args.component == "all"
             else [args.component])
    rc = 0
    for c in comps:
        try:
            r = run(c, split=args.split, limit=args.limit, device=args.device,
                    rerank=not args.no_rerank, pool=args.pool,
                    concept=args.concept or None,
                    reader_model=args.reader_model or None)
            print(json.dumps(r, indent=1, ensure_ascii=False)[:2000], flush=True)
        except Exception as exc:
            print(f"{c} FAILED: {type(exc).__name__}: {exc}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
