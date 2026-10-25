"""Per-stage evaluation on **gold input** (BUILD_GUIDE 9.2).

Each stage is scored with the previous stage's *gold* output, so a bad reader
cannot be blamed for a bad segmenter. Gold comes from the ledger: a unit knows
which event(s) it renders (`Unit.gold_event_ids`), and the event carries the
exact label set.

`sae stage segment|reader|slots|linker|retrieve`
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from engine.config import MODELS_DIR, load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query, Source, utcnow
from eval import metrics as M
from gen.ledger import build_ledger
from gen.qa_from_ledger import build_questions
from gen.render import reader_status, reader_type, render_ledger
from gen.ledger import reader_gold
from gen.splice import write_manifest

STAGES = ("segment", "reader", "slots", "linker", "retrieve")


def _reader_override(reader: str | None,
                    model_dir: str | None = None) -> dict | None:
    """--reader maps to a cfg override. `warm` is NOT a config backend value
    (a new DEFAULTS value would change the config hash and force a gate
    re-record for an optional experiment): it selects the trained backend
    pointed at the warm checkpoint's dir. `--reader-model` generalises that to
    any checkpoint (models/reader_v2 in the B-13 follow-up) without adding a
    config key."""
    if reader == "model":
        out = {"backend": "model"}
        if model_dir:
            out["model_dir"] = str(model_dir)
        return out
    if reader == "warm":
        return {"backend": "model", "model_dir": str(MODELS_DIR / "reader_warm")}
    if reader == "rules":
        return {"backend": "rules"}
    return None


def _dir_suffix(linker: str | None, reader: str | None,
                model_dir: str | None = None) -> str:
    """Ablations get their own artefact directory (the CI tripwire once silently
    overwrote headline artefacts); the rules baseline keeps the plain name.
    A custom --reader-model names the dir after the checkpoint, so `model` of
    models/reader_v2 can never collide with the plain `_rmodel` rows."""
    suf = "" if (linker or "rules") == "rules" else f"_{linker}"
    if model_dir and reader in ("model", "warm"):
        return suf + f"_r{Path(str(model_dir)).name}"
    return suf + {"model": "_rmodel", "warm": "_rwarm"}.get(reader or "rules", "")


def _build(args):
    ledger = build_ledger(n_projects=args.projects, seed=args.seed)
    manifest = write_manifest(repo_root() / "data/splits/ledger.json",
                              ledger.projects, seed=args.seed)
    sources = render_ledger(ledger, seed=args.seed)
    return ledger, manifest, sources


def _ingest(ledger, sources, args, root: Path) -> Engine:
    """Ingest the whole corpus. Scoring is restricted to one split's projects by
    the caller; the *store* must hold every project, because the cross-project
    distractors are the thing that makes "same slot" a hard problem (7.1.1)."""
    # ruler pin: stage metrics are per-component rulers recorded pre-D-041
    # with rerank none; the §3 rows must stay comparable to their artefacts
    overrides = {"store": {"path": str(root / f"stage-{args.seed}.db")},
                 "embed": {"backend": "hash"},
                 "retrieve": {"rerank": {"backend": "none"}}}
    backend = getattr(args, "linker", None)
    if backend:
        overrides["link"] = {"backend": backend}
    ro = _reader_override(getattr(args, "reader", None),
                          getattr(args, "reader_model", None))
    if ro:
        overrides["reader"] = ro
    cfg = load_config(getattr(args, "config", None), overrides=overrides)
    for p in root.glob(f"stage-{args.seed}.db*"):
        p.unlink()
    eng = Engine(cfg, path=root / f"stage-{args.seed}.db")
    for s in sources:
        eng.ingest(s)
    return eng


def run_stage(stage: str, args) -> int:
    if stage not in STAGES:
        print(f"unknown stage {stage!r}; choose from {STAGES}")
        return 2
    split = getattr(args, "split", "test") or "test"
    suffix = _dir_suffix(getattr(args, "linker", None), getattr(args, "reader", None),
                         getattr(args, "reader_model", None))
    root = repo_root() / "runs" / f"stage_{stage}_{split}_{args.seed}{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    ledger, manifest, sources = _build(args)
    result = {"stage": stage, "seed": args.seed, "split": split,
              "n_projects": len(ledger.projects), "generated_at": utcnow().isoformat()}
    eng = _ingest(ledger, sources, args, root)
    try:
        fn = {"segment": _segment, "reader": _reader, "slots": _slots,
              "linker": _linker, "retrieve": _retrieve}[stage]
        result.update(fn(eng, ledger, sources, set(manifest["split"][split]), args))
        # read-back, not echo: if link.backend said nli and the head was absent,
        # the pipeline fell back to rules and the artefact must say so (D-011)
        result["linker_backend"] = getattr(eng.ingest_pipeline, "linker_backend", "rules")
        result["reader_backend"] = getattr(eng, "reader_backend", "rules")
    finally:
        eng.close()
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                                 default=str), encoding="utf-8")
    lines = [f"# stage `{stage}` — seed {args.seed}"]
    for k, v in result.items():
        if k in ("stage", "seed", "split", "n_projects", "generated_at"):
            continue
        lines.append(f"- {k}: {v}")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, default=str)[:2500])
    print("\nwrote", root)
    return 0


# --- segment ---------------------------------------------------------------
def _segment(eng, ledger, sources, holdout, args) -> dict:
    from engine.segment import turns as T

    t0 = time.perf_counter()
    all_units = eng.store.all_units()
    elapsed = time.perf_counter() - t0
    reconstructed = broken = 0
    for u in all_units:
        got = eng.store.with_sentences(u)
        if not got.sentences:
            broken += 1
            continue
        ok = all(s.slice(got.text) == s.surface for s in got.sentences)
        reconstructed += int(ok)
        broken += int(not ok)
    # cross-speaker: a unit may never merge turns
    cross = 0
    for src in sources:
        by_id = {t.turn_index: t for t in src.turns}
        for u in eng.store.units_of_source(src.source_id):
            if u.turn_index is not None and u.turn_index in by_id:
                if by_id[u.turn_index].speaker not in (None, u.speaker):
                    cross += 1
    # determinism: segmenting the same source twice yields the same ids
    again = eng.segment(sources[0])
    first = eng.segment(sources[0])
    deterministic = [u.unit_id for u in first] == [u.unit_id for u in again]
    n = max(1, reconstructed + broken)
    return {
        "units": len(all_units),
        "span_reconstruction_rate": round(reconstructed / n, 4),
        "span_broken": broken,
        "cross_speaker_units": cross,
        "unit_ids_deterministic": deterministic,
        "ingest_seconds_per_1k_units": round(elapsed / max(1, len(all_units)) * 1000, 3),
        "mean_tokens_per_unit": round(
            sum(T.count_tokens(u.text) for u in all_units) / max(1, len(all_units)), 2),
    }


# --- reader ----------------------------------------------------------------
def _reader(eng, ledger, sources, holdout, args) -> dict:
    gold_kind, gold_type, gold_status = [], [], []
    pred_kind, pred_type, pred_status = [], [], []
    for u in eng.store.all_units():
        if not u.gold_event_ids or u.project not in holdout:
            continue
        ev = ledger.event(u.gold_event_ids[0])
        # The gold triple comes from the ledger's ONE table. This used to
        # hard-code `"query" if ev.kind == "question"`, which contradicted 7.3
        # (a meeting question is `info`, not a user query) and would have scored
        # the reader's now-correct labelling as an error.
        gold_kind.append(reader_gold(ev)[0])
        gold_type.append(reader_type(ev))
        gold_status.append(reader_status(ev))
        pred_kind.append(u.kind or "info")
        pred_type.append(u.type or "other")
        pred_status.append(u.status or "mention")
    if not gold_kind:
        return {"note": "no gold-labelled units in this split"}
    kind_acc = sum(a == b for a, b in zip(gold_kind, pred_kind)) / len(gold_kind)
    type_f1, _ = M.macro_f1(gold_type, pred_type)
    status_f1, per_class = M.macro_f1(gold_status, pred_status)
    return {
        "n": len(gold_kind),
        "kind_accuracy": round(kind_acc, 4),
        "type_macro_f1": type_f1,
        "status_macro_f1": status_f1,
        "status_per_class": {k: v["f1"] for k, v in per_class.items()},
        "targets": {"kind": 0.97, "status_ledger": 0.85},
    }


# --- slots -----------------------------------------------------------------
def _slots(eng, ledger, sources, holdout, args) -> dict:
    from engine.slots.regex_dates import parse_amount, parse_date

    date_ok = date_n = owner_ok = owner_n = subj_ok = subj_n = 0
    amount_ok = amount_n = 0
    for u in eng.store.all_units():
        if not u.gold_event_ids or u.project not in holdout:
            continue
        ev = ledger.event(u.gold_event_ids[0])
        if ev.due:
            date_n += 1
            date_ok += int(u.slots.due == ev.due.isoformat())
        # NB: no second date branch. Comparing parse_date(unit.text) against
        # parse_date(gold_value) would score the parser against itself — a
        # circular metric that reports 1.0 no matter what it parses. True
        # parser correctness lives in tests/unit/test_slots_dates.py.
        if ev.person and ev.kind in ("assignment", "commitment"):
            owner_n += 1
            owner_ok += int(u.slots.owner == ev.person)
        if ev.subject_key:
            subj_n += 1
            subj_ok += int(u.slots.subject == ev.subject_key)
        if ev.kind in ("confirmed", "changed") and (ev.value or "").endswith("€"):
            amount_n += 1
            amount_ok += int(parse_amount(u.text) == parse_amount(ev.value))
    def rate(ok, n):
        return round(ok / n, 4) if n else None
    return {"date_exact_match": rate(date_ok, date_n), "date_cases": date_n,
            "owner_precision": rate(owner_ok, owner_n), "owner_cases": owner_n,
            "subject_exact_match": rate(subj_ok, subj_n), "subject_cases": subj_n,
            "amount_exact_match": rate(amount_ok, amount_n), "amount_cases": amount_n,
            "targets": {"date": 0.95, "owner": 0.90}}


# --- linker ----------------------------------------------------------------
def _linker(eng, ledger, sources, holdout, args) -> dict:
    """Relation P/R against gold pairs, unit-level, with supersedes precision."""
    from engine.link.supersede import evaluate, same_slot

    unit_ids_by_event: dict[str, list[str]] = {}
    for u in eng.store.all_units():
        for eid in u.gold_event_ids:
            unit_ids_by_event.setdefault(eid, []).append(u.unit_id)

    gold_pairs: list[tuple[str, str, str]] = []
    for e in ledger.events:
        if e.project not in holdout:
            continue
        if e.supersedes and e.supersedes in unit_ids_by_event \
                and e.event_id in unit_ids_by_event:
            for nid in unit_ids_by_event[e.event_id]:
                for oid in unit_ids_by_event[e.supersedes]:
                    gold_pairs.append((nid, oid, "supersedes"))

    project_of = {u.unit_id: u.project for u in eng.store.all_units()}
    units_by_id = {u.unit_id: u for u in eng.store.all_units()}
    pred_triples = []
    for u in eng.store.all_units():
        if u.project not in holdout:
            continue
        for e in eng.store.edges_for(u.unit_id, direction="from"):
            # DENOMINATOR MUST MATCH THE GOLD SET. The stage store holds every
            # project, but gold pairs are authored only for the holdout; scoring
            # full-corpus predictions against holdout gold reported supersedes
            # precision 0.221 when the true figure was 0.81 (measured: only 13 of
            # 68 predicted edges disagreed with gold_relation).
            if project_of.get(e.to_id) not in holdout:
                continue
            pred_triples.append((e.from_id, e.to_id, e.relation))
    pred = set(pred_triples)
    gold_set = set(gold_pairs)
    prf = M.relation_prf(sorted(pred), sorted(gold_set))
    sup = prf.get("supersedes", {})
    # 9.1 also asks for relation Macro-F1 and a confusion matrix, which
    # `relation_prf` cannot produce: its gold set is the supersedes-only pair
    # list, so every other relation would be scored against an empty gold column
    # and read as a perfect zero. The matrix is therefore built from
    # `ledger.gold_relation`, which labels any event pair.
    #
    # Two traps here, both found by measuring the first version of this block:
    #
    # * `unrelated` is NOT a writable relation. The store holds only related
    #   pairs, so an unrelated pair produces no edge and can never be "predicted".
    #   Including it in the label space guarantees a 0.0 class and drags macro-F1
    #   down by arithmetic -- the first run reported 0.1523 with `unrelated: 0.0`,
    #   which looked like a broken linker and was a broken ruler. What an
    #   unrelated pair *does* mean is measured separately, below, as the false-edge
    #   rate: an edge written where gold says nothing belongs.
    # * The matrix covers only pairs with exactly one gold label, so its coverage
    #   is reported beside it and cannot be read as "all predictions".
    #
    # Edge order is from=newer, to=older, matching `gold_relation(new, old)`.
    writable = ("supports", "contradicts", "supersedes", "creates_action", "answers")
    matrix: dict[str, dict[str, int]] = {}
    aligned_gold: list[str] = []
    aligned_pred: list[str] = []
    edges_on_unrelated = 0
    unlabelled = 0
    for from_id, to_id, relation in sorted(pred):
        fu, tu = units_by_id.get(from_id), units_by_id.get(to_id)
        if fu is None or tu is None or not fu.gold_event_ids or not tu.gold_event_ids:
            unlabelled += 1
            continue
        labels: set[str] = set()
        for ef in fu.gold_event_ids:
            for et in tu.gold_event_ids:
                try:
                    labels.add(ledger.gold_relation(ef, et))
                except KeyError:
                    continue        # an event id outside this ledger
        if len(labels) != 1:
            unlabelled += 1
            continue                # ambiguous: no single gold label to compare to
        gold_rel, = labels
        if gold_rel == "unrelated":
            edges_on_unrelated += 1
            continue
        matrix.setdefault(gold_rel, {})
        matrix[gold_rel][relation] = matrix[gold_rel].get(relation, 0) + 1
        aligned_gold.append(gold_rel)
        aligned_pred.append(relation)
    rel_macro_f1, rel_per_class = (M.macro_f1(aligned_gold, aligned_pred)
                                   if aligned_gold else (None, {}))
    refusals = [e.guard for u in eng.store.all_units() if u.project in holdout
                for e in eng.store.edges_for(u.unit_id, direction="from")
                if e.guard and e.guard not in ("OK", "G0_model")]
    false_supersede = len([p for p in pred
                           if p[2] == "supersedes" and p not in gold_set])
    return {
        "gold_supersede_pairs": len(gold_set),
        "predicted_supersede": len([p for p in pred if p[2] == "supersedes"]),
        "supersedes_precision": sup.get("precision"),
        "supersedes_recall": sup.get("recall"),
        "false_supersede_edges": false_supersede,
        "relation_prf": prf,
        # macro-F1 over the five relations the linker can actually write; `unrelated`
        # is excluded because it is the absence of an edge, not a prediction
        "relation_macro_f1": rel_macro_f1,
        "relation_macro_f1_label_space": list(writable),
        "relation_per_class": {k: v["f1"] for k, v in rel_per_class.items()},
        "relation_confusion": {g: dict(sorted(p.items()))
                               for g, p in sorted(matrix.items())},
        # the denominator stated beside the matrix, because 108 labelled edges out
        # of 154 predicted is a different claim from "all predictions"
        "relation_confusion_coverage": {
            "labelled_edges": len(aligned_gold),
            "predicted_edges": len(pred),
            "unlabelled_edges": unlabelled,
            "share": round(len(aligned_gold) / len(pred), 4) if pred else None,
            "not_labelled_because": "no gold event id on a side, the pair maps to "
                                    "more than one gold label, or gold says unrelated",
        },
        # the meaningful version of "did we invent a relation": an edge written on a
        # pair the ledger says has nothing to do with each other
        "edges_on_unrelated_pairs": edges_on_unrelated,
        "false_edge_rate": round(edges_on_unrelated / len(pred), 4) if pred else None,
        "guard_refusals": len(refusals),
        "target_supersedes_precision": 0.95,
    }


# --- retrieve --------------------------------------------------------------
def _retrieve(eng, ledger, sources, holdout, args) -> int | dict:
    from eval.temporal import _score_system, gold_unit_map

    qa = build_questions(ledger, seed=args.seed)
    items = [q for q in qa.items if q["project"] in holdout
             and q["kind"] in ("current_decision", "history", "proposal_vs_decision",
                               "owner", "date", "unanswerable")]
    rows = _score_system(eng, items, gold_unit_map(eng.store), ledger, eng.cfg)
    out = {k: v for k, v in rows.items() if not k.startswith("_")
           and k != "per_question"}
    out["n_questions"] = len(items)
    return out
