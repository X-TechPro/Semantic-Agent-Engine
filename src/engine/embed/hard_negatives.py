"""Training triples for the embedder fine-tune (BUILD_GUIDE 6.4).

Three sources, mixed in the ratios the guide names:

* **constructed minimal pairs** (>= 25% of the negatives). These come from
  `gen.hard_negs`, i.e. authored from the ledger, so the gold is exact. The guide
  is emphatic that these are *never* run through the NV-Retriever "is the
  negative too similar, drop it" filter — being almost identical to the positive
  is the entire point. That filter is for *mined* negatives, and this module
  mines none, which is why no similarity filter appears here.
* **question -> the sentence that answers it** (synthetic, from the QA
  generator). Anchors and positives from the same renderer are the closest thing
  to the deployment task the engine has.
* **general French pairs** (20-30%): MASSIVE fr-FR, apache-2.0, verified live.
  Anti-forgetting: without it the encoder drifts into "only my templates exist".
  Pair = two utterances with the same intent label, which is a real
  semantic-equivalence signal in French and needs no annotation.

Two exclusions, both from checking rather than assuming:

* `manu/fquad2_test` (apache-2.0) has **no train split** — only test/valid, so
  training on it is evaluating on the test set. Excluded.
* `irds/mmarco_fr` and `french-datasets/ahmadSiddiqi_miracl` resolve but state
  **no licence**, and 0.5 forbids claiming one. Excluded.

`repeat` minimal pairs are dropped from the triples: both sides assert the same
value, so their gold relation is `supports` and ranking one over the other is a
coin flip that would teach the encoder noise (the same reason `eval.temporal`
excludes them from the accuracy denominator).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from gen.ledger import Ledger, build_ledger
from gen.qa_from_ledger import build_questions
from gen.splice import write_manifest

#: general-language share the guide asks for, and the floor for constructed
#: negatives. Checked in `mix_report`, which the caller turns into a hard error:
#: a mix that quietly drifted is a config key nothing reads.
#:
#: The guide's two ratios are read as: >= 25% of *rows* carry a constructed
#: negative (only constructed rows have one), and 20-30% of every epoch's rows are
#: general-language. Those are stated as batch/epoch shares, so a fixed row count
#: is the closest honest implementation — and it is why `general_limit` is
#: DERIVED from the synthetic count below rather than hard-coded: with 1200 general
#: rows against ~80 constructed, the 25% floor fails by construction.
GENERAL_SHARE = (0.20, 0.30)
CONSTRUCTED_MIN = 0.25
#: pairs per train project from `gen.hard_negs` (before cue filtering). Higher
#: than the QA draw because only ~7/8 of flips keep a discriminating cue.
CONSTRUCTED_PER_PROJECT = 20
MASSIVE_ID = "mteb/amazon_massive_intent"
MASSIVE_LANG = "fr"


@dataclass
class TripletData:
    anchors: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)      # "" = in-batch only
    source: list[str] = field(default_factory=list)          # per-row provenance
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.anchors)

    def rows(self) -> list[dict[str, str]]:
        return [{"anchor": a, "positive": p, "negative": n, "source": s}
                for a, p, n, s in zip(self.anchors, self.positives,
                                      self.negatives, self.source)]


def unit_text_by_event(ledger: Ledger, projects: set[str], seed: int,
                       run_dir: Path) -> dict[str, str]:
    """event_id -> the sentence the engine actually indexed for it.

    Taken from a real ingest rather than re-derived from the event, so the
    encoder trains on the strings retrieval will see (including the segmenter's
    own splitting), and not on a parallel copy of the generator.
    """
    from engine.engine import Engine
    from gen.render import render_meeting

    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "embed_triplets.db"
    for stale in run_dir.glob("embed_triplets.db*"):
        stale.unlink()
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": "hash"},
                                 "reader": {"backend": "rules"}})
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    eng = Engine(cfg, path=db, roster=roster, linker_enabled=False)
    for m in ledger.meetings:
        if m.project in projects:
            eng.ingest(render_meeting(ledger, m.meeting_id, seed=seed))
    out: dict[str, str] = {}
    for u in eng.store.all_units():
        for eid in u.gold_event_ids:
            out.setdefault(eid, u.text)
    eng.close()
    for stale in run_dir.glob("embed_triplets.db*"):
        stale.unlink()
    return out


def constructed_rows(ledger: Ledger, projects: set[str], seed: int,
                     per_project: int = 8) -> tuple[list[str], list[str], list[str]]:
    """(anchor, positive, negative) from the authored minimal pairs."""
    from gen.hard_negs import build_hard_negatives

    a: list[str] = []
    p: list[str] = []
    n: list[str] = []
    for hn in build_hard_negatives(ledger, seed=seed, per_project=per_project):
        if hn.project not in projects:
            continue
        if not hn.cue or hn.meta.get("scoreable", True) is False:
            continue
        a.append(hn.anchor)
        p.append(hn.positive)
        n.append(hn.negative)
    return a, p, n


def qa_rows(ledger: Ledger, projects: set[str], texts: dict[str, str],
            seed: int, per_project: int = 8) -> tuple[list[str], list[str]]:
    """(question, gold sentence) pairs. Anchor and positive share no wording."""
    a: list[str] = []
    p: list[str] = []
    for item in build_questions(ledger, seed=seed, per_project=per_project).items:
        if item["project"] not in projects or not item.get("gold_event_ids"):
            continue
        pos = [texts.get(e) for e in item["gold_event_ids"] if texts.get(e)]
        if not pos:
            continue            # the gold unit was filtered as chatter
        a.append(item["question"])
        p.append(pos[0])
    return a, p


#: the MASSIVE French slice is ~11.5k rows, so it is filtered once per process.
#: Re-filtering 587k rows on every call is pure waste, and the shuffle below is
#: seeded, so caching cannot make two runs differ.
_FR_CACHE: dict[str, list[tuple[str, str]]] = {}
#: disjoint halves, keyed on a stable digest of the sentence. `build` draws
#: BUCKET_TRAIN and the fine-tune's anti-forgetting probe draws BUCKET_PROBE, so
#: the probe measures retention rather than memorisation — two seeds over one
#: corpus would NOT be disjoint (the same sentences reappear in a new order).
BUCKET_TRAIN = "train"
BUCKET_PROBE = "probe"


def _bucket(text: str) -> str:
    import hashlib

    return BUCKET_TRAIN if int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big") % 2 \
        else BUCKET_PROBE


def general_rows(limit: int = 1200, seed: int = 42,
                 bucket: str = BUCKET_TRAIN) -> tuple[list[str], list[str]]:
    """French same-intent pairs from MASSIVE (apache-2.0, verified live).

    A good anti-forgetting source for a reason worth stating: MASSIVE is a
    *paraphrase* corpus, so two rows sharing an intent are the same meaning in
    different words — which is the general skill 6.4 wants preserved while the
    encoder learns our ledger's distinctions.

    `label` in this mirror is already the intent NAME ("alarm_set"); the numeric
    column is `label_index`. Grouping by `int(label)` raises ValueError, which is
    how the first version of this function failed.
    """
    if not _FR_CACHE:
        import datasets

        ds = datasets.load_dataset(MASSIVE_ID, split="train")
        fr = ds.filter(lambda x: x == MASSIVE_LANG, input_columns=["lang"])
        _FR_CACHE["rows"] = list(zip(fr["label"], fr["text"]))

    by_intent: dict[str, list[str]] = {}
    for lbl, text in _FR_CACHE["rows"]:
        if _bucket(text) != bucket:
            continue
        by_intent.setdefault(str(lbl), []).append(text)
    rng = random.Random(sub_seed(seed, "massive"))
    a: list[str] = []
    p: list[str] = []
    for texts in by_intent.values():
        uniq = list(dict.fromkeys(t.strip() for t in texts if len(t.strip()) > 8))
        if len(uniq) < 2:
            continue
        rng.shuffle(uniq)
        for i in range(0, len(uniq) - 1, 2):
            a.append(uniq[i])
            p.append(uniq[i + 1])
            if len(a) >= limit:
                return a, p
    return a, p


def sub_seed(seed: int, tag: str) -> int:
    from engine.seeding import subseed

    return subseed(seed, "embed", tag)


def mix_report(data: TripletData) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(data.source)
    total = max(1, len(data))
    general = counts.get("general_fr", 0) / total
    constructed = counts.get("constructed", 0) / total
    lo, hi = GENERAL_SHARE
    return {
        "n_rows": len(data), "by_source": dict(counts),
        "general_share": round(general, 4), "constructed_share": round(constructed, 4),
        "constructed_negatives": sum(1 for n in data.negatives if n),
        "ok": bool(lo <= general <= hi and constructed >= CONSTRUCTED_MIN),
    }


def build(projects: int = 20, seed: int = 42, general_share: float = 0.25,
          per_project: int = 8, with_general: bool = True,
          run_dir: Path | None = None) -> TripletData:
    """Assemble the train-split triples. Dev and test projects never appear.

    9.5: tuning data may not touch the test split, and here it may not touch dev
    either — dev is where the fine-tuned encoder gets chosen against the frozen
    one, so any dev sentence in the training mix would leak the selection.

    The general-language budget is DERIVED from the synthetic row count so the
    guide's 20-30% band and the >=25% constructed floor can hold at the same
    time. A fixed 1200-row cap could not: with ~180 constructed pairs it puts the
    constructed share near 12%, so the mix would violate the rule the docstring
    claims to enforce. `general_share` is therefore a ratio, not a row count.
    """
    root = repo_root()
    run_dir = run_dir or (root / "runs" / "embed_finetune")
    if not (GENERAL_SHARE[0] <= general_share <= GENERAL_SHARE[1]):
        raise SystemExit(f"general share {general_share} outside 6.4's "
                         f"{GENERAL_SHARE} band")
    ledger = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                        seed=seed)["split"]
    train = set(sp["train"])
    if not train:
        raise SystemExit("no train projects; raise --projects")

    texts = unit_text_by_event(ledger, train, seed, run_dir)
    data = TripletData(meta={"seed": seed, "train_projects": sorted(train),
                             "base_split": {k: len(v) for k, v in sp.items()}})

    ca, cp, cn = constructed_rows(ledger, train, seed,
                                  per_project=CONSTRUCTED_PER_PROJECT)
    data.anchors += ca
    data.positives += cp
    data.negatives += cn
    data.source += ["constructed"] * len(ca)

    qa, qp = qa_rows(ledger, train, texts, seed, per_project=per_project)
    data.anchors += qa
    data.positives += qp
    data.negatives += [""] * len(qa)
    data.source += ["qa"] * len(qa)

    if with_general:
        synthetic = len(data)
        limit = int(round(synthetic * general_share / (1.0 - general_share)))
        ga, gp = general_rows(limit=limit, seed=seed)
        data.anchors += ga
        data.positives += gp
        data.negatives += [""] * len(ga)
        data.source += ["general_fr"] * len(ga)

    rng = random.Random(sub_seed(seed, "shuffle"))
    order = list(range(len(data)))
    rng.shuffle(order)
    take = lambda col: [col[i] for i in order]      # noqa: E731 (one-liner, used 4x)
    data.anchors = take(data.anchors)
    data.positives = take(data.positives)
    data.negatives = take(data.negatives)
    data.source = take(data.source)
    data.meta["mix"] = mix_report(data)
    if not data.meta["mix"]["ok"]:
        raise SystemExit(
            f"training mix violates 6.4: {data.meta['mix']} — constructed rows "
            f"must be >= {CONSTRUCTED_MIN:.0%} and general-language within "
            f"{GENERAL_SHARE}; raise --constructed-per-project or lower "
            f"--general-share")
    return data


__all__ = ["TripletData", "build", "mix_report", "constructed_rows", "qa_rows",
           "general_rows", "unit_text_by_event", "GENERAL_SHARE",
           "CONSTRUCTED_MIN", "MASSIVE_ID"]
