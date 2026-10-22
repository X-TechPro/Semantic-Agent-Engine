"""Concept-attention kill test (BUILD_GUIDE 6.13 "run this FIRST").

The claim under test is not "concept attention is nice" but:

    can the vector space ALONE rank the newer, in-force unit first?

If it already can, M5 has no headroom and must be dropped. If it cannot but the
relation layer can, geometry is the wrong place to put that responsibility. The
scenario is the guide's, verbatim: "launch in December" (1 Oct) -> "actually
January" (5 Oct) -> distractors; query "when is the launch?"; measured with **no
validity filter and no reranker**, and with two dated sources because that is
how ingestion actually sees a superseding statement (an earlier draft of this
file re-stamped unit dates in place, which is not a real ingest path).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query, Source, Turn

TARGET_TEXT = "Finalement le lancement du projet Atlas passe en janvier."
STALE_TEXT = "Le lancement du projet Atlas est prévu en décembre."
QUERY = "Quand est le lancement du projet Atlas ?"

DISTRACTORS = [
    ("Sophie", "Le budget du projet Atlas est de 1,5 M€."),
    ("Julien", "On pourrait peut-être revoir le périmètre du projet Atlas."),
    ("Nadia", "Le prestataire retenu pour le projet Atlas est Nova Cloud."),
    ("Marie", "Attention, le risque de conformité du projet Atlas est élevé."),
    ("Karim", "Le rapport de cadrage du projet Atlas part fin novembre."),
]


def _sources() -> list[Source]:
    early = Source(source_id="meeting-kill-1", source_type="meeting",
                   source_date=date(2026, 10, 1), project="Atlas",
                   default_speaker="Marie",
                   turns=[Turn(speaker="Marie", text=STALE_TEXT, turn_index=0,
                               project="Atlas")])
    later = Source(source_id="meeting-kill-2", source_type="meeting",
                   source_date=date(2026, 10, 5), project="Atlas",
                   default_speaker="Karim",
                   turns=[Turn(speaker="Karim", text=TARGET_TEXT, turn_index=0,
                               project="Atlas")]
                   + [Turn(speaker=s, text=t, turn_index=i + 1, project="Atlas")
                      for i, (s, t) in enumerate(DISTRACTORS)])
    return [early, later]


def run(seed: int = 42, embed_backend: str = "hash") -> dict:
    root = repo_root()
    out_dir = root / "runs" / f"killtest_{seed}_{embed_backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "kill.db"
    for p in list(out_dir.glob("kill.db*")):
        p.unlink(missing_ok=True)
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": embed_backend}})
    eng = Engine(cfg, path=db, linker_enabled=True)
    for s in _sources():
        eng.ingest(s)

    q = Query(text=QUERY, project="Atlas")
    # three views of the same corpus: pure geometry, plain hybrid RAG, and the
    # typed-relation system
    dense_only = [(uid, eng.store.get(uid).text)
                  for uid, _ in eng.store.neighbors(
                      eng.embedder.embed([QUERY], role="query")[0], 5)]
    hybrid_a = [h.unit.text for h in eng.retrieve(q, k=5, linker_enabled=False)]
    system_b = [h.unit.text for h in eng.retrieve(q, k=5)]

    def rank(rows: list[str], needle: str) -> int | None:
        for i, t in enumerate(rows, start=1):
            if needle in t:
                return i
        return None

    dense_texts = [t for _, t in dense_only]
    stale_head = [u for u in eng.store.all_units() if STALE_TEXT in u.text]
    verdict = {
        "seed": seed, "embed_backend": embed_backend, "query": QUERY,
        "dense_only_order": dense_texts,
        "hybrid_A_order": hybrid_a,
        "system_B_order": system_b,
        "dense_only_rank_of_in_force": rank(dense_texts, TARGET_TEXT),
        "dense_only_rank_of_stale": rank(dense_texts, STALE_TEXT),
        "A_rank_of_in_force": rank(hybrid_a, TARGET_TEXT),
        "A_rank_of_stale": rank(hybrid_a, STALE_TEXT),
        "B_rank_of_in_force": rank(system_b, TARGET_TEXT),
        "B_rank_of_stale": rank(system_b, STALE_TEXT),
        "stale_unit_is_superseded": bool(stale_head) and not stale_head[0].is_current,
        "space_alone_solves_it": (rank(dense_texts, TARGET_TEXT) == 1
                                  and rank(dense_texts, STALE_TEXT) != 1),
    }
    verdict["m5_recommendation"] = _recommend(verdict)
    (out_dir / "result.json").write_text(json.dumps(verdict, indent=1,
                                                    ensure_ascii=False),
                                          encoding="utf-8")
    eng.close()
    return verdict


def _recommend(v: dict) -> str:
    """6.13's gate, stated exactly: drop ONLY if the space already does it.

    An earlier version returned "DROP" when the *relation layer* solved the case
    end-to-end. That inverts the gate: 6.13 asks whether the vector space alone
    ranks the in-force unit first, and the measured answer here is no (stale
    December ranks 1st, January 2nd), so the concept layer has precisely the
    headroom it was proposed to fill and stays in scope -- to be judged by 9.7's
    "beats the plain version by > 3 points", not by whether something else
    already works.
    """
    if v["space_alone_solves_it"]:
        return ("DROP concept attention (6.13): the frozen space already ranks "
                "the in-force unit first with no filter and no reranker, so "
                "there is nothing for the refinement to add.")
    if v["B_rank_of_in_force"] == 1:
        return ("KEEP M5 IN SCOPE: the space alone fails the decisive test, so "
                "refined vectors have headroom. The relation layer also solves "
                "this case, so the honest bar is 9.7/6.13's >=3 pts on "
                "temporal@1 over System B -- not 'does anything work'.")
    return ("KEEP M5 IN SCOPE: neither the space nor the relation layer solves "
            "it yet, so run the full temporal set before committing compute.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed", default="hash", choices=["hash", "st"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    res = run(seed=args.seed, embed_backend=args.embed)
    print(f"query: {res['query']}\n")
    print("dense-only (no filter, no reranker):")
    for i, t in enumerate(res["dense_only_order"], start=1):
        print(f"  {i}. {t[:74]}")
    print(f"\n  in-force rank : {res['dense_only_rank_of_in_force']}")
    print(f"  stale rank    : {res['dense_only_rank_of_stale']}")
    print(f"  A rank        : {res['A_rank_of_in_force']} (stale {res['A_rank_of_stale']})")
    print(f"  B rank        : {res['B_rank_of_in_force']} (stale {res['B_rank_of_stale']})")
    print(f"  stale unit marked superseded: {res['stale_unit_is_superseded']}")
    print(f"  space alone solves it       : {res['space_alone_solves_it']}")
    print(f"\nM5 recommendation: {res['m5_recommendation']}")
