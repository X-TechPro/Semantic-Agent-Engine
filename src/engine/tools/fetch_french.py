"""Fetch the pinned French corpora for `eval.frbench` (real-data check, 7.2).

Idempotent: each dataset lands in `data/external/<id>/` with a PROVENANCE.json
pinning the repo revision, the exact files, the fetch date and the licence *as the
card states it* -- never as the guide claimed it. If the provenance file already
records a revision, that revision is honoured, so re-running months later yields
byte-identical inputs and the recorded numbers stay comparable (the artefact-identity
rule the split manifest already follows).

Usage:
    PYTHONPATH=src python -m engine.tools.fetch_french [--only xnli|massive|miracl|summre]
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from engine.config import repo_root

HF = "https://huggingface.co/api/datasets/{}"


def _sha(rid: str) -> str:
    with urllib.request.urlopen(HF.format(rid), timeout=30) as r:
        return json.load(r)["sha"]


def _fetch(folder: str, rid: str, files: list[str], licence_note: str,
           repo_type: str = "dataset") -> dict:
    dest = repo_root() / "data" / "external" / folder
    prov_path = dest / "PROVENANCE.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        sha = prov.get("revision") or _sha(rid)
    else:
        sha = _sha(rid)
        prov = {}
    from huggingface_hub import hf_hub_download
    got = []
    for f in files:
        p = dest / f
        if p.exists() and p.stat().st_size > 0:
            got.append((f, p.stat().st_size, "cached"))
            continue
        local = hf_hub_download(repo_id=rid, filename=f, repo_type=repo_type,
                                local_dir=str(dest), revision=sha)
        got.append((f, Path(local).stat().st_size, "downloaded"))
    prov.update({"repo": rid, "revision": sha, "files": files,
                 "licence_card": licence_note,
                 "provenance_note": "licence recorded as the card states it; "
                                    "`NOT STATED` is never filled from the guide",
                 "use_rule": "test-only unless documented in docs/DATASETS.md"})
    prov_path.write_text(json.dumps(prov, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    return {"folder": folder, "repo": rid, "revision": sha[:10], "files": got}


SETS = {
    "xnli": dict(folder="facebook__xnli", rid="facebook/xnli",
                 licence_note="NOT STATED on the card",
                 files=["fr/test-00000-of-00001.parquet",
                        "fr/validation-00000-of-00001.parquet", "README.md"]),
    "massive": dict(folder="mteb__amazon_massive_intent",
                    rid="mteb/amazon_massive_intent",
                    licence_note="apache-2.0 (mirror; canonical AmazonScience/massive card: cc-by-4.0)",
                    files=["test/fr.json.gz", "validation/fr.json.gz", "README.md"]),
    "miracl": dict(folder="miracl__miracl", rid="miracl/miracl",
                   licence_note="apache-2.0",
                   files=["miracl-v1.0-fr/topics/topics.miracl-v1.0-fr-dev.tsv",
                          "miracl-v1.0-fr/qrels/qrels.miracl-v1.0-fr-dev.tsv"]),
    "miracl_corpus": dict(folder="miracl__miracl-corpus", rid="miracl/miracl-corpus",
                          licence_note="apache-2.0",
                          # the whole French pool: qrels candidates (gold AND the
                          # official hard negatives) are scattered across all 30
                          # shards; a 1-shard slice left 340/343 queries unjudgeable
                          # and a 28-shard guess still missed 94 judged candidates
                          # (measured 2026-10-21 -- `range(28)` was an assumption;
                          # the repo actually ships docs-0..docs-29)
                          files=[f"miracl-corpus-v1.0-fr/docs-{i}.jsonl.gz"
                                 for i in range(30)]),
    "summre": dict(folder="linagora__summ-re", rid="linagora/SUMM-RE",
                   licence_note="cc-by-sa-4.0",
                   files=["data/example/example-00000-of-00001.parquet"]),
}


def fetch(only: str | None = None) -> dict:
    out = {}
    for name, spec in SETS.items():
        if only and only not in (name, "all"):
            continue
        out[name] = _fetch(**spec)
        for f, size, state in out[name]["files"]:
            print(f"  {state:>10}  {size:>12,}  {name}/{f}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, choices=[*SETS, "all"])
    args = ap.parse_args()
    r = fetch(args.only)
    print(json.dumps(r, indent=1)[:400])
