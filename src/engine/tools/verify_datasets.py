"""`python -m engine.tools.verify_datasets` — regenerate docs/DATASETS.md rows.

Same shape as verify_models, for datasets: hit the live HF API, read the
`license:` front-matter key from the card, and report a name that does not
resolve as a failure rather than an assumption. BUILD_GUIDE 7.2 lists dataset
ids that were never checked, and two of them are wrong.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

API = "https://huggingface.co/api/datasets/{did}"
RAW = "https://huggingface.co/datasets/{did}/raw/main/README.md"
UA = {"User-Agent": "semantic-agent-engine/0.1 (licence verification)"}

#: (id asserted by BUILD_GUIDE 7.2, ids actually worth trying, guide's claim)
CANDIDATES: list[tuple[str, list[str], str]] = [
    ("linagora/SummRe", ["linagora/SummRe", "linagora/SUMM-RE",
                         "datasets-CNRS/summ-re-asru"], "CC-BY-SA-4.0"),
    ("qanastek/FREDSum", ["qanastek/FREDSum", "linagora/FREDSum"], "Apache-2.0"),
    ("MASSIVE (fr-FR)", ["AmazonScience/massive", "massive",
                         "mteb/amazon_massive_intent", "Universal-NLG/Massive"],
     "CC BY 4.0"),
    ("XNLI (pilot)", ["facebook/xnli", "climate_fever/xnli"], "CC BY-NC 4.0"),
    ("mMARCO-fr", ["unicamp-dl/mmarco", "mteb/mmarco"], "not stated"),
    ("MIRACL-fr", ["miracl/miracl", "miracl/miracl-corpus", "unicamp-dl/miracl",
                   "mteb/miracl", "french-datasets/ahmadSiddiqi_miracl"], "not stated"),
    ("FQuAD", ["illuin-tech/fquad", "manu/fquad2_test", "LightOn/FQuAD-dev"],
     "CC BY-NC-SA"),
    ("NanoMTEB-French", ["LeaderboardSpace/NanoMTEB-French",
                         "hakari-bench/NanoMTEB-French"], "not stated"),
    ("LightOn finemem-cc", ["LightOn/finemem-cc", "lightonai/Lighton-RR",
                            "LightOn/embedding-modernbert-800m"], "Apache-2.0"),
    ("LongMemEval", ["xiaowu0162/longmemeval", "xiaowu0162/longmemeval-v2"],
     "Apache/MIT"),
    ("MemoryAgentBench", ["ai-hyz/MemoryAgentBench", "Robin076/MemoryAgentBench"],
     "Apache/MIT"),
    ("belebele", ["facebook/belebele", "GoogleBelebele/belebele"], "CC BY-NC 4.0"),
    ("TIAGE", ["tiage-org/tiage", "getomski/tiage"], "MIT"),
    ("SuperDialseg", ["bfurri/superdialseg", "GaChM/UFR_Benchmark_Discourse_Seg"],
     "not stated"),
    ("AMI meetings", ["dpshien/AMI-meetings-qa", "askam/mlx_ami"], "CC BY-SA-ish"),
]


@dataclass(slots=True)
class Row:
    asserted: str
    tried: list[str] = field(default_factory=list)
    resolved: str = ""
    licence: str = ""
    licence_source: str = ""
    guide_claim: str = ""
    agrees: bool | None = None
    note: str = ""


def _get(url: str, timeout: float = 25.0) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        return f"__HTTP_{exc.code}__".encode()
    except Exception:
        return None


def _licence_from_card(did: str) -> tuple[str, str]:
    """(licence, where it was found) from the README front matter."""
    raw = _get(RAW.format(did=did))
    if raw is None or raw.startswith(b"__HTTP_"):
        return "", "no card"
    txt = raw.decode("utf-8", "replace")
    m = re.search(r"^---\s*\n(.*?)\n---", txt, re.DOTALL)
    block = m.group(1) if m else txt[:4000]
    lic = re.search(r"(?im)^\s*licen[cs]e\s*:\s*[\"']?([\w.\- ]{2,40})", block)
    if lic:
        return lic.group(1).strip().rstrip("\"'"), "card front-matter"
    lic2 = re.search(r"(?i)licen[cs]e[s]?\D{0,3}((?:CC|Apache|MIT|BSD|ODC|DbCL)"
                     r"[\w.\-+]*)", txt[:8000])
    if lic2:
        return lic2.group(1).strip(), "card body"
    return "", "not stated"


def probe(asserted: str, ids: list[str], claim: str) -> Row:
    row = Row(asserted=asserted, tried=list(ids), guide_claim=claim)
    for did in ids:
        meta = _get(API.format(did=did))
        if meta is None or meta.startswith(b"__HTTP_"):
            continue
        try:
            json.loads(meta)
        except Exception:
            continue
        row.resolved = did
        row.licence, row.licence_source = _licence_from_card(did)
        break
    if not row.resolved:
        row.note = ("no id resolved unauthenticated (gated, agreement required, "
                    "or the guide's name is wrong)")
        row.agrees = None
        return row
    norm = row.licence.lower().replace("_", "-").replace(" ", "")
    claim_norm = claim.lower().replace(" ", "").replace("_", "-")
    if not row.licence:
        row.agrees = None
        row.note = "resolves but states no licence — do not claim one"
    elif claim_norm in ("", "notstated"):
        # The guide asserted nothing here, so a stated card licence is
        # information, not a contradiction — "NO" would imply the guide claimed
        # otherwise (the range(28) lesson applied to claims).
        row.agrees = None
        row.note = f"guide stated no licence; card states {row.licence}"
    else:
        # Normalise before comparing: `cc-by-4.0` (card style) and `CC BY 4.0`
        # (guide style) are the same licence, but a literal substring check called
        # it a CONFLICT and the generated DATASETS.md printed a false disagreement
        # -- the exact inversion of what this tool exists to do.
        strip = re.sub(r"[^a-z0-9]", "", norm)
        strip_claim = re.sub(r"[^a-z0-9]", "", claim_norm)
        row.agrees = (strip in strip_claim or strip_claim in strip
                      or ("bysa" in strip and "bysa" in strip_claim)
                      or ("bync" in strip and "bync" in strip_claim)
                      or ("mit" in strip and "mit" in strip_claim)
                      or ("apache" in strip and "apache" in strip_claim))
        if not row.agrees:
            row.note = f"CONFLICT: card says {row.licence}, guide says {claim}"
    return row


def _usage_state(root) -> list[str]:
    """What is actually used today, computed from disk -- not from a constant.

    The generator previously hardcoded "**Nothing external.**", which became a
    lie the moment `eval/frbench.py` published its first XNLI verdict. An
    instruction that survives regeneration can rot; this section reads the
    provenance files and the artefacts instead.
    """
    ext = sorted((root / "data" / "external").glob("*/PROVENANCE.json"))
    if not ext:
        return ["**Nothing external.** Every measured result in `runs/` comes "
                "from the synthetic ledger (`data/ledger/`), so all headline "
                "numbers are **internal-validity only**.", ""]
    L = [f"**{len(ext)} external corpus/corpora fetched**, revision-pinned in "
         "`data/external/*/PROVENANCE.json` and consumed by `eval/frbench.py`:", ""]
    for p in ext:
        try:
            prov = json.loads(p.read_text())
        except Exception:
            L.append(f"- `{p.parent.name}`: PROVENANCE.json unreadable")
            continue
        L.append(f"- `{prov.get('repo')}` rev `{str(prov.get('revision'))[:10]}` — "
                 f"card licence: {prov.get('licence_card', 'NOT STATED')} — "
                 f"{len(prov.get('files', []))} file(s)")
    fb = sorted((root / "runs" / "frbench").glob("*/result.json"))
    if fb:
        L += ["", f"- {len(fb)} real-French component verdict(s) on disk in "
              "`runs/frbench/*/result.json` (`make frbench`). These numbers are "
              "component-level external checks."]
    # The headline-external claim is COMPUTED, never asserted — three states,
    # whatever the disk actually shows (the "Nothing external" rot this section
    # exists to prevent).
    gold = root / "data" / "gold" / "summre_temporal_test.json"
    sm = sorted((root / "runs" / "frbench").glob("summre_*/result.json"))
    if sm:
        try:
            r = json.loads(sm[-1].read_text())
            arms = r.get("arms") or {}
            fmt = lambda v: "not measured" if v is None else f"{v:.0%}"
            A = (arms.get("baseline_A") or {}).get("new_above_old")
            B = (arms.get("system_B") or {}).get("new_above_old")
            n = (arms.get("system_B") or {}).get("n_pairs")
            L += ["", f"- The headline layer has an external number: newer claim "
                  f"ranked first on A {fmt(A)} vs B {fmt(B)} over {n} hand-labelled "
                  "SUMM-RE supersedes pairs (`runs/frbench/summre_*`, test-only, "
                  "read once; mechanism-reach caveats in REPORT.md §6d)."]
        except Exception as exc:                      # unreadable = say so
            L += ["", f"- SUMM-RE artefact unreadable ({type(exc).__name__}); no "
                  "external headline number is claimed."]
    elif gold.exists():
        L += ["", "- The SUMM-RE hand-labelled temporal set is BUILT "
              "(`data/gold/summre_temporal_test.json`); the one A-vs-B test read "
              "has not been run, so the headline itself stays synthetic until "
              "`sae frbench --component summre` spends it."]
    elif fb:
        L += ["", "- Headline temporal@1 stays synthetic: no hand-labelled "
              "external supersede set on disk, and none in `runs/`."]
    L.append("")
    return L


def render(rows: list[Row], root=None) -> str:
    if root is None:
        from engine.config import repo_root
        root = repo_root()
    L = ["# DATASETS.md — provenance and licences", "",
         f"Generated by `python -m engine.tools.verify_datasets` (live HF API, "
         f"{len(rows)} asserted corpora). A name that does not resolve is reported "
         f"as such — it is not assumed to exist, and an absent licence is not "
         f"invented from the guide.", "",
         "| guide id (7.2) | resolves as | licence (card) | guide claim | agree? |",
         "|---|---|---|---|---|"]
    for r in rows:
        agree = "—" if r.agrees is None else ("yes" if r.agrees else "**NO**")
        L.append(f"| `{r.asserted}` | "
                 + (f"`{r.resolved}`" if r.resolved else f"none of {r.tried}")
                 + f" | {r.licence or '—'} | {r.guide_claim} | {agree} |")
    L += ["", "## Notes", ""]
    for r in rows:
        if r.note:
            L.append(f"- `{r.asserted}` → {r.note}")
    conflicts = [r for r in rows if r.agrees is False]
    unverified = [r for r in rows if r.agrees is None]
    L += ["", "## What is actually used today", ""] + _usage_state(root) + [
          f"- {len(conflicts)} licence conflict(s) with BUILD_GUIDE 7.2: "
          + (", ".join(f"`{r.asserted}` (card: {r.licence}, guide: "
                       f"{r.guide_claim})" for r in conflicts) or "none"),
          f"- {len(unverified)} unresolved or licence-unstated row(s): "
          + (", ".join(f"`{r.asserted}`" for r in unverified) or "none"), "",
          "A `—` in the licence column means the card states none. It is written "
          "as `not stated`, never filled from the guide: an unverified licence is "
          "a licence that could turn out non-commercial (the XNLI and FQuAD cases "
          "in 7.2 are exactly that risk).", "", "## Reproduce", "",
          "```bash", "PYTHONPATH=src python -m engine.tools.verify_models    "
          "# models  -> docs/MODELCARDS.md",
          "PYTHONPATH=src python -m engine.tools.verify_datasets  # datasets -> "
          "this file", "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from engine.config import repo_root

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(repo_root() / "docs" / "DATASETS.md"))
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    rows = []
    for asserted, ids, claim in CANDIDATES:
        r = probe(asserted, ids, claim)
        flag = "OK" if r.resolved else "NONE"
        print(f"[{flag:>4}] {asserted:22} -> {r.resolved or '-':38} "
              f"licence={r.licence or '?'}")
        rows.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    print(f"\nwrote {out}")
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in rows], indent=1,
                                              ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
