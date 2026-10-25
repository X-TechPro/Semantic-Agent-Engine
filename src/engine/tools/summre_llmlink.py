"""Whole-session LLM linking — the user's "tiny LLM reads the meeting" idea.

What was tested before and is NOT this: D-026 probed segment->short-key
extraction (0/6 usable; the models' failure mode was writing summaries) and
D-039 fitted a pair classifier over ~38k pairs per meeting (recall 0.45,
precision 0.0009). Here the model sees the WHOLE meeting at once and PROPOSES
typed relations by segment index — the Hindsight retain step, run locally on a
small model. Zero-shot: the prompt carries no SUMM-RE text (no examples from
any scored corpus).

Bars, declared BEFORE the run (this docstring is the declaration):
  G1  pooled over the six labelled meetings (train 5 + sel): supersedes
      precision >= 0.70 AND recall >= 0.30 — D-039's own bars, exact
      (old, new) pair match. Clears -> the sealed holdout is read EXACTLY
      ONCE, same bars (G2). Fails -> nothing is wired, holdout never read.
Instrument guards, not tuning knobs: malformed JSON is a recorded rejection;
a truncated PROMPT or a truncated ANSWER aborts the run loudly (a first run
cut a 14 KB answer at num_predict=4096 and the parser scored it as zero
proposals — an instrument fault, never again a "capability" number); spent-test
meetings are unreachable and holdout rows are built only after G1 clears.

Run:  PYTHONPATH=src python -u -m engine.tools.summre_llmlink [--model qwen3.5:2b]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

from engine.config import repo_root
from engine.tools.summre_threads import (SPLITS, SPENT_TEST, _load)

OLLAMA = "http://localhost:11434"
RELATIONS = ("supersedes", "contradicts", "supports")
GOLD_FILES = {"train": "summre_temporal_train.json",
              "sel": "summre_temporal_sel.json",
              "holdout": "summre_temporal_holdout.json"}
BARS = {"precision": 0.70, "recall": 0.30}

PROMPT_TMPL = """You extract decision-change relations from a meeting transcript. \
Answer with JSON only.

Below is a French meeting transcript. Each line is "[segment_index] (speaker) text".

Find every case where a LATER statement changes the group's position on something \
an EARLIER statement had said:
- "supersedes": the new statement REPLACES the old decision or value (a change of \
venue, date, budget, vendor, scope: "finalement on part sur...", "c'est annulé", ...).
- "contradicts": two statements cannot both hold, and neither replaces the other.
- "supports": the new statement confirms or reinforces the old one.

Rules: only report relations between real segment indices you can see, with \
old < new. Do not invent indices, do not report a relation twice. Output JSON:
{{"relations": [{{"old": <int>, "new": <int>, "relation": "supersedes|contradicts|supports", "why": "<short reason>"}}]}}
If you find none, output {{"relations": []}}.

TRANSCRIPT:
{transcript}
"""


def transcript(segs: list[dict]) -> str:
    return "\n".join(f"[{i}] ({s.get('speaker', '?')}) {s['text'].strip()}"
                     for i, s in enumerate(segs))


def ask(model: str, prompt: str, num_ctx: int = 32768) -> dict:
    # no num_predict cap: a first run truncated a 14 KB answer mid-JSON at
    # 4096 tokens and the parse guard scored it as zero proposals — the
    # instrument destroying the measurement it exists to take. done_reason
    # below is the loud guard instead.
    body = {"model": model, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "num_ctx": num_ctx},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(OLLAMA + "/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def check_not_truncated(resp: dict, prompt: str, meeting: str) -> None:
    """French averages ~4 chars/token; if the server saw far fewer tokens than
    the prompt contains, the transcript was cut and the score is fiction."""
    seen = resp.get("prompt_eval_count") or 0
    expect = len(prompt) / 4.0
    if seen < 0.75 * expect:
        raise SystemExit(f"{meeting}: context truncated — server processed "
                         f"{seen} prompt tokens, transcript needs ~{expect:.0f}. "
                         "Raise num_ctx or the run is void.")


def parse_relations(raw: str, n_segments: int):
    """-> (kept, rejected). Guards: shape, known relation, old < new, in range."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], [{"raw": raw[:400], "reject": f"bad json: {e}"}]
    rows = doc.get("relations") if isinstance(doc, dict) else doc
    if not isinstance(rows, list):
        return [], [{"raw": str(doc)[:400], "reject": "no relations list"}]
    kept, rej, seen = [], [], set()
    for r in rows:
        if not isinstance(r, dict):
            rej.append({"row": r, "reject": "row not an object"})
            continue
        try:
            old, new = int(r["old"]), int(r["new"])
        except (KeyError, TypeError, ValueError):
            rej.append({"row": r, "reject": "non-integer index"})
            continue
        rel = str(r.get("relation", "")).strip().lower()
        if rel not in RELATIONS:
            rej.append({"row": r, "reject": f"unknown relation {rel!r}"})
            continue
        if not (0 <= old < new < n_segments):
            rej.append({"row": r, "reject": "not ordered/in range"})
            continue
        key = (old, new, rel)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"old": old, "new": new, "relation": rel,
                     "why": str(r.get("why", ""))[:200]})
    return kept, rej


def load_gold(root: Path, split: str) -> dict[str, set]:
    doc = json.loads((root / "data/gold" / GOLD_FILES[split])
                     .read_text(encoding="utf-8"))
    out = {}
    for mm, d in doc["meetings"].items():
        out[mm] = {(r["old"], r["new"], r["relation"])
                   for th in d["threads"] for r in th["relations"]}
    return out


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    return {"n_pred": len(pred), "n_gold": len(gold), "tp": tp,
            "precision": round(tp / len(pred), 4) if pred else 0.0,
            "recall": round(tp / len(gold), 4) if gold else None}


def run_meetings(model: str, meetings: list[str], root: Path, gold: dict,
                 segs_by_meeting: dict, num_ctx: int, raw_dir: Path) -> dict:
    per, all_pred, all_gold = {}, set(), set()
    for mm in meetings:
        if mm in SPENT_TEST:
            raise ValueError(f"{mm} is the spent external test — unreachable")
        segs = segs_by_meeting[mm]
        prompt = PROMPT_TMPL.format(transcript=transcript(segs))
        resp = ask(model, prompt, num_ctx)
        if resp.get("done_reason") == "length":
            raise SystemExit(
                f"{mm}: server stopped the answer at the context limit — a "
                "cut-off JSON must never be scored as zero proposals. Raise "
                "--num-ctx; the run is void.")
        check_not_truncated(resp, prompt, mm)
        raw = resp.get("message", {}).get("content", "")
        (raw_dir / f"{mm}.json").write_text(raw, encoding="utf-8")
        kept, rej = parse_relations(raw, len(segs))
        pred = {(k["old"], k["new"], k["relation"]) for k in kept}
        sup_pred = {(o, n) for o, n, r in pred if r == "supersedes"}
        sup_gold = {(o, n) for o, n, r in gold[mm] if r == "supersedes"}
        m = prf(sup_pred, sup_gold)
        m.update({"n_proposals": len(kept), "n_rejected": len(rej),
                  "prompt_tokens": resp.get("prompt_eval_count"),
                  "by_relation": {r: sum(1 for k in kept if k["relation"] == r)
                                  for r in RELATIONS}})
        per[mm] = {"metrics": m, "pred": sorted(sup_pred),
                   "gold": sorted(sup_gold), "rejected": rej[:20]}
        all_pred |= {(mm, o, n) for o, n in sup_pred}
        all_gold |= {(mm, o, n) for o, n in sup_gold}
        print(f"{mm}: supersedes P={m['precision']} R={m['recall']} "
              f"(pred {m['n_pred']}, gold {m['n_gold']}, tp {m['tp']}; "
              f"{m['n_rejected']} rejected)", flush=True)
    return {"per_meeting": per, "pooled": prf(all_pred, all_gold)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3.5:2b")
    ap.add_argument("--num-ctx", type=int, default=32768)
    args = ap.parse_args(argv)
    root = repo_root()
    for mm in SPENT_TEST:
        if mm in SPLITS.get("train", []) or mm in SPLITS.get("sel", []):
            raise SystemExit("split discipline broken: spent test inside train/sel")

    # The scored set is defined by the temporal GOLD, not by SPLITS: D-044
    # expanded train to 9 meetings but only the original 5 (+ sel 047c) carry
    # labelled relation pairs. A meeting without gold cannot be scored.
    gold = load_gold(root, "train") | load_gold(root, "sel")
    meetings = sorted(gold)
    if not set(meetings) <= set(SPLITS["train"]) | set(SPLITS["sel"]):
        raise SystemExit("temporal gold contains a meeting outside train/sel")
    segs_all = _load(root)
    run_dir = root / "runs" / "llm_linker"
    raw_dir = run_dir / f"raw-{date.today()}-{args.model.replace(':', '-')}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out: dict = {"date": date.today().isoformat(), "model": args.model,
                 "mechanism": "whole-meeting zero-shot proposal of typed "
                              "relations by segment index (ollama, temp 0, "
                              "format=json)",
                 "bars": {"G1_pooled_supersedes": BARS}}
    res = run_meetings(args.model, meetings, root, gold, segs_all,
                       args.num_ctx, raw_dir)
    out["train_sel"] = res
    p, r = res["pooled"]["precision"], res["pooled"]["recall"]
    g1 = p >= BARS["precision"] and (r or 0) >= BARS["recall"]
    out["bars"]["G1_pooled_supersedes"]["cleared"] = g1
    verdict = "refused"
    if g1:
        ho = sorted(SPLITS["holdout"])
        ho_gold = load_gold(root, "holdout")
        ho_res = run_meetings(args.model, ho, root, ho_gold, segs_all,
                              args.num_ctx, raw_dir)
        hp, hr = ho_res["pooled"]["precision"], ho_res["pooled"]["recall"]
        g2 = hp >= BARS["precision"] and (hr or 0) >= BARS["recall"]
        out["holdout"] = {"read": "once", **ho_res,
                          "G2_cleared": g2}
        verdict = "pass" if g2 else "refused"
        out["verdict"] = ("CLEARED G1+G2" if g2 else
                          "REFUSED at G2 — sel/train gain did not transfer "
                          "to the sealed holdout")
    else:
        out["verdict"] = ("REFUSED at G1 — whole-session LLM proposals did "
                          "not clear precision>=0.70 / recall>=0.30 pooled on "
                          "train+sel; the sealed holdout was never read")
    pth = run_dir / f"result-{out['date']}-{args.model.replace(':', '-')}-{verdict}.json"
    pth.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pooled_train_sel": res["pooled"], "G1": g1,
                      "verdict": out["verdict"]}, indent=1))
    print("wrote", pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
