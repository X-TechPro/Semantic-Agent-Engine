"""Inject realistic ASR/diarization errors into the synthetic meeting corpora.

Three axes requested by the piscine mentor:
  * acronyms   - formatting variants, case, dropped letters, confusable swaps;
  * named entities - accents, typos, hyphen/space splits, case;
  * diarization - speaker swaps, mislabels, drops, merged labels, anonymous
                  SPEAKER_N labels for a whole meeting.

Message ids and gold evidence ids are never changed, so the same queries remain
scoreable. Segment boundaries (merge/split of turns) are not simulated.

Usage:
    python3 make_noisy_corpora.py --data-dir data --out-dir data/noisy --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")
ENTITY = re.compile(r"\b[A-ZÀ-ÖØ-Þ][a-zà-ÿ]+(?:[ -][A-ZÀ-ÖØ-Þ][a-zà-ÿ]+)*\b")
KEEP = {"OK", "OUI", "NON", "M", "MME", "M.", "NON", "OUI", "TV", "IA"}
ACCENT = str.maketrans({
    "à": "a", "â": "a", "ä": "a", "ç": "c", "é": "e", "è": "e", "ê": "e",
    "ë": "e", "î": "i", "ï": "i", "ô": "o", "ö": "o", "ù": "u", "û": "u",
    "ü": "u", "ÿ": "y", "ñ": "n", "æ": "ae", "œ": "oe",
    "À": "A", "Â": "A", "Ä": "A", "Ç": "C", "É": "E", "È": "E", "Ê": "E",
    "Ë": "E", "Î": "I", "Ï": "I", "Ô": "O", "Ö": "O", "Ù": "U", "Û": "U",
    "Ü": "U", "Ÿ": "Y", "Ñ": "N", "Æ": "AE", "Œ": "OE",
})


def deaccent(s: str) -> str:
    return s.translate(ACCENT)


def typo(s: str, rng: random.Random) -> str:
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.4:
        return s[:i] + s[i + 1:]
    if op < 0.8:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice("aeiouéèàbcdfgklmnprstvz") + s[i + 1:]


def acronym_map(ms: list[dict], qs: list[dict]) -> dict[str, str]:
    """Full product/org names -> their natural acronym (LRS, AN, ND...)."""
    phrases: set[str] = set()
    for m in ms:
        if m.get("title"):
            phrases.add(m["title"])
    for q in qs:
        for s in ENTITY.findall(q.get("query", "")):
            if " " in s:
                phrases.add(s)
    out = {}
    for p in phrases:
        words = [w for w in re.split(r"[\s-]+", p) if w]
        letters = "".join(w[0] for w in words if w[0].isupper())
        if 2 <= len(letters) <= 5 and len(words) > 1:
            out[p] = letters.upper()
    return out


def inject_acronyms(text: str, mapping: dict[str, str], rng: random.Random,
                    rate: float) -> tuple[str, int]:
    n = 0
    for full, acr in mapping.items():
        if full in text and rng.random() < rate:
            text = text.replace(full, acr)
            n += 1
    return text, n


def corrupt_acronyms(text: str, rng: random.Random, rate: float, pool: list[str]) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        s = m.group(0)
        if s in KEEP or rng.random() >= rate:
            return s
        n += 1
        op = rng.random()
        if op < 0.25:
            return " ".join(s)
        if op < 0.45:
            return ".".join(s) + "."
        if op < 0.60:
            return s.lower()
        if op < 0.75:
            return typo(s, rng)
        if pool:
            return rng.choice(pool)
        return s.lower()

    return ACRONYM.sub(repl, text), n


def corrupt_entities(text: str, rng: random.Random, rate: float,
                     entity_set: set[str]) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        s = m.group(0)
        # only real entities: names/titles/query entities or multi-word names,
        # not every capitalised sentence start
        if s in KEEP or rng.random() >= rate:
            return s
        if s not in entity_set and " " not in s and "-" not in s:
            return s
        n += 1
        op = rng.random()
        if op < 0.25:
            return deaccent(s)
        if op < 0.55:
            return typo(s, rng)
        if op < 0.75 and "-" in s:
            return s.replace("-", " ")
        if op < 0.90 and " " in s:
            return s.replace(" ", "-")
        return s[0].lower() + s[1:]
    return ENTITY.sub(repl, text), n


def corrupt_speakers(messages: list[dict], rng: random.Random,
                     scale: float = 1.0) -> Counter:
    counts = Counter()
    speakers = []
    for m in messages:
        sp = (m.get("speaker") or "").strip()
        if sp and sp.lower() not in {"inconnu", "speaker"} and sp not in speakers:
            speakers.append(sp)
    if not speakers:
        return counts

    # whole-meeting anonymisation: ASR gives SPEAKER_01.. instead of names
    anon = rng.random() < 0.25
    if anon:
        order = {sp: f"SPEAKER_{i:02d}" for i, sp in enumerate(speakers, 1)}
        for m in messages:
            m["speaker"] = order.get(m.get("speaker"), m.get("speaker"))
        counts["anonymised_meeting"] += 1
        speakers = list(order.values())

    for i, m in enumerate(messages):
        sp = m.get("speaker") or ""
        r = rng.random()
        if r < 0.06 * scale and i + 1 < len(messages):
            nxt = messages[i + 1].get("speaker") or ""
            m["speaker"], messages[i + 1]["speaker"] = nxt, sp
            counts["swap"] += 1
        elif r < 0.11 * scale:
            m["speaker"] = rng.choice([s for s in speakers if s != sp] or speakers)
            counts["mislabel"] += 1
        elif r < 0.15 * scale:
            m["speaker"] = "Inconnu"
            counts["drop"] += 1
        elif r < 0.17 * scale and sp:
            other = rng.choice([s for s in speakers if s != sp] or speakers)
            m["speaker"] = f"{sp} / {other}"
            counts["merged_label"] += 1
        elif r < 0.22 * scale and sp:
            m["speaker"] = deaccent(sp) if rng.random() < 0.5 else typo(sp, rng)
            counts["name_variant"] += 1
    return counts


def process(name: str, data_dir: Path, out_dir: Path, seed: int,
            acr_rate: float, ent_rate: float, diar_scale: float = 1.0) -> dict:
    rng = random.Random(seed + sum(map(ord, name)))
    meetings = json.loads((data_dir / f"{name}_meetings.json").read_text(encoding="utf-8"))
    queries = json.loads((data_dir / f"{name}_queries.json").read_text(encoding="utf-8"))

    ms = meetings["meetings"] if isinstance(meetings, dict) else meetings
    qs = queries["queries"] if isinstance(queries, dict) else queries

    titles = [m.get("title", "") for m in ms]
    speakers = sorted({(x.get("speaker") or "") for m in ms for x in m["messages"]})
    acr_pool = sorted({a for m in ms for x in m["messages"]
                       for a in ACRONYM.findall(x.get("text", ""))
                       if a not in KEEP and len(a) <= 5
                       and a not in speakers})
    entity_set = set(speakers)
    for t in titles:
        entity_set.update(ENTITY.findall(t))
        entity_set.update(w for w in re.findall(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ-]{2,}\b", t))
    for q in qs:
        entity_set.update(ENTITY.findall(q.get("query", "")))
    entity_set = {e for e in entity_set if len(e) > 1 and e not in KEEP}
    acr_map = acronym_map(ms, qs)
    acr_pool = sorted(set(acr_pool) | set(acr_map.values()))

    counts = Counter()
    for m in ms:
        counts.update(corrupt_speakers(m["messages"], rng, diar_scale))
        for x in m["messages"]:
            t, inj = inject_acronyms(x.get("text", ""), acr_map, rng, 0.35)
            t, a = corrupt_acronyms(t, rng, acr_rate, acr_pool)
            t, e = corrupt_entities(t, rng, ent_rate, entity_set)
            x["text"] = t
            counts["acronym_injected"] += inj
            counts["acronym"] += a
            counts["entity"] += e
            if x.get("speaker"):
                x["speaker"] = (deaccent(x["speaker"]) if rng.random() < 0.10
                                else x["speaker"])
                if rng.random() < 0.10:
                    x["speaker"] = typo(x["speaker"], rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": f"{name}_meetings.json",
        "seed": seed,
        "acronym_rate": acr_rate,
        "entity_rate": ent_rate,
        "diarization_scale": diar_scale,
        "counts": dict(counts),
        "test_queries": sum(1 for q in qs if q.get("split") == "test"),
    }
    if isinstance(meetings, dict):
        meetings = {**meetings, "meetings": ms, "corruption": meta}
    else:
        meetings = {"meetings": ms, "corruption": meta}
    (out_dir / f"{name}_meetings.json").write_text(
        json.dumps(meetings, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"{name}_queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/noisy"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--acronym-rate", type=float, default=0.5)
    ap.add_argument("--entity-rate", type=float, default=0.30)
    ap.add_argument("--diar-scale", type=float, default=1.0)
    args = ap.parse_args()
    for name in ("v8", "v9", "v10"):
        meta = process(name, args.data_dir, args.out_dir, args.seed,
                       args.acronym_rate, args.entity_rate, args.diar_scale)
        print(name, json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
