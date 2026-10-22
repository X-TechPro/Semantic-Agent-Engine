"""Post-hoc verification of rendered output (6.11 `agent/verify.py`).

Deterministic first, model second:
  1. every claim must cite >= 1 unit id (schema-enforced, re-checked here)
  2. every **name / date / number** appearing in the output must appear in the
     cited evidence (exact or accent-folded fuzzy match)
  3. every e-mail address must exist in the contacts directory

Anything else is an *unsupported claim*; the caller regenerates once and then
falls back to the template renderer. This module is what produces the
unsupported-claim rate metric in 9.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from engine.schemas import Unit
from engine.slots.subjects import fold

_NAME_RE = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ'’-]{2,}\b")
_NUM_RE = re.compile(r"\b\d[\d\s,./]*\d\b|\b\d+\b")
_DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
           "septembre", "octobre", "novembre", "décembre")
_STOP_NAMES = {"Le", "La", "Les", "Bonjour", "Attention", "Finalement", "C", "On",
               "Nous", "Vous", "Il", "Elle", "Ce", "Cette", "Du", "Des", "En"}


@dataclass
class VerifyResult:
    ok: bool
    unsupported_names: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)

    @property
    def unsupported_claim_rate(self) -> float:
        if not self.unsupported_claims and not (self.unsupported_names
                                                or self.unsupported_numbers):
            return 0.0
        total = max(1, len(self.unsupported_claims))
        return round(min(1.0, (len(self.unsupported_names)
                               + len(self.unsupported_numbers)) / total), 4)


def evidence_text(units: list[Unit]) -> str:
    """Everything a rendered answer may legitimately contain.

    6.11 requires provenance *in the output* (source, date, speaker, status), so
    those fields are part of the allowed evidence — checking only `unit.text`
    made the verifier reject the very dates it asked the renderer to print
    (measured: `numbers=['01','05','10','12','2026']` from "2026-10-05").
    Slot values count too: an extracted due date is derived from the sentence.
    """
    parts: list[str] = []
    for u in units:
        bits = [u.text, u.source_date.isoformat(), u.source_id]
        for extra in (u.speaker, u.project, u.type, u.status,
                      u.slots.owner, u.slots.subject, u.slots.date_value,
                      u.slots.amount, u.slots.due, u.slots.action):
            if extra:
                bits.append(str(extra))
        bits += list(u.slots.persons)
        parts.append(" ".join(bits))
    return fold(" ".join(parts))


#: words the *renderer* owns — section labels and connective scaffolding. These
#: are not claims about the world, so they are not "unsupported names".
SCAFFOLD = {
    "contexte", "historique", "décisions", "decision", "fait", "faits",
    "responsables", "références", "récapitulatif", "chronologie", "point",
    "points", "suivant", "ouverture", "ouvertures", "note", "noter",
    "bonjour", "cordialement", "merci", "sujet", "demande", "version",
    "atelier", "proposition", "propositions", "hypothèse", "hypothèses",
    "condition", "condition", "validé", "validée", "annulé", "acté",
    "remplacé", "vigueur", "éléments", "cir-dessus", " Confirmation",
    "à", "au", "du", "des", "un", "une", "les", "la", "le", "de", "l",
}


def check_output(output_text: str, units: list[Unit], *,
                 contacts: dict[str, str] | None = None,
                 extra_allow: list[str] | None = None) -> VerifyResult:
    ev = evidence_text(units)
    allowed = set(extra_allow or [])
    allowed_scaffold = {fold(w) for w in SCAFFOLD}
    names: list[str] = []
    for n in set(_NAME_RE.findall(output_text)):
        if n in _STOP_NAMES or fold(n) in allowed_scaffold:
            continue
        if fold(n) not in ev and fold(n) not in {fold(a) for a in allowed}:
            names.append(n)
    numbers: list[str] = []
    for num in set(_NUM_RE.findall(output_text)):
        compact = fold(num).replace(" ", "")
        if compact in ev.replace(" ", "") or _month_near(num, ev):
            continue
        if any(fold(num) in fold(a) for a in allowed):
            continue
        numbers.append(num)
    emails = sorted(set(_EMAIL_RE.findall(output_text)))
    bad_emails = []
    if contacts is not None:
        known = {e.lower() for e in contacts.values() if e}
        bad_emails = [e for e in emails if e.lower() not in known]
    claims = [ln for ln in _sentences(output_text)]
    unsupported_claims = [c for c in claims if not cites_unit(c)]
    ok = not names and not bad_emails and not numbers
    return VerifyResult(ok=ok, unsupported_names=sorted(names),
                        unsupported_numbers=sorted(numbers),
                        unsupported_claims=unsupported_claims,
                        emails=emails)


def cites_unit(text: str) -> bool:
    return bool(re.search(r"u-[0-9a-f]+", text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _month_near(num: str, ev: str) -> bool:
    """A rendered date like "5 janvier 2027" is not a bare number match."""
    return any(m in fold(num) for m in _MONTHS) and bool(ev)


def verify_agent_output(payload: dict, units: list[Unit],
                        contacts: dict[str, str] | None = None
                        ) -> VerifyResult:
    """Walk a payload dict and verify all human-readable fields at once."""
    texts: list[str] = []
    _collect_text(payload, texts)
    all_units = units
    out = None
    for t in texts:
        r = check_output(t, all_units, contacts=contacts)
        out = r if out is None else _merge(out, r)
    return out or VerifyResult(ok=True)


def _collect_text(node, bucket: list[str]) -> None:
    if isinstance(node, str):
        if len(node.strip()) > 3:
            bucket.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k in ("evidence", "path", "unit_ids", "source_unit_ids", "id"):
                continue
            _collect_text(v, bucket)
    elif isinstance(node, list):
        for v in node:
            _collect_text(v, bucket)


def _merge(a: VerifyResult, b: VerifyResult) -> VerifyResult:
    return VerifyResult(
        ok=a.ok and b.ok,
        unsupported_names=sorted(set(a.unsupported_names) | set(b.unsupported_names)),
        unsupported_numbers=sorted(set(a.unsupported_numbers) | set(b.unsupported_numbers)),
        unsupported_claims=a.unsupported_claims + b.unsupported_claims,
        emails=sorted(set(a.emails) | set(b.emails)))
