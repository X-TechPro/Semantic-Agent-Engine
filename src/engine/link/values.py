"""Deterministic value/change signals for real (ASR) French.

The linker's guards compare *values* (`supersede.comparable_values`), but on real
speech the value is often a spoken number ("le onze", "vingt et un juin") or a
bare contrast ("non pas maître gims"), and `regex_dates` only reads digits. This
module supplies the missing surface layer so the guards can see a change at all.

Everything here is a signal, never an authority: `supersede.evaluate` still
decides, and a witness only substitutes for a missing global subject key — it
never overrides an existing one (D-046).
"""
from __future__ import annotations

import re

from engine.slots import subjects as S

#: 1-31 as said in French meetings. Compound forms are matched longest-first so
#: "vingt et un" cannot be consumed as "vingt".
_SPOKEN: dict[int, tuple[str, ...]] = {
    1: ("un", "une"), 2: ("deux",), 3: ("trois",), 4: ("quatre",), 5: ("cinq",),
    6: ("six",), 7: ("sept",), 8: ("huit",), 9: ("neuf",), 10: ("dix",),
    11: ("onze",), 12: ("douze",), 13: ("treize",), 14: ("quatorze",),
    15: ("quinze",), 16: ("seize",), 17: ("dix sept", "dix-sept"),
    18: ("dix huit", "dix-huit"), 19: ("dix neuf", "dix-neuf"),
    20: ("vingt",), 21: ("vingt et un", "vingt-et-un"),
    22: ("vingt deux", "vingt-deux"), 23: ("vingt trois", "vingt-trois"),
    24: ("vingt quatre", "vingt-quatre"), 25: ("vingt cinq", "vingt-cinq"),
    26: ("vingt six", "vingt-six"), 27: ("vingt sept", "vingt-sept"),
    28: ("vingt huit", "vingt-huit"), 29: ("vingt neuf", "vingt-neuf"),
    30: ("trente",), 31: ("trente et un", "trente-et-un"),
}

#: unambiguous as standalone tokens; "un"/"une" are deliberately absent (articles)
_SPOKEN_SAFE = {n: forms for n, forms in _SPOKEN.items() if n != 1}

_FORMS = sorted(
    ((form, n) for n, forms in _SPOKEN_SAFE.items() for form in forms),
    key=lambda t: -len(t[0]))
_SPOKEN_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(f) for f, _ in _FORMS) + r")(?!\w)")
_DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)?")

#: phrases that assert a change of state rather than restate one
CHANGE_CUE_RE = re.compile(
    r"finalement|au lieu|plut[oô]t|repouss|annul|report|retien|retenu|"
    r"d[ée]cid|valid|act[ée]|chang|on passe|passe [àa]|d[ée]cal|abandonn|"
    r"remplac|non pas|sinon|en fait|du coup on|on part|on reste|on prend|"
    r"c'est bon|ça sera|ce sera", re.IGNORECASE)

_NEG_RE = re.compile(r"\b(non|pas|plus|jamais|aucun|sans)\b", re.IGNORECASE)

#: clause boundary for scope tests. Deliberately not "et": it is too common to
#: split on, and a negation before "et" usually still governs the first value.
_CLAUSE_SPLIT = re.compile(
    r"(?:[.;!?,]|\bmais\b|\bdonc\b|\balors\b|\bparce que\b|\bcar\b|\bpuis\b|"
    r"\bensuite\b|\bdu coup\b)", re.IGNORECASE)

#: Negative cues, in increasing ambiguity. `ne ... pas/plus/...` is the reliable
#: form; verbless ASR forms ("pas de budget", "pas encore validé") need a
#: following determiner/participle. Bare "plus" is NOT here: "plus de budget"
#: can mean *more* budget. `sans doute`/`sans problème` are affirmative.
_NEG_PATTERNS = (
    re.compile(r"\b(?:ne\s+|n['’]\s*)(?:\w+\s+){0,6}"
               r"(?:pas|plus|jamais|rien|aucun(?:e)?s?|personne|gu[eè]re)\b",
               re.IGNORECASE),
    re.compile(r"\bnon\s+pas\b", re.IGNORECASE),
    re.compile(r"\bpas\s+(?:de|d['’]|du|des|un|une|le|la|les|ce|cette|ces|"
               r"encore|assez|trop|toujours|vraiment|pour|à|a|valid|act|"
               r"pr[ée]vu|possible|question|besoin)\b", re.IGNORECASE),
    re.compile(r"\bsans\s+(?!doute\b|probl[eè]me\b|compter\b|que\b)\w+",
               re.IGNORECASE),
    re.compile(r"(?<!si )\b(?:aucun(?:e)?s?|jamais|rien)\b", re.IGNORECASE),
    re.compile(r"\bnon\b(?!\s+pas)", re.IGNORECASE),
)

#: scaffolding that carries no topic/value on its own
SCAFFOLD: frozenset[str] = frozenset({
    "projet", "budget", "lancement", "date", "prestataire", "stack", "technique",
    "rapport", "tarif", "recrutement", "conformite", "perimetre", "strategie",
    "prevu", "finalement", "environnements", "risque", "marche", "francophones",
    "coup", "enfin", "voila", "donc", "euh", "ben", "ouais", "oui", "truc",
    "chose", "peut", "etre", "faire", "fait", "aussi", "apres", "avant", "quand",
    "comme", "plus", "bien", "tout", "tous", "toute", "dans", "avec", "pour",
    "mais", "elle", "nous", "vous", "ils", "elles", "cest", "est", "sont",
    "hein", "alors", "donc", "genre", "truc",
})

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(text: str) -> str:
    return S.fold(text or "").replace("-", " ").replace("\u2019", "'")


def spoken_numbers(text: str) -> frozenset[str]:
    """Number tokens as said: digits stay digits, French words become digits."""
    t = fold(text)
    out = set(_DIGITS_RE.findall(t))
    for m in _SPOKEN_RE.finditer(t):
        word = m.group(1)
        for form, n in _FORMS:
            if form == word:
                out.add(str(n))
                break
    return frozenset(out)


def normalise_numbers(text: str) -> str:
    """Rewrite spoken numbers as digits so `regex_dates` can read them."""
    def rep(m: re.Match) -> str:
        word = m.group(1)
        for form, n in _FORMS:
            if form == word:
                return str(n)
        return word
    return _SPOKEN_RE.sub(rep, fold(text))


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(fold(text))
            if len(w) > 3 and w not in SCAFFOLD}


def has_change_cue(text: str) -> bool:
    return bool(CHANGE_CUE_RE.search(text or ""))


def is_negated(text: str) -> bool:
    return bool(_NEG_RE.search(text or ""))


def _polarity_of(clause: str) -> str | None:
    if any(p.search(clause) for p in _NEG_PATTERNS):
        return "negative"
    # "plus de budget" without "ne" is genuinely ambiguous (more vs no more).
    # Refusing to label it is safer than inventing a contradiction.
    if re.search(r"\bplus\s+de\b", clause):
        return None
    return "affirmative"


def polarity(text: str, subject: str | None = None) -> str | None:
    """Affirmative / negative for the clause that carries this slot.

    `subject` is the resolved key (`atlas.budget`). When it maps to a known
    attribute, only the clause containing that attribute's surface form is
    tested, so "le budget est de 2 M€, mais la salle n'est pas prête" stays
    affirmative for `budget`. Without a match the whole text is used, which is
    the old sentence-level behaviour. `None` means genuinely undecidable.
    """
    hay = fold(text)
    if not hay.strip():
        return None
    if subject:
        attr = subject.rsplit(".", 1)[-1]
        forms = S.ATTRIBUTE_LEXICON.get(attr, [attr])
        needles = [fold(f) for f in forms if f]
        for clause in _CLAUSE_SPLIT.split(hay):
            if any(n in clause for n in needles):
                return _polarity_of(clause)
    return _polarity_of(hay)


def slot_signature(text: str) -> str | None:
    """The coarse slot a sentence can be *about*, from its own words.

    Attribute first (the existing 13-entry lexicon), then a typed value, then a
    bare number. Returns None when the sentence asserts nothing slot-shaped —
    the same "undecidable, do not guess" posture as `slots/rules.py`.
    """
    from engine.slots.regex_dates import parse_amount, parse_date
    from datetime import date

    attr = S.match_attribute(text)
    if attr:
        return f"a:{attr}"
    if parse_date(normalise_numbers(text), date(2000, 1, 1))[0]:
        return "date"
    if parse_amount(text):
        return "amount"
    if spoken_numbers(text):
        return "number"
    return None


def value_witness(new_text: str, old_text: str) -> str | None:
    """What kind of change separates two statements, using text alone.

    `attribute`/`date`/`amount` are strong (a real slot or typed value moves);
    `number`/`negation` are weak and callers must demand supporting context.
    """
    from datetime import date

    from engine.slots.regex_dates import parse_amount, parse_date

    an, ao = S.match_attribute(new_text), S.match_attribute(old_text)
    if ao and an == ao:
        return "attribute"
    ref = date(2000, 1, 1)
    dn = parse_date(normalise_numbers(new_text), ref)[0]
    do = parse_date(normalise_numbers(old_text), ref)[0]
    if do and dn and dn != do:
        return "date"
    mn, mo = parse_amount(new_text), parse_amount(old_text)
    if mo and mn and mn != mo:
        return "amount"
    sn, so = spoken_numbers(new_text), spoken_numbers(old_text)
    if so and sn and sn != so:
        return "number"
    if is_negated(new_text) != is_negated(old_text):
        return "negation"
    return None


def pair_slot_witness(new_text: str, old_text: str, *, min_share: int = 2) -> str | None:
    """A witness strong enough to stand in for a missing subject key.

    Weak witnesses (number, negation) are admitted only when the two sentences
    share at least `min_share` content words, so two unrelated numbers in one
    meeting do not become a replacement.
    """
    wit = value_witness(new_text, old_text)
    if wit in ("attribute", "date", "amount"):
        return wit
    if wit and len(content_words(new_text) & content_words(old_text)) >= min_share:
        return wit
    return None
