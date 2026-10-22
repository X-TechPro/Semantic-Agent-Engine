"""Deterministic French date/number parsing relative to `source_date` (6.3).

This is where temporal bugs hide, so it is table-tested (tests/unit/
test_slots_dates.py). Everything resolves *against the date the words were
said*, never against today — that is the bi-temporal rule of 2.2.

Accepted resolution orders (most specific first):
  explicit numeric (22 novembre / 5 janvier 2027 / 22/11) → month qualifier
  (fin décembre, mi-mars, courant avril) → relative weekday (jeudi prochain,
  d'ici vendredi) → relative span (dans 3 jours, la semaine prochaine) →
  deictic (hier, demain) → bare month (décembre) → bare day-of-month (le 22).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {
    "janvier": 1, "janv": 1, "février": 2, "fevrier": 2, "févr": 2, "fevr": 2,
    "mars": 3, "avril": 4, "avr": 4, "mai": 5, "juin": 6, "juillet": 7,
    "juil": 7, "août": 8, "aout": 8, "septembre": 9, "sept": 9,
    "octobre": 10, "oct": 10, "novembre": 11, "nov": 11, "décembre": 12,
    "decembre": 12, "déc": 12, "dec": 12,
}
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
             "août", "septembre", "octobre", "novembre", "décembre"]
WDAYS = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4,
         "samedi": 5, "dimanche": 6}
# French week convention: Monday is day 1, Sunday is day 7.
_ISO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

# Longest-first: an alternation is leftmost-first, so without the sort "nov."
# would work but "décembre" could be consumed as "déc" (measured: an abbreviated
# month fell through to the day-of-month rule and answered 2026-10-22 for
# "Le 22 nov. est la date", which 6.1 lists as a required case).
MONTHS_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_DAYS = {"jour": 1, "jours": 1, "semaine": 7, "semaines": 7, "mois": 30,
         "an": 365, "ans": 365}


def mk(year: int, month: int, day: int) -> date:
    """Clamped date: 31 février becomes the last valid day, never an exception."""
    last = calendar.monthrange(year, month)[1]
    return date(year, month, max(1, min(day, last)))


def roll_forward(d: date, ref: date) -> date:
    """A bare month/day means the *next* occurrence at or after `ref`."""
    if d >= ref:
        return d
    return mk(d.year + 1, d.month, d.day)


def next_weekday(ref: date, weekday: int, *, strict: bool = True) -> date:
    """Next occurrence of `weekday` after `ref` (strict: never the same day)."""
    delta = (_ISO[weekday] - _ISO[ref.weekday()]) % 7
    if delta == 0:
        delta = 7 if strict else 0
    return ref + timedelta(days=delta)


def add_months(ref: date, n: int) -> date:
    total = ref.month - 1 + n
    return mk(ref.year + total // 12, total % 12 + 1, ref.day)


def month_anchor(year: int, month: int, qualifier: str) -> date:
    """début -> 5th, mi -> 15th, fin/courant -> last day, plain -> 1st."""
    if qualifier in ("fin", "courant"):
        return mk(year, month, calendar.monthrange(year, month)[1])
    if qualifier == "mi":
        return mk(year, month, 15)
    if qualifier in ("début", "debut"):
        return mk(year, month, 5)
    return mk(year, month, 1)


# --- individual recognisers, each returning (iso, rule) or None -------------
def _try_iso(t: str, ref: date):
    """ISO first: `2026-11-03` must never be read by the day/month rule.

    Measured defect: the renderer emits due dates as ISO, and `_try_numeric`
    matched `26-11` inside `2026-11-03`, returning 2027-03-11 — an invented
    future date that would have reached a user's draft e-mail.
    """
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12:
        return None
    return mk(y, mo, d).isoformat(), "iso"


def _try_day_month(t: str, ref: date):
    m = re.search(rf"\b(?:le\s+|du\s+)?(\d{{1,2}})(?:er)?\s+({MONTHS_RE})\b", t)
    if not m:
        return None
    day, month = int(m.group(1)), MONTHS[m.group(2)]
    if not 1 <= day <= 31:
        return None
    return roll_forward(mk(ref.year, month, day), ref).isoformat(), "day_month"


def _try_month_day_year(t: str, ref: date):
    m = re.search(rf"\b({MONTHS_RE})\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?\b", t)
    if not m:
        return None
    month, day = MONTHS[m.group(1)], int(m.group(2))
    year = int(m.group(3)) if m.group(3) else ref.year
    return mk(year, month, day).isoformat(), "month_day_year"


def _try_numeric(t: str, ref: date):
    """22/11, 22-11-2027, 22.11. Comma is a decimal separator in French, so it
    is deliberately *not* a date separator here (1,5 M€ must not read as 1 May)."""
    m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b", t)
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    if m.group(3):
        year = int(m.group(3))
        year += 2000 if year < 100 else 0
        return mk(year, month, day).isoformat(), "numeric_ymd"
    return roll_forward(mk(ref.year, month, day), ref).isoformat(), "numeric_dm"


def _try_month_qualifier(t: str, ref: date):
    m = re.search(rf"\b(fin|début|debut|mi|courant)\s*[-\s]+({MONTHS_RE})\b", t)
    if not m:
        return None
    qual, month = m.group(1), MONTHS[m.group(2)]
    cand = month_anchor(ref.year, month, qual)
    if qual == "mi":
        cand = roll_forward(cand, ref)
    elif cand < ref:
        nxt = month_anchor(cand.year + 1, month, qual)
        cand = nxt
    return cand.isoformat(), "month_qualifier"


def _try_relative_weekday(t: str, ref: date):
    m = re.search(rf"\b({ '|'.join(WDAYS)})\s+(prochain|prochaine|dernier|dernière)\b", t)
    if not m:
        return None
    wd = WDAYS[m.group(1)]
    if m.group(2).startswith("dernier") or m.group(2) == "dernière":
        delta = (_ISO[ref.weekday()] - _ISO[wd]) % 7
        return (ref - timedelta(days=delta or 7)).isoformat(), "weekday_dernier"
    return next_weekday(ref, wd).isoformat(), "weekday_prochain"


def _try_d_ici(t: str, ref: date):
    m = re.search(rf"\bd['’]ici\s+(?:le\s+|fin\s+)?({ '|'.join(WDAYS)}|{MONTHS_RE})\b", t)
    if not m:
        return None
    tok = m.group(1)
    if tok in WDAYS:
        return next_weekday(ref, WDAYS[tok]).isoformat(), "d_ici_weekday"
    return roll_forward(mk(ref.year, MONTHS[tok], 1), ref).isoformat(), "d_ici_month"


def _try_in_n(t: str, ref: date):
    m = re.search(r"\bdans\s+(\d+)\s*(jours?|semaines?|mois|ans?)\b", t)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if unit.startswith("mois"):
        return add_months(ref, n).isoformat(), "dans_n_mois"
    return (ref + timedelta(days=n * _DAYS[unit])).isoformat(), f"dans_n_{unit.rstrip('s')}"


def _try_week_month_next(t: str, ref: date):
    if re.search(r"\b(?:la\s+)?semaine prochaine\b", t):
        return (ref + timedelta(days=7)).isoformat(), "semaine_prochaine"
    if re.search(r"\ble mois prochain\b", t):
        return add_months(ref, 1).isoformat(), "mois_prochain"
    if re.search(r"\bl['']année prochaine\b", t):
        return add_months(ref, 12).isoformat(), "annee_prochaine"
    if re.search(r"\b(?:cette|la) semaine[-\s]ci\b", t):
        return ref.isoformat(), "cette_semaine"
    return None


def _try_deictic(t: str, ref: date):
    if "avant-hier" in t:
        return (ref - timedelta(days=2)).isoformat(), "avant_hier"
    if "après-demain" in t:
        return (ref + timedelta(days=2)).isoformat(), "apres_demain"
    if re.search(r"\bhier\b", t):
        return (ref - timedelta(days=1)).isoformat(), "hier"
    if re.search(r"\baujourd'hui\b|\baujourd['’]hui\b", t):
        return ref.isoformat(), "aujourdhui"
    if re.search(r"\bdemain\b", t):
        return (ref + timedelta(days=1)).isoformat(), "demain"
    return None


def _try_bare_month(t: str, ref: date):
    # a month name not attached to a day number and not preceded by a qualifier
    if re.search(rf"\b(?:fin|début|debut|mi|courant)\s+({MONTHS_RE})\b", t):
        return None
    for m in re.finditer(rf"\b({MONTHS_RE})\b", t):
        month = MONTHS[m.group(1)]
        pre = t[max(0, m.start() - 4): m.start()]
        if re.search(r"\d\s*$", pre):
            continue
        return roll_forward(mk(ref.year, month, 1), ref).isoformat(), "bare_month"
    return None


def _try_day_of_month(t: str, ref: date):
    m = re.search(r"\ble\s+(3[01]|2\d|1\d|[1-9])\b", t)
    if not m:
        return None
    day = int(m.group(1))
    cand = mk(ref.year, ref.month, day)
    if cand < ref:
        y, mo = (ref.year + 1, 1) if ref.month == 12 else (ref.year, ref.month + 1)
        cand = mk(y, mo, day)
    return cand.isoformat(), "day_of_month"


_TRIERS = [_try_iso, _try_day_month, _try_month_day_year, _try_numeric,
           _try_month_qualifier,
           _try_relative_weekday, _try_d_ici, _try_in_n, _try_week_month_next,
           _try_deictic, _try_bare_month, _try_day_of_month]


def parse_date(text: str, source_date: date, *, as_of: date | None = None
               ) -> tuple[str | None, str]:
    """(ISO date or None, rule name). Most specific recogniser wins."""
    ref = as_of or source_date
    t = (text or "").lower().replace("\u2019", "'")
    if not t.strip():
        return None, "none"
    for fn in _TRIERS:
        got = fn(t, ref)
        if got:
            return got
    return None, "none"


# --- amounts ----------------------------------------------------------------
# Two measured defects fixed here:
#   * unit alternation was `k|m|...|milliards?`, and a regex alternation is
#     leftmost-first, so "2 milliards d'euros" matched the bare `m` -> 2,000,000
#     instead of 2,000,000,000. Longest form first.
#   * "12 000 €" (French thousands separator) matched only "12" -> 12.
_MONEY_RE = re.compile(
    r"(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?)"
    r"\s*(milliards?|millions?|milliers?|mds?|k|m)?"
    r"\s*(?:€|euros?|francs?)?", re.IGNORECASE)
_CURRENCY_HINT = re.compile(r"€|euros?|millions?\s+d['’]euros?|budg|co[ûu]t|tarif|prix",
                            re.IGNORECASE)
_SEP = str.maketrans({" ": "", "\u00a0": "", "\u202f": ""})


def parse_amount(text: str) -> str | None:
    """'1,5 M€' -> '1500000'; '240 k€' -> '240000'; '3 millions' -> '3000000'.

    Integer euros, so the minimal pair 1,5 M€ / 15 M€ is *not* equal — the
    linker's same-value test depends on that. A number with no money context
    ("30 personnes") returns None instead of pretending to be an amount.
    """
    t = (text or "").replace("\u00a0", " ").replace("\u202f", " ")
    m = _MONEY_RE.search(t)
    if not m:
        return None
    num_s, unit = m.group(1), (m.group(2) or "").lower()
    has_currency = bool(unit) or re.search(r"€|euros?|francs?", m.group(0),
                                            re.IGNORECASE) is not None
    if not has_currency and not _CURRENCY_HINT.search(t):
        return None
    num = float(num_s.translate(_SEP).replace(",", "."))
    if unit.startswith("milliard"):
        mult = 1_000_000_000.0
    elif unit.startswith("millier") or unit.startswith("k"):
        mult = 1_000.0
    elif unit:                      # "million(s)", "m", "md", "mds"
        mult = 1_000_000.0
    else:
        mult = 1.0
    return str(int(round(num * mult)))


def parse_due(text: str, source_date: date) -> tuple[str | None, str]:
    """`due` prefers deadline language, else falls back to any date in the unit."""
    t = (text or "").lower()
    m = re.search(r"(?:pour|avant|d['’]ici|échéance|au plus tard|deadline|jusqu'au)\s+"
                  r"([^.;]{1,40})", t)
    if m:
        iso, rule = parse_date(m.group(0), source_date)
        if iso:
            return iso, "deadline_" + rule
    return parse_date(t, source_date)


def format_fr(d: date) -> str:
    return f"{d.day} {MONTHS_FR[d.month - 1]} {d.year}"
