"""French date/amount parser table (BUILD_GUIDE 12.1: >= 60 cases).

Expectations follow the documented semantics in engine/slots/regex_dates.py:
  bare month -> first day of the next occurrence; début -> 5th; mi -> 15th;
  fin/courant -> last day; weekday words -> strictly after the reference date.
Every case resolves *against source_date*, never against today.
"""

from __future__ import annotations

import pytest
from datetime import date

from engine.slots.regex_dates import parse_amount, parse_date

pytestmark = pytest.mark.unit

REF = date(2026, 10, 1)          # a Thursday

# (text, source_date, expected ISO or None)
DATE_CASES: list[tuple[str, date, str | None]] = [
    # --- explicit day + month ---
    ("le 22 novembre", date(2026, 10, 1), "2026-11-22"),
    ("Le 22 nov. est la date", date(2026, 10, 1), "2026-11-22"),
    ("réunion le 5 janvier", date(2026, 10, 1), "2027-01-05"),
    ("point le 3 décembre", date(2026, 10, 1), "2026-12-03"),
    ("on se voit le 1er mars", date(2026, 10, 1), "2027-03-01"),
    ("le 31 octobre", date(2026, 10, 1), "2026-10-31"),
    # a day already passed this year rolls to next year
    ("le 15 janvier", date(2026, 6, 1), "2027-01-15"),
    ("le 2 juin", date(2026, 2, 1), "2026-06-02"),
    # --- month + day + year ---
    ("janvier 6, 2027", date(2026, 10, 1), "2027-01-06"),
    ("novembre 22 2026", date(2026, 10, 1), "2026-11-22"),
    # --- ISO ---
    ("échéance 2026-11-03", date(2026, 10, 1), "2026-11-03"),
    ("2027-01-06", date(2026, 10, 1), "2027-01-06"),
    # the regression this rule was added for: the day/month rule used to read
    # `26-11` inside `2026-11-03` and answer 2027-03-11
    ("Le budget est acté pour 2026-12-31", date(2026, 10, 1), "2026-12-31"),
    # --- numeric dm ---
    ("22/11", date(2026, 10, 1), "2026-11-22"),
    ("prévu le 03.01", date(2026, 10, 1), "2027-01-03"),
    ("2026-11-22 deadline", date(2026, 10, 1), "2026-11-22"),
    # --- month qualifiers ---
    ("fin décembre", date(2026, 10, 1), "2026-12-31"),
    ("fin mars", date(2026, 10, 1), "2027-03-31"),
    ("début janvier", date(2026, 10, 1), "2027-01-05"),
    ("mi-novembre", date(2026, 10, 1), "2026-11-15"),
    ("courant novembre", date(2026, 10, 1), "2026-11-30"),
    ("courant février", date(2026, 10, 1), "2027-02-28"),
    # --- bare month -> next occurrence, first day ---
    ("décembre", date(2026, 10, 1), "2026-12-01"),
    ("en janvier", date(2026, 10, 1), "2027-01-01"),
    ("le lancement est prévu en octobre", date(2026, 10, 1), "2026-10-01"),
    ("c'est pour mai", date(2026, 10, 1), "2027-05-01"),
    # --- relative weekdays (strictly after ref; ref is a Thursday) ---
    ("jeudi prochain", date(2026, 10, 1), "2026-10-08"),
    ("lundi prochain", date(2026, 10, 1), "2026-10-05"),
    ("vendredi prochain", date(2026, 10, 1), "2026-10-02"),
    ("mardi prochain", date(2026, 10, 1), "2026-10-06"),
    ("d'ici vendredi", date(2026, 10, 1), "2026-10-02"),
    ("d'ici lundi", date(2026, 10, 1), "2026-10-05"),
    ("d'ici la semaine prochaine", date(2026, 10, 1), "2026-10-08"),
    ("jeudi dernier", date(2026, 10, 8), "2026-10-01"),
    # --- spans ---
    ("dans 3 jours", date(2026, 10, 1), "2026-10-04"),
    ("dans 10 jours", date(2026, 10, 1), "2026-10-11"),
    ("dans 2 semaines", date(2026, 10, 1), "2026-10-15"),
    ("dans 1 mois", date(2026, 10, 1), "2026-11-01"),
    ("dans 3 mois", date(2026, 11, 30), "2027-02-28"),
    ("la semaine prochaine", date(2026, 10, 1), "2026-10-08"),
    ("le mois prochain", date(2026, 10, 31), "2026-11-30"),
    ("l'année prochaine", date(2026, 10, 1), "2027-10-01"),
    # --- deictic ---
    ("hier", date(2026, 10, 2), "2026-10-01"),
    ("demain", date(2026, 10, 2), "2026-10-03"),
    ("aujourd'hui", date(2026, 10, 2), "2026-10-02"),
    ("avant-hier", date(2026, 10, 5), "2026-10-03"),
    ("après-demain", date(2026, 10, 5), "2026-10-07"),
    # --- day of month ---
    ("le 22", date(2026, 10, 1), "2026-10-22"),
    ("le 5", date(2026, 10, 8), "2026-11-05"),
    ("le 31", date(2026, 1, 15), "2026-01-31"),
    # --- no date at all ---
    ("Le budget est validé.", date(2026, 10, 1), None),
    ("bientôt", date(2026, 10, 1), None),
    ("prochainement on verra", date(2026, 10, 1), None),
    ("on en parle", date(2026, 10, 1), None),
    ("Marie s'occupe du dossier", date(2026, 10, 1), None),
    # decimals must NOT be read as dates (the 1,5 M€ trap)
    ("Le budget est de 1,5 M€.", date(2026, 10, 1), None),
    ("hausse de 2,3 %", date(2026, 10, 1), None),
    # --- the same words against a different reference date ---
    ("jeudi prochain", date(2026, 10, 4), "2026-10-08"),
    # "fin décembre" said on 20 Dec is the 31st of *this* year, not next year
    ("fin décembre", date(2026, 12, 20), "2026-12-31"),
    ("décembre", date(2026, 12, 20), "2027-12-01"),
    ("le 22 novembre", date(2026, 11, 25), "2027-11-22"),
    ("dans 3 jours", date(2026, 12, 30), "2027-01-02"),
]

assert len(DATE_CASES) >= 60, "12.1 requires >= 60 date cases"


@pytest.mark.parametrize("text,ref,expected", DATE_CASES,
                         ids=[f"{i}:{c[0][:18]}" for i, c in enumerate(DATE_CASES)])
def test_date_resolution(text, ref, expected):
    got, rule = parse_date(text, ref)
    assert got == expected, f"{text!r} @ {ref} -> {got} ({rule}), expected {expected}"


AMOUNT_CASES = [
    ("Le budget est de 1,5 M€.", "1500000"),
    ("budget de 15 M€", "15000000"),
    ("240 k€", "240000"),
    ("450 k€", "450000"),
    ("on met 3,2 M€", "3200000"),
    ("3 millions d'euros", "3000000"),
    ("950 k€", "950000"),
    ("180 k€", "180000"),
    ("Le tarif est de 12 000 €", "12000"),
    ("2 milliards d'euros", "2000000000"),
    # a bare number with no money context is not an amount
    ("30 personnes étaient là", None),
    ("le 22 novembre", None),
    ("2 équipes", None),
]


@pytest.mark.parametrize("text,expected", AMOUNT_CASES)
def test_amount_normalisation(text, expected):
    assert parse_amount(text) == expected, text


def test_minimal_pair_amounts_are_not_equal():
    """12.3: 1,5 M€ vs 15 M€ must not collide in the same-value test."""
    assert parse_amount("1,5 M€") != parse_amount("15 M€")


def test_minimal_pair_dates_are_not_equal():
    """12.3: 22 novembre vs 2 novembre."""
    a = parse_date("le 22 novembre", REF)[0]
    b = parse_date("le 2 novembre", REF)[0]
    assert a != b


def test_same_text_different_reference_gives_different_date():
    """Bi-temporal rule: resolution is relative to when it was said.

    Uses a bare month, where roll-forward genuinely changes the year. The
    earlier version of this test used "fin décembre" at two references, but
    both correctly resolve to 2026-12-31 — the assertion was wrong, not the
    parser.
    """
    early = parse_date("décembre", date(2026, 10, 1))[0]
    late = parse_date("décembre", date(2026, 12, 20))[0]
    assert early == "2026-12-01" and late == "2027-12-01"
    assert early != late
    # and a qualifier inside the same month stays on this year's last day
    assert parse_date("fin décembre", date(2026, 12, 20))[0] == "2026-12-31"
