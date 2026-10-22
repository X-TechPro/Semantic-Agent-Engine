"""The NLI head's scope IS the safety property (D-023), so it is pinned here.

Measured reason the head exists at all (D-022): on XNLI-fr the rules fall-through
sits at chance (contra-vs-all AUC 0.4544; rules answered `unrelated` on
4906/5010 pairs) while the fine-tuned head reaches test macro-F1 0.7986. The
danger of wiring it is the one 6.11 names: a model that *authorises* structure.
These tests assert the head is asked **only** on the text-only fall-through, is
clamped to the three text relations, and can neither veto a slot-agreed
judgement nor mint a `supersedes` — so `supersede.evaluate`'s inputs cannot move
and nli-vs-rules stays a clean ablation.

No torch, no checkpoint: the head is a stub with `.relate()`. `build_linker`
rows use paths that cannot exist (or a monkeypatched loader) so the real GPU
weights never load inside the unit tier.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.config import DEFAULTS, repo_root
from engine.link.rules import RuleLinker
from engine.link.supervised import NLILinker, build_linker
from engine.schemas import Slots, Span, Unit, utcnow

pytestmark = pytest.mark.unit


class _Head:
    """Stub head: always answers `rel`, records every consultation."""

    def __init__(self, rel: str):
        self.rel = rel
        self.calls: list[tuple[str, str]] = []

    def relate(self, new_text: str, old_text: str) -> str:
        self.calls.append((new_text, old_text))
        return self.rel


def _u(uid: str, text: str, *, project="atlas", subject=None,
       amount=None, type="fact", status="confirmed") -> Unit:
    return Unit(
        unit_id=uid, source_id=f"s-{uid}", source_type="meeting",
        source_date=date(2026, 2, 1), ingested_at=utcnow(), text=text,
        sentences=[Span.whole(text)], project=project, kind="info",
        type=type, status=status, slots=Slots(subject=subject, amount=amount))


def _linker(rel: str = "contradicts") -> tuple[NLILinker, _Head]:
    head = _Head(rel)
    return NLILinker(head, topical_min=0.0), head


DISJOINT = ("Le déploiement a glissé à juin",
            "La migration du cluster est reportée")


# --- the head's one permitted surface -----------------------------------------
def test_head_answers_only_the_text_only_fallthrough():
    link, head = _linker("contradicts")
    new, old = _u("n1", DISJOINT[0]), _u("o1", DISJOINT[1])
    assert link.propose(new, old) == "contradicts"
    # orientation is pinned: the call carries (new, old) exactly as frbench
    # scored the pair (premise = older, hypothesis = newer) when it read test
    assert head.calls == [(DISJOINT[0], DISJOINT[1])]
    assert link.head_routes == [("n1", "o1", "unrelated", "contradicts")]


def test_head_output_is_clamped_to_the_three_text_relations():
    # a head that hallucinates `supersedes` must not mint one (6.11); the pair
    # keeps rules' fall-through answer and the audit still records the attempt
    link, head = _linker("supersedes")
    new, old = _u("n2", DISJOINT[0]), _u("o2", DISJOINT[1])
    assert link.propose(new, old) == "unrelated"
    assert head.calls and link.head_routes[0][3] == "supersedes"


# --- what the head may NEVER touch ---------------------------------------------
def test_cross_project_refusal_is_not_delegated():
    link, head = _linker("supports")          # head would agree, and must not be asked
    new, old = _u("n3", "On a signé le contrat", project="atlas"), \
               _u("o3", "On a signé le contrat", project="boreas")
    assert link.propose(new, old) == "unrelated"
    assert head.calls == []


def test_slot_agreement_is_never_delegated_in_either_direction():
    # (a) equal subject + identical value: rules says supports; a head shouting
    #     contradicts cannot veto the slot-ratified judgement
    link, head = _linker("contradicts")
    same = _u("n4", "Le budget est de 45 k€", subject="atlas.budget",
              amount="45 k€")
    old_same = _u("o4", "Le budget est de 45 k€", subject="atlas.budget",
                  amount="45 k€")
    assert link.propose(same, old_same) == "supports"
    assert head.calls == []
    # (b) equal subject + changed value: rules' supersedes proposal survives
    #     untouched — the head neither mints nor blocks it
    link2, head2 = _linker("unrelated")
    changed = _u("n5", "Le budget est de 80 k€", subject="atlas.budget",
                 amount="80 k€")
    assert link2.propose(changed, old_same) == "supersedes"
    assert head2.calls == []


def test_known_different_subjects_are_not_delegated():
    link, head = _linker("supports")
    new = _u("n6", "Le prestataire retenu est Nova Cloud",
             subject="atlas.vendor")
    old = _u("o6", "Le budget est de 45 k€", subject="atlas.budget")
    assert link.propose(new, old) == "unrelated"
    assert head.calls == []


def test_two_questions_never_reach_the_head():
    link, head = _linker("supports")
    new = _u("n7", "Où en est le déploiement ?", type="question")
    old = _u("o7", "Le budget est-il encore à 45 k€ ?", type="question")
    assert link.propose(new, old) == "unrelated"
    assert head.calls == []


def test_structural_predicate_covers_every_rule_branch_by_name():
    link, _ = _linker()
    slot = _u("n8", "x", subject="atlas.budget")
    same = _u("o8", "x", subject="atlas.budget")
    assert link._structural(slot, same, "supports") is True       # same_slot
    assert link._structural(slot, same, "supersedes") is True     # not a text rel
    plain = _u("n9", DISJOINT[0])
    plain_old = _u("o9", DISJOINT[1])
    assert link._structural(plain, plain_old, "unrelated") is False


def test_linker_link_path_is_inherited_unchanged():
    """Only propose() is overridden: the guard pipeline in RuleLinker.link is
    the authority, and a copy of it here would be a second table to drift."""
    assert NLILinker.link is RuleLinker.link


# --- build_linker honesty --------------------------------------------------------
def _cfg(**over):
    return {"link": {**DEFAULTS["link"], **over}}


def test_build_linker_nli_without_checkpoint_says_rules_fallback(tmp_path):
    linker, label = build_linker(_cfg(backend="nli",
                                      nli_model_dir=str(tmp_path / "absent")))
    assert isinstance(linker, RuleLinker) and not isinstance(linker, NLILinker)
    assert label.startswith("rules[fallback: no head at"), label


def test_build_linker_nli_with_broken_checkpoint_names_the_failure(tmp_path):
    d = tmp_path / "linker_nli"
    d.mkdir()
    (d / "weights.pt").write_bytes(b"not a checkpoint")  # meta.json missing
    linker, label = build_linker(_cfg(backend="nli", nli_model_dir=str(d)))
    assert isinstance(linker, RuleLinker)
    assert "unusable" in label and "FileNotFoundError" in label, label


def test_build_linker_nli_reports_the_head_when_it_loads(monkeypatch, tmp_path):
    from engine.link import nli_finetune

    monkeypatch.setattr(nli_finetune.NLITextHead, "from_dir",
                        classmethod(lambda cls, path=None, device=None: _Head("supports")))
    linker, label = build_linker(_cfg(backend="nli",
                                      nli_model_dir=str(tmp_path / "linker_nli")))
    assert isinstance(linker, NLILinker)
    assert label == "nli-head(linker_nli)", label


def test_build_linker_default_is_rules_and_untouched():
    linker, label = build_linker({"link": dict(DEFAULTS["link"])})
    assert type(linker) is RuleLinker and label == "rules"


def test_nli_artifact_dir_is_suffixed_so_the_baseline_cannot_be_clobbered():
    """The stage runner writes `_nli`-suffixed dirs and the report refuses a
    non-rules `linker_backend` as its default row — both halves of artefact
    identity, checked without running an ingest."""
    import argparse
    from eval.report import _stage

    ns = argparse.ArgumentParser()
    ns.add_argument("--linker", default=None, choices=["rules", "logreg", "nli"])
    assert ns.parse_args(["--linker", "nli"]).linker == "nli"
    assert ns.parse_args([]).linker is None       # baseline invocations unchanged

    runs = repo_root() / "runs"
    row = _stage(runs, "linker")
    assert row is not None and row.get("linker_backend") in (None, "rules"), (
        "the report's default linker stage row must be the rules baseline")
