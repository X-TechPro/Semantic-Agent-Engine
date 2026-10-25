"""Trained linker behind the same 2.2 interface (BUILD_GUIDE 6.8 baseline (b)).

`SupervisedLinker` replaces only the *proposal* step of `RuleLinker`. The
deterministic guards in `supersede.evaluate` still run inside `link()`, so a
model can never authorise a supersede the guards refuse (6.11).

Selection is config-driven: `link.backend: rules | logreg | nli`. If a model file
is absent or its feature order does not match, `build_linker` falls back to rules
and reports which one it gave you — a run must not be able to claim "trained"
while silently using rules.
"""

from __future__ import annotations

from pathlib import Path

from engine.config import repo_root
from engine.link.model import PairClassifier
from engine.link.rules import RuleLinker, _QUESTION_RE, propose as rules_propose
from engine.link.supersede import same_slot
from engine.schemas import Unit

DEFAULT_MODEL = "models/linker_lr.json"


class SupervisedLinker(RuleLinker):
    """Rules linker whose proposal step is a trained pair classifier."""

    name = "logreg"

    def __init__(self, classifier: PairClassifier, *, guards: bool = True,
                 min_score: float = 0.55, topical_min: float = 0.5,
                 equal_max: float = 0.8, word_only_supersedes: bool = True,
                 prefer_assertive_head: bool = True,
                 unknown_cardinality: str = "allow"):
        # `build_linker` passes the whole guard configuration to whichever linker
        # it builds. This signature accepted four of the six keys, so every call
        # with `link.backend: logreg` raised
        # `TypeError: __init__() got an unexpected keyword argument
        # 'word_only_supersedes'` — the trained linker could never be constructed,
        # and no measured run could have used it. The two guard flags also have to
        # reach `RuleLinker`: a model whose link stage runs with different guards
        # than the rules baseline is not the like-for-like comparison 6.8 asks for.
        super().__init__(guards=guards, min_score=min_score,
                         topical_min=topical_min, equal_max=equal_max,
                         word_only_supersedes=word_only_supersedes,
                         prefer_assertive_head=prefer_assertive_head,
                         unknown_cardinality=unknown_cardinality)
        self.clf = classifier
        self.last_scores: list[tuple[str, str, float, str]] = []

    def propose(self, new: Unit, old: Unit, *, topical_min: float = 0.5,
                equal_max: float = 0.8, thread: bool = False) -> str:
        base = rules_propose(new, old, topical_min=topical_min,
                             equal_max=equal_max, thread=thread)
        if base == "supersedes":
            # the rules already want it; the guard is still the authority
            self.last_scores.append((new.unit_id, old.unit_id, 1.0, "rules"))
            return base
        p = self.clf.score(new, old)
        self.last_scores.append((new.unit_id, old.unit_id, p, "model"))
        if self.clf.proposes_supersede(new, old):
            return "supersedes"
        return base


class NLILinker(RuleLinker):
    """Rules linker whose **text-only fall-through** is answered by a trained head.

    Measured reason (D-022): on XNLI-fr the shipped fall-through
    (`_lex_overlap > 0.45`) sits at chance (contra-vs-all AUC 0.4544, rules
    answered `unrelated` on 4906/5010) while the same encoder fine-tuned on
    XNLI-fr **train** reaches test macro-F1 0.7986 — and the shipped
    cross-encoder proved this is a polarity problem, not a data problem (it ranks
    contradictions *below* unrelated pairs), so a relevance score could not simply
    be wired in. This head is trained for exactly the three-way call.

    The scope is the safety property: the head speaks only where NO structural
    signal exists — same project, no slot agreement, no known-different subjects,
    no speech-act answer, not two questions. Structural answers and every
    `supersedes` proposal go through untouched, so `supersede.evaluate`'s inputs
    cannot move and nli-vs-rules stays a clean ablation (6.11). And there is one
    decision table, not two: this delegates to `rules_propose` rather than
    restating it — the reader-gold failure began by duplicating a table.
    """

    name = "nli"

    def __init__(self, head, *, guards: bool = True, min_score: float = 0.55,
                 topical_min: float = 0.5, equal_max: float = 0.8,
                 word_only_supersedes: bool = True,
                 prefer_assertive_head: bool = True,
                 unknown_cardinality: str = "allow"):
        super().__init__(guards=guards, min_score=min_score,
                         topical_min=topical_min, equal_max=equal_max,
                         word_only_supersedes=word_only_supersedes,
                         prefer_assertive_head=prefer_assertive_head,
                         unknown_cardinality=unknown_cardinality)
        self.head = head  # object with .relate(new_text, old_text) -> relation str
        #: (new_id, old_id, rules_base, head_answer) for every consultation —
        #: the audit needs to see when the head was asked at all
        self.head_routes: list[tuple[str, str, str, str]] = []

    def _structural(self, new: Unit, old: Unit, base: str) -> bool:
        """True when rules' answer came from structure, not from word overlap."""
        if base not in ("supports", "unrelated"):
            return True                       # supersedes / answers / creates_action
        if new.project and old.project and new.project != old.project:
            return True                       # deliberate cross-project refusal
        if same_slot(new, old):
            return True                       # slot logic incl. the ratification branch
        sn = (new.slots.subject or "").strip()
        so = (old.slots.subject or "").strip()
        if sn and so and sn != so:
            return True                       # line-403: different attribute = unrelated
        if _QUESTION_RE.search(new.text) and _QUESTION_RE.search(old.text):
            return True                       # two questions do not corroborate
        return False

    def propose(self, new: Unit, old: Unit, *, topical_min: float = 0.5,
                equal_max: float = 0.8, thread: bool = False) -> str:
        base = rules_propose(new, old, topical_min=topical_min,
                             equal_max=equal_max, thread=thread)
        if self._structural(new, old, base):
            return base
        rel = self.head.relate(new.text, old.text)   # premise = older, hyp = newer
        self.head_routes.append((new.unit_id, old.unit_id, base, rel))
        return rel if rel in ("supports", "contradicts", "unrelated") else base


def build_linker(cfg):
    """Return a linker per config, plus the backend actually in use."""
    link_cfg = dict(cfg["link"]) if "link" in cfg else {}
    common = dict(guards=bool(link_cfg.get("guards", True)),
                  min_score=float(link_cfg.get("supersede_min_score", 0.55)),
                  topical_min=float(link_cfg.get("value_topical_min", 0.0)),
                  equal_max=float(link_cfg.get("value_equal_max", 0.8)),
                  word_only_supersedes=bool(link_cfg.get("word_only_supersedes", True)),
                  prefer_assertive_head=bool(link_cfg.get("prefer_assertive_head",
                                                        True)),
                  unknown_cardinality=str(link_cfg.get("unknown_cardinality",
                                                       "allow")))
    want = str(link_cfg.get("backend", "rules")).lower()
    if want in ("logreg", "lr", "model", "supervised"):
        raw = link_cfg.get("model_path") or DEFAULT_MODEL
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root() / path
        clf = PairClassifier.load(path)
        if clf is not None:
            return SupervisedLinker(clf, **common), f"logreg({path.name})"
        return RuleLinker(**common), f"rules[fallback: no model at {path}]"
    if want in ("nli", "nli-head", "text"):
        raw = link_cfg.get("nli_model_dir") or "models/linker_nli"
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root() / path
        from engine.link import nli_finetune   # torch lives here; late import
        try:
            head = nli_finetune.NLITextHead.from_dir(path)
        except Exception as exc:  # noqa: BLE001 — the label names what broke
            return (RuleLinker(**common),
                    f"rules[fallback: {path.name} unusable: {type(exc).__name__}]")
        if head is None:
            return RuleLinker(**common), f"rules[fallback: no head at {path}]"
        return NLILinker(head, **common), f"nli-head({path.name})"
    return RuleLinker(**common), "rules"


__all__ = ["SupervisedLinker", "NLILinker", "build_linker", "DEFAULT_MODEL"]
