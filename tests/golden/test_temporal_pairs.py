"""The 12.3 temporal minimal pairs, as executable golden tests.

These are the scenarios the guide marks "never drop": the December -> January
launch update, forward *and* reverse ingestion order, cancel/reinstate, and the
hypothesis-after-decision case. A rule set that improves a precision metric by
vetoing these is worse than useless, which is what the kill test caught.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.config import load_config
from engine.engine import Engine
from engine.link.supersede import evaluate
from engine.schemas import Query, Source, Turn
from helpers import make_unit

pytestmark = pytest.mark.golden

DEC = "Le lancement du projet Atlas est prévu en décembre."
JAN = "Finalement le lancement du projet Atlas passe en janvier."
Q = "Quand est le lancement du projet Atlas ?"


def _two_sources(tmp_path, name, ordered):
    """One source per speaking date — bi-temporal truth needs real dates."""
    db = tmp_path / f"{name}.db"
    # CPU-tier pin: the shipped cross-encoder (D-041) is a GPU model; golden
    # rulers score the deterministic path, the model is gated by make demo.
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": "hash"},
                                 "retrieve": {"rerank": {"backend": "none"}}})
    eng = Engine(cfg, path=db)
    for i, (speaker, text, d) in enumerate(ordered):
        eng.ingest(Source(source_id=f"{name}-{i}", source_type="meeting",
                          source_date=d, project="Atlas", default_speaker=speaker,
                          turns=[Turn(speaker=speaker, text=text, turn_index=0,
                                      project="Atlas")]))
    return eng


D1, D5 = date(2026, 10, 1), date(2026, 10, 5)


class TestLaunchDateScenario:
    def test_january_supersedes_december(self, tmp_path):
        eng = _two_sources(tmp_path, "fwd", [("Karim", DEC, D1),
                                              ("Karim", JAN, D5)])
        dec = next(u for u in eng.store.all_units() if DEC in u.text)
        jan = next(u for u in eng.store.all_units() if JAN in u.text)
        assert dec.superseded_by == jan.unit_id, (
            f"December not superseded (its status={dec.status!r} type={dec.type!r}); "
            f"the 12.3 flagship scenario must be replaced by the January unit")
        assert dec.valid_to == D5
        assert jan.is_current

    def test_query_returns_january_first(self, tmp_path):
        eng = _two_sources(tmp_path, "fwd2", [("Karim", DEC, D1),
                                              ("Karim", JAN, D5)])
        hits = eng.retrieve(Query(text=Q, project="Atlas"))
        assert hits, "expected retrieval hits"
        assert JAN in hits[0].unit.text, \
            f"rank-1 was {hits[0].unit.text!r}; the in-force unit must win"

    def test_reverse_ingestion_order_gives_the_same_answer(self, tmp_path):
        """12.3 out-of-order: ingest January first, December later (older doc)."""
        eng = _two_sources(tmp_path, "rev", [("Karim", JAN, D5),
                                             ("Karim", DEC, D1)])
        dec = next(u for u in eng.store.all_units() if DEC in u.text)
        jan = next(u for u in eng.store.all_units() if JAN in u.text)
        assert jan.is_current, "the newer statement must stay in force"
        assert not dec.is_current or dec.superseded_by is None or \
            dec.source_date < jan.source_date
        hits = eng.retrieve(Query(text=Q, project="Atlas"))
        assert hits and JAN in hits[0].unit.text

    def test_graph_expansion_returns_the_chain(self, tmp_path):
        eng = _two_sources(tmp_path, "chain", [("Karim", DEC, D1),
                                               ("Karim", JAN, D5)])
        hits = eng.retrieve(Query(text=Q, project="Atlas", mode="history"),
                            k=10)
        texts = " ".join(h.unit.text for h in hits)
        assert DEC in texts and JAN in texts, \
            "a history query must be able to surface both generations"


class TestGuardDoesNotVetoLegitimateSupersedes:
    def test_a_statement_without_status_words_is_still_replaceable(self):
        """The measured regression: gating on old.status == "mention" vetoed this
        pair entirely, making System B identical to Baseline A on the flagship
        scenario. Only a real answer/question may not be replaced."""
        old = make_unit(DEC, unit_id="u-dec", source_date="2026-10-01",
                        type="fact", status="mention")
        new = make_unit(JAN, unit_id="u-jan", source_date="2026-10-05",
                        type="decision", status="confirmed")
        from engine.schemas import Slots

        old = old.model_copy(update={"slots": Slots(subject="atlas.launch_date",
                                                    date_value="2026-12-01")})
        new = new.model_copy(update={"slots": Slots(subject="atlas.launch_date",
                                                    date_value="2027-01-01")})
        assert evaluate(new, old, "supersedes")[0] == "supersedes"

    def test_an_answer_is_not_replaced_by_a_later_decision(self):
        old = make_unit("Pour le budget, on est à 1,5 M€.", unit_id="u-ans",
                        source_date="2026-10-01", type="answer", status="confirmed")
        new = make_unit("Le budget est révisé à 2,4 M€.", unit_id="u-dec2",
                        source_date="2026-10-05", type="decision", status="confirmed")
        from engine.schemas import Slots

        old = old.model_copy(update={"slots": Slots(subject="atlas.budget",
                                                     amount="1500000")})
        new = new.model_copy(update={"slots": Slots(subject="atlas.budget",
                                                     amount="2400000")})
        assert evaluate(new, old)[0] != "supersedes"


class TestCancelThenReinstate:
    def test_cancel_then_reinstate_keeps_one_head(self, tmp_path):
        steps = [("Marie", "Le budget d'Atlas est validé à 1,5 M€.", date(2026, 10, 1)),
                 ("Sophie", "Finalement le budget d'Atlas est annulé.", date(2026, 10, 5)),
                 ("Karim", "Le budget d'Atlas revient à 1,5 M€.", date(2026, 10, 9))]
        eng = _two_sources(tmp_path, "rein", steps)
        heads = [u for u in eng.store.all_units() if u.is_current]
        assert len(heads) == 1, [u.text for u in heads]
        assert "revient" in heads[0].text
        eng.close()


class TestHypothesisAfterDecision:
    def test_hypothesis_never_replaces_the_decision(self, tmp_path):
        steps = [("Marie", "Le périmètre d'Atlas est validé: phase pilote.",
                  date(2026, 10, 1)),
                 ("Sophie", "On pourrait peut-être revoir le périmètre d'Atlas.",
                  date(2026, 10, 5))]
        eng = _two_sources(tmp_path, "hyp", steps)
        dec = next(u for u in eng.store.all_units() if "phase pilote" in u.text)
        hyp = next(u for u in eng.store.all_units() if "peut-être" in u.text)
        assert hyp.status == "hypothesis"
        assert dec.is_current, "a hypothesis must not supersede a confirmed decision"
        eng.close()
