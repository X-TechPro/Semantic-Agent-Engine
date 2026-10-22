"""Property tests (BUILD_GUIDE 12.2) — invariants, not examples.

The store's guarantees are universal claims, so they get hypothesis-driven
checks rather than one hand-picked row each.
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import date, timedelta
from hypothesis import HealthCheck, given, settings, strategies as st

from engine.schemas import Slots, utcnow
from engine.store.sqlite_store import SQLiteStore
from helpers import make_edge, make_unit

pytestmark = [pytest.mark.property]

# units with distinct dates keep the "final valid set" order-invariant (12.2)
unit_st = st.builds(
    lambda i, day, amount, project, status: make_unit(
        f"Le budget est de {amount} €.", source_id=f"meeting-{project}",
        unit_id=f"u-{project}-{i}", source_date=date(2026, 1, 1) + timedelta(days=day),
        project=project, status=status, type="decision",
        slots=Slots(subject=f"{project}.budget", amount=str(amount))),
    i=st.integers(0, 5000),
    day=st.integers(0, 400),
    amount=st.integers(100, 999),
    project=st.sampled_from(["alpha", "beta", "gamma"]),
    status=st.sampled_from(["confirmed", "proposed"]),
)


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "p.db")
    yield s
    s.close()


def _vecs(units, dim=16, seed=7):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((len(units), dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestNeverDelete:
    @settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.lists(unit_st, min_size=1, max_size=12, unique_by=lambda u: u.unit_id))
    def test_superseding_never_removes_a_unit(self, tmp_path_factory, units):
        s = SQLiteStore(tmp_path_factory.mktemp("nd") / "p.db")
        try:
            s.add_units(units)
            before = {u.unit_id for u in s.all_units()}
            ordered = sorted(units, key=lambda u: u.source_date)
            for old, new in zip(ordered, ordered[1:]):
                if old.slots.subject != new.slots.subject or old.project != new.project:
                    continue
                s.mark_superseded(old.unit_id, new.unit_id, new.source_date)
            after = {u.unit_id for u in s.all_units()}
            assert before == after, "a unit disappeared"
            assert s.count()["units"] == len(units)
        finally:
            s.close()


class TestBiTemporalOrder:
    def test_valid_to_is_never_before_valid_from(self, store):
        u = make_unit("Le budget est de 100 €.", unit_id="u-1",
                      source_date="2026-05-05",
                      slots=Slots(subject="alpha.budget", amount="100"))
        store.add_units([u])
        newer = make_unit("Le budget est de 200 €.", unit_id="u-2",
                          source_date="2026-06-01",
                          slots=Slots(subject="alpha.budget", amount="200"))
        store.add_units([newer])
        out = store.mark_superseded("u-1", "u-2", newer.source_date)
        assert out.valid_to >= out.valid_from

    def test_store_rejects_an_illegal_valid_window(self, store):
        """valid_to < valid_from is refused on insert *and* on update (6.5)."""
        bad = make_unit("fenêtre illégale", unit_id="u-bad",
                        source_date="2026-05-05").model_copy(
            update={"valid_from": date(2026, 5, 5), "valid_to": date(2026, 1, 1)})
        with pytest.raises(ValueError):
            store.add_units([bad])
        assert store.get("u-bad") is None, "the rejected row must not be half-written"

        good = make_unit("fenêtre correcte", unit_id="u-good",
                         source_date="2026-05-05")
        store.add_units([good])
        with pytest.raises(ValueError):
            store.update_unit(good.model_copy(update={"valid_to": date(2026, 1, 1)}))
        assert store.get("u-good").valid_to is None, "the bad update must not persist"


class TestOneHeadPerSlot:
    """6.5: at most one *valid* confirmed unit per (project, subject_key), unless
    the pair is an explicit conflict — a same-date same-status clash is kept as a
    contradiction (6.8.3) and is therefore marked by a contradicts edge."""

    def test_chained_supersede_leaves_one_head(self, store):
        chain = [make_unit(f"Le budget est de {100 * (i + 1)} €.",
                           unit_id=f"u-c{i}", source_date=f"2026-0{i + 1}-10",
                           slots=Slots(subject="alpha.budget",
                                       amount=str(100 * (i + 1))))
                 for i in range(3)]
        store.add_units(chain)
        store.mark_superseded("u-c0", "u-c1", chain[1].source_date)
        store.mark_superseded("u-c1", "u-c2", chain[2].source_date)
        current = [u for u in store.history("alpha.budget") if u.is_current]
        assert [u.unit_id for u in current] == ["u-c2"]
        assert store.current("alpha.budget").unit_id == "u-c2"

    def test_conflict_is_the_only_way_to_get_two_heads(self, store):
        a = make_unit("Le budget est de 100 €.", unit_id="u-a",
                      source_date="2026-03-01",
                      slots=Slots(subject="alpha.budget", amount="100"))
        b = make_unit("Le budget est de 200 €.", unit_id="u-b",
                      source_date="2026-03-01",
                      slots=Slots(subject="alpha.budget", amount="200"))
        store.add_units([a, b])
        store.add_edges([make_edge("u-a", "u-b", "contradicts", 0.7)])
        current = [u for u in store.history("alpha.budget") if u.is_current]
        assert len(current) == 2, "an explicit conflict must survive both ways"
        assert store.edges_for("u-a", direction="from", relation="contradicts")


class TestRebuildIdempotence:
    @settings(max_examples=10, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.lists(unit_st, min_size=3, max_size=14, unique_by=lambda u: u.unit_id))
    def test_rebuild_returns_identical_top_k(self, tmp_path_factory, units):
        s = SQLiteStore(tmp_path_factory.mktemp("rb") / "p.db")
        try:
            s.add_units(units)
            vecs = _vecs(units, seed=11)
            s.add_vectors([u.unit_id for u in units], vecs)
            q = vecs[0]
            s.rebuild_index()
            first = s.neighbors(q, 5)
            s.rebuild_index()
            second = s.neighbors(q, 5)
            assert [u for u, _ in first] == [u for u, _ in second]
            assert all(abs(a - b) < 1e-6 for (_, a), (_, b) in zip(first, second))
        finally:
            s.close()


class TestEdgeIntegrity:
    def test_edge_to_an_unknown_unit_is_refused(self, store):
        u = make_unit("seule", unit_id="u-only")
        store.add_units([u])
        with pytest.raises(KeyError):
            store.add_edges([make_edge("u-only", "u-ghost", "supports")])
        with pytest.raises(KeyError):
            store.add_edges([make_edge("u-ghost", "u-only", "supersedes")])


class TestChainAcyclic:
    def test_supersede_chain_terminates_and_has_no_repeat(self, store):
        ids = [f"u-{i}" for i in range(4)]
        units = [make_unit(f"budget {i}", unit_id=uid,
                           source_date=f"2026-0{i + 1}-0{i + 1}",
                           slots=Slots(subject="alpha.budget", amount=str(i)))
                 for i, uid in enumerate(ids)]
        store.add_units(units)
        for old, new in zip(ids, ids[1:]):
            nxt = store.get(new)
            store.mark_superseded(old, new, nxt.source_date)
        chain = store.supersede_chain("u-0")
        assert chain == ids
        assert len(chain) == len(set(chain))

    def test_a_cycle_is_not_walked_twice(self, store):
        a = make_unit("a", unit_id="u-x", source_date="2026-01-01")
        b = make_unit("b", unit_id="u-y", source_date="2026-02-01")
        store.add_units([a, b])
        store.mark_superseded("u-x", "u-y", b.source_date)
        # hostile direct write: point the head back at the tail
        store.update_unit(store.get("u-y").model_copy(
            update={"superseded_by": "u-x", "valid_to": b.source_date}))
        chain = store.supersede_chain("u-x")
        assert len(chain) == len(set(chain)), f"cycle walked twice: {chain}"


class TestIngestionOrderInvariance:
    def test_final_valid_set_does_not_depend_on_ingestion_order(self, tmp_path):
        units = [make_unit(f"Le budget est de {100 * (i + 1)} €.",
                           unit_id=f"u-o{i}", source_date=f"2026-0{i + 2}-01",
                           project="alpha",
                           slots=Slots(subject="alpha.budget",
                                       amount=str(100 * (i + 1))))
                 for i in range(3)]
        results = []
        for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
            db = SQLiteStore(tmp_path / f"o{order[0]}{order[1]}{order[2]}.db")
            try:
                db.add_units([units[i] for i in order])
                ordered = sorted(units, key=lambda u: u.source_date)
                for old, new in zip(ordered, ordered[1:]):
                    db.mark_superseded(old.unit_id, new.unit_id, new.source_date)
                results.append(sorted(u.unit_id for u in db.all_units()
                                      if u.is_current))
            finally:
                db.close()
        assert results[0] == results[1] == results[2] == ["u-o2"]


class TestStoreCounts:
    def test_counts_aggregate_persisted_rows(self, store):
        u = make_unit("une unité", unit_id="u-n1")
        store.add_units([u])
        store.add_vectors(["u-n1"], np.ones((1, 8), dtype="float32"))
        store.add_edges([make_edge("u-n1", "u-n1", "supports")])
        c = store.count()
        assert c["units"] == 1 and c["vectors"] == 1 and c["edges"] == 1
        assert c["sentences"] == 1

    def test_reingesting_the_same_unit_is_idempotent(self, store):
        u = make_unit("encore", unit_id="u-dup")
        store.add_units([u])
        store.add_units([u])
        store.add_units([u])
        assert store.count()["units"] == 1
