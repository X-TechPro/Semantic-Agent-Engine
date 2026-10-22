"""Concept attention (6.13 / M5) — the model, the losses, the job, the guards.

This package exists to be *refused* as much as to be run. Four of its five
modules were written after measurements that said the naive version is wrong, and
every one of those measurements is pinned here so it cannot regress quietly:

* the topic vector must keep the **base dimension** (retrieval embeds a query with
  the frozen encoder and searches the refined column; a different width would
  return nothing, looking like "no results" rather than a broken model);
* the head is a **residual with a structural cap**, because a soft drift penalty
  was measurably outbid (drift_p95 1.2238 against a 0.25 contract, cosine-to-base
  0.0105 at init);
* the **drift hinge has a NaN gradient at zero distance**, and zero distance is
  exactly what the zero-initialised residual produces at step 0;
* the refine job must **refuse to publish** an unanchored space, and the gate uses
  p95, not the mean (mean 0.1055 passed while max 0.5310 breached).

Runs offline: a hand-built store with hash vectors and a 32-dim, 1-layer model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from engine.concept import losses as L
from engine.concept.data import build_sets, edge_pairs, gap_days, label_counts
from engine.concept.model import (KINDS, SOURCE_TYPES, STATUSES, TYPES,
                                  ConceptAttention, ConceptConfig, log_gap_days)
from engine.concept.refine_job import clear_refined, refine
from engine.concept.train import CHECKPOINT_VERSION, _pad, _prefix, load, save, train
from engine.retrieve.dense import DenseRetriever
from engine.schemas import Slots
from helpers import make_edge, make_unit

pytestmark = pytest.mark.unit

DIM = 32


def _vecs(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, DIM), dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_store):
    """A hand-built corpus: a value chain, a proposed-vs-confirmed pair (the hard
    negative 6.13 names), and a second project for the split guards."""
    units = [
        make_unit("Le budget est à 1,8 M€.", unit_id="u-old", source_date="2026-01-05",
                  project="Atlas", type="fact", status="confirmed",
                  slots=Slots(subject="atlas.budget.value", amount="1,8 M€"),
                  superseded_by="u-new"),
        make_unit("Le budget est à 2,4 M€.", unit_id="u-new", source_date="2026-06-01",
                  project="Atlas", type="decision", status="confirmed",
                  slots=Slots(subject="atlas.budget.value", amount="2,4 M€")),
        make_unit("Proposition de budget à 2,4 M€.", unit_id="u-prop",
                  source_date="2026-05-01", project="Atlas", type="decision",
                  status="proposed",
                  slots=Slots(subject="atlas.budget.value", amount="2,4 M€")),
        make_unit("Le prestataire retenu est Nova Cloud.", unit_id="u-vendor",
                  source_date="2026-04-01", project="Atlas", type="decision",
                  status="confirmed", slots=Slots(subject="atlas.vendor.value")),
        make_unit("Le budget d'Iris est à 900 k€.", unit_id="u-iris",
                  source_date="2026-03-01", project="Iris", type="fact",
                  status="confirmed", slots=Slots(subject="iris.budget.value")),
    ]
    assert tmp_store.add_units(units) == len(units)
    ids = [u.unit_id for u in units]
    tmp_store.add_vectors(ids, _vecs(len(ids)), kind="base")
    tmp_store.add_edges([make_edge("u-new", "u-old", "supersedes", 0.9, "G1"),
                         make_edge("u-new", "u-vendor", "supports", 0.6)])
    return tmp_store


def _cfg(**kw):
    return ConceptConfig(dim=DIM, d_model=8, layers=1, heads=4,
                         max_context=kw.pop("max_context", 4), **kw)


def _batch(b=2, n=4, mask=None):
    base = F.normalize(torch.randn(b, n, DIM), dim=-1)
    return {"x": base, "src": torch.ones(b, n, dtype=torch.long),
            "kind": torch.ones(b, n, dtype=torch.long),
            "type": torch.ones(b, n, dtype=torch.long),
            "status": torch.ones(b, n, dtype=torch.long),
            "bucket": torch.ones(b, n, dtype=torch.long),
            "log_gap": torch.full((b, n), log_gap_days(31.0)),
            "pos": torch.arange(n).repeat(b, 1),
            "mask": mask if mask is not None else torch.ones(b, n, dtype=torch.int)}


# ------------------------------------------------------------------- the model

def test_refined_vector_keeps_the_base_dimension_and_is_unit_length():
    """6.13 stores refined vectors in the same index a base-space query searches."""
    m = ConceptAttention(_cfg())
    out = m(_batch())["topic"]
    assert out.shape[-1] == DIM
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_the_head_is_a_residual_so_it_starts_as_the_measured_baseline():
    """0.4: keep the baseline until the trained version beats it — literally, by
    making step 0 *be* the baseline. Measured: without the sum and with zero-init the
    model emitted all-zero vectors (cosine-to-base 0.000000, drift 1.0)."""
    m = ConceptAttention(_cfg())
    batch = _batch()
    with torch.no_grad():
        out = m(batch)
        cos = (out["topic"] * batch["x"]).sum(-1)
        drift = (out["topic"] - batch["x"]).norm(dim=-1)
    assert cos.min() > 0.999999
    assert drift.max() < 1e-4


def test_max_step_caps_how_far_a_vector_may_move_even_when_trained_hard():
    """The structural bound replaces the penalty that was measurably outbid."""
    cfg = _cfg(max_step=0.2)
    m = ConceptAttention(cfg)
    batch = _batch()
    opt = torch.optim.AdamW(m.parameters(), lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        out = m(batch)
        # push the delta as hard as possible in one arbitrary direction
        (-(out["topic"][:, 0] * torch.randn(2, DIM)).sum()).backward()
        opt.step()
    with torch.no_grad():
        drift = (m(batch)["topic"] - batch["x"]).norm(dim=-1)
    assert float(drift.max()) <= cfg.max_step + 0.02, f"drift escaped the cap: {drift.max()}"


def test_masked_slots_contribute_nothing_and_the_first_slot_is_never_masked():
    batch = _batch(mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.int))
    out = ConceptAttention(_cfg())(batch)
    assert torch.equal(out["topic_masked"][0, 2:], torch.zeros(2, DIM))
    assert out["keep"][0].tolist() == [1.0, 1.0, 0.0, 0.0]
    # a fully-masked row would NaN the attention softmax; slot 0 is always kept
    assert batch["mask"][:, 0].min() == 1


def test_forward_is_deterministic_and_validity_is_a_probability():
    m = ConceptAttention(_cfg())
    b = _batch()
    with torch.no_grad():                      # also keeps the suite warning-free
        a1, a2 = m(b), m(b)
    assert torch.equal(a1["topic"], a2["topic"])
    assert 0.0 <= float(a1["validity"].min()) and float(a1["validity"].max()) <= 1.0


def test_every_head_receives_a_gradient_including_the_state_vector():
    """6.13 hands `state` to the linker. Measured before the triplet term existed:
    `out_state.weight.grad is None`, i.e. the checkpoint shipped an untrained
    projection behind a documented input."""
    m = ConceptAttention(_cfg())
    batch = _batch()
    pos = torch.tensor([[1], [2]])
    neg = torch.tensor([[2], [3]])
    tgt = torch.full((2, 4), -1.0)
    tgt[:, 0] = 1.0
    tgt[:, 1] = 0.0
    out = m(batch)
    tot, _ = L.total({"contrastive": L.contrastive(out["topic"], batch["x"],
                                                  batch["mask"], pos, neg),
                      "validity": L.validity(out["validity"], tgt, batch["mask"]),
                      "incremental": L.incremental(out["topic"], out["topic"],
                                                   batch["mask"]),
                      "state": L.triplet(out["state"], batch["mask"], pos, neg),
                      "drift": L.drift(out["topic"], batch["x"], batch["mask"])},
                     dict.fromkeys(L.TERMS, 1.0))
    tot.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"heads trained by nothing: {missing}"


def test_index_of_gives_unknown_and_missing_labels_the_reserved_row():
    assert ConceptAttention.index_of(TYPES, "decision") == TYPES.index("decision") + 1
    assert ConceptAttention.index_of(TYPES, None) == 0
    assert ConceptAttention.index_of(TYPES, "not_a_label") == 0


def test_config_rejects_a_model_width_the_heads_cannot_divide():
    with pytest.raises(ValueError, match="not divisible by heads"):
        ConceptConfig(dim=DIM, d_model=30, heads=4)


def test_log_gap_is_signed_bounded_and_none_is_distinguishable():
    assert log_gap_days(31) > 0 and log_gap_days(-31) < 0
    assert log_gap_days(None) == 0.0
    assert log_gap_days(10000) < 10                      # log, not linear
    assert log_gap_days(0) == 0.0


# -------------------------------------------------------------------- losses

def test_contrastive_rewards_the_positive_and_penalises_the_hard_negative():
    """Orthonormal base rows make the arithmetic checkable by hand: one positive and
    one other candidate gives log(2) = 0.6931 when the centre sits on the negative,
    and 0.0 when it sits on the positive."""
    base = torch.eye(4, DIM).unsqueeze(0)
    mask = torch.ones(1, 4, dtype=torch.int)
    pos, neg = torch.tensor([[1]]), torch.tensor([[2]])

    def topic_on(i):
        t = base.clone()
        t[0, 0] = base[0, i]
        return t

    assert L.contrastive(topic_on(1), base, mask, pos, neg) == pytest.approx(0.0, abs=1e-6)
    assert L.contrastive(topic_on(2), base, mask, pos, neg) == pytest.approx(
        float(np.log(2)), abs=1e-4)


def test_losses_return_zero_rather_than_a_number_when_there_is_nothing_to_learn():
    base = torch.eye(4, DIM).unsqueeze(0)
    mask = torch.ones(1, 4, dtype=torch.int)
    empty = torch.tensor([[-1]])
    assert float(L.contrastive(base, base, mask, empty, torch.tensor([[1]]))) == 0.0
    unlabelled = torch.full((1, 4), -1.0)
    assert float(L.validity(torch.full((1, 4), 0.5), unlabelled, mask)) == 0.0
    assert float(L.triplet(base, mask, empty, torch.tensor([[1]]))) == 0.0


def test_validity_ignores_unlabelled_rows_instead_of_reading_minus_one_as_zero():
    base = torch.eye(4, DIM)
    tgt = torch.full((1, 4), -1.0)
    tgt[0, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.int)
    good = torch.full((1, 4), 0.95)
    bad = torch.full((1, 4), 0.05)
    assert float(L.validity(good, tgt, mask)) < float(L.validity(bad, tgt, mask))
    # a -1 target must never reach BCE: it would train toward 0 and invent labels
    assert float(L.validity(bad, torch.full((1, 4), -1.0), mask)) == 0.0


def test_triplet_excludes_the_centre_from_being_its_own_pair():
    base = torch.eye(4, DIM).unsqueeze(0)
    mask = torch.ones(1, 4, dtype=torch.int)
    # index 0 is the centre: passing it as the positive must contribute nothing
    assert float(L.triplet(base, mask, torch.tensor([[0]]), torch.tensor([[2]]))) == 0.0


def test_drift_is_zero_identical_and_finite_graded_at_exactly_zero_distance():
    """The NaN regression: `relu(sqrt(0) - max)` returns 0.0 forward and NaN
    backward (0 x inf), and the zero-initialised residual makes distance exactly 0 at
    step 0 — so without the epsilon every training run NaNs on its first backward."""
    base = torch.eye(4, DIM)
    mask = torch.ones(1, 4, dtype=torch.int)
    r = base.clone().requires_grad_(True)
    val = L.drift(r, base, mask, max_norm=0.25)
    assert float(val.detach()) == 0.0
    g, = torch.autograd.grad(val, r)
    assert torch.isfinite(g).all(), "sqrt-at-zero produced a non-finite gradient"


def test_drift_hinge_adds_the_excess_beyond_the_bound():
    a = torch.eye(4, DIM)
    b = F.normalize(torch.eye(4, DIM) + 0.6, dim=-1)
    mask = torch.ones(1, 4, dtype=torch.int)
    plain = float(L.drift(b, a, mask, max_norm=10.0))
    hinged = float(L.drift(b, a, mask, max_norm=0.05))
    assert hinged > plain


def test_total_reports_every_term_including_the_ones_weighted_off():
    """An ablation that turns a term off must still show its number, so a dropped
    row in the table cannot be mistaken for a term that was never measured."""
    parts = {"contrastive": torch.tensor(1.0), "validity": torch.tensor(2.0),
             "incremental": torch.tensor(3.0), "state": torch.tensor(4.0),
             "drift": torch.tensor(5.0)}
    tot, seen = L.total(parts, {"contrastive": 1.0, "drift": 0.0})
    assert float(tot) == 1.0
    assert seen["drift"] == 5.0 and seen["w_drift"] == 0.0
    assert seen["total"] == 1.0
    assert set(seen) == set(parts) | {f"w_{k}" for k in parts} | {"total"}


# ---------------------------------------------------------------- data builder

def test_build_sets_uses_the_stores_own_edges_and_labels(store):
    sets = build_sets(store, k=3)
    assert sets, "a store with vectors must produce context sets"
    for s in sets:
        assert s.ids[0] == s.centre_id, "the centre must be slot 0"
        assert s.n == len(s.ids) == s.base.shape[0]
    counts = label_counts(sets)
    assert counts["positives"] > 0, "the supersedes/supports edges must pair up"
    assert counts["hard_negatives"] > 0, (
        "u-prop is the same slot with a different status: the named hard negative")
    assert 0.0 < counts["validity_supervised_rate"] <= 1.0
    assert all(k in counts for k in ("sets", "sets_with_positive",
                                     "sets_with_hard_negative"))


def test_validity_targets_are_only_assertive_current_or_superseded(store):
    s = next(x for x in build_sets(store, k=4) if x.centre_id == "u-new")
    assert s.validity[s.pos_of["u-new"]] == 1.0          # current + confirmed
    assert s.validity[s.pos_of["u-old"]] == 0.0          # superseded
    # u-prop is proposed: neither in force nor superseded, so it must be unlabelled
    assert s.pos_of["u-prop"] not in s.validity


def test_restrict_to_an_unmatched_project_yields_no_sets(store):
    assert build_sets(store, k=2, restrict={"Nowhere"}) == []


def test_gap_days_and_edge_pairs_agree_with_the_store(store):
    pairs = edge_pairs(store)
    assert pairs[("u-new", "u-old")] == "supersedes"
    assert gap_days(__import__("datetime").date(2026, 6, 1),
                    __import__("datetime").date(2026, 1, 5)) == 147.0
    assert gap_days(__import__("datetime").date(2026, 6, 1), None) is None


# ------------------------------------------------------------------- training

def test_pad_and_prefix_agree_on_a_real_store(store):
    sets = build_sets(store, k=3)
    batch = _pad(sets)
    assert batch["x"].shape[0] == len(sets)
    assert batch["x"].shape[1] == max(s.n for s in sets)
    assert batch["positives"].min().item() in (-1, 1)     # padded with -1, never 0
    assert (batch["negatives"] != 0).all(), "0 is the centre: cannot be a fill value"
    short = _prefix(batch)
    assert short["x"].shape[1] == batch["x"].shape[1] - 1
    # a pair index that pointed at the dropped slot must become -1, not a stale row
    assert (short["positives"] < short["x"].shape[1]).all()
    assert (short["negatives"] < short["x"].shape[1]).all()


def test_train_reports_curves_and_selects_the_best_epoch_on_dev(store):
    rep = train(store, cfg=_cfg(), seed=42, epochs=2, batch_size=2, k_neighbors=2,
                train_projects={"Atlas"}, dev_projects={"Atlas"}, device="cpu")
    assert len(rep["curve"]) == 2
    assert rep["n_dev_sets"] > 0
    assert 0 <= rep["best_epoch"] < 2
    assert set(rep["weights"]) == set(L.TERMS)
    assert rep["embed_model_version"] == store.embed_version
    first, last = (rep["curve"][0]["train"]["total"], rep["curve"][-1]["train"]["total"])
    assert last <= first, f"train loss did not decrease ({first} -> {last})"
    assert "total" in rep["curve"][0]["dev"]


def test_train_refuses_a_split_name_that_matches_nothing(store):
    """Falling through to `restrict=None` would train on every project, and the
    curve would still look healthy."""
    with pytest.raises(ValueError, match="matched no stored units"):
        train(store, cfg=_cfg(), epochs=1, device="cpu", train_projects={"Nowhere"})
    with pytest.raises(ValueError, match="matched no stored units"):
        train(store, cfg=_cfg(), epochs=1, device="cpu", train_projects={"Atlas"},
              dev_projects={"Nowhere"})


def test_train_rejects_a_typo_in_the_loss_weights():
    """`total()` reads weights with .get(name, 0.0), so an unknown key would have
    silently disabled a term -- the inert-knob bug (D-011) in a config's clothes."""
    with pytest.raises(ValueError, match="unknown concept loss term"):
        train(None, cfg=_cfg(), weights={"driftt": 5.0})


def test_an_all_zero_weights_ablation_is_honoured_not_replaced_by_defaults(store):
    """A deliberate "every term off" run must stay off.

    This is a pin against a future rewrite to `weights or defaults`, not a claim
    that one was broken here: a dict of five zeros is non-empty, so truthiness
    already honoured it (measured). What would be silently swallowed is an empty
    dict written to mean "no terms", and that case is asserted below too.

    `total()` still reports every term's value at weight 0, so the ablation is
    auditable rather than looking like a run that produced zero loss.
    """
    rep = train(store, cfg=_cfg(), seed=42, epochs=1, batch_size=2, k_neighbors=2,
                weights=dict.fromkeys(L.TERMS, 0.0), device="cpu")
    assert all(v == 0.0 for v in rep["weights"].values()), rep["weights"]
    assert rep["curve"][0]["train"]["total"] == 0.0
    # the terms were still measured at zero weight, so the ablation is auditable
    assert rep["curve"][0]["train"]["drift"] >= 0.0
    assert rep["curve"][0]["train"]["contrastive"] >= 0.0

    empty = train(store, cfg=_cfg(), seed=42, epochs=1, batch_size=2, k_neighbors=2,
                  weights={}, device="cpu")
    assert empty["weights"] == {"contrastive": 1.0, "validity": 1.0,
                                "incremental": 0.5, "state": 0.5, "drift": 1.0}, \
        "an empty weights dict means 'no opinion', and must fall back to defaults"


def test_train_refuses_a_store_with_no_vectors_at_all(tmp_path):
    """Better to raise than to train on an empty corpus and report a converged
    curve built from nothing."""
    from engine.store.sqlite_store import SQLiteStore

    empty = SQLiteStore(tmp_path / "empty.db")
    try:
        with pytest.raises(RuntimeError, match="no context sets"):
            train(empty, cfg=_cfg(), epochs=1, device="cpu")
    finally:
        empty.close()


def test_checkpoint_round_trips_and_refuses_a_different_embedder(tmp_path, store):
    rep = train(store, cfg=_cfg(), seed=42, epochs=1, batch_size=2, k_neighbors=2,
                train_projects={"Atlas"}, device="cpu")
    path = save(rep["model"], rep, tmp_path / "c.pt")
    model, meta = load(path, expect_embed_version=store.embed_version)
    assert meta["epochs"] == 1 and "config" not in meta
    assert model.cfg.dim == DIM
    import json
    blob = torch.load(path, map_location="cpu", weights_only=True)   # not weights_only=False
    assert blob["version"] == CHECKPOINT_VERSION
    assert blob["config"]["max_step"] == rep["config"].max_step, \
        "the movement bound must survive save/load or the anchoring contract floats"
    with pytest.raises(ValueError, match="refusing to refine"):
        load(path, expect_embed_version="some-other-encoder")


# --------------------------------------------------------------------- refine

def _trained(store):
    rep = train(store, cfg=_cfg(max_step=0.2), seed=42, epochs=6, batch_size=2,
                k_neighbors=2, train_projects={"Atlas"}, dev_projects={"Atlas"},
                device="cpu")
    return rep["model"]


def n_refined(s):
    return s.db.execute(
        "SELECT COUNT(*) c FROM vectors WHERE refined IS NOT NULL").fetchone()["c"]


def test_refine_writes_the_column_and_rebuilds_only_that_index(store):
    model = _trained(store)
    rpt = refine(store, model, k=2, device="cpu", writes=True, max_drift=10.0)
    assert rpt["n_refined"] == rpt["n_units"] == 5
    assert rpt["anchored"] is True
    assert n_refined(store) == 5
    assert rpt["rebuilt_index"] == 5
    assert rpt["unit_length_min"] == pytest.approx(1.0, abs=1e-3)


def test_refine_refuses_an_unanchored_space_and_writes_nothing(store):
    """Measured: cosine_to_base 0.0105 with drift 1.4063 against a 0.25 bound while
    both columns still returned 5 hits — a ranking that looks alive and is not."""
    model = _trained(store)
    rpt = refine(store, model, k=2, device="cpu", writes=True, max_drift=0.001)
    assert rpt["status"] == "refused_unanchored"
    assert rpt["anchored"] is False
    assert n_refined(store) == 0, "a refused job must not touch the store"
    assert "rebuilt_index" not in rpt
    assert "cannot compare to" in rpt["warning"]


def test_the_anchoring_gate_uses_p95_not_the_mean(store):
    """drift_mean 0.1055 passed while drift_max was 0.5310, 2.1x the bound: the
    hinge in the loss is per-row, so the contract has to be per-row too."""
    model = _trained(store)
    bound = 0.2
    rpt = refine(store, model, k=2, device="cpu", writes=False, max_drift=bound)
    assert rpt["drift_p95"] <= bound + 1e-6, rpt
    assert rpt["cosine_to_base_mean"] >= rpt["cosine_to_base_floor_required"]
    # the floor is the bound's own geometry, not a second invented number
    assert rpt["cosine_to_base_floor_required"] == pytest.approx(1 - bound**2 / 2, abs=1e-4)


def test_writes_false_measures_without_touching_the_store(store):
    model = _trained(store)
    rpt = refine(store, model, k=2, device="cpu", writes=False, max_drift=10.0)
    assert rpt["anchored"] is True and "rebuilt_index" not in rpt
    assert n_refined(store) == 0


def test_clear_refined_returns_the_store_to_the_frozen_space(store):
    model = _trained(store)
    refine(store, model, k=2, device="cpu", writes=True, max_drift=10.0)
    assert clear_refined(store) == 5
    assert n_refined(store) == 0
    assert not (store.path.with_suffix(".refined.faiss")).exists()


def test_refine_on_a_store_without_vectors_says_so(tmp_store):
    """An empty store is a real state (a fresh ingest that has not embedded yet);
    the job must name it instead of dividing by a zero-length matrix."""
    rpt = refine(tmp_store, ConceptAttention(_cfg()), k=2, device="cpu", writes=True)
    assert rpt == {"n_units": 0, "n_refined": 0, "note": "no base vectors to refine"}
    assert n_refined(tmp_store) == 0


# --------------------------------------------------------- the training corpus

def test_corpus_excludes_test_projects_from_the_store_used_for_training(tmp_path,
                                                                       monkeypatch):
    """§9.5 by construction, not by argument.

    `corpus.build_corpus` writes `data/splits/ledger.json` and a db under `runs/`,
    so the repo root is redirected here: a unit test that rewrote the canonical
    split manifest would change what every later run considers `test`, which is the
    artefact-identity failure this project already had once.
    """
    from engine.concept import corpus

    monkeypatch.setattr(corpus, "repo_root", lambda: tmp_path)
    built = corpus.build_corpus(n_projects=3, seed=42, store_path=tmp_path / "c.db",
                                device="cpu")
    try:
        eng = built["engine"]
        assert built["train"] and built["dev"] and built["test"], \
            "a 3-project ledger must still populate all three splits"
        assert not (built["train"] & built["test"] or built["dev"] & built["test"])
        present = {u.project for u in eng.store.all_units() if u.project}
        assert not (present & built["test"]), \
            f"test projects leaked into the training corpus: {present & built['test']}"
        assert built["n_meetings_ingested"] > 0
        assert built["counts"]["units"] > 0
        assert (tmp_path / "data" / "splits" / "ledger.json").exists(), \
            "the manifest was written beside the temp root, not the repo"
    finally:
        built["engine"].close()


def test_train_store_saves_a_loadable_checkpoint_that_the_refiner_accepts(tmp_path,
                                                                         monkeypatch):
    """The end-to-end M5 path: corpus -> fit -> save -> load -> refine.

    Runs the real config model (dim 256, 2 layers) on a 3-project ledger so the
    numbers below describe the shipped shape, not a toy one. The checkpoint goes to
    a temp path: writing it to `models/concept/concept.pt` from a unit test would
    overwrite the trained artefact the report cites.
    """
    from engine.concept import corpus
    from engine.concept.train import load

    monkeypatch.setattr(corpus, "repo_root", lambda: tmp_path)
    ckpt = tmp_path / "models" / "concept.pt"
    rep = corpus.train_store(n_projects=3, seed=42, epochs=1, checkpoint=ckpt,
                             device="cpu")
    assert ckpt.exists() and rep["bytes"] > 0
    assert rep["corpus"]["test_projects_absent"], rep["corpus"]
    assert rep["embed_version"] == "hash-256-35"
    assert rep["labels"]["sets"] > 0

    a = rep["anchor_probe"]
    assert a["anchored"] is True, (
        f"a freshly fitted model must start at the baseline and stay inside the "
        f"drift bound (cos={a['cosine_to_base_mean']} floor={a['cosine_to_base_floor_required']} "
        f"drift_p95={a['drift_p95']} bound={a['max_drift_allowed']})")

    model, meta = load(ckpt, expect_embed_version=rep["embed_version"])
    assert meta["best_epoch"] == rep["best_epoch"]
    assert model.cfg.dim == 256


# ------------------------------------------------------- retrieval-side guards

def test_dense_retriever_refuses_to_claim_refined_when_only_base_exists(store):
    """`vector_matrix("refined")` falls back to base for NULL rows, so without this
    the refined arm would silently measure the frozen baseline and print `refined`."""
    with pytest.raises(ValueError, match="has none"):
        DenseRetriever(store, _HashEmbedderLike(), use="refined")
    assert DenseRetriever(store, _HashEmbedderLike(), use="base").use == "base"


def test_dense_retriever_accepts_refined_once_the_column_is_filled(store):
    model = _trained(store)
    refine(store, model, k=2, device="cpu", writes=True, max_drift=10.0)
    d = DenseRetriever(store, _HashEmbedderLike(), use="refined")
    hits = d.search("budget", k=3)
    assert hits and len(hits) <= 3


def test_unknown_vector_column_is_rejected_not_silently_used(store):
    with pytest.raises(ValueError, match="unknown vector column"):
        DenseRetriever(store, _HashEmbedderLike(), use="refinedd")


class _HashEmbedderLike:
    """A stand-in embedder. Deliberately free of `hash()`: Python randomises
    `hash(str)` per process, so a hash-seeded vector would make retrieval-order
    assertions flaky in exactly the way `tests/helpers.py` warns about."""

    version = "hash-test"
    dim = DIM

    def embed(self, texts, role="doc"):
        return _vecs(len(texts), seed=7)

    def embed_query(self, text):
        return self.embed([text])[0]
