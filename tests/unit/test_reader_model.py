"""The trained reader (6.2.3 - 6.2.6) runs end to end on a toy encoder.

`engine/read/model.py` was at 15% coverage: every one of its defects this project
hit — heads re-created per epoch so early stopping could never improve, a
temperature pinned at the clamp floor turning ECE into an artifact of
over-confidence, a checkpoint loaded under a label order that no longer matched
the schema — lived in code no test executed. These tests build a 16-hidden-unit
BERT and a 30-token vocab **locally** (nothing is downloaded), so the real torch
path runs on CPU in about a second.

The pure helpers are tested against hand-computed values, because that is the
only way to tell a formula bug from a training run that happens to converge.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from engine.read.model import (HEADS, LABEL_SMOOTHING, MASK, ReaderModel,
                               apply_temperature, encode_inputs, evaluate_heads,
                               fit_temperature, gold_vector, mean_confidence, nll)

pytestmark = pytest.mark.unit


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ gold_vector

def test_gold_vector_maps_labels_to_head_indices():
    v = gold_vector("info", "fact", "confirmed", "answer")
    assert v["kind"] == HEADS["kind"].index("info")
    assert v["intent"] == HEADS["intent"].index("answer")
    assert set(v) == set(HEADS)


def test_gold_vector_masks_an_unknown_intent_but_not_an_unknown_status():
    """Intent is gold only on query rows (6.2.2), so None is a legitimate mask.
    A *wrong* label is never legitimate and must raise."""
    assert gold_vector("info", "fact", "confirmed", None)["intent"] == MASK
    with pytest.raises(KeyError):
        gold_vector("info", "fact", "definitely_not_a_status")


# ---------------------------------------------------------------- encode_inputs

def test_encode_inputs_prefixes_the_speaker_and_honours_the_context_knob(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    plain = encode_inputs(tok, ["bonjour", "au revoir"], ["Marie", None], 32)
    with_ctx = encode_inputs(tok, ["bonjour", "au revoir"], ["Marie", None], 32,
                             context=["[a]", None], with_context=True)
    ids = tok(["[Marie] bonjour", "au revoir"], padding=True, truncation=True,
              max_length=32, return_tensors="pt")

    def real(enc, i):
        """The tokens that are not padding — batch padding length varies with the
        longest row, so comparing padded rows compares the padding, not the text."""
        m = enc["attention_mask"][i]
        return enc["input_ids"][i][:int(m.sum())].tolist()

    assert real(plain, 0) == real(ids, 0)      # "[Marie] bonjour"
    assert real(plain, 1) == real(ids, 1)      # unprefixed row stays unprefixed
    # the knob actually changes the row it is applied to...
    assert real(with_ctx, 0) != real(plain, 0)
    # ...and does not leak into the row whose context is empty
    assert real(with_ctx, 1) == real(plain, 1)


# -------------------------------------------------------------- class weights

def test_class_weights_rare_upweight_and_never_divide_by_zero():
    labels = np.array([0, 0, 0, 0, 1])
    w = _weights(labels, 2)
    assert w[1] > w[0]
    assert np.isfinite(w).all()


def test_class_weights_handles_a_fully_masked_head():
    """An intent head on statement-only rows sees -1 everywhere; the weights must
    not come back nan, because they are multiplied into the loss."""
    w = _weights(np.full(6, MASK), 8)
    assert np.isfinite(w).all() and w.mean() == pytest.approx(1.0)


def _weights(labels, n):
    from engine.read.model import _class_weights

    return _class_weights(labels, n)


# ------------------------------------------------------------- temperature math

def test_apply_temperature_matches_softmax_at_one_and_degenerates_correctly():
    logits = np.array([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    assert np.allclose(apply_temperature(logits, 1.0), softmax(logits))
    hot = apply_temperature(logits, 0.01)
    assert np.allclose(hot.max(axis=1), 1.0)          # one-hot, not calibrated
    cold = apply_temperature(logits, 1e3)
    assert np.allclose(cold, 1.0 / logits.shape[1], atol=1e-3)


def test_apply_temperature_is_shift_invariant_and_overflow_safe():
    logits = np.array([[1000.0, 999.0], [-900.0, -1200.0]])
    out = apply_temperature(logits, 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(axis=1), 1.0)


def test_nll_and_mean_confidence_edge_cases():
    perfect = np.array([[5.0, -5.0], [0.0, 9.0]])
    assert nll(perfect, np.array([0, 1])) < 1e-3
    assert nll(perfect, np.array([MASK, MASK])) == 0.0      # nothing to score
    assert mean_confidence(np.array([[0.7, 0.3]])) == pytest.approx(0.7)
    assert mean_confidence(np.zeros((0, 2))) == 0.0


def test_fit_temperature_sharpens_an_underconfident_head_and_stays_in_bounds():
    """The head is right but its probabilities are flat, so the NLL optimum is
    T < 1. Recitting the bound check matters: the first version clamped at 1e-2
    and produced a fake calibration."""
    rng = np.random.default_rng(42)
    gold = rng.integers(0, 3, size=200)
    logits = np.zeros((200, 3)) + 0.02          # near-uniform
    logits[np.arange(200), gold] += 0.25        # right, but timidly
    t = fit_temperature(logits, gold)
    assert 0.05 <= t <= 20.0
    assert t < 1.0
    assert nll(logits, gold, t) < nll(logits, gold, 1.0)


def test_fit_temperature_on_all_masked_rows_is_neutral():
    assert fit_temperature(np.zeros((3, 4)), np.full(3, MASK)) == 1.0


# -------------------------------------------------------------------------- ECE

def test_ece_is_zero_for_a_perfectly_calibrated_head_and_high_for_an_overconfident_one():
    from engine.read.model import _ece

    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    honest = np.array([[0.5, 0.5]] * 4 + [[0.5, 0.5]] * 4)   # 50% conf, 50% acc
    assert _ece(honest, gold) == pytest.approx(0.0, abs=1e-9)
    over = np.array([[0.99, 0.01]] * 8)
    assert _ece(over, gold) > 0.4                            # 99% conf, 50% acc
    assert _ece(np.zeros((0, 2)), np.array([])) == 0.0


def test_evaluate_heads_scores_only_unmasked_rows():
    gold = {"kind": np.array([0, 1, MASK]), "intent": np.array([2, MASK, MASK])}
    pred = {"kind": np.array([0, 0, 0]), "intent": np.array([2, 2, 2])}
    out = evaluate_heads(gold, pred)
    assert out["kind"]["n"] == 2 and out["kind"]["accuracy"] == 0.5
    assert out["intent"]["n"] == 1                      # only the labelled row
    assert "ece" not in out["kind"]                     # no probs passed in


def test_evaluate_heads_reports_an_empty_head_without_raising():
    out = evaluate_heads({"status": np.full(4, MASK)},
                         {"status": np.zeros(4, dtype=int)})
    assert out["status"]["n"] == 0 and out["status"]["macro_f1"] is None


# --------------------------------------------------------------- toy fixtures
#
# The tokenizer here is a real failure mode made visible: the first version built
# a `BertTokenizerFast` from a hand-written vocab file and reported
# `vocab_size = 5` — only the special tokens. Every word became `[UNK]`, so the
# "separable" toy set carried no signal at all, and the fit test measured chance
# while the tests that compared a tokenizer against itself passed vacuously. A
# WordLevel tokenizer trained on the toy corpus gives 21 real tokens.

_TINY_CORPUS = ["dec 0 1 2 3 4 5", "abc 0 1 2 3 4 5",
                "bonjour au revoir le budget est valide marie Marie p q r s t"]


def _tiny_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(_TINY_CORPUS, trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


def _tiny_model(tmp_path, heads=None, temperatures=None):
    """ReaderModel on a 16-unit BERT: real torch, no download, ~1 s to fit."""
    import torch

    from transformers import BertConfig, BertModel

    # seeded HERE, not only inside `fit`: the encoder's weights are drawn when it
    # is constructed, which happens before `fit` reseeds. Without this the toy
    # network started from different weights depending on which test ran first,
    # and the accuracy assertion passed in a single-file run and failed in the
    # full suite — a flaky test that looked like a model defect.
    torch.manual_seed(0)
    np.random.seed(0)
    tok = _tiny_tokenizer(tmp_path)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=64,
                     # dropout off: on a 24-row toy set it is pure noise, and two
                     # of these tests assert a measured accuracy floor
                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    enc = BertModel(cfg)
    assert tok.tokenize("dec 3") == ["dec", "3"], "toy vocab must carry the signal"
    m = ReaderModel(encoder=enc, tokenizer=tok, heads=heads,
                    encoder_id="toy-bert", max_length=32,
                    temperatures=temperatures, device="cpu", hidden_size=16)
    return m, tok


def _toy_rows(n=24):
    """A toy set where every head has at least two classes AND 6.2.2's masking.

    `intent` is gold only on query rows; the rest are -1. The first version of
    this fixture labelled `kind` as one constant class on all 24 rows, and that
    degenerate head contributed a constant gradient the 16-unit toy encoder could
    not fight off: `type` collapsed to a single prediction and measured chance.
    A head with one class is a bug in the test, not a property of the model.
    """
    texts, kind, type_, status, intent = [], [], [], [], []
    for i in range(n):
        query = i % 2 == 1                      # 'abc …' is the query cue
        texts.append(f"{'abc' if query else 'dec'} {i % 5}")
        kind.append(HEADS["kind"].index("query" if query else "info"))
        type_.append(HEADS["type"].index("fact" if query else "decision"))
        status.append(HEADS["status"].index("proposed" if query else "confirmed"))
        if i % 4 == 1:
            intent.append(HEADS["intent"].index("answer"))
        elif query:
            intent.append(HEADS["intent"].index("compare"))
        else:
            intent.append(MASK)                 # not a query: no gold intent
    return (texts,
            {"kind": np.array(kind), "type": np.array(type_),
             "status": np.array(status), "intent": np.array(intent)},
            np.array(["Marie" if i % 3 else None for i in range(n)]))


# ------------------------------------------------------------------------ fit

def test_fit_learns_a_separable_signal_and_records_its_settings(tmp_path):
    m, _tok = _tiny_model(tmp_path)
    texts, labels, speakers = _toy_rows()
    log = []
    meta = m.fit(texts, labels, speakers=speakers, epochs=6, batch_size=8, lr=1e-3,
                 head_lr=5e-3, seed=42, log=log.append)
    # 24 rows / 8 = 3 steps per epoch, 6 epochs
    assert meta["steps"] == 18 and meta["n_train"] == len(texts)
    assert meta["epochs"] == 6 and meta["device"] == "cpu"

    losses = [float(line.split("loss")[1]) for line in log]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    preds, probs = m.predict_batch(texts, speakers)
    # `kind` is the head the toy cue determines outright, so it is the honest place
    # to assert learning. The assertion is "it separated", not "it hit a number":
    # an accuracy threshold invented before the measurement is what made the first
    # version of this test fail at exactly chance for a fixture reason, not a model
    # one. A collapsed head — every row the same class — is the failure mode here.
    keep = labels["kind"] >= 0
    assert len(set(preds["kind"][keep].tolist())) == len(HEADS["kind"])
    acc = (preds["kind"][keep] == labels["kind"][keep]).mean()
    # The classes are balanced (12 info / 12 query), so predicting one class for
    # every row scores 0.5 — the exact failure 6.2.4 warns about. Measured under
    # this fixed seed: 0.833 with all four heads competing for one 16-unit
    # encoder, 1.0 with the `kind` head alone. The floor below is between chance
    # and the measurement, not a number invented before the run.
    assert acc > 0.75, f"toy reader should beat the majority baseline, got {acc}"
    assert probs["kind"].shape == (len(texts), len(HEADS["kind"]))
    assert np.allclose(probs["kind"].sum(axis=1), 1.0)


def test_fit_does_not_re_create_heads_that_already_match_the_schema(tmp_path):
    """Regression: `fit` used to rebuild every head on every call, so a
    train → validate → train loop threw away the weights each epoch and early
    stopping could never make the model better than epoch 1."""
    m, _ = _tiny_model(tmp_path)
    texts, labels, _ = _toy_rows(12)
    m.fit(texts, labels, epochs=1, batch_size=8)
    before = {k: v for k, v in m.heads["kind"].named_parameters()}
    m.fit(texts, labels, epochs=1, batch_size=8)
    assert m.heads["kind"].state_dict()["weight"].shape == (len(HEADS["kind"]), 16)
    # same Parameter objects, i.e. the head survived the second call
    assert all(a is b for (a, b) in zip(before.values(),
                                        dict(m.heads["kind"].named_parameters()).values()))


def test_fit_replaces_a_head_whose_label_count_no_longer_matches(tmp_path):
    """A checkpoint from a 2-class `kind` era cannot keep its head when the
    schema has a different number of labels: the weights would be read against
    the wrong label count. `fit` must rebuild that one head."""
    m, _ = _tiny_model(tmp_path, heads={"kind": torch.nn.Linear(16, 7)})
    texts, labels, _ = _toy_rows(8)
    m.fit(texts, labels, epochs=1, batch_size=8)
    assert m.heads["kind"].out_features == len(HEADS["kind"])


def test_fit_skips_a_head_with_no_labelled_row_in_the_batch(tmp_path):
    """The intent head is masked on every statement row. A batch that is 100%
    masked used to produce a float loss and then crash on `.backward()`."""
    m, _ = _tiny_model(tmp_path)
    texts = ["abc 1", "abc 2", "abc 3", "abc 4"]
    labels = {"kind": np.array([1, 1, 1, 1]),
              "intent": np.full(4, MASK)}
    m.fit(texts, labels, epochs=1, batch_size=4)
    assert "intent" in m.heads                     # head exists, was never trained


def test_fit_can_freeze_the_encoder(tmp_path):
    m, _ = _tiny_model(tmp_path)
    texts, labels, _ = _toy_rows(8)
    meta = m.fit(texts, labels, epochs=1, batch_size=8, freeze_encoder=True)
    assert meta["freeze_encoder"] is True
    assert all(not p.requires_grad for p in m.encoder.parameters())


# ------------------------------------------------------------------ calibrate

def test_calibrate_fits_per_head_temperatures_and_reports_the_diagnostic(tmp_path):
    m, _ = _tiny_model(tmp_path)
    texts, labels, _ = _toy_rows(16)
    # fit first: an unfitted model has no heads, so `logits_batch` would return
    # {} and this test would pass by measuring nothing
    m.fit(texts, labels, epochs=1, batch_size=8)
    logits = m.logits_batch(texts)
    assert set(logits) == set(labels)
    before = dict(m.temperatures)
    rep = m.calibrate(logits, labels)
    assert set(rep) == set(logits)
    for name, r in rep.items():
        assert {"temperature", "accuracy", "nll_before", "nll_after",
                "mean_conf_before", "mean_conf_after", "ece_before", "ece_after",
                "pinned_at_bound", "confidence_gap"} <= set(r)
        assert 0.05 <= r["temperature"] <= 20.0
        assert r["nll_after"] <= r["nll_before"] + 1e-6     # NLL must not get worse
    assert m.temperatures != before or not before
    # probabilities now follow the fitted temperature, not the raw logits
    _preds, probs = m.predict_batch(texts)
    assert np.allclose(probs["kind"], apply_temperature(logits["kind"],
                                                        m.temperatures["kind"]))


def test_calibrate_ignores_a_head_with_no_development_gold(tmp_path):
    m, _ = _tiny_model(tmp_path)
    logits = m.logits_batch(["abc 1", "dec 2"])
    rep = m.calibrate(logits, {"kind": np.full(2, MASK)})
    assert rep == {} and m.temperatures == {}


# --------------------------------------------------------------- persistence

def test_save_then_load_round_trips_predictions_and_refuses_a_label_drift(tmp_path):
    m, _ = _tiny_model(tmp_path)
    texts, labels, speakers = _toy_rows(12)
    m.fit(texts, labels, speakers=speakers, epochs=2, batch_size=8)
    d = tmp_path / "reader"
    m.save(d)
    (d / "meta.json").write_text(
        json.dumps({**json.loads((d / "meta.json").read_text()),
                    "encoder_id": str(_save_encoder(tmp_path, m))}), encoding="utf-8")
    again = ReaderModel.load(d, device="cpu")
    p1, _ = m.predict_batch(texts, speakers)
    p2, _ = again.predict_batch(texts, speakers)
    assert (p1["kind"] == p2["kind"]).all()
    assert again.temperatures == m.temperatures
    assert again.train_meta["epochs"] == 2
    assert again.hidden_size == 16

    # a checkpoint whose head order no longer matches the schema must be refused,
    # not silently mis-indexed
    meta = json.loads((d / "meta.json").read_text())
    meta["heads"] = {"kind": list(reversed(HEADS["kind"])), "type": HEADS["type"],
                     "status": HEADS["status"], "intent": HEADS["intent"]}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="retrain"):
        ReaderModel.load(d, device="cpu")


def _save_encoder(tmp_path, m):
    """Persist the toy encoder so `load` can rebuild it without a download."""
    d = tmp_path / "toy-encoder"
    m.encoder.save_pretrained(d)
    return d


def test_save_writes_the_label_order_and_smoothing_the_run_used(tmp_path):
    m, _ = _tiny_model(tmp_path)
    texts, labels, _ = _toy_rows(8)
    m.fit(texts, labels, epochs=1, batch_size=8)
    d = tmp_path / "r2"
    m.save(d)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["heads"] == HEADS
    assert meta["train_meta"]["label_smoothing"] == LABEL_SMOOTHING
    assert (d / "weights.pt").exists() and (d / "tokenizer").exists()


def test_logit_batches_are_independent_of_batch_size(tmp_path):
    """A padded batch must not change a row's score, or the eval harness's
    numbers depend on how it happened to chunk."""
    m, _ = _tiny_model(tmp_path)
    texts, labels, _ = _toy_rows(10)
    m.fit(texts, labels, epochs=1, batch_size=8)
    a = m.logits_batch(texts, batch_size=10)
    b = m.logits_batch(texts, batch_size=1)
    assert np.allclose(a["kind"], b["kind"], atol=1e-4)
