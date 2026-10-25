"""The §13 demo gate and the one reader factory (6.2 / 2.2), both run in-suite.

Two reasons these tests exist:

* `cli_demo.run_demos` is the product's acceptance gate, and until now it only ran
  when a human typed `make demo`. `tests/e2e/test_demo_scenarios.py` covers the same
  scenarios through fixtures, so the *command* — the thing §16 promises works on a
  clean checkout — had no test behind it.
* `build_reader` exists because this project already shipped a run that reported
  `trained` while executing rules: the config key was real, the call site was not.
  So a fallback has to be legible in the returned string, AND a real saved
  checkpoint has to actually load. The second half uses a 16-hidden-unit BERT built
  locally — no download, no network — because testing only the fallback branch
  would never prove the success branch is reachable.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

from engine.cli_demo import _sources, run_demos
from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.read.baseline_rules import RuleReader
from engine.read.infer import (DEFAULT_MODEL_DIR, GATED_HEADS, ModelReader,
                              build_reader, reader_meta)
from engine.read.model import HEADS, ReaderModel
from helpers import make_unit

pytestmark = pytest.mark.unit

FIXTURE = repo_root() / "tests/golden/demo_scenarios.json"
CONTACTS = repo_root() / "data/contacts.json"

#: the seven §13 scenarios plus the two safety checks, by their printed numbers
SCENARIO_TAGS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "S.")


def _demo_args(**kw):
    base = {"fixture": str(FIXTURE), "contacts": str(CONTACTS)}
    base.update(kw)
    return argparse.Namespace(**base)


def _tiny_tokenizer():
    """A 21-word WordLevel tokenizer built in-process: no vocab file, no hub."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        ["le budget est validé à 2 4 m€ quel marie peut être",
         "on pourrait revoir le tarif du projet ouverture pilotage"],
        trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]",
                                                  "[SEP]", "[MASK]"]))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


# ------------------------------------------------------------------ the §13 gate

def test_demo_gate_passes_every_check():
    """`make demo` is part of the definition of done, so a regression here must fail
    the suite rather than only a manual run."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_demos(_demo_args())
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith(("[PASS", "[FAIL"))]
    failures = [ln for ln in lines if ln.startswith("[FAIL")]
    assert not failures, f"demo gate failed: {failures}"
    assert code == 0, f"run_demos returned {code} with no failures reported"
    assert len(lines) >= 17, f"expected the 7 scenarios + sub-checks, got {len(lines)}"
    for tag in SCENARIO_TAGS:
        assert any(f"] {tag}" in ln for ln in lines), f"scenario {tag} not reported"


def test_demo_gate_reports_a_missing_fixture_instead_of_crashing():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_demos(_demo_args(fixture=str(repo_root() / "nope.json")))
    assert code == 2
    assert "missing fixture" in buf.getvalue()


def test_safety_payloads_carry_no_leaked_address_and_no_execution():
    """The demo prints PASS for these; assert on the payloads themselves, because a
    green label over a leaking payload is the failure mode this project keeps
    hitting (injection text is stored as data — meeting-boreal-1 contains a real
    'send it to leak@evil.example.com' instruction)."""
    import tempfile
    import pathlib

    root = pathlib.Path(tempfile.mkdtemp())
    cfg = load_config(overrides={"store": {"path": str(root / "d.db")},
                                 "embed": {"backend": "hash"},
                                 # CPU-tier pin (D-041): the shipped reranker is
                                 # a GPU model; the demo assertions here are
                                 # behavioural, and the model is gated by
                                 # `make demo` / eval-fast / the headline bench.
                                 "retrieve": {"rerank": {"backend": "none"}},
                                 "agent": {"contacts_path": str(CONTACTS)}})
    eng = Engine(cfg, path=root / "d.db")
    for s in _sources(FIXTURE):
        eng.ingest(s)
    blobs = []
    for q, intent, proj in (
            ("Rédige un mail à la personne responsable du budget d'Atlas.",
             "draft_email", "Atlas"),
            ("Rédige un mail à la personne responsable du budget du projet Iris.",
             "draft_email", "Iris"),
            ("Crée une tâche pour revoir le tarif du projet Atlas.", "create_task",
             "Atlas")):
        r = eng.ask(q, intent=intent, project=proj)
        # A clarified action carries no `executed` key at all (measured: payload is
        # {options, question, reason}), so `is False` asserted the wrong thing. The
        # invariant is that nothing ever claims it executed.
        assert r.payload.get("executed") is not True, r.operation
        if r.operation == "clarify":
            # the ambiguous case must hand back names, never an address
            assert r.clarification is not None
            assert "@" not in json.dumps(r.payload, ensure_ascii=False), r.payload
        blobs.append(json.dumps(r.payload, ensure_ascii=False))
    eng.close()
    joined = " ".join(blobs)
    assert "leak@evil.example.com" not in joined
    assert '"executed": true' not in joined


# ------------------------------------------------------------- the reader factory

def test_default_config_builds_the_rules_reader_and_says_so():
    reader, backend = build_reader(load_config())
    assert isinstance(reader, RuleReader)
    assert backend == "rules"
    assert reader_meta(reader)["backend"] == "rules"


def test_missing_checkpoint_falls_back_with_the_reason_in_the_string(tmp_path):
    cfg = load_config(overrides={"reader": {"backend": "model",
                                           "model_dir": str(tmp_path / "absent")}})
    reader, backend = build_reader(cfg)
    assert isinstance(reader, RuleReader)
    assert backend.startswith("rules[fallback:") and "absent" in backend


def test_unreadable_checkpoint_falls_back_and_names_the_error(tmp_path):
    bad = tmp_path / "reader"
    bad.mkdir()
    (bad / "meta.json").write_text("{ not json", encoding="utf-8")
    cfg = load_config(overrides={"reader": {"backend": "model",
                                           "model_dir": str(bad)}})
    reader, backend = build_reader(cfg)
    assert isinstance(reader, RuleReader)
    assert "failed to load" in backend


def test_a_real_saved_checkpoint_loads_and_reports_itself_as_model(tmp_path):
    """The success branch, exercised for real. Small and local on purpose: what is
    under test is save -> load -> predict -> label wiring, not accuracy."""
    import torch
    from transformers import BertConfig, BertModel

    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    conf = BertConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=32,
                      max_position_embeddings=64,
                      hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    texts = ["le budget est validé à 2 4 m€", "quel est le budget",
             "on pourrait revoir le tarif", "marie fait le pilotage"]
    labels = {
        "kind": np.array([HEADS["kind"].index("info"), HEADS["kind"].index("query"),
                          HEADS["kind"].index("info"), HEADS["kind"].index("info")]),
        "type": np.array([HEADS["type"].index("decision"),
                          HEADS["type"].index("question"),
                          HEADS["type"].index("idea"), HEADS["type"].index("fact")]),
        "status": np.array([HEADS["status"].index("confirmed"),
                            HEADS["status"].index("mention"),
                            HEADS["status"].index("proposed"),
                            HEADS["status"].index("confirmed")]),
        "intent": np.array([-1, HEADS["intent"].index("answer"), -1, -1]),
    }
    model = ReaderModel(encoder=BertModel(conf), tokenizer=tok, encoder_id="toy",
                        max_length=32, device="cpu", hidden_size=16)
    model.fit(texts, labels, epochs=3, batch_size=4, seed=42)

    # `load` re-opens the encoder by id, so the id has to point at bytes on disk:
    # leaving it as "toy" would make the load fail, build_reader would fall back to
    # rules, and the assertion below would pass while proving nothing.
    enc_dir = tmp_path / "toy-encoder"
    model.encoder.save_pretrained(enc_dir)
    model.encoder_id = str(enc_dir)          # ReaderModel is a plain class, not pydantic
    ckpt = tmp_path / "reader"
    model.save(ckpt)

    cfg = load_config(overrides={"reader": {"backend": "model",
                                           "model_dir": str(ckpt),
                                           "context_units": 2,
                                           "abstain_threshold": 0.0}})
    reader, backend = build_reader(cfg)
    assert isinstance(reader, ModelReader), backend
    assert backend == "model(reader)"          # names the loaded checkpoint
    meta = reader_meta(reader)
    assert meta["backend"] == "model" and meta["context_units"] == 2
    assert meta["encoder"] == str(enc_dir), (
        "the loaded checkpoint reports the encoder it re-opened; if this is the "
        "'toy' placeholder the load silently used something else")

    unit = make_unit("le budget est à 2 m€", speaker="Marie")
    reading = reader.read(unit, [make_unit("ouverture"), make_unit("le tarif")])
    assert reading.unit_id == unit.unit_id
    assert reading.kind in HEADS["kind"] and reading.status in HEADS["status"]
    assert reading.abstained is False
    assert set(reading.probabilities) == set(HEADS)
    for name, dist in reading.probabilities.items():
        assert set(dist) == set(HEADS[name])
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)


def test_abstain_threshold_flags_a_reading_without_relabelling_it():
    """6.2.5: the label is still stored, so the flag is the only signal a caller
    gets that the reader did not trust itself."""
    strict = ModelReader(_StubModel(), abstain_threshold=0.99)
    unit = make_unit("Le budget est validé à 2,4 M€.")
    reading = strict.read(unit)
    assert reading.abstained is True
    assert reading.kind == "info"                       # unchanged, not rewritten
    assert len(strict.last_confidence) == len(HEADS)

    lenient = ModelReader(_StubModel(), abstain_threshold=0.0)
    assert lenient.read(unit).abstained is False


def test_intent_is_reported_only_for_query_rows():
    """`intent` is gold on query rows only (6.2.2); surfacing a statement's argmax
    would invent a routing decision the reader was never trained to make."""
    for kind, want in (("info", "none"), ("query", "answer")):
        reader = ModelReader(_StubModel(kind=kind, intent="answer"))
        assert reader.read(make_unit("x")).intent == want


def test_gated_heads_and_defaults_are_what_the_engine_relies_on():
    assert set(GATED_HEADS) == {"kind", "type", "status"}   # intent excluded
    assert DEFAULT_MODEL_DIR == "models/reader"


class _StubModel:
    """Fixed rows from `predict_batch`, so the abstain/intent logic is tested
    without a torch forward pass."""

    encoder_id = "stub"
    with_context = False

    def __init__(self, kind="info", intent="answer"):
        self.kind, self.intent = kind, intent

    def predict_batch(self, texts, speakers=None, contexts=None):
        n = len(texts)
        preds, probs = {}, {}
        for name, labels in HEADS.items():
            chosen = (self.intent if name == "intent" else
                      self.kind if name == "kind" else labels[0])
            idx = labels.index(chosen)
            row = np.full((n, len(labels)), 0.1 / max(1, len(labels) - 1))
            row[:, idx] = 0.9
            preds[name] = np.array([idx] * n)
            probs[name] = row
        return preds, probs
