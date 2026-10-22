"""Offline boundaries: the LLM transport, the FTS5 index, FAISS, and the abstain head.

Four modules sat at 0-60% because each one needs something the rest of the suite
doesn't provide — an HTTP response, a virtual table, a vector index, a trained
checkpoint. None of them needs it:

* `agent/llm.py` is the only place this project talks to a model, and its contract
  is a *refusal* contract: no key configured must raise before any socket opens,
  an unreachable endpoint must surface as `LLMUnavailable`, and a model that wraps
  its answer in prose or a fence must still be parsed rather than trusted raw.
  Patching `urlopen` tests all of that without a server.
* `store/fts.py` is the comparison row for bm25s (6.6). Its `bm25()` sign flip is
  the kind of one-character bug that silently inverts a ranking and still returns
  plausible-looking hits.
* `store/faiss_index.py`'s filtered search over-fetches then falls back to exact
  brute force. If the fallback never fires, filtered queries quietly return fewer
  than `k` hits and the recall numbers are measuring the index, not the retriever.
* `read/infer.ModelReader` decides `abstained` from a min over the gated heads.
  A fake model makes that decision testable without torch — the real checkpoint
  load path is covered in `test_demo_and_reader_factory.py`.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from engine.agent import llm as L
from engine.config import load_config
from engine.read.infer import GATED_HEADS, ModelReader, reader_meta
from engine.read.model import HEADS
from engine.segment import turns as T
from engine.schemas import Reading
from engine.store.faiss_index import FaissVectors
from engine.store.fts import FtsIndex, LexicalHit, _light_stem, fts_quote, fts_quote_or, tokenize
from helpers import make_unit

pytestmark = pytest.mark.unit


# ------------------------------------------------------------- the LLM transport

class _Resp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    """Intercept `urlopen`; nothing leaves the process."""
    box = {}

    def install(payload, error=None):
        def fake(req, timeout=None):
            box["url"] = req.full_url
            box["body"] = json.loads(req.data.decode())
            box["headers"] = dict(req.headers)
            box["timeout"] = timeout
            if error:
                raise error
            return _Resp(payload)

        monkeypatch.setattr(L.urllib.request, "urlopen", fake)
        return box

    return install


def test_template_backend_is_the_llm_off_row_and_says_nothing():
    b = L.TemplateBackend()
    assert b.name == "template"
    assert b.generate("anything", schema={"a": 1}, system="s") is None
    # `available` is name-based, so template counts as available: the renderer
    # *is* usable, it just never produces text. Asserted, not wished.
    assert b.available is True


def test_base_backend_is_abstract_and_reports_unavailable():
    base = L.LLMBackend()
    assert base.available is False
    with pytest.raises(NotImplementedError):
        base.generate("x")


def test_openai_backend_refuses_without_a_key_before_any_request(sent, monkeypatch):
    box = sent({"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.delenv("SAE_LLM_KEY", raising=False)
    b = L.OpenAICompatibleBackend(api_key=None)
    with pytest.raises(L.LLMUnavailable, match="SAE_LLM_KEY not set"):
        b.generate("prompt")
    assert box == {}, "a request was sent without a key"


def test_openai_backend_sends_bearer_and_parses_content(sent, monkeypatch):
    monkeypatch.delenv("SAE_LLM_KEY", raising=False)
    box = sent({"choices": [{"message": {"content": '{"answer": "2,4 M€"}'}}]})
    b = L.OpenAICompatibleBackend(endpoint="http://127.0.0.1:9/v1", api_key="sk-test")
    out = b.generate("question", system="be brief",
                     schema={"properties": {"answer": {"type": "string"}}})
    assert out == {"answer": "2,4 M€"}
    assert box["url"].endswith("/chat/completions")
    assert box["headers"]["Authorization"] == "Bearer sk-test"
    assert box["body"]["temperature"] == 0.0
    assert box["body"]["messages"][0]["role"] == "system"
    assert box["body"]["response_format"]["json_schema"]["strict"] is True


def test_ollama_backend_happy_path_and_endpoint_from_env(sent, monkeypatch):
    monkeypatch.setenv("SAE_OLLAMA_URL", "http://127.0.0.1:11434")
    box = sent({"response": '```json\n{"answer": "ok"}\n```'})
    b = L.OllamaBackend(model="m", timeout=7.0)
    assert b.generate("p", schema={"properties": {}}) == {"answer": "ok"}
    assert box["url"] == "http://127.0.0.1:11434/api/generate"
    assert box["body"]["stream"] is False and box["body"]["options"]["temperature"] == 0.0
    assert box["timeout"] == 7.0


def test_unreachable_endpoint_becomes_llm_unavailable(sent):
    sent(None, error=OSError("connection refused"))
    with pytest.raises(L.LLMUnavailable, match="connection refused"):
        L.OllamaBackend(endpoint="http://127.0.0.1:1").generate("p")


def test_build_llm_maps_config_names_and_refuses_the_unknown():
    def cfg(backend):
        return {"agent": {"llm": {"backend": backend, "model_id": None}}}

    assert isinstance(L.build_llm(cfg("template")), L.TemplateBackend)
    assert isinstance(L.build_llm(cfg(None)), L.TemplateBackend)
    assert isinstance(L.build_llm(cfg("ollama")), L.OllamaBackend)
    assert isinstance(L.build_llm(cfg("openai")), L.OpenAICompatibleBackend)
    assert isinstance(L.build_llm(cfg("openai_compatible")), L.OpenAICompatibleBackend)
    assert L.build_llm({}).name == "template"          # no agent section at all
    with pytest.raises(ValueError, match="unknown llm backend"):
        L.build_llm(cfg("gpt99"))
    # the shipped default must be the one that cannot call home
    assert load_config()["agent"]["llm"]["backend"] == "template"


@pytest.mark.parametrize("text,want", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go: {"a": 1} hope that helps', {"a": 1}),
    ('{"a": 1, "b": 2}', {"a": 1, "b": 2}),
    ("", None),
    ("no json at all", None),
    ("[1, 2]", None),                      # a list is not an answer object
    ("{oops", None),
])
def test_coerce_json_survives_model_formatting(text, want):
    """No schema passed, so nothing is filtered here — key dropping is the next
    test's subject. Pass `want is None` through without an `or` shortcut: a
    legitimate `{}` result must not be confused with a refusal."""
    got = L._coerce_json(text, None)
    assert (got is None) == (want is None), f"{text!r} -> {got!r}"
    if want is not None:
        assert got == want


def test_coerce_json_drops_keys_the_schema_never_declared():
    """A model that answers with an extra field must not get it trusted downstream;
    but a model that answers with ONLY unexpected fields returns them, because
    'nothing matched' and 'empty object' are different events."""
    schema = {"properties": {"answer": {"type": "string"}}}
    assert L._coerce_json('{"answer": "x", "recipient": "boss@evil.com"}', schema) \
        == {"answer": "x"}
    assert L._coerce_json('{"other": 1}', schema) == {"other": 1}


# ------------------------------------------------------------------- FTS5 index

@pytest.fixture
def fts_db():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE sentences (unit_id text, sent_id integer, start integer, "
               "end integer, text text)")
    db.execute("CREATE VIRTUAL TABLE sentences_fts USING fts5(text, unit_id, sent_id, "
               "tokenize='unicode61 remove_diacritics 2')")
    rows = [("u1", 0, 0, 20, "Le budget d'Atlas est validé à 2,4 M€."),
            ("u2", 0, 0, 18, "Le lancement est prévu en décembre."),
            ("u2", 1, 18, 30, "Marie s'occupe du pilotage.")]
    db.executemany("INSERT INTO sentences VALUES (?,?,?,?,?)", rows)
    db.commit()
    yield db
    db.close()


def test_tokenize_folds_accents_keeps_digits_and_is_stemmable():
    assert "novembre" in tokenize("Réunion en Novembre 2026")
    assert "2026" in tokenize("Réunion en Novembre 2026")
    assert tokenize("prévu", strip_accents=False)[0].startswith("pr")
    assert tokenize("validé") == ["valide"]           # accent folding is on
    assert tokenize("documents") != tokenize("documents", stem=True)
    assert tokenize("") == []


def test_light_stem_leaves_short_tokens_alone():
    for tok in ("les", "de", "est", "a"):
        assert _light_stem(tok) == tok
    assert _light_stem("fonctionnements") == "fonctionn"
    assert _light_stem("porteuses") == "port"


def test_fts_quote_and_or_helpers():
    """Tokens are lower-cased on purpose: the index is analysed with
    `remove_diacritics 2` and FTS5 matching is case-insensitive, so quoting the
    user's original casing would just be noise in the emitted query."""
    assert fts_quote("budget Atlas") == '"budget" AND "atlas"'
    assert fts_quote_or("budget Atlas") == '"budget" OR "atlas"'
    assert fts_quote("Prévu À Atlas") == '"prevu" AND "a" AND "atlas"'
    assert fts_quote("!!!") == "" and fts_quote_or("   ") == ""


def test_fts_sync_and_search_flips_bm25_sign(fts_db):
    idx = FtsIndex(fts_db)
    assert idx.sync_from_sentences() == 3
    hits = idx.search("budget", k=5)
    assert hits and isinstance(hits[0], LexicalHit)
    # bm25() returns negative relevance; the module negates it, so a real match
    # must score POSITIVE — an unsigned flip would rank the worst hit first.
    assert all(h.score > 0 for h in hits), [h.score for h in hits]
    assert hits[0].unit_id == "u1"
    assert hits[0].evidence
    assert idx.search("nothingmatches", k=5) == []
    assert idx.search("", k=5) == []                   # empty query, not a crash


def test_fts_require_all_false_widens_the_result_set(fts_db):
    idx = FtsIndex(fts_db)
    idx.sync_from_sentences()
    both = idx.search("budget décembre", require_all=True)
    either = idx.search("budget décembre", require_all=False)
    assert {h.unit_id for h in both} <= {h.unit_id for h in either}
    assert len(either) >= len(both)


# ---------------------------------------------------------------------- FAISS

def _vec(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, dim), dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_faiss_rebuild_search_and_persist(tmp_path):
    import faiss

    ids = [f"u{i}" for i in range(8)]
    mat = _vec(8, 16)
    fv = FaissVectors(tmp_path / "store.db")
    fv.rebuild(ids, mat, rowid_of={u: i for i, u in enumerate(ids)})
    assert len(fv) == 8
    hits = fv.search(mat[0], k=3)
    assert hits and hits[0][0] == "u0"                  # exact index finds itself
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)   # normalised inner product
    assert (tmp_path / "store.base.faiss").exists()

    again = FaissVectors(tmp_path / "store.db")
    assert again.load("base") is True and again.dim == 16
    assert again.load("refined") is False               # nothing written for it


def test_faiss_rejects_a_query_that_is_out_of_dimension(tmp_path):
    """`rebuild` records the index dimension, so a mismatched query must raise
    rather than return an empty list: empty would look like 'nothing matched'."""
    fv = FaissVectors(tmp_path / "s.db")
    fv.rebuild(["u1"], _vec(1, 16), rowid_of={"u1": 0})
    with pytest.raises(Exception):
        fv.search(np.ones(7, dtype="float32"), k=1)


def test_faiss_filter_escalates_to_brute_force_when_overfetch_is_short(tmp_path):
    """OVERFETCH is 8: with 64 vectors and k=4 the index returns 32 candidates, so
    allowing only the *last* unit forces the exact-fallback path."""
    ids = [f"u{i}" for i in range(64)]
    mat = _vec(64, 24, seed=3)
    fv = FaissVectors(tmp_path / "f.db")
    fv.rebuild(ids, mat, rowid_of={u: i for i, u in enumerate(ids)})

    hits = fv.search(mat[-1], k=4, allow_rowids={63})
    assert [h[0] for h in hits] == ["u63"], "the allowed unit must survive filtering"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    # nothing allowed at all is an empty answer, not an exception
    assert fv.search(mat[-1], k=4, allow_rowids=set()) == []


def test_faiss_empty_rebuild_clears_the_index(tmp_path):
    fv = FaissVectors(tmp_path / "e.db")
    fv.rebuild(["u1"], _vec(1, 8), rowid_of={"u1": 0})
    assert len(fv) == 1
    fv.rebuild([], np.zeros((0, 8), dtype="float32"), rowid_of={})
    assert len(fv) == 0 and fv.search(np.ones(8, dtype="float32"), k=3) == []
    fv.close()
    assert len(fv) == 0


# ---------------------------------------------------------- abstain / intent head

class _FakeModel:
    encoder_id = "fake-encoder"
    with_context = False

    def __init__(self, kind="info", intent="none", conf=0.9):
        self.kind, self.intent, self.conf = kind, intent, conf

    def predict_batch(self, texts, speakers=None, contexts=None):
        preds, probs = {}, {}
        for name, labels in HEADS.items():
            chosen = (self.kind if name == "kind" else
                      self.intent if name == "intent" else labels[0])
            idx = labels.index(chosen)
            row = np.full((len(texts), len(labels)),
                          (1.0 - self.conf) / max(1, len(labels) - 1))
            row[:, idx] = self.conf
            preds[name] = np.array([idx] * len(texts))
            probs[name] = row
        return preds, probs


def test_low_confidence_sets_abstained_without_changing_the_label():
    """6.2.5: a wrong label is still written to the unit, so the flag is the only
    place the doubt survives."""
    unit = make_unit("Le budget est validé à 2,4 M€.")
    r = ModelReader(_FakeModel(conf=0.5), abstain_threshold=0.8).read(unit)
    assert isinstance(r, Reading) and r.abstained is True
    assert r.kind == "info" and r.status in HEADS["status"]

    confident = ModelReader(_FakeModel(conf=0.95), abstain_threshold=0.8).read(unit)
    assert confident.abstained is False
    assert ModelReader(_FakeModel(conf=0.1), abstain_threshold=0.0).read(unit) \
        .abstained is False, "a zero threshold must disable abstention entirely"


def test_intent_is_only_ever_none_on_statement_rows():
    """The intent head is gold for query rows only, so a statement's argmax must
    never reach the router as a routing decision."""
    statement = make_unit("Le budget est validé.")
    query = make_unit("Qui s'occupe du budget ?")
    assert ModelReader(_FakeModel(kind="info", intent="compare")).read(statement).intent \
        == "none"
    assert ModelReader(_FakeModel(kind="query", intent="compare")).read(query).intent \
        == "compare"


def test_context_units_truncates_to_the_last_n_and_zero_disables_it():
    ctx = [make_unit(f"u{i}") for i in range(5)]
    seen = {}

    class _Spy(_FakeModel):
        def predict_batch(self, texts, speakers=None, contexts=None):
            seen["ctx"] = list(contexts or [])
            return super().predict_batch(texts, speakers, contexts)

    ModelReader(_Spy(), context_units=2).read(make_unit("x"), ctx)
    assert seen["ctx"] == ["u3 | u4"], seen["ctx"]

    seen.clear()
    ModelReader(_Spy(), context_units=0).read(make_unit("x"), ctx)
    assert seen["ctx"] == [None]
    seen.clear()
    ModelReader(_Spy(), context_units=3).read(make_unit("x"), None)
    assert seen["ctx"] == [None], "no context supplied must not invent one"


def test_reader_meta_records_what_the_reader_actually_is():
    strict = ModelReader(_FakeModel(), context_units=2, abstain_threshold=0.6)
    meta = reader_meta(strict)
    assert meta == {"backend": "model", "encoder": "fake-encoder",
                    "with_context": False, "context_units": 2,
                    "abstain_threshold": 0.6}
    from engine.read.baseline_rules import RuleReader

    assert reader_meta(RuleReader())["backend"] == "rules"
    assert "intent" not in GATED_HEADS


def test_last_confidence_is_recorded_for_every_head():
    r = ModelReader(_FakeModel(conf=0.7))
    r.read(make_unit("x"))
    assert set(r.last_confidence) == set(HEADS)
    assert all(0.0 <= v <= 1.0 for v in r.last_confidence.values())


# ---------------------------------------------------------------- turns helper

def test_split_sentences_offsets_slice_back_to_the_surface():
    """6.1's span-reconstruction invariant: if `text[start:end]` ever drifts from
    `surface`, BM25 evidence and the citation shown to the user disagree."""
    text = "Le budget est validé à 2,4 M€. Marie l'a revu ! Et le risque ? Rien."
    parts = T.split_sentences(text)
    assert len(parts) >= 3
    for s in parts:
        assert text[s.start:s.end].strip() == s.text.strip()
    assert T.split_sentences("") == []


def test_token_window_splitter_keeps_every_word_exactly_once():
    """The unpunctuated-ASR path (6.1) must not drop or duplicate words: those units
    are what BM25 indexes and what a citation quotes. `min_tokens` swallows a
    short tail into the previous window, so the last window can be longer."""
    words = " ".join(f"mot{i}" for i in range(60))
    windows = T.split_by_token_window(words, max_tokens=10, min_tokens=3)
    assert len(windows) == 6
    joined = " ".join(w.text for w in windows)
    assert joined == words, "words reordered, dropped or duplicated"
    # offsets reconstruct the original exactly
    assert "".join(words[w.start:w.end] if i == 0 else f" {words[w.start:w.end]}"
                   for i, w in enumerate(windows))
    # a tail shorter than min_tokens is merged, never left as a 1-word unit
    tail = T.split_by_token_window(" ".join(f"w{i}" for i in range(11)),
                                   max_tokens=10, min_tokens=3)
    assert len(tail) == 1 and T.count_tokens(tail[0].text) == 11
    assert T.split_by_token_window("", 10) == []


def test_punctuation_predicates_agree_with_the_segmenter_choice():
    """`is_unpunctuated` decides whether the token-window path or the sentence path
    runs, so the two predicates must not contradict each other on the same text."""
    assert T.is_unpunctuated("le budget validé") is True
    assert T.has_sentence_end("le budget validé") is False
    assert T.is_unpunctuated("Le budget est validé.") is False
    assert T.has_sentence_end("Le budget est validé.") is True
    # ASR output ending in a bare number still counts as unpunctuated
    assert T.is_unpunctuated("réunion dix heures 30") is True
    assert T.is_unpunctuated("réunion à 10h ?") is False


def test_repair_glued_abbreviations_moves_a_tail_without_inventing_text():
    """The chunklet splitter can glue an abbreviation's trailing period onto the
    next sentence; the repair must move words, never create or lose them."""
    well_formed = ["Le budget est validé.", "Marie le pilote."]
    assert T.repair_glued_abbreviations(well_formed) == well_formed

    glued = ["La réunion est à 10 h", "N. Marie valide le budget"]
    fixed = T.repair_glued_abbreviations(glued)
    src = sorted(" ".join(glued).split())
    out = sorted(" ".join(fixed).split())
    assert src == out, f"words invented or dropped: {glued} -> {fixed}"
    assert all(s.strip() for s in fixed)


def test_count_tokens_counts_words_not_characters():
    assert T.count_tokens("") == 0
    assert T.count_tokens("un deux trois") == 3
    assert T.count_tokens("Le budget d'Atlas est validé à 2,4 M€.") > 5
