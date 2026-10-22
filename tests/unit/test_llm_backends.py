"""LLM backends (6.11) — the boundary between "template-only" and a real endpoint.

`agent/llm.py` was 92 statements at 0% coverage. That is the file whose whole job is
to *refuse*: it must never invent an answer, never send a key it wasn't given, and
never trust a model's JSON. None of that is exercised by the rest of the suite,
because every measured path runs `agent.llm.backend: template`.

No network here: `urlopen` is replaced with a recording stub, so the tests assert
the request that *would* go out (endpoint, body, headers) and how each reply shape
is coerced or rejected.
"""

from __future__ import annotations

import json

import pytest

from engine.agent import llm as L
from engine.config import load_config

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def transport(monkeypatch):
    """Capture the request, replay a chosen reply."""
    seen = {}

    def install(payload):
        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            seen["headers"] = dict(req.headers)
            seen["timeout"] = timeout
            return FakeResponse(payload)

        monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
        return seen

    return install


# ------------------------------------------------------------------ selection

def test_build_llm_defaults_to_template_so_nothing_can_call_home():
    """The measured configuration must not be able to reach a network."""
    cfg = load_config()
    assert cfg["agent"]["llm"]["backend"] == "template"
    backend = L.build_llm(cfg)
    assert isinstance(backend, L.TemplateBackend)
    assert backend.generate("anything") is None      # template == no generation


def test_build_llm_unknown_backend_raises_inst_of_silently_disabling():
    cfg = {"agent": {"llm": {"backend": "gpt9", "model_id": None}}}
    with pytest.raises(ValueError, match="unknown llm backend"):
        L.build_llm(cfg)


def test_template_backend_is_the_base_contracts_only_generate():
    base = L.LLMBackend()
    with pytest.raises(NotImplementedError):
        base.generate("x")


# ------------------------------------------------------------------- ollama

def test_ollama_request_shape_and_reply_parsing(transport):
    seen = transport({"response": '{"answer": "2,4 M€", "extra": 1}'})
    b = L.OllamaBackend(model="m1", endpoint="http://127.0.0.1:11434")
    out = b.generate("budget ?", schema={"properties": {"answer": {}}})
    assert seen["url"].endswith("/api/generate")
    assert seen["body"]["model"] == "m1" and seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0.0     # deterministic by fiat
    assert seen["body"]["format"] == {"properties": {"answer": {}}}
    assert out == {"answer": "2,4 M€"}          # `extra` dropped: schema is strict


def test_ollama_without_schema_returns_object_untouched(transport):
    transport({"response": '{"a": 1}'})
    b = L.OllamaBackend(endpoint="http://x")
    assert b.generate("p") == {"a": 1}


def test_ollama_endpoint_comes_from_env_not_the_repo(transport, monkeypatch):
    """6.11: endpoints are host config, never committed values. `transport` is
    installed first so no request can escape even if construction misbehaves."""
    seen = transport({"response": "{}"})
    monkeypatch.setenv("SAE_OLLAMA_URL", "http://127.0.0.1:9999")
    b = L.OllamaBackend()
    assert b.endpoint == "http://127.0.0.1:9999"
    b.generate("p")
    assert seen["url"].startswith("http://127.0.0.1:9999/")


def test_unreachable_ollama_raises_llm_unavailable(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(L.urllib.request, "urlopen", boom)
    with pytest.raises(L.LLMUnavailable, match="connection refused"):
        L.OllamaBackend(endpoint="http://x").generate("p")


# -------------------------------------------------------------------- openai

def test_openai_refuses_without_a_key_and_never_logs_one(transport, monkeypatch):
    """The pre-request guard is the point: it must fail before any socket opens.

    The env var is deleted explicitly: the constructor falls back to
    `SAE_LLM_KEY`, so on a machine that happens to have it set this test would
    silently be testing the *success* path while claiming to check the refusal.
    """
    monkeypatch.delenv("SAE_LLM_KEY", raising=False)
    seen = transport({"choices": [{"message": {"content": "{}"}}]})
    b = L.OpenAICompatibleBackend(api_key=None)
    assert b.api_key is None
    with pytest.raises(L.LLMUnavailable, match="SAE_LLM_KEY not set"):
        b.generate("p")
    assert seen == {}                              # nothing was sent


def test_openai_sends_bearer_and_system_prompt(transport):
    seen = transport({"choices": [{"message": {"content": '{"answer": "ok"}'}}]})
    b = L.OpenAICompatibleBackend(model="g", endpoint="http://api/v1", api_key="sk-1")
    out = b.generate("user text", schema={"properties": {"answer": {}}},
                     system="be brief")
    assert seen["headers"].get("Authorization") == "Bearer sk-1"
    roles = [m["role"] for m in seen["body"]["messages"]]
    assert roles == ["system", "user"]
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert out == {"answer": "ok"}


def test_openai_http_error_becomes_llm_unavailable(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("401")

    monkeypatch.setattr(L.urllib.request, "urlopen", boom)
    with pytest.raises(L.LLMUnavailable):
        L.OpenAICompatibleBackend(api_key="k", endpoint="http://x").generate("p")


# ---------------------------------------------------- reply coercion (both paths)

@pytest.mark.parametrize("text,want", [
    ('{"a": 1}', {"a": 1}),                                  # plain json
    ('```json\n{"a": 1}\n```', {"a": 1}),                   # fenced
    ('```\n{"a": 1}\n```', {"a": 1}),                        # fence w/o language
    ('Sure! {"a": 1} hope that helps', {"a": 1}),           # prose wrapped json
    ('', None),                                              # empty
    ('no json here', None),                                  # garbage
    ('[1,2]', None),                                         # array, not object
    ('{"a":', None),                                         # truncated
])
def test_coerce_json_accepts_shapes_and_refuses_the_rest(text, want):
    assert L._coerce_json(text, None) == want


def test_coerce_json_drops_keys_the_schema_did_not_declare():
    """A model that answers with extra fields must not have them trusted into an
    operation argument."""
    out = L._coerce_json('{"answer": "ok", "recipient": "x@y"}',
                         {"properties": {"answer": {}}})
    assert out == {"answer": "ok"}


def test_coerce_json_keeps_unsanitisable_output_rather_than_losing_it():
    """If the schema filters *everything* away, returning the object beats returning
    nothing — the caller's verifier still checks the fields."""
    assert L._coerce_json('{"other": 1}', {"properties": {"answer": {}}}) \
        == {"other": 1}
