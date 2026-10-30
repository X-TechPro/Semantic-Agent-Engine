"""LLM access behind one narrow interface (6.11).

The LLM may only *render* text from an evidence bundle it is handed, and may
never decide retrieval, relations or routing. Backends:

  * ``template`` (default) — no LLM at all. Rendering falls back to the
    deterministic templates, which is also the "LLM-off" ablation row and what
    every test runs on.
  * ``ollama`` / ``openai`` — HTTP to a loopback endpoint; used only when
    explicitly configured, and always followed by `agent.verify`.

No key is ever read from a config file committed here; endpoints come from env.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


class LLMBackend:
    name = "none"

    def generate(self, prompt: str, schema: dict[str, Any] | None = None,
                 system: str | None = None) -> dict[str, Any] | None:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self.name != "none"


class TemplateBackend(LLMBackend):
    name = "template"

    def generate(self, prompt, schema=None, system=None):
        return None  # signals "use the deterministic renderer"


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str = "qwen3.5:4b",
                 endpoint: str | None = None, timeout: float = 120.0,
                 extra_options: dict[str, Any] | None = None):
        self.model = model
        self.endpoint = endpoint or os.environ.get("SAE_OLLAMA_URL",
                                                   "http://127.0.0.1:11434")
        self.timeout = timeout
        self.extra_options = dict(extra_options or {})

    def generate(self, prompt, schema=None, system=None):
        body = {"model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.0, **self.extra_options}}
        if schema is not None:
            body["format"] = schema
        req = urllib.request.Request(f"{self.endpoint}/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:  # boundary: external process, so we tolerate absence explicitly
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        text = raw.get("response", "")
        return _coerce_json(text, schema)


class OpenAICompatibleBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", endpoint: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0):
        self.model = model
        self.endpoint = endpoint or os.environ.get("SAE_LLM_URL",
                                                   "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("SAE_LLM_KEY")
        self.timeout = timeout

    def generate(self, prompt, schema=None, system=None):
        if not self.api_key:
            raise LLMUnavailable("SAE_LLM_KEY not set")
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": 0.0}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema":
                                       {"name": "out", "schema": schema,
                                        "strict": True}}
        req = urllib.request.Request(f"{self.endpoint}/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        return _coerce_json(raw["choices"][0]["message"]["content"], schema)


class LLMUnavailable(RuntimeError):
    pass


def build_llm(cfg) -> LLMBackend:
    backend = (cfg["agent"]["llm"]["backend"] if "agent" in cfg else "template")
    if backend in (None, "template"):
        return TemplateBackend()
    if backend == "ollama":
        # extra_options exists for callers that need it. It is NOT set here on
        # purpose: think:False was tried for the subject extractor and measured
        # no difference on qwen3.5:4b (16-51 s/call either way — smoke +
        # ctx-smoke, 2026-10-21), so nothing sold it as a speed fix and the
        # shipped default stays plain.
        return OllamaBackend(model=cfg["agent"]["llm"].get("model_id") or "qwen3.5:4b")
    if backend in ("openai", "openai_compatible"):
        return OpenAICompatibleBackend(model=cfg["agent"]["llm"].get("model_id")
                                       or "gpt-4o-mini")
    raise ValueError(f"unknown llm backend {backend!r}")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _coerce_json(text: str, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    if schema is None:
        return obj
    # strict-schema: anything unexpected is dropped rather than trusted
    props = schema.get("properties", {})
    return {k: v for k, v in obj.items() if k in props} or obj
