"""
LLM backends for the query parser.

Written against stdlib urllib on purpose - no new dependency, and it works on
Kaggle where package installs are slow.

Three properties that matter more than the model choice:

  1. Every backend can fail. `complete()` returns None instead of raising, and
     the caller falls back to the rule parser.
  2. Every call is cached to disk. Rehearse your demo queries once and they
     are answered from cache forever after. A rate limit cannot kill you on
     stage.
  3. The default backend is offline. Nothing depends on a network call unless
     someone explicitly opts in via SATQUERY_LLM_BACKEND.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import config


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

class ParseCache:
    """Content-addressed cache of prompt -> completion."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.LLM_CACHE_PATH
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    @staticmethod
    def key(backend: str, model: str, prompt: str) -> str:
        raw = f"{backend}|{model}|{prompt}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        try:
            self.path.write_text(json.dumps(self._data, indent=0))
        except Exception:
            pass          # a cache write failing must never break a query

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------

class LLMBackend(Protocol):
    backend_id: str
    model: str

    def available(self) -> bool: ...
    def complete(self, prompt: str) -> str | None: ...


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict | None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class OfflineBackend:
    """Default. Never calls out. Forces the rule-parser fallback path."""

    backend_id = "offline"
    model = "none"

    def available(self) -> bool:
        return False

    def complete(self, prompt: str) -> str | None:
        return None


class OpenAICompatBackend:
    """Any OpenAI-compatible chat endpoint: Groq, OpenRouter, local vLLM."""

    backend_id = "openai_compat"

    def __init__(self, base: str | None = None, model: str | None = None,
                 key: str | None = None, timeout: float | None = None) -> None:
        self.base = (base or config.OPENAI_COMPAT_BASE).rstrip("/")
        self.model = model or config.OPENAI_COMPAT_MODEL
        self.key = key or config.OPENAI_COMPAT_KEY
        self.timeout = timeout or config.LLM_TIMEOUT_S

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, prompt: str) -> str | None:
        if not self.available():
            return None
        data = _post_json(
            f"{self.base}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {self.key}"},
            self.timeout,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError):
            return None


class GeminiBackend:
    """Google AI Studio free tier."""

    backend_id = "gemini"

    def __init__(self, model: str | None = None, key: str | None = None,
                 timeout: float | None = None) -> None:
        self.model = model or config.GEMINI_MODEL
        self.key = key or config.GEMINI_KEY
        self.timeout = timeout or config.LLM_TIMEOUT_S

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, prompt: str) -> str | None:
        if not self.available():
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.key}")
        data = _post_json(
            url,
            {"contents": [{"parts": [{"text": prompt}]}],
             "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200}},
            {"Content-Type": "application/json"},
            self.timeout,
        )
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (TypeError, KeyError, IndexError):
            return None


_BACKENDS = {
    "offline": OfflineBackend,
    "openai_compat": OpenAICompatBackend,
    "gemini": GeminiBackend,
}


def get_backend(name: str | None = None) -> LLMBackend:
    """Build the configured backend. Unknown names degrade to offline."""
    key = (name or config.LLM_BACKEND).lower()
    return _BACKENDS.get(key, OfflineBackend)()
