"""Cached optional-provider clients (OpenAI / Anthropic / Gemini).

Every LLM/VLM module used to construct its own client per call; this is the
single place that builds and caches them (and validates the API key).
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from imagecb.config import SETTINGS

_lock = threading.Lock()
_openai_client: Optional[Any] = None
_anthropic_client: Optional[Any] = None
_genai_client: Optional[Any] = None


def get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        with _lock:
            if _openai_client is None:
                if not SETTINGS.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY is not configured")
                from openai import OpenAI

                _openai_client = OpenAI(api_key=SETTINGS.openai_api_key)
    return _openai_client


def get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        with _lock:
            if _anthropic_client is None:
                if not SETTINGS.anthropic_api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY is not configured")
                import anthropic

                _anthropic_client = anthropic.Anthropic(
                    api_key=SETTINGS.anthropic_api_key
                )
    return _anthropic_client


def get_genai_client() -> Any:
    """Cached Google GenAI client for Nano Banana image editing."""
    global _genai_client
    if _genai_client is None:
        with _lock:
            if _genai_client is None:
                from google import genai

                from imagecb.models.secrets import get_gemini_api_key

                _genai_client = genai.Client(api_key=get_gemini_api_key())
    return _genai_client


def reset_provider_clients() -> None:
    """Drop cached clients (tests / key rotation)."""
    global _openai_client, _anthropic_client, _genai_client
    with _lock:
        _openai_client = None
        _anthropic_client = None
        _genai_client = None
