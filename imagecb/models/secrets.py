"""Load secrets from env or AWS Secrets Manager (Gemini / Nano Banana)."""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached_gemini_key: Optional[str] = None
_gemini_resolved = False


def parse_gemini_secret_string(raw: str) -> str:
    """Accept a plaintext API key or JSON with common key names."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Gemini secret string is empty")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini secret JSON is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError("Gemini secret JSON must be an object")
        for key in (
            "api_key",
            "API_KEY",
            "GEMINI_API_KEY",
            "gemini_api_key",
            "GeminiApiKey",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(
            "Gemini secret JSON must include api_key, API_KEY, GEMINI_API_KEY, "
            "or gemini_api_key"
        )
    return text


def _fetch_from_secrets_manager() -> str:
    import boto3

    # Uses the default credential chain (env keys, shared config, instance/task
    # role) — same path as S3 and Bedrock inside Docker/ECS.
    client = boto3.client(
        "secretsmanager",
        region_name=SETTINGS.gemini_secret_region,
    )
    resp = client.get_secret_value(SecretId=SETTINGS.gemini_secret_name)
    if "SecretString" in resp and resp["SecretString"]:
        return parse_gemini_secret_string(resp["SecretString"])
    binary = resp.get("SecretBinary")
    if binary:
        return parse_gemini_secret_string(bytes(binary).decode("utf-8"))
    raise RuntimeError(
        f"Secrets Manager secret {SETTINGS.gemini_secret_name!r} has no SecretString"
    )


def _safe_error_message(exc: BaseException) -> str:
    """Public diagnostic text — never include secret values."""
    name = type(exc).__name__
    text = str(exc).strip() or name
    # Truncate long AWS error blobs; keep the actionable prefix.
    if len(text) > 280:
        text = text[:277] + "..."
    return f"{name}: {text}" if not text.startswith(name) else text


def get_gemini_api_key(*, force_refresh: bool = False) -> str:
    """Return the Gemini API key (env first, else Secrets Manager). Cached."""
    global _cached_gemini_key, _gemini_resolved
    if not force_refresh and _gemini_resolved and _cached_gemini_key:
        return _cached_gemini_key
    with _lock:
        if not force_refresh and _gemini_resolved and _cached_gemini_key:
            return _cached_gemini_key
        env_key = (SETTINGS.gemini_api_key or "").strip()
        if env_key:
            _cached_gemini_key = env_key
            _gemini_resolved = True
            return _cached_gemini_key
        try:
            _cached_gemini_key = _fetch_from_secrets_manager()
        except Exception as exc:
            logger.exception(
                "Failed to load Gemini API key from Secrets Manager "
                "(secret=%s region=%s)",
                SETTINGS.gemini_secret_name,
                SETTINGS.gemini_secret_region,
            )
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY or grant "
                f"secretsmanager:GetSecretValue on secret "
                f"{SETTINGS.gemini_secret_name!r} in {SETTINGS.gemini_secret_region}. "
                f"Underlying error: {_safe_error_message(exc)}"
            ) from None
        _gemini_resolved = True
        return _cached_gemini_key


def is_nano_banana_available() -> bool:
    """True when a Gemini key can be resolved (does not call Gemini)."""
    try:
        get_gemini_api_key()
        return True
    except Exception:
        return False


def nano_banana_status(*, force_refresh: bool = False) -> dict:
    """Availability plus safe diagnostics (no secret values)."""
    base = {
        "available": False,
        "model": SETTINGS.nano_banana_model,
        "source": None,
        "secret_name": SETTINGS.gemini_secret_name,
        "secret_region": SETTINGS.gemini_secret_region,
        "error": None,
    }
    env_key = (SETTINGS.gemini_api_key or "").strip()
    if env_key:
        try:
            get_gemini_api_key(force_refresh=force_refresh)
            base["available"] = True
            base["source"] = "env"
            return base
        except Exception as exc:
            base["source"] = "env"
            base["error"] = _safe_error_message(exc)
            return base
    try:
        get_gemini_api_key(force_refresh=force_refresh)
        base["available"] = True
        base["source"] = "secrets_manager"
        return base
    except Exception as exc:
        base["source"] = "secrets_manager"
        base["error"] = _safe_error_message(exc)
        return base


def reset_gemini_secret_cache() -> None:
    """Drop cached key (tests / rotation)."""
    global _cached_gemini_key, _gemini_resolved
    with _lock:
        _cached_gemini_key = None
        _gemini_resolved = False
