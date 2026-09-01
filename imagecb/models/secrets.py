"""Load secrets from env or AWS Secrets Manager (Gemini / Nano Banana)."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached_config: Optional["GeminiConfig"] = None
_config_resolved = False

_KEY_JSON_FIELDS = (
    "api_key",
    "API_KEY",
    "GEMINI_API_KEY",
    "gemini_api_key",
    "GeminiApiKey",
)
_PROJECT_JSON_FIELDS = (
    "project_id",
    "project",
    "GCP_PROJECT_ID",
    "gcp_project_id",
    "vertex_project",
)
_LOCATION_JSON_FIELDS = (
    "location",
    "vertex_location",
    "GEMINI_VERTEX_LOCATION",
)


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    project: Optional[str]
    location: str
    backend: str  # "vertex" | "google_ai"

    @property
    def is_vertex(self) -> bool:
        return self.backend == "vertex"


def _first_string(payload: dict, keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_secret_payload(raw: str) -> tuple[str, Optional[str], Optional[str]]:
    """Return (api_key, project, location) from plaintext or JSON."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Gemini secret string is empty")
    if not text.startswith("{"):
        return text, None, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini secret JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("Gemini secret JSON must be an object")
    api_key = _first_string(payload, _KEY_JSON_FIELDS)
    if not api_key:
        raise ValueError(
            "Gemini secret JSON must include api_key, API_KEY, GEMINI_API_KEY, "
            "or gemini_api_key"
        )
    project = _first_string(payload, _PROJECT_JSON_FIELDS)
    location = _first_string(payload, _LOCATION_JSON_FIELDS)
    return api_key, project, location


def parse_gemini_secret_string(raw: str) -> str:
    """Accept a plaintext API key or JSON with common key names."""
    return _parse_secret_payload(raw)[0]


def _build_config(
    *,
    api_key: str,
    project: Optional[str],
    location: Optional[str],
) -> GeminiConfig:
    project_id = (project or "").strip() or None
    region = (location or SETTINGS.gemini_vertex_location or "us-central1").strip()
    if not region:
        region = "us-central1"
    backend = "vertex" if project_id else "google_ai"
    return GeminiConfig(
        api_key=api_key.strip(),
        project=project_id,
        location=region,
        backend=backend,
    )


def parse_gemini_secret_config(raw: str) -> GeminiConfig:
    """Parse a secret string into a full Gemini config (key + optional Vertex fields)."""
    key, project, location = _parse_secret_payload(raw)
    return _build_config(api_key=key, project=project, location=location)


_AWS_CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


@contextmanager
def _without_blank_aws_credential_env() -> Iterator[None]:
    """Drop blank AWS key env vars so boto3 can fall back to the instance role."""
    saved: dict[str, str] = {}
    for name in _AWS_CREDENTIAL_ENV_VARS:
        value = os.environ.get(name)
        if value is not None and not value.strip():
            saved[name] = value
            del os.environ[name]
    try:
        yield
    finally:
        os.environ.update(saved)


def _fetch_secret_string_from_secrets_manager() -> str:
    import boto3

    with _without_blank_aws_credential_env():
        client = boto3.client(
            "secretsmanager",
            region_name=SETTINGS.gemini_secret_region,
        )
        resp = client.get_secret_value(SecretId=SETTINGS.gemini_secret_name)
    if "SecretString" in resp and resp["SecretString"]:
        return resp["SecretString"]
    binary = resp.get("SecretBinary")
    if binary:
        return bytes(binary).decode("utf-8")
    raise RuntimeError(
        f"Secrets Manager secret {SETTINGS.gemini_secret_name!r} has no SecretString"
    )


def _fetch_from_secrets_manager() -> str:
    """Return the API key from Secrets Manager (legacy helper for tests)."""
    return parse_gemini_secret_string(_fetch_secret_string_from_secrets_manager())


def _safe_error_message(exc: BaseException) -> str:
    """Public diagnostic text — never include secret values."""
    name = type(exc).__name__
    text = str(exc).strip() or name
    if len(text) > 280:
        text = text[:277] + "..."
    return f"{name}: {text}" if not text.startswith(name) else text


def _apply_env_vertex_overrides(
    project: Optional[str], location: Optional[str]
) -> tuple[Optional[str], str]:
    env_project = (SETTINGS.gemini_vertex_project or "").strip() or None
    env_location = (SETTINGS.gemini_vertex_location or "us-central1").strip()
    return env_project or project, env_location or location or "us-central1"


def get_gemini_vertex_config(*, force_refresh: bool = False) -> GeminiConfig:
    """Return Gemini API key plus Vertex project/location (env first, else SM). Cached."""
    global _cached_config, _config_resolved
    if not force_refresh and _config_resolved and _cached_config is not None:
        return _cached_config
    with _lock:
        if not force_refresh and _config_resolved and _cached_config is not None:
            return _cached_config
        env_key = (SETTINGS.gemini_api_key or "").strip()
        if env_key:
            project, location = _apply_env_vertex_overrides(None, None)
            _cached_config = _build_config(
                api_key=env_key, project=project, location=location
            )
            _config_resolved = True
            return _cached_config
        try:
            raw = _fetch_secret_string_from_secrets_manager()
            key, project, location = _parse_secret_payload(raw)
            project, location = _apply_env_vertex_overrides(project, location)
            _cached_config = _build_config(
                api_key=key, project=project, location=location
            )
        except Exception as exc:
            logger.exception(
                "Failed to load Gemini config from Secrets Manager "
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
        _config_resolved = True
        return _cached_config


def get_gemini_api_key(*, force_refresh: bool = False) -> str:
    """Return the Gemini API key (env first, else Secrets Manager). Cached."""
    return get_gemini_vertex_config(force_refresh=force_refresh).api_key


def is_nano_banana_available() -> bool:
    """True when Gemini config resolves (does not call Gemini)."""
    try:
        get_gemini_vertex_config()
        return True
    except Exception:
        return False


def _status_base() -> dict:
    return {
        "available": False,
        "model": SETTINGS.nano_banana_model,
        "backend": None,
        "project_id": None,
        "location": SETTINGS.gemini_vertex_location,
        "source": None,
        "secret_name": SETTINGS.gemini_secret_name,
        "secret_region": SETTINGS.gemini_secret_region,
        "error": None,
    }


def nano_banana_status(*, force_refresh: bool = False) -> dict:
    """Availability plus safe diagnostics (no secret values)."""
    base = _status_base()
    env_key = (SETTINGS.gemini_api_key or "").strip()
    source = "env" if env_key else "secrets_manager"
    base["source"] = source
    try:
        config = get_gemini_vertex_config(force_refresh=force_refresh)
        base["available"] = True
        base["backend"] = config.backend
        base["project_id"] = config.project
        base["location"] = config.location
        base["error"] = None
        return base
    except Exception as exc:
        base["error"] = _safe_error_message(exc)
        return base


def reset_gemini_secret_cache() -> None:
    """Drop cached config (tests / rotation)."""
    global _cached_config, _config_resolved
    with _lock:
        _cached_config = None
        _config_resolved = False
