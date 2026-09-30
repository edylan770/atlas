"""Nano Banana secrets parsing, pending edits, and edit API."""

from __future__ import annotations

import hashlib
import io
from base64 import b64encode
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from imagecb.api.edit_sessions import EditSession, EditTurn, clear_edit_sessions
from imagecb.api.edit_routes import EditTurnRequest, edit_turn
from imagecb.api import rate_limit
from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.models.secrets import (
    _fetch_from_secrets_manager,
    _without_blank_aws_credential_env,
    get_gemini_vertex_config,
    nano_banana_status,
    parse_gemini_secret_config,
    parse_gemini_secret_string,
    reset_gemini_secret_cache,
)
from imagecb.models.providers import get_genai_client, reset_provider_clients
from imagecb.models.image_edit import ImageEditError, _provider_error, edit_image, generate_image
from imagecb.pending_edits import (
    PendingEditIngestError,
    accept_pending_edit,
    create_pending_edit,
    decline_pending_edit,
    list_pending_edits,
)
from imagecb.storage import blob_store, metadata_db
from imagecb.storage.metadata_db import ImageRecord, session_scope
from imagecb.telemetry.schema import ensure_telemetry_schema


@pytest.fixture(autouse=True)
def _reset():
    reset_gemini_secret_cache()
    reset_provider_clients()
    clear_edit_sessions()
    rate_limit.reset()
    yield
    reset_gemini_secret_cache()
    reset_provider_clients()
    clear_edit_sessions()
    rate_limit.reset()


def _png_bytes(color=(10, 20, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _patch_settings(settings):
    return (
        patch("imagecb.config.SETTINGS", settings),
        patch("imagecb.storage.blob_store.SETTINGS", settings),
        patch("imagecb.storage.metadata_db.SETTINGS", settings),
        patch("imagecb.pending_edits.SETTINGS", settings),
        patch("imagecb.images.SETTINGS", settings),
    )


def _open_tmp_db(settings):
    """Point metadata_db at a fresh sqlite under the patched SETTINGS."""
    metadata_db.dispose_engine()
    metadata_db.reopen_engine()
    ensure_telemetry_schema()


def test_parse_gemini_secret_plaintext():
    assert parse_gemini_secret_string("  abc-key  ") == "abc-key"


def test_parse_gemini_secret_json_variants():
    assert parse_gemini_secret_string('{"api_key":"k1"}') == "k1"
    assert parse_gemini_secret_string('{"API_KEY":"k0"}') == "k0"
    assert parse_gemini_secret_string('{"GEMINI_API_KEY":"k2"}') == "k2"
    assert parse_gemini_secret_string('{"gemini_api_key":"k3"}') == "k3"


def test_parse_gemini_secret_rejects_empty_json():
    with pytest.raises(ValueError):
        parse_gemini_secret_string("{}")


def test_parse_gemini_secret_config_vertex_json():
    raw = (
        '{"api_key":"k1","project_id":"spatial-airship-460318-g1",'
        '"location":"us-central1"}'
    )
    config = parse_gemini_secret_config(raw)
    assert config.api_key == "k1"
    assert config.project == "spatial-airship-460318-g1"
    assert config.location == "us-central1"
    assert config.backend == "vertex_express"
    assert config.is_vertex is True


def test_parse_gemini_secret_config_vertex_express_plaintext_key(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(SETTINGS, gemini_vertex_express=False),
    )
    config = parse_gemini_secret_config("AQ.test-vertex-express-key")
    assert config.backend == "vertex_express"
    assert config.project is None


def test_parse_gemini_secret_config_vertex_express_forced_by_env(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(SETTINGS, gemini_vertex_express=True),
    )
    config = parse_gemini_secret_config("plain-developer-key")
    assert config.backend == "vertex_express"


def test_parse_gemini_secret_config_google_ai_plaintext_without_markers(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(SETTINGS, gemini_vertex_express=False),
    )
    config = parse_gemini_secret_config("plain-developer-key")
    assert config.backend == "google_ai"


def test_get_gemini_vertex_config_from_env(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(
            SETTINGS,
            gemini_api_key="env-key",
            gemini_vertex_project="my-gcp-project",
            gemini_vertex_location="us-central1",
        ),
    )
    config = get_gemini_vertex_config(force_refresh=True)
    assert config.api_key == "env-key"
    assert config.project == "my-gcp-project"
    assert config.backend == "vertex_express"


def test_get_genai_client_uses_vertex_when_project_set(monkeypatch):
    captured: dict = {}

    class _FakeClient:
        pass

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return _FakeClient()

    settings = replace(
        SETTINGS,
        gemini_api_key="vertex-key",
        gemini_vertex_project="my-gcp-project",
        gemini_vertex_location="us-central1",
    )
    monkeypatch.setattr("imagecb.models.secrets.SETTINGS", settings)
    monkeypatch.setattr("google.genai.Client", _fake_client)

    client = get_genai_client()
    assert isinstance(client, _FakeClient)
    assert captured["vertexai"] is True
    assert captured["api_key"] == "vertex-key"
    assert "project" not in captured
    assert "location" not in captured


def test_nano_banana_status_from_env(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(SETTINGS, gemini_api_key="env-key-xyz"),
    )
    status = nano_banana_status(force_refresh=True)
    assert status["available"] is True
    assert status["source"] == "env"
    assert status["backend"] == "google_ai"
    assert status["error"] is None


def test_nano_banana_status_vertex_from_env(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(
            SETTINGS,
            gemini_api_key="env-key-xyz",
            gemini_vertex_project="proj-123",
            gemini_vertex_location="us-central1",
        ),
    )
    status = nano_banana_status(force_refresh=True)
    assert status["available"] is True
    assert status["backend"] == "vertex_express"
    assert status["project_id"] == "proj-123"
    assert status["location"] == "us-central1"


@pytest.mark.parametrize("as_base64", [False, True])
def test_edit_image_extracts_inline_image(monkeypatch, as_base64):
    output = _png_bytes(color=(90, 80, 70))
    payload = b64encode(output).decode("ascii") if as_base64 else output
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(data=payload),
                            text=None,
                        )
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        parts=None,
        prompt_feedback=None,
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: response)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    result = edit_image(_png_bytes(), "make it warmer")
    assert Image.open(io.BytesIO(result)).getpixel((0, 0)) == (90, 80, 70)


def test_generate_image_extracts_inline_image(monkeypatch):
    output = _png_bytes(color=(12, 34, 56))
    captured: dict = {}

    def _generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                inline_data=SimpleNamespace(data=output),
                                text=None,
                            )
                        ]
                    ),
                    finish_reason="STOP",
                )
            ],
            parts=None,
            prompt_feedback=None,
        )

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    result = generate_image("a blue chart on a white background")
    assert Image.open(io.BytesIO(result)).getpixel((0, 0)) == (12, 34, 56)
    parts = captured["contents"][0].parts
    assert len(parts) == 1
    assert getattr(parts[0], "inline_data", None) is None


def test_edit_image_reports_text_only_response(monkeypatch):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=None,
                            text="I cannot create that image.",
                        )
                    ]
                ),
                finish_reason="SAFETY",
            )
        ],
        parts=None,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: response)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    with pytest.raises(ImageEditError) as caught:
        edit_image(_png_bytes(), "unsafe request")
    assert caught.value.code == "no_image"
    assert "SAFETY" in str(caught.value)
    assert "I cannot create that image" in str(caught.value)


def test_edit_image_classifies_provider_permission_error(monkeypatch):
    def _raise(**_kwargs):
        raise RuntimeError("403 PERMISSION_DENIED")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_raise)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    with pytest.raises(ImageEditError) as caught:
        edit_image(_png_bytes(), "make it blue")
    assert caught.value.code == "permission_denied"
    assert "image-model access" in str(caught.value)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("401 Unauthorized: invalid api key", "authentication_failed"),
        ("403 PERMISSION_DENIED", "permission_denied"),
        ("404 model not found", "model_unavailable"),
        ("429 RESOURCE_EXHAUSTED quota exceeded", "quota_exceeded"),
        ("Deadline exceeded: request timed out", "provider_timeout"),
        ("boom from SDK", "provider_error"),
    ],
)
def test_provider_error_classifies_common_failures(message, code):
    err = _provider_error(RuntimeError(message))
    assert err.code == code


def test_edit_image_raises_invalid_image_for_undecodable_inline(monkeypatch):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(data=b"not-a-real-image"),
                            text=None,
                        )
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        parts=None,
        prompt_feedback=None,
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: response)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    with pytest.raises(ImageEditError) as caught:
        edit_image(_png_bytes(), "make it blue")
    assert caught.value.code == "invalid_image"


def test_edit_image_empty_candidates_is_no_image(monkeypatch):
    response = SimpleNamespace(
        candidates=[],
        parts=None,
        prompt_feedback=None,
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: response)
    )
    config = SimpleNamespace(backend="vertex_express")
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_gemini_vertex_config", lambda: config
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.get_genai_client", lambda: fake_client
    )

    with pytest.raises(ImageEditError) as caught:
        edit_image(_png_bytes(), "make it blue")
    assert caught.value.code == "no_image"


def _corpus_client(tmp_path, *, settings_overrides=None, edit_image_return=None):
    """Build a TestClient with a one-image corpus and Nano Banana mocked available."""
    overrides = settings_overrides or {}
    settings_kwargs = {
        "blob_storage_backend": "local",
        "data_dir": tmp_path / "data",
        "image_cache_dir": tmp_path / "data" / "images",
        "uploads_dir": tmp_path / "data" / "uploads",
        "sqlite_path": tmp_path / "data" / "test.db",
        "s3_prefix": "imagecb",
        "admin_api_key": "test-admin-secret",
        "gemini_api_key": "fake-gemini-key",
        "llm_rate_limit_per_minute": 0,
    }
    settings_kwargs.update(overrides)
    settings = replace(SETTINGS, **settings_kwargs)
    settings.ensure_dirs()
    (tmp_path / "data" / "images").mkdir(parents=True, exist_ok=True)

    image_id = "corpus-1"
    png = _png_bytes()
    image_path = tmp_path / "data" / "images" / f"{image_id}.png"
    image_path.write_bytes(png)

    edit_return = (
        edit_image_return
        if edit_image_return is not None
        else _png_bytes(color=(200, 100, 50))
    )
    patches = _patch_settings(settings)
    stack = (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patch("imagecb.api.auth.SETTINGS", settings),
        patch("imagecb.api.rate_limit.SETTINGS", settings),
        patch("imagecb.models.secrets.SETTINGS", settings),
        patch("imagecb.api.edit_sessions.SETTINGS", settings),
        patch(
            "imagecb.api.edit_routes.nano_banana_status",
            return_value={"available": True},
        ),
        patch("imagecb.models.image_edit.edit_image", return_value=edit_return),
    )
    return settings, image_id, png, stack


def test_edit_turn_returns_actionable_provider_error(monkeypatch):
    session = EditSession(
        source_image_id="source-1",
        original_image_png=_png_bytes(),
        working_image_png=_png_bytes(),
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.nano_banana_status",
        lambda: {"available": True},
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.get_edit_session",
        lambda _session_id: session,
    )
    monkeypatch.setattr(
        "imagecb.models.image_edit.edit_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ImageEditError("authentication_failed", "Gemini rejected the API key.")
        ),
    )

    with pytest.raises(HTTPException) as caught:
        edit_turn("session-1", EditTurnRequest(prompt="make it blue"))
    assert caught.value.status_code == 502
    assert "[authentication_failed]" in str(caught.value.detail)


def _session_with_turn() -> EditSession:
    png = _png_bytes()
    session = EditSession(
        source_image_id="source-1",
        original_image_png=png,
        working_image_png=png,
    )
    session.turns.append(EditTurn(prompt="first", result_image_png=png))
    return session


def test_edit_turn_rejects_concurrent_request_on_same_session(monkeypatch):
    session = _session_with_turn()
    monkeypatch.setattr(
        "imagecb.api.edit_routes.nano_banana_status", lambda: {"available": True}
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.get_edit_session", lambda _sid: session
    )
    calls = []
    monkeypatch.setattr(
        "imagecb.models.image_edit.edit_image",
        lambda *_a, **_k: calls.append(1) or _png_bytes(),
    )

    assert session.lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as caught:
            edit_turn("session-1", EditTurnRequest(prompt="again"))
    finally:
        session.lock.release()
    assert caught.value.status_code == 409
    assert calls == []
    assert len(session.turns) == 1


def test_edit_turn_enforces_turn_cap(monkeypatch):
    session = _session_with_turn()
    monkeypatch.setattr(
        "imagecb.api.edit_routes.nano_banana_status", lambda: {"available": True}
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.get_edit_session", lambda _sid: session
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.SETTINGS",
        replace(SETTINGS, edit_session_max_turns=1),
    )
    with pytest.raises(HTTPException) as caught:
        edit_turn("session-1", EditTurnRequest(prompt="one too many"))
    assert caught.value.status_code == 409
    assert "limit" in str(caught.value.detail)
    assert not session.lock.locked()


def test_submit_stages_exactly_one_pending(monkeypatch):
    from imagecb.api.edit_routes import submit_session

    session = _session_with_turn()
    monkeypatch.setattr(
        "imagecb.api.edit_routes.get_edit_session", lambda _sid: session
    )
    staged = []
    monkeypatch.setattr(
        "imagecb.api.edit_routes.create_pending_edit",
        lambda **kwargs: staged.append(kwargs) or {"pending_id": "p1"},
    )
    monkeypatch.setattr("imagecb.api.edit_routes.delete_edit_session", lambda _sid: None)

    # An in-flight submit holds the lock: a concurrent one must not stage.
    assert session.lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as busy:
            submit_session("session-1")
    finally:
        session.lock.release()
    assert busy.value.status_code == 409

    assert submit_session("session-1")["pending"] == {"pending_id": "p1"}
    with pytest.raises(HTTPException) as again:
        submit_session("session-1")
    assert again.value.status_code == 409
    assert len(staged) == 1


def test_discard_session_frees_it():
    from imagecb.api.edit_routes import discard_session
    from imagecb.api.edit_sessions import create_edit_session, get_edit_session

    session_id, _ = create_edit_session(
        source_image_id="source-1", working_image_png=_png_bytes()
    )
    assert get_edit_session(session_id) is not None
    assert discard_session(session_id) == {"ok": True}
    assert get_edit_session(session_id) is None
    assert discard_session(session_id) == {"ok": True}


def test_without_blank_aws_credential_env_strips_empty_values(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "  ")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")

    with _without_blank_aws_credential_env():
        assert "AWS_ACCESS_KEY_ID" not in __import__("os").environ
        assert "AWS_SECRET_ACCESS_KEY" not in __import__("os").environ
        assert __import__("os").environ.get("AWS_SESSION_TOKEN") == "token"

    assert __import__("os").environ.get("AWS_ACCESS_KEY_ID") == ""
    assert __import__("os").environ.get("AWS_SECRET_ACCESS_KEY") == "  "


def test_fetch_from_secrets_manager_with_blank_aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "")

    class _FakeClient:
        def get_secret_value(self, *, SecretId: str):
            assert SecretId == "gemini"
            return {"SecretString": "sm-key-from-aws"}

    def _fake_boto3_client(service_name, *, region_name):
        assert service_name == "secretsmanager"
        assert region_name == "us-east-1"
        return _FakeClient()

    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(
            SETTINGS,
            gemini_secret_name="gemini",
            gemini_secret_region="us-east-1",
        ),
    )
    monkeypatch.setattr("boto3.client", _fake_boto3_client)

    assert _fetch_from_secrets_manager() == "sm-key-from-aws"


def test_nano_banana_status_reports_sm_error(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(
            SETTINGS,
            gemini_api_key=None,
            gemini_secret_name="gemini",
            gemini_secret_region="us-east-1",
        ),
    )

    def _boom():
        raise RuntimeError("AccessDeniedException: not allowed")

    monkeypatch.setattr(
        "imagecb.models.secrets._fetch_secret_string_from_secrets_manager", _boom
    )
    status = nano_banana_status(force_refresh=True)
    assert status["available"] is False
    assert status["source"] == "secrets_manager"
    assert "secret_name" not in status
    assert "secret_region" not in status
    assert "AccessDenied" in (status["error"] or "")


def test_pending_create_and_decline(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
    )
    settings.ensure_dirs()
    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        _open_tmp_db(settings)

        pending = create_pending_edit(
            source_image_id="src-1",
            image_bytes=_png_bytes(),
            last_prompt="make it blue",
        )
        assert pending["source_image_id"] == "src-1"
        assert pending["last_prompt"] == "make it blue"
        assert list_pending_edits()
        staged = pending["staged_ref"]
        assert blob_store.exists(staged)

        decline_pending_edit(pending["pending_id"])
        assert list_pending_edits() == []
        assert not blob_store.exists(staged)


def test_pending_accept_sets_parent_and_clears_staging(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        admin_api_key="admin",
    )
    settings.ensure_dirs()

    new_id = "ingested-from-pending"

    def fake_ingest(paths, **_kwargs):
        data = paths[0].read_bytes()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        content_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=new_id,
                    content_hash=content_hash,
                    image_path=str(tmp_path / "data" / "images" / f"{new_id}.png"),
                    source_file=str(paths[0]),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )
        (tmp_path / "data" / "images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "images" / f"{new_id}.png").write_bytes(data)
        return {"images_added": 1}

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.ingest.ingest_paths", side_effect=fake_ingest
    ):
        _open_tmp_db(settings)

        pending = create_pending_edit(
            source_image_id="parent-img",
            image_bytes=_png_bytes(color=(1, 2, 3)),
            last_prompt="edit me",
        )
        staged = pending["staged_ref"]
        result = accept_pending_edit(pending["pending_id"])
        assert result["new_image_id"] == new_id
        assert result["source_image_id"] == "parent-img"
        rec = metadata_db.get_record(new_id)
        assert rec is not None
        assert rec.parent_image_id == "parent-img"
        assert list_pending_edits() == []
        assert not blob_store.exists(staged)


def test_pending_accept_created_image_skips_parent(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        admin_api_key="admin",
    )
    settings.ensure_dirs()

    new_id = "ingested-from-create"
    seen_names: list[str] = []

    def fake_ingest(paths, **_kwargs):
        seen_names.append(paths[0].name)
        data = paths[0].read_bytes()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        content_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=new_id,
                    content_hash=content_hash,
                    image_path=str(tmp_path / "data" / "images" / f"{new_id}.png"),
                    source_file=str(paths[0]),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )
        (tmp_path / "data" / "images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "images" / f"{new_id}.png").write_bytes(data)
        return {"images_added": 1}

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.ingest.ingest_paths", side_effect=fake_ingest
    ):
        _open_tmp_db(settings)

        pending = create_pending_edit(
            source_image_id="",
            image_bytes=_png_bytes(color=(4, 5, 6)),
            last_prompt="a red bar chart",
        )
        result = accept_pending_edit(pending["pending_id"])
        assert result["new_image_id"] == new_id
        assert result["source_image_id"] == ""
        rec = metadata_db.get_record(new_id)
        assert rec is not None
        assert rec.parent_image_id is None
        assert seen_names[0].startswith("nano-banana-created-")


def test_pending_accept_keeps_pending_when_ingest_adds_nothing(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        admin_api_key="admin",
    )
    settings.ensure_dirs()

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.ingest.ingest_paths", return_value={"images_added": 0, "errors": 1}
    ):
        _open_tmp_db(settings)

        pending = create_pending_edit(
            source_image_id="parent-img",
            image_bytes=_png_bytes(color=(7, 8, 9)),
            last_prompt="edit me",
        )
        with pytest.raises(PendingEditIngestError):
            accept_pending_edit(pending["pending_id"])
        remaining = list_pending_edits()
        assert [p["pending_id"] for p in remaining] == [pending["pending_id"]]
        assert blob_store.exists(pending["staged_ref"])


def test_create_image_session_then_submit(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        admin_api_key="test-admin-secret",
        gemini_api_key="fake-gemini-key",
        llm_rate_limit_per_minute=0,
    )
    settings.ensure_dirs()
    created_png = _png_bytes(color=(20, 30, 40))
    refined_png = _png_bytes(color=(50, 60, 70))

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.api.auth.SETTINGS", settings
    ), patch("imagecb.api.rate_limit.SETTINGS", settings), patch(
        "imagecb.models.secrets.SETTINGS", settings
    ), patch(
        "imagecb.api.edit_routes.nano_banana_status",
        return_value={"available": True},
    ), patch(
        "imagecb.models.image_edit.generate_image",
        return_value=created_png,
    ), patch(
        "imagecb.models.image_edit.edit_image",
        return_value=refined_png,
    ):
        _open_tmp_db(settings)
        client = TestClient(create_app())
        created = client.post(
            "/api/edit/sessions/create",
            json={"prompt": "a blue dashboard"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        session_id = body["session_id"]
        assert body["source_image_id"] == ""
        assert body["turn_count"] == 1
        assert body["turns"][0]["prompt"] == "a blue dashboard"

        turn_img = client.get(body["turns"][0]["image_url"])
        assert turn_img.status_code == 200
        assert turn_img.content == created_png

        refined = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "make the bars taller"},
        )
        assert refined.status_code == 200, refined.text
        assert refined.json()["turn_count"] == 2

        submitted = client.post(f"/api/edit/sessions/{session_id}/submit")
        assert submitted.status_code == 200, submitted.text
        pending = submitted.json()["pending"]
        assert pending["source_image_id"] == ""
        assert pending["last_prompt"] == "make the bars taller"
        assert list_pending_edits()[0]["source_image_id"] == ""


def test_revise_create_session_rewinds_to_base_image(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        gemini_api_key="fake-gemini-key",
        llm_rate_limit_per_minute=0,
    )
    settings.ensure_dirs()
    created_png = _png_bytes(color=(20, 30, 40))
    refined_png = _png_bytes(color=(50, 60, 70))
    revised_png = _png_bytes(color=(80, 90, 100))
    edit_calls: list[tuple[bytes, str]] = []

    def fake_edit(image_bytes, prompt):
        edit_calls.append((image_bytes, prompt))
        if prompt == "wider margins":
            return revised_png
        return refined_png

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.api.rate_limit.SETTINGS", settings
    ), patch(
        "imagecb.api.edit_routes.nano_banana_status",
        return_value={"available": True},
    ), patch(
        "imagecb.models.image_edit.generate_image",
        return_value=created_png,
    ), patch(
        "imagecb.models.image_edit.edit_image",
        side_effect=fake_edit,
    ):
        client = TestClient(create_app())
        created = client.post(
            "/api/edit/sessions/create",
            json={"prompt": "a blue dashboard"},
        )
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]

        refined = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "make the bars taller"},
        )
        assert refined.status_code == 200, refined.text

        revised = client.post(
            f"/api/edit/sessions/{session_id}/revise",
            json={"prompt": "wider margins", "base_turn_index": 0},
        )
        assert revised.status_code == 200, revised.text
        payload = revised.json()
        assert payload["turn_count"] == 2
        assert [turn["prompt"] for turn in payload["turns"]] == [
            "a blue dashboard",
            "wider margins",
        ]
        assert edit_calls[-1] == (created_png, "wider margins")

        working = client.get(f"/api/edit/sessions/{session_id}/image")
        assert working.status_code == 200
        assert working.content == revised_png
        assert working.content != refined_png


def test_edit_session_turn_submit_and_admin_decline(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        admin_api_key="test-admin-secret",
        gemini_api_key="fake-gemini-key",
        llm_rate_limit_per_minute=0,
    )
    settings.ensure_dirs()
    (tmp_path / "data" / "images").mkdir(parents=True, exist_ok=True)

    image_id = "corpus-1"
    png = _png_bytes()
    image_path = tmp_path / "data" / "images" / f"{image_id}.png"
    image_path.write_bytes(png)

    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.api.auth.SETTINGS", settings
    ), patch("imagecb.api.rate_limit.SETTINGS", settings), patch(
        "imagecb.models.secrets.SETTINGS", settings
    ), patch(
        "imagecb.api.edit_routes.nano_banana_status",
        return_value={"available": True},
    ), patch(
        "imagecb.models.image_edit.edit_image",
        return_value=_png_bytes(color=(200, 100, 50)),
    ):
        _open_tmp_db(settings)
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=image_id,
                    content_hash="hash-corpus-1",
                    image_path=str(image_path),
                    source_file=str(image_path),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )

        client = TestClient(create_app())
        created = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]
        assert created.json()["original_image_url"] == (
            f"/api/edit/sessions/{session_id}/original"
        )

        original = client.get(f"/api/edit/sessions/{session_id}/original")
        assert original.status_code == 200
        assert original.content == png

        turn = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "make background green"},
        )
        assert turn.status_code == 200, turn.text
        assert turn.json()["turn_count"] == 1
        turn_payload = turn.json()
        assert turn_payload["turns"][0]["image_url"] == (
            f"/api/edit/sessions/{session_id}/turns/0/image"
        )

        edited_png = _png_bytes(color=(200, 100, 50))
        turn_img = client.get(turn_payload["turns"][0]["image_url"])
        assert turn_img.status_code == 200
        assert turn_img.content == edited_png
        assert turn_img.content != png

        img = client.get(f"/api/edit/sessions/{session_id}/image")
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")
        assert img.content == edited_png

        submitted = client.post(f"/api/edit/sessions/{session_id}/submit")
        assert submitted.status_code == 200, submitted.text
        pending_id = submitted.json()["pending"]["pending_id"]

        listed = client.get(
            "/api/admin/pending-edits",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
        assert listed.status_code == 200
        assert any(i["pending_id"] == pending_id for i in listed.json()["items"])

        declined = client.post(
            f"/api/admin/pending-edits/{pending_id}/decline",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
        assert declined.status_code == 200
        assert list_pending_edits() == []


def test_create_session_returns_503_when_unavailable(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path / "data",
        image_cache_dir=tmp_path / "data" / "images",
        uploads_dir=tmp_path / "data" / "uploads",
        sqlite_path=tmp_path / "data" / "test.db",
        s3_prefix="imagecb",
        gemini_api_key="fake-gemini-key",
        llm_rate_limit_per_minute=0,
    )
    settings.ensure_dirs()
    patches = _patch_settings(settings)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
        "imagecb.api.rate_limit.SETTINGS", settings
    ), patch(
        "imagecb.api.edit_routes.nano_banana_status",
        return_value={
            "available": False,
            "error": "Secrets Manager AccessDenied",
        },
    ):
        client = TestClient(create_app())
        res = client.post("/api/edit/sessions", json={"image_id": "missing"})
        assert res.status_code == 503
        assert "unavailable" in res.json()["detail"].lower()
        assert "AccessDenied" in res.json()["detail"]


def test_edit_api_missing_session_submitted_and_empty_submit(tmp_path):
    settings, image_id, _png, stack = _corpus_client(tmp_path)
    with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[
        6
    ], stack[7], stack[8], stack[9], stack[10]:
        _open_tmp_db(settings)
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=image_id,
                    content_hash="hash-corpus-1",
                    image_path=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_file=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )

        client = TestClient(create_app())

        missing = client.post(
            "/api/edit/sessions/does-not-exist/turn",
            json={"prompt": "make it blue"},
        )
        assert missing.status_code == 404

        created = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        empty_submit = client.post(f"/api/edit/sessions/{session_id}/submit")
        assert empty_submit.status_code == 400
        assert "at least once" in empty_submit.json()["detail"].lower()

        turn = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "make it blue"},
        )
        assert turn.status_code == 200

        submitted = client.post(f"/api/edit/sessions/{session_id}/submit")
        assert submitted.status_code == 200

        # Submit deletes the session; further turns/submits must 404.
        after = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "again"},
        )
        assert after.status_code == 404

        second_submit = client.post(f"/api/edit/sessions/{session_id}/submit")
        assert second_submit.status_code == 404


def test_edit_turn_returns_409_when_session_already_submitted(monkeypatch):
    session = EditSession(
        source_image_id="source-1",
        original_image_png=_png_bytes(),
        working_image_png=_png_bytes(),
        submitted=True,
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.nano_banana_status",
        lambda: {"available": True},
    )
    monkeypatch.setattr(
        "imagecb.api.edit_routes.get_edit_session",
        lambda _session_id: session,
    )

    with pytest.raises(HTTPException) as caught:
        edit_turn("session-1", EditTurnRequest(prompt="make it blue"))
    assert caught.value.status_code == 409
    assert "already submitted" in str(caught.value.detail).lower()


def test_edit_create_rate_limited(tmp_path):
    settings, image_id, _png, stack = _corpus_client(
        tmp_path, settings_overrides={"llm_rate_limit_per_minute": 1}
    )
    with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[
        6
    ], stack[7], stack[8], stack[9], stack[10]:
        _open_tmp_db(settings)
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=image_id,
                    content_hash="hash-corpus-1",
                    image_path=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_file=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )

        client = TestClient(create_app())
        first = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert first.status_code == 200, first.text
        second = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert second.status_code == 429
        assert "Retry-After" in second.headers


def test_edit_session_ttl_eviction_returns_404(tmp_path, monkeypatch):
    from imagecb.api import edit_sessions as edit_sessions_mod

    settings, image_id, _png, stack = _corpus_client(
        tmp_path, settings_overrides={"edit_session_ttl_sec": 1}
    )
    clock = {"now": 1000.0}

    def fake_monotonic():
        return clock["now"]

    monkeypatch.setattr("imagecb.api.edit_sessions.time.monotonic", fake_monotonic)

    with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[
        6
    ], stack[7], stack[8], stack[9], stack[10]:
        _open_tmp_db(settings)
        with session_scope() as s:
            s.add(
                ImageRecord(
                    image_id=image_id,
                    content_hash="hash-corpus-1",
                    image_path=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_file=str(tmp_path / "data" / "images" / f"{image_id}.png"),
                    source_type="image",
                    created_at=datetime.utcnow(),
                )
            )

        client = TestClient(create_app())
        created = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        # Dataclass default_factory keeps a direct ref to real time.monotonic;
        # stamp last_used onto the fake clock so TTL eviction is deterministic.
        with edit_sessions_mod._lock:
            edit_sessions_mod._sessions[session_id].last_used = clock["now"]

        # Advance past TTL; creating another session triggers eviction.
        clock["now"] = 1002.5
        other = client.post("/api/edit/sessions", json={"image_id": image_id})
        assert other.status_code == 200

        expired = client.get(f"/api/edit/sessions/{session_id}")
        assert expired.status_code == 404

        expired_turn = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "too late"},
        )
        assert expired_turn.status_code == 404