"""Nano Banana secrets parsing, pending edits, and edit API."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagecb.api.edit_sessions import clear_edit_sessions
from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.models.secrets import (
    nano_banana_status,
    parse_gemini_secret_string,
    reset_gemini_secret_cache,
)
from imagecb.pending_edits import (
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
    clear_edit_sessions()
    yield
    reset_gemini_secret_cache()
    clear_edit_sessions()


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


def test_nano_banana_status_from_env(monkeypatch):
    monkeypatch.setattr(
        "imagecb.models.secrets.SETTINGS",
        replace(SETTINGS, gemini_api_key="env-key-xyz"),
    )
    status = nano_banana_status(force_refresh=True)
    assert status["available"] is True
    assert status["source"] == "env"
    assert status["error"] is None


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
        "imagecb.models.secrets._fetch_from_secrets_manager", _boom
    )
    status = nano_banana_status(force_refresh=True)
    assert status["available"] is False
    assert status["source"] == "secrets_manager"
    assert status["secret_name"] == "gemini"
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
        "imagecb.models.secrets.is_nano_banana_available", return_value=True
    ), patch(
        "imagecb.api.edit_routes.is_nano_banana_available", return_value=True
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

        turn = client.post(
            f"/api/edit/sessions/{session_id}/turn",
            json={"prompt": "make background green"},
        )
        assert turn.status_code == 200, turn.text
        assert turn.json()["turn_count"] == 1

        img = client.get(f"/api/edit/sessions/{session_id}/image")
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")

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
