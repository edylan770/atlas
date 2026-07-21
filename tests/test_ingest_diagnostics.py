"""Tests for authenticated ingest runtime diagnostics."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

from imagecb.config import SETTINGS


def test_preflight_reports_the_failing_dependency(monkeypatch):
    from imagecb import ingest_diagnostics

    monkeypatch.setattr(ingest_diagnostics, "_writable_data_dir", lambda: "writable")
    monkeypatch.setattr(ingest_diagnostics, "_sqlite_round_trip", lambda: "sqlite ok")
    monkeypatch.setattr(ingest_diagnostics, "_s3_round_trip", lambda: "s3 ok")
    monkeypatch.setattr(
        ingest_diagnostics,
        "_image_embedding_probe",
        lambda: (_ for _ in ()).throw(PermissionError("InvokeModel denied")),
    )
    monkeypatch.setattr(ingest_diagnostics, "_text_embedding_probe", lambda: "text ok")
    monkeypatch.setattr(ingest_diagnostics, "_caption_probe", lambda: "caption ok")
    monkeypatch.setattr(ingest_diagnostics, "runtime_diagnostics", lambda: {"build_id": "test"})

    result = ingest_diagnostics.run_ingest_preflight()

    assert result["ok"] is False
    failed = [check for check in result["checks"] if not check["ok"]]
    assert len(failed) == 1
    assert failed[0]["name"] == "image_embedding"
    assert failed[0]["detail"] == "PermissionError: InvokeModel denied"


def test_runtime_diagnostics_redacts_credentials(tmp_path, monkeypatch):
    from imagecb import ingest_diagnostics
    from imagecb.storage import metadata_db

    if metadata_db._engine is not None:
        metadata_db._engine.dispose()
    metadata_db._engine = None
    metadata_db._SessionLocal = None
    configured = replace(
        SETTINGS,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "diagnostics.db",
        uploads_dir=tmp_path / "uploads",
        s3_bucket="atlas-test-bucket",
    )
    monkeypatch.setattr(metadata_db, "SETTINGS", configured)
    monkeypatch.setattr(ingest_diagnostics, "SETTINGS", configured)
    monkeypatch.setenv("APP_BUILD_ID", "test-build")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-appear")

    result = ingest_diagnostics.runtime_diagnostics()

    assert result["build_id"] == "test-build"
    assert result["s3_bucket"] == "atlas-test-bucket"
    assert "must-not-appear" not in repr(result)

    metadata_db._engine.dispose()
    metadata_db._engine = None
    metadata_db._SessionLocal = None


def test_admin_diagnostics_requires_authentication():
    from imagecb.api.server import create_app

    configured = replace(SETTINGS, admin_api_key="test-admin-secret")
    payload = {
        "build_id": "build-1",
        "runtime_id": "host:1",
        "runner": {"alive": True},
    }
    with patch("imagecb.api.auth.SETTINGS", configured), patch(
        "imagecb.ingest_diagnostics.runtime_diagnostics",
        return_value=payload,
    ):
        client = TestClient(create_app())
        denied = client.get("/api/admin/ingest/diagnostics")
        allowed = client.get(
            "/api/admin/ingest/diagnostics",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["build_id"] == "build-1"
    assert allowed.headers["cache-control"] == "no-store, max-age=0"
