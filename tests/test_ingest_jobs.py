"""Tests for durable ingest jobs and their HTTP API."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from imagecb.config import SETTINGS


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    from imagecb import ingest_jobs
    from imagecb.storage import metadata_db

    if metadata_db._engine is not None:
        metadata_db._engine.dispose()
    metadata_db._engine = None
    metadata_db._SessionLocal = None

    configured = replace(
        SETTINGS,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "jobs.db",
        uploads_dir=tmp_path / "uploads",
    )
    monkeypatch.setattr(metadata_db, "SETTINGS", configured)
    monkeypatch.setattr(ingest_jobs, "SETTINGS", configured)
    ingest_jobs.ensure_job_schema()
    yield ingest_jobs

    if metadata_db._engine is not None:
        metadata_db._engine.dispose()
    metadata_db._engine = None
    metadata_db._SessionLocal = None


def test_job_cancel_is_durable_and_idempotent(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    job = isolated_jobs.create_job(
        "job-1",
        [source],
        {"workers": 1},
    )
    assert job["status"] == "queued"

    first = isolated_jobs.request_cancel("job-1")
    second = isolated_jobs.request_cancel("job-1")
    assert first is not None and first["status"] == "cancelled"
    assert second is not None and second["status"] == "cancelled"
    assert isolated_jobs.get_job("job-1")["finished_at"] is not None


def test_interrupted_running_job_is_requeued(isolated_jobs, tmp_path):
    from imagecb.storage.metadata_db import IngestJob, session_scope

    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-2", [source], {"workers": 1})
    with session_scope() as session:
        row = session.get(IngestJob, "job-2")
        assert row is not None
        row.status = "running"
        row.started_at = datetime.utcnow()

    isolated_jobs._recover_interrupted_jobs()
    recovered = isolated_jobs.get_job("job-2")
    assert recovered is not None
    assert recovered["status"] == "queued"
    assert recovered["error"] == "Recovered after server restart"


def test_claim_records_runner_and_phase(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-phase", [source], {"workers": 1})

    claimed = isolated_jobs._claim_next_job("host:123:runner")

    assert claimed is not None
    job = isolated_jobs.get_job("job-phase")
    assert job is not None
    assert job["status"] == "running"
    assert job["phase"] == "claimed"
    assert job["runner_id"] == "host:123:runner"


def test_runner_records_cancelled_partial_stats(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-3", [source], {"workers": 1})
    claimed = isolated_jobs._claim_next_job()
    assert claimed is not None

    stats = {"cancelled": True, "images_seen": 2, "images_added": 1}
    with patch("imagecb.ingest.ingest_paths_batched", return_value=stats):
        isolated_jobs.IngestJobRunner()._execute(*claimed)

    job = isolated_jobs.get_job("job-3")
    assert job is not None
    assert job["status"] == "cancelled"
    assert job["stats"]["images_added"] == 1


def test_runner_fails_job_when_all_processing_fails(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-failed", [source], {"workers": 1})
    claimed = isolated_jobs._claim_next_job("test-runner")
    assert claimed is not None

    stats = {
        "images_seen": 1,
        "images_added": 0,
        "images_updated": 0,
        "skipped_duplicates": 0,
        "errors": 1,
        "timeouts": 1,
        "last_error": "Image processing exceeded the configured 30s timeout",
    }
    with patch("imagecb.ingest.ingest_paths_batched", return_value=stats):
        isolated_jobs.IngestJobRunner()._execute(*claimed)

    job = isolated_jobs.get_job("job-failed")
    assert job is not None
    assert job["status"] == "failed"
    assert "timeout" in (job["error"] or "")
    assert job["phase"] == "failed"


def test_create_and_cancel_job_api():
    from imagecb.api.server import create_app

    configured = replace(SETTINGS, admin_api_key="test-admin-secret")
    job = {
        "job_id": "job-api",
        "status": "queued",
        "files": ["x.png"],
        "files_total": 1,
        "options": {"workers": 2},
        "cancellable": True,
    }
    with patch("imagecb.api.auth.SETTINGS", configured), patch(
        "imagecb.api.routes.save_uploads_from_files",
        new=AsyncMock(return_value=([Path("x.png")], [])),
    ), patch("imagecb.api.routes.new_job_id", return_value="job-api"), patch(
        "imagecb.api.routes.create_job", return_value=job
    ), patch("imagecb.api.routes.request_cancel", return_value={**job, "status": "cancelled"}):
        client = TestClient(create_app())
        response = client.post(
            "/api/ingest/jobs",
            files=[("files", ("x.png", b"png", "image/png"))],
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
        assert response.status_code == 202
        assert response.json()["job_id"] == "job-api"

        cancelled = client.post(
            "/api/ingest/jobs/job-api/cancel",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


def test_job_list_and_detail_are_consistent_and_not_cached():
    from imagecb.api.server import create_app

    configured = replace(SETTINGS, admin_api_key="test-admin-secret")
    job = {
        "job_id": "job-visible",
        "status": "running",
        "files": ["x.png"],
        "files_total": 1,
        "phase": "image_embedding",
        "status_detail": "Calling image embedding model",
        "runner_id": "host:1:runner",
        "cancellable": True,
    }
    with patch("imagecb.api.auth.SETTINGS", configured), patch(
        "imagecb.api.routes.list_jobs", return_value=[job]
    ), patch("imagecb.api.routes.get_job", return_value=job):
        client = TestClient(create_app())
        headers = {"X-Admin-Api-Key": "test-admin-secret"}
        listed = client.get("/api/ingest/jobs", headers=headers)
        detail = client.get("/api/ingest/jobs/job-visible", headers=headers)

    assert listed.status_code == 200
    assert detail.status_code == 200
    assert listed.json()["jobs"][0] == detail.json()
    assert listed.headers["cache-control"] == "no-store, max-age=0"
    assert detail.headers["cache-control"] == "no-store, max-age=0"


def test_staging_job_not_claimable_until_started(isolated_jobs, tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    job = isolated_jobs.create_job(
        "job-stage",
        [a],
        {"workers": 1},
        status="staging",
    )
    assert job["status"] == "staging"
    assert job["cancellable"] is True
    assert isolated_jobs._claim_next_job() is None

    appended = isolated_jobs.append_job_files("job-stage", [b], stage_errors=["warn"])
    assert appended is not None
    assert appended["files_total"] == 2
    assert appended["stage_errors"] == ["warn"]
    assert appended["status"] == "staging"
    assert isolated_jobs._claim_next_job() is None

    started = isolated_jobs.start_job("job-stage")
    assert started is not None
    assert started["status"] == "queued"
    claimed = isolated_jobs._claim_next_job()
    assert claimed is not None
    assert claimed[0] == "job-stage"
    assert len(claimed[1]) == 2


def test_cancel_staging_job_cleans_stage_dir(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-stg-cancel", [source], {"workers": 1}, status="staging")
    stage_dir = isolated_jobs.job_stage_dir("job-stg-cancel")
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "source.png").write_bytes(b"png")

    cancelled = isolated_jobs.request_cancel("job-stg-cancel")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert not stage_dir.exists()


def test_start_job_rejects_empty_or_non_staging(isolated_jobs, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    isolated_jobs.create_job("job-queued", [source], {"workers": 1}, status="queued")
    with pytest.raises(ValueError, match="not staging"):
        isolated_jobs.start_job("job-queued")

    isolated_jobs.create_job("job-empty", [], {"workers": 1}, status="staging")
    with pytest.raises(ValueError, match="no staged files"):
        isolated_jobs.start_job("job-empty")


def test_staging_append_start_job_api():
    from imagecb.api.server import create_app

    configured = replace(SETTINGS, admin_api_key="test-admin-secret")
    staging_job = {
        "job_id": "job-chunk",
        "status": "staging",
        "files": ["a.png"],
        "files_total": 1,
        "options": {"workers": 2},
        "cancellable": True,
        "stage_errors": [],
    }
    appended_job = {
        **staging_job,
        "files": ["a.png", "b.png"],
        "files_total": 2,
    }
    started_job = {**appended_job, "status": "queued"}

    with patch("imagecb.api.auth.SETTINGS", configured), patch(
        "imagecb.api.routes.save_uploads_from_files",
        new=AsyncMock(side_effect=[([Path("a.png")], []), ([Path("b.png")], [])]),
    ), patch("imagecb.api.routes.new_job_id", return_value="job-chunk"), patch(
        "imagecb.api.routes.create_job", return_value=staging_job
    ) as mock_create, patch(
        "imagecb.api.routes.get_job",
        side_effect=[staging_job, staging_job, appended_job],
    ), patch(
        "imagecb.api.routes.append_job_files", return_value=appended_job
    ), patch("imagecb.api.routes.start_job", return_value=started_job):
        client = TestClient(create_app())
        headers = {"X-Admin-Api-Key": "test-admin-secret"}

        created = client.post(
            "/api/ingest/jobs",
            files=[("files", ("a.png", b"a", "image/png"))],
            data={"start": "false"},
            headers=headers,
        )
        assert created.status_code == 202
        assert created.json()["status"] == "staging"
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs.get("status") == "staging"

        appended = client.post(
            "/api/ingest/jobs/job-chunk/files",
            files=[("files", ("b.png", b"b", "image/png"))],
            headers=headers,
        )
        assert appended.status_code == 200
        assert appended.json()["files_total"] == 2

        started = client.post(
            "/api/ingest/jobs/job-chunk/start",
            headers=headers,
        )
        assert started.status_code == 200
        assert started.json()["status"] == "queued"


def test_append_files_rejects_non_staging_job():
    from imagecb.api.server import create_app

    configured = replace(SETTINGS, admin_api_key="test-admin-secret")
    queued = {
        "job_id": "job-q",
        "status": "queued",
        "files": ["a.png"],
        "files_total": 1,
        "cancellable": True,
        "stage_errors": [],
    }
    with patch("imagecb.api.auth.SETTINGS", configured), patch(
        "imagecb.api.routes.get_job", return_value=queued
    ):
        client = TestClient(create_app())
        response = client.post(
            "/api/ingest/jobs/job-q/files",
            files=[("files", ("b.png", b"b", "image/png"))],
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
    assert response.status_code == 409
    assert "not accepting uploads" in response.json()["detail"]
