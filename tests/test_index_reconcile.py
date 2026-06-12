"""Tests for ingest concurrency guard and readiness API."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.ingest import IngestInProgressError


@pytest.fixture
def client():
    return TestClient(create_app())


def test_ready_returns_200_when_healthy(client):
    from imagecb.repair import IndexHealthReport

    healthy = IndexHealthReport(
        total_records=1,
        chroma_vectors=1,
        missing_cache_count=0,
        stores_in_sync=True,
        is_healthy=True,
        bm25_doc_count=1,
    )
    with patch("imagecb.repair.assess_index_health", return_value=healthy):
        res = client.get("/api/ready")
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is True
    assert body["stores_in_sync"] is True


def test_ready_returns_503_when_unhealthy(client):
    from imagecb.repair import IndexHealthReport

    unhealthy = IndexHealthReport(
        total_records=2,
        chroma_vectors=1,
        missing_cache_count=0,
        missing_chroma_count=1,
        stores_in_sync=False,
        is_healthy=False,
    )
    with patch("imagecb.repair.assess_index_health", return_value=unhealthy):
        res = client.get("/api/ready")
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail["ready"] is False


def test_status_includes_store_counts(client):
    from imagecb.repair import IndexHealthReport

    report = IndexHealthReport(
        total_records=5,
        chroma_vectors=5,
        text_vector_count=4,
        bm25_doc_count=5,
        missing_cache_count=0,
        stores_in_sync=True,
        is_healthy=True,
    )
    with patch("imagecb.repair.assess_index_health", return_value=report):
        res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["indexed_count"] == 5
    assert body["total_records"] == 5
    assert body["bm25_doc_count"] == 5
    assert body["stores_in_sync"] is True


def test_ingest_returns_409_when_already_running():
    from dataclasses import replace
    from pathlib import Path

    from imagecb.config import SETTINGS

    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    with patch("imagecb.api.auth.SETTINGS", patched), patch(
        "imagecb.api.routes.save_uploads_from_files",
        return_value=([Path("/tmp/x.png")], []),
    ), patch(
        "imagecb.api.routes.ingest_paths",
        side_effect=IngestInProgressError("Another ingest is already in progress"),
    ):
        client = TestClient(create_app())
        res = client.post(
            "/api/ingest",
            files=[("files", ("x.png", b"png", "image/png"))],
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
    assert res.status_code == 409


def test_bm25_list_ids_and_count():
    import imagecb.storage.bm25_index as bm25_module
    from imagecb.storage.bm25_index import BM25Index

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(bm25_module, "_index", None)
    idx = bm25_module.get_index()
    idx.build(["a", "b"], ["hello world", "foo bar"])
    assert idx.count() == 2
    assert idx.list_ids() == {"a", "b"}
    assert idx.is_loaded() is True
    monkeypatch.undo()
