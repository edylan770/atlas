"""Tests for ingest concurrency guard and readiness API."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.ingest import IngestInProgressError


@pytest.fixture(autouse=True)
def _fresh_status_cache():
    """/status and /ready cache the health report; isolate it per test."""
    from imagecb.api import routes as _routes

    _routes.reset_status_cache()
    yield
    _routes.reset_status_cache()



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


def test_status_indexed_count_follows_sqlite_when_stores_diverge(client):
    from imagecb.repair import IndexHealthReport

    report = IndexHealthReport(
        total_records=12,
        chroma_vectors=7,
        text_vector_count=7,
        bm25_doc_count=12,
        missing_cache_count=0,
        missing_chroma_count=5,
        stores_in_sync=False,
        is_healthy=False,
    )
    with patch("imagecb.repair.assess_index_health", return_value=report):
        res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["indexed_count"] == 12
    assert body["total_records"] == 12
    assert body["chroma_vectors"] == 7
    assert body["stores_in_sync"] is False


def test_status_health_failure_keeps_store_semantics(client):
    with patch(
        "imagecb.repair.assess_index_health",
        side_effect=RuntimeError("health scan failed"),
    ), patch(
        "imagecb.api.routes.metadata_db.count_active_records",
        return_value=12,
    ), patch(
        "imagecb.api.routes.vector_store.count",
        return_value=7,
    ):
        res = client.get("/api/status")

    assert res.status_code == 200
    body = res.json()
    assert body["indexed_count"] == 12
    assert body["total_records"] == 12
    assert body["chroma_vectors"] == 7
    assert body["is_healthy"] is False
    assert body["stores_in_sync"] is False


def test_catalog_count_uses_full_sqlite_corpus_not_page_or_chroma(client):
    with patch(
        "imagecb.api.routes.metadata_db.list_catalog_records",
        return_value=[],
    ), patch(
        "imagecb.api.routes.metadata_db.count_active_records",
        return_value=12,
    ), patch(
        "imagecb.api.routes.vector_store.count",
        side_effect=AssertionError("catalog count must not read Chroma"),
    ):
        res = client.get("/api/corpus/catalog?limit=1")

    assert res.status_code == 200
    assert res.json()["items"] == []
    assert res.json()["indexed_count"] == 12


def test_ingest_response_separates_corpus_and_chroma_counts():
    from dataclasses import replace
    from pathlib import Path

    from imagecb.config import SETTINGS

    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    stats = {
        "elapsed_sec": 0,
        "images_added": 1,
        "images_updated": 0,
    }
    with patch("imagecb.api.auth.SETTINGS", patched), patch(
        "imagecb.api.routes.save_uploads_from_files",
        return_value=([Path("/tmp/x.png")], []),
    ), patch(
        "imagecb.api.routes.ingest_paths",
        return_value=stats,
    ), patch(
        "imagecb.api.routes.cleanup_staged_uploads",
    ), patch(
        "imagecb.api.routes.metadata_db.count_active_records",
        return_value=12,
    ), patch(
        "imagecb.api.routes.vector_store.count",
        return_value=7,
    ):
        client = TestClient(create_app())
        res = client.post(
            "/api/ingest",
            files=[("files", ("x.png", b"png", "image/png"))],
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )

    assert res.status_code == 200
    body = res.json()
    assert body["indexed_count"] == 12
    assert body["chroma_vectors"] == 7


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
