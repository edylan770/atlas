"""Admin corpus API tests."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.admin import curation
from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.storage.metadata_db import ImageRecord, get_engine, upsert_image
from imagecb.telemetry.schema import ensure_telemetry_schema


@pytest.fixture(autouse=True)
def _db():
    get_engine()
    ensure_telemetry_schema()
    yield


@pytest.fixture
def admin_client():
    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    with patch("imagecb.api.auth.SETTINGS", patched):
        yield TestClient(create_app())


def _record(image_id: str, *, caption_quality: str = "ok") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"h-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="/tmp/report.pptx",
        source_type="pptx",
        caption_quality=caption_quality,
        created_at=datetime.utcnow(),
    )


def test_list_corpus_images():
    iid = f"corpus-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(iid))
    images = curation.list_corpus_images()
    ids = {img["image_id"] for img in images}
    assert iid in ids
    row = next(img for img in images if img["image_id"] == iid)
    assert row["image_url"] == f"/api/images/{iid}"


def test_corpus_images_endpoint(admin_client):
    iid = f"api-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(iid))
    res = admin_client.get(
        "/api/admin/corpus/images",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    ids = {img["image_id"] for img in res.json()["images"]}
    assert iid in ids


@patch("imagecb.admin.routes.duplicates.find_duplicate_clusters")
def test_duplicate_clusters_returns_error_not_500(mock_find, admin_client):
    mock_find.side_effect = RuntimeError("chroma unavailable")
    res = admin_client.get(
        "/api/admin/corpus/duplicate-clusters",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["clusters"] == []
    assert "chroma" in (body.get("error") or "").lower()


@patch("imagecb.admin.routes.audit.append_audit")
@patch("imagecb.repair.regenerate_caption")
def test_regenerate_caption_endpoint(mock_regenerate, mock_audit, admin_client):
    mock_regenerate.return_value = {
        "image_id": "img-1",
        "caption_quality": "ok",
        "needs_regeneration": False,
        "caption_short": "New caption",
    }
    res = admin_client.post(
        "/api/admin/images/img-1/regenerate-caption",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["caption_short"] == "New caption"
    mock_regenerate.assert_called_once_with("img-1")
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["action"] == "regenerate_caption"


@patch("imagecb.repair.regenerate_caption")
def test_regenerate_caption_not_found(mock_regenerate, admin_client):
    mock_regenerate.side_effect = ValueError("image not found")
    res = admin_client.post(
        "/api/admin/images/missing/regenerate-caption",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 404


@patch("imagecb.admin.routes.audit.append_audit")
@patch("imagecb.repair.reindex_image")
def test_reindex_endpoint(mock_reindex, mock_audit, admin_client):
    mock_reindex.return_value = {
        "image_id": "img-1",
        "reindexed": True,
        "caption_short": "Stored caption",
        "caption_quality": "ok",
    }
    res = admin_client.post(
        "/api/admin/images/img-1/reindex",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["reindexed"] is True
    mock_reindex.assert_called_once_with("img-1")
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["action"] == "reindex"


@patch("imagecb.repair.reindex_image")
def test_reindex_not_found(mock_reindex, admin_client):
    mock_reindex.side_effect = ValueError("image not found")
    res = admin_client.post(
        "/api/admin/images/missing/reindex",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 404


@patch("imagecb.repair.assess_index_health")
def test_corpus_health_endpoint(mock_assess, admin_client):
    from imagecb.repair import IndexHealthReport

    mock_assess.return_value = IndexHealthReport(
        total_records=10,
        chroma_vectors=10,
        missing_cache_count=0,
        failed_caption_count=2,
        weak_caption_count=3,
        needs_regeneration_count=5,
        is_healthy=False,
    )
    res = admin_client.get(
        "/api/admin/corpus/health",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total_images"] == 10
    assert body["failed_caption_count"] == 2
    assert body["weak_caption_count"] == 3
    assert body["needs_regeneration_count"] == 5
    assert body["is_healthy"] is False
    mock_assess.assert_called_once_with(include_weak=True)


def test_corpus_images_filter_weak(admin_client):
    ok_id = f"ok-{uuid.uuid4().hex[:8]}"
    weak_id = f"weak-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(ok_id, caption_quality="ok"))
    upsert_image(_record(weak_id, caption_quality="weak"))

    res = admin_client.get(
        "/api/admin/corpus/images?caption_quality=weak",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    ids = {img["image_id"] for img in res.json()["images"]}
    assert weak_id in ids
    assert ok_id not in ids


def test_corpus_images_invalid_quality_filter(admin_client):
    res = admin_client.get(
        "/api/admin/corpus/images?caption_quality=bad",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 400


@patch("imagecb.admin.routes.audit.append_audit")
@patch("imagecb.repair.repair_failed_captions")
def test_repair_captions_failed_scope(mock_repair, mock_audit, admin_client):
    mock_repair.return_value = {
        "attempted": 2,
        "repaired": 2,
        "errors": 0,
        "elapsed_sec": 1.2,
        "scope": "failed",
    }
    res = admin_client.post(
        "/api/admin/corpus/repair-captions?scope=failed",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["repaired"] == 2
    mock_repair.assert_called_once_with(scope="failed")
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["action"] == "repair_captions"
    assert mock_audit.call_args.kwargs["target_id"] == "failed"


@patch("imagecb.admin.routes.audit.append_audit")
@patch("imagecb.repair.repair_failed_captions")
def test_repair_captions_weak_scope(mock_repair, mock_audit, admin_client):
    mock_repair.return_value = {
        "attempted": 3,
        "repaired": 1,
        "errors": 2,
        "elapsed_sec": 4.5,
        "scope": "weak",
    }
    res = admin_client.post(
        "/api/admin/corpus/repair-captions?scope=weak",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["errors"] == 2
    mock_repair.assert_called_once_with(scope="weak")
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["target_id"] == "weak"
