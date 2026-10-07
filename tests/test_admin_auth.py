"""Admin API key authentication."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.config import SETTINGS


@pytest.fixture
def client():
    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    with patch("imagecb.api.auth.SETTINGS", patched):
        yield TestClient(create_app())


def test_admin_summary_requires_key(client):
    res = client.get("/api/admin/analytics/summary")
    assert res.status_code == 401


def test_admin_summary_rejects_wrong_key(client):
    res = client.get(
        "/api/admin/analytics/summary",
        headers={"X-Admin-Api-Key": "wrong"},
    )
    assert res.status_code == 403


def test_admin_summary_accepts_key(client):
    res = client.get(
        "/api/admin/analytics/summary",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    assert "total_searches" in res.json()


def test_admin_content_gaps_requires_key(client):
    res = client.get("/api/admin/analytics/content-gaps")
    assert res.status_code == 401


def test_admin_content_gaps_accepts_key(client, monkeypatch, tmp_path):
    from imagecb.telemetry import s3_store

    settings = replace(
        SETTINGS,
        admin_api_key="test-admin-secret",
        blob_storage_backend="local",
        data_dir=tmp_path,
        s3_prefix="imagecb",
        weak_result_score_threshold=0.25,
        content_gap_similarity_threshold=0.82,
        telemetry_retention_days=90,
        telemetry_default_window_days=90,
    )
    monkeypatch.setattr(s3_store, "SETTINGS", settings)
    monkeypatch.setattr("imagecb.storage.blob_store.SETTINGS", settings)
    monkeypatch.setattr("imagecb.admin.analytics.SETTINGS", settings)
    monkeypatch.setattr("imagecb.admin.content_gaps.SETTINGS", settings)
    monkeypatch.setattr(
        "imagecb.admin.content_gaps._embed_unique_labels",
        lambda labels: None,
    )
    s3_store.invalidate_quality_cache()

    # Seed two identical zero-result gaps so a theme appears.
    from datetime import datetime
    import uuid

    for i in range(2):
        created = datetime.utcnow()
        s3_store.put_search_event(
            {
                "id": str(uuid.uuid4()),
                "created_at": created.isoformat(),
                "query_text": "org chart",
                "user_id": f"u{i}",
                "session_id": None,
                "search_kind": "chat",
                "served_image_ids": [],
                "result_count": 0,
                "top_score": None,
                "top_score_kind": None,
                "parsed_semantic_query": None,
                "has_interaction": False,
            }
        )

    with patch("imagecb.api.auth.SETTINGS", settings):
        client2 = TestClient(create_app())
        res = client2.get(
            "/api/admin/analytics/content-gaps?days=90",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["gap_event_count"] == 2
    assert body["clustering_mode"] == "exact"
    assert len(body["themes"]) == 1
    assert "org chart" in body["themes"][0]["theme_label"].lower()
    assert body["themes"][0]["search_count"] == 2
    assert "summary" in body["themes"][0]
    assert "example_queries" in body["themes"][0]
