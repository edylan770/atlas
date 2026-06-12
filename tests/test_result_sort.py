"""Tests for result sort ordering."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.sort import (
    InvalidSortError,
    parse_sort,
    resolve_sort,
    sort_image_records,
    sort_ranked_results,
)
from imagecb.storage.metadata_db import ImageRecord, get_engine, upsert_image


def _record(
    image_id: str,
    *,
    created_at: datetime | None = None,
    image_name: str | None = None,
    source_file: str = "/tmp/report.pptx",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"h-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file=source_file,
        source_type="pptx",
        created_at=created_at or datetime.utcnow(),
        image_name=image_name,
    )


def _ranked(
    image_id: str,
    *,
    score: float = 0.5,
    created_at: datetime | None = None,
    image_name: str | None = None,
    source_file: str = "/tmp/report.pptx",
) -> RankedResult:
    rec = _record(
        image_id,
        created_at=created_at,
        image_name=image_name,
        source_file=source_file,
    )
    return RankedResult(
        image_id=image_id,
        score=score,
        record=rec,
        provenance_line="",
    )


def test_resolve_sort_defaults():
    assert resolve_sort(None, is_search=True) == "relevance"
    assert resolve_sort(None, is_search=False) == "newest"
    assert resolve_sort("name", is_search=True) == "name"


def test_parse_sort_rejects_unknown():
    with pytest.raises(InvalidSortError):
        parse_sort("invalid")


def test_sort_ranked_results_relevance_preserves_order():
    results = [_ranked("a", score=0.9), _ranked("b", score=0.1)]
    ordered = sort_ranked_results(results, "relevance")
    assert [r.image_id for r in ordered] == ["a", "b"]


def test_sort_ranked_results_newest():
    older = datetime(2020, 1, 1)
    newer = datetime(2024, 1, 1)
    results = [
        _ranked("old", created_at=older),
        _ranked("new", created_at=newer),
    ]
    ordered = sort_ranked_results(results, "newest")
    assert [r.image_id for r in ordered] == ["new", "old"]


def test_sort_ranked_results_oldest():
    older = datetime(2020, 1, 1)
    newer = datetime(2024, 1, 1)
    results = [
        _ranked("new", created_at=newer),
        _ranked("old", created_at=older),
    ]
    ordered = sort_ranked_results(results, "oldest")
    assert [r.image_id for r in ordered] == ["old", "new"]


def test_sort_ranked_results_name():
    results = [
        _ranked("b", image_name="Zebra"),
        _ranked("a", image_name="Apple"),
    ]
    ordered = sort_ranked_results(results, "name")
    assert [r.image_id for r in ordered] == ["a", "b"]


def test_sort_ranked_results_source():
    results = [
        _ranked("b", source_file="/tmp/z-deck.pptx"),
        _ranked("a", source_file="/tmp/a-deck.pptx"),
    ]
    ordered = sort_ranked_results(results, "source")
    assert [r.image_id for r in ordered] == ["a", "b"]


def test_sort_image_records_newest():
    base = datetime.utcnow()
    records = [
        _record("old", created_at=base - timedelta(days=1)),
        _record("new", created_at=base),
    ]
    ordered = sort_image_records(records, "newest")
    assert [r.image_id for r in ordered] == ["new", "old"]


@pytest.fixture
def admin_client():
    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    with patch("imagecb.api.auth.SETTINGS", patched):
        yield TestClient(create_app())


@pytest.fixture(autouse=True)
def _db():
    get_engine()
    yield


def test_catalog_endpoint_sort_oldest():
    older_id = f"old-{uuid.uuid4().hex[:8]}"
    newer_id = f"new-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(older_id, created_at=datetime(2020, 6, 1)))
    upsert_image(_record(newer_id, created_at=datetime(2024, 6, 1)))

    client = TestClient(create_app())
    res = client.get("/api/corpus/catalog?limit=200&sort=oldest")
    assert res.status_code == 200
    ids = [item["image_id"] for item in res.json()["items"]]
    assert ids.index(older_id) < ids.index(newer_id)


def test_admin_corpus_sort_name(admin_client):
    id_b = f"name-b-{uuid.uuid4().hex[:8]}"
    id_a = f"name-a-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(id_b, image_name="Zulu chart"))
    upsert_image(_record(id_a, image_name="Alpha chart"))

    res = admin_client.get(
        "/api/admin/corpus/images?sort=name",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 200
    ids = [img["image_id"] for img in res.json()["images"]]
    assert ids.index(id_a) < ids.index(id_b)


def test_admin_corpus_invalid_sort(admin_client):
    res = admin_client.get(
        "/api/admin/corpus/images?sort=bad",
        headers={"X-Admin-Api-Key": "test-admin-secret"},
    )
    assert res.status_code == 400
