"""Soft delete evicts from searchable set."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from imagecb.admin import curation
from imagecb import repair
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.hybrid import search
from imagecb.storage.metadata_db import (
    ImageRecord,
    get_engine,
    get_record,
    get_records,
    session_scope,
    upsert_image,
)
from imagecb.telemetry.schema import ensure_telemetry_schema


def _record(image_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"h-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="/tmp/x.pptx",
        source_type="pptx",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def delete_target():
    get_engine()
    ensure_telemetry_schema()
    image_id = f"to-delete-{uuid.uuid4().hex[:8]}"
    upsert_image(_record(image_id))
    yield image_id


@patch("imagecb.admin.curation.vector_store")
@patch("imagecb.admin.curation.rebuild_bm25_active")
def test_soft_delete_marks_record_and_calls_vector_delete(
    mock_bm25, mock_vs, delete_target
):
    curation.soft_delete_image(image_id=delete_target, actor="admin-test")
    mock_vs.delete.assert_called_once_with([delete_target])
    mock_bm25.assert_called_once()

    assert get_record(delete_target) is None
    with session_scope() as s:
        from sqlalchemy import select

        row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == delete_target)
        ).scalar_one()
        assert row.deleted_at is not None


@patch("imagecb.admin.curation.vector_store")
@patch("imagecb.admin.curation.rebuild_bm25_active")
@patch("imagecb.repair.assess_index_health")
def test_soft_delete_unrecoverable_only_targets_broken(
    mock_assess, mock_bm25, mock_vs
):
    get_engine()
    ensure_telemetry_schema()
    bad_id = f"bad-{uuid.uuid4().hex[:8]}"
    good_id = f"good-{uuid.uuid4().hex[:8]}"
    bad = _record(bad_id)
    good = _record(good_id)
    upsert_image(bad)
    upsert_image(good)

    mock_report = MagicMock()
    mock_report.unrecoverable_records = [bad]
    mock_assess.return_value = mock_report

    stats = curation.soft_delete_unrecoverable(actor="admin-test")
    assert stats["candidates"] == 1
    assert stats["deleted"] == 1
    assert stats["skipped"] == 0
    assert stats["image_ids"] == [bad_id]
    mock_vs.delete.assert_called_once_with([bad_id])
    mock_vs.delete_text.assert_called_once_with([bad_id])
    mock_bm25.assert_called_once()

    assert get_record(bad_id) is None
    assert get_record(good_id) is not None
    with session_scope() as s:
        from sqlalchemy import select

        bad_row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == bad_id)
        ).scalar_one()
        good_row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == good_id)
        ).scalar_one()
        assert bad_row.deleted_at is not None
        assert good_row.deleted_at is None


def test_soft_delete_unrecoverable_uses_real_health_classification(tmp_path):
    get_engine()
    ensure_telemetry_schema()
    bad_id = f"missing-{uuid.uuid4().hex[:8]}"
    recoverable_id = f"recoverable-{uuid.uuid4().hex[:8]}"
    present_id = f"present-{uuid.uuid4().hex[:8]}"
    ids = [bad_id, recoverable_id, present_id]

    source = tmp_path / "recoverable.pptx"
    source.write_bytes(b"source")
    cached = tmp_path / "present.png"
    cached.write_bytes(b"png")
    records = [
        ImageRecord(
            image_id=bad_id,
            content_hash=f"h-{bad_id}",
            image_path=str(tmp_path / "missing-cache.png"),
            source_file=str(tmp_path / "missing-source.pptx"),
            source_type="pptx",
            created_at=datetime.utcnow(),
        ),
        ImageRecord(
            image_id=recoverable_id,
            content_hash=f"h-{recoverable_id}",
            image_path=str(tmp_path / "missing-recoverable-cache.png"),
            source_file=str(source),
            source_type="pptx",
            created_at=datetime.utcnow(),
        ),
        ImageRecord(
            image_id=present_id,
            content_hash=f"h-{present_id}",
            image_path=str(cached),
            source_file=str(tmp_path / "missing-present-source.pptx"),
            source_type="pptx",
            created_at=datetime.utcnow(),
        ),
    ]
    for record in records:
        upsert_image(record)

    def active_test_records(*, include_deleted=False):
        return get_records(ids, include_deleted=include_deleted)

    with patch("imagecb.repair.get_all_records", side_effect=active_test_records), patch(
        "imagecb.repair.vector_store"
    ) as health_vectors, patch("imagecb.repair.bm25_index") as health_bm25, patch(
        "imagecb.admin.curation.vector_store"
    ) as delete_vectors, patch(
        "imagecb.admin.curation.rebuild_bm25_active"
    ) as rebuild_bm25, patch("imagecb.admin.curation.append_audit") as mock_audit:
        health_vectors.count.return_value = len(ids)
        health_vectors.list_ids.return_value = set(ids)
        health_vectors.list_text_ids.return_value = set()
        health_bm25.count.return_value = len(ids)
        health_bm25.list_ids.return_value = set(ids)

        health = repair.assess_index_health(include_weak=False)
        assert [row.image_id for row in health.unrecoverable_records] == [bad_id]
        assert recoverable_id not in {
            row.image_id for row in health.unrecoverable_records
        }

        first = curation.soft_delete_unrecoverable(actor="admin-test")
        second = curation.soft_delete_unrecoverable(actor="admin-test")

    assert first == {
        "candidates": 1,
        "deleted": 1,
        "skipped": 0,
        "image_ids": [bad_id],
    }
    assert second == {
        "candidates": 0,
        "deleted": 0,
        "skipped": 0,
        "image_ids": [],
    }
    delete_vectors.delete.assert_called_once_with([bad_id])
    delete_vectors.delete_text.assert_called_once_with([bad_id])
    rebuild_bm25.assert_called_once()
    assert mock_audit.call_count == 2
    assert get_record(bad_id) is None
    assert get_record(recoverable_id) is not None
    assert get_record(present_id) is not None


@patch("imagecb.admin.curation.append_audit")
@patch("imagecb.admin.curation.vector_store")
@patch("imagecb.admin.curation.rebuild_bm25_active")
@patch("imagecb.repair.assess_index_health")
def test_soft_delete_unrecoverable_reports_stale_candidates_as_skipped(
    mock_assess, mock_bm25, mock_vs, mock_audit
):
    get_engine()
    ensure_telemetry_schema()
    image_id = f"stale-{uuid.uuid4().hex[:8]}"
    record = _record(image_id)
    record.deleted_at = datetime.utcnow()
    upsert_image(record)
    mock_report = MagicMock()
    mock_report.unrecoverable_records = [record]
    mock_assess.return_value = mock_report

    stats = curation.soft_delete_unrecoverable(actor="admin-test")

    assert stats == {
        "candidates": 1,
        "deleted": 0,
        "skipped": 1,
        "image_ids": [],
    }
    mock_vs.delete.assert_not_called()
    mock_vs.delete_text.assert_not_called()
    mock_bm25.assert_not_called()
    assert mock_audit.call_args.kwargs["details"]["candidates"] == 1
    assert mock_audit.call_args.kwargs["details"]["deleted"] == 0
    assert mock_audit.call_args.kwargs["details"]["skipped"] == 1


@patch("imagecb.retrieval.hybrid.metadata_db.get_active_image_ids", return_value=["active-1"])
@patch("imagecb.retrieval.hybrid.vector_store")
@patch("imagecb.retrieval.hybrid.bm25_index")
@patch("imagecb.retrieval.hybrid.get_text_embedder")
@patch("imagecb.retrieval.hybrid.get_embedder")
def test_hybrid_excludes_deleted_when_only_active_ids(
    mock_embedder,
    mock_text_embedder,
    mock_bm25,
    mock_vs,
    _active,
):
    mock_embedder.return_value.embed_text.return_value = [MagicMock()]
    mock_text_embedder.return_value.embed_query.return_value = MagicMock()
    mock_vs.query.return_value = [("active-1", 0.9)]
    mock_vs.query_text.return_value = []
    mock_bm25.get_index.return_value.query.return_value = []

    spec = QuerySpec(semantic_query="test", raw_text="test")
    search(spec)
    mock_vs.query.assert_called_once()
    call_kwargs = mock_vs.query.call_args.kwargs
    assert call_kwargs.get("allowed_ids") == ["active-1"]
    text_kwargs = mock_vs.query_text.call_args.kwargs
    assert text_kwargs.get("allowed_ids") == ["active-1"]
