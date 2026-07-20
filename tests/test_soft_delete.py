"""Soft delete evicts from searchable set."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from imagecb.admin import curation
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.hybrid import search
from imagecb.storage.metadata_db import ImageRecord, get_engine, get_record, session_scope, upsert_image
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
    assert stats["deleted"] == 1
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
