"""Tests for batch caption quality rescan and single-image regenerate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from imagecb.caption.quality import CAPTION_FAILED
from imagecb.models.vlm import CaptionJSON, GroundedCaption, InterpretiveCaption, SearchTerms
from imagecb.repair import regenerate_caption, reindex_image, rescan_caption_quality
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _make_record(**overrides) -> ImageRecord:
    base = dict(
        image_id="img-1",
        content_hash="abc",
        image_path="/tmp/img-1.png",
        source_file="photo.jpg",
        source_type="image",
        image_name="Photo",
        caption_short="A photo",
        caption_detailed="",
        scene="",
        objects_json=serialize_list([]),
        tags_json=serialize_list(["photo"]),
        recommended_cases_json=serialize_list(["photo"]),
        search_aliases_json=serialize_list([]),
        caption_quality="ok",
    )
    base.update(overrides)
    return ImageRecord(**base)


@patch("imagecb.repair.get_all_records")
@patch("imagecb.repair._persist_record")
def test_rescan_flags_weak_legacy_ok(mock_persist, mock_get_all):
    mock_get_all.return_value = [_make_record()]
    stats = rescan_caption_quality()
    assert stats["scanned"] == 1
    assert stats["updated"] == 1
    assert stats["weak"] == 1
    mock_persist.assert_called_once()


@patch("imagecb.repair.get_all_records")
@patch("imagecb.repair._persist_record")
def test_rescan_skips_unchanged(mock_persist, mock_get_all):
    record = _make_record(
        caption_short=CAPTION_FAILED,
        caption_detailed="VLM error: timeout",
        caption_quality="failed",
    )
    mock_get_all.return_value = [record]
    stats = rescan_caption_quality()
    assert stats["failed"] == 1
    assert stats["updated"] == 0
    mock_persist.assert_not_called()


@patch("imagecb.repair.refresh_text_vector", return_value=True)
@patch("imagecb.repair.bm25_index.rebuild_from_records")
@patch("imagecb.repair._upsert_record_embedding")
@patch("imagecb.repair._persist_record")
@patch("imagecb.repair.get_embedder")
@patch("imagecb.repair._caption_with_retry")
@patch("imagecb.repair._load_cached_image")
@patch("imagecb.storage.metadata_db.get_record")
def test_regenerate_caption_reembeds(
    mock_get_record,
    mock_load_image,
    mock_caption_retry,
    mock_get_embedder,
    mock_persist,
    mock_upsert,
    mock_bm25,
    mock_text_vector,
):
    record = _make_record(caption_quality="weak")
    mock_get_record.return_value = record
    mock_load_image.return_value = MagicMock()
    mock_caption_retry.return_value = CaptionJSON(
        image_name="Updated Photo",
        grounded=GroundedCaption(objects=["person"], scene="office"),
        interpretive=InterpretiveCaption(
            theme="workplace",
            use_case="HR",
            short_caption="Colleagues collaborating in a bright office space",
            detailed_description="Several people stand near desks in an open office.",
        ),
        search=SearchTerms(
            tags=["office", "people", "team"],
            recommended_cases=["office team photo", "workplace collaboration", "staff meeting"],
            aliases=["colleagues", "workplace"],
        ),
        caption_quality="ok",
    )
    embedder = MagicMock()
    embedder.embed_image_with_context.return_value = np.zeros((1, 8), dtype=np.float32)
    mock_get_embedder.return_value = embedder

    result = regenerate_caption("img-1", rebuild_bm25=True)

    assert result["caption_quality"] == "ok"
    assert result["needs_regeneration"] is False
    assert result["image_name"] == "Updated Photo"
    mock_persist.assert_called_once()
    mock_upsert.assert_called_once()
    mock_bm25.assert_called_once()


@patch("imagecb.repair.refresh_text_vector", return_value=True)
@patch("imagecb.repair.bm25_index.rebuild_from_records")
@patch("imagecb.repair._upsert_record_embedding")
@patch("imagecb.repair.get_embedder")
@patch("imagecb.repair._load_cached_image")
@patch("imagecb.storage.metadata_db.get_record")
def test_reindex_image_reembeds(
    mock_get_record,
    mock_load_image,
    mock_get_embedder,
    mock_upsert,
    mock_bm25,
    mock_text_vector,
):
    record = _make_record(caption_short="Stored caption", caption_quality="ok")
    mock_get_record.return_value = record
    mock_load_image.return_value = MagicMock()
    embedder = MagicMock()
    embedder.embed_image_with_context.return_value = np.zeros((1, 8), dtype=np.float32)
    mock_get_embedder.return_value = embedder

    result = reindex_image("img-1", rebuild_bm25=True)

    assert result["reindexed"] is True
    assert result["caption_short"] == "Stored caption"
    assert result["caption_quality"] == "ok"
    mock_upsert.assert_called_once()
    mock_bm25.assert_called_once()
