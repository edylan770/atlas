"""Tests for image-to-image similar search pipeline."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.hybrid import rrf_merge
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.similar import _filter_by_min_score, _fuse_and_rank, search_similar
from imagecb.storage.metadata_db import ImageRecord


def _record(image_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"data/images/{image_id}.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        source_modified_at=datetime(2024, 9, 15),
        source_created_at=None,
        author=None,
        slide_index=1,
        page_index=None,
        slide_title=None,
        slide_notes=None,
        ocr_text=None,
        caption_short="Test caption",
        caption_detailed=None,
        objects_json=None,
        tags_json=None,
        scene=None,
        text_overlay_summary=None,
        created_at=datetime.utcnow(),
    )


def test_rrf_merge_blends_two_ranked_lists():
    visual = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    text = [("c", 0.95), ("a", 0.5), ("d", 0.4)]
    merged = rrf_merge(visual, text, k=60)
    ids = [c.image_id for c in merged]
    assert ids[0] == "a"
    assert "d" in ids


def test_filter_by_min_score_uses_dense_score():
    results = [
        RankedResult(
            image_id="a",
            score=0.5,
            record=_record("a"),
            provenance_line="a",
            score_kind="dense",
        ),
        RankedResult(
            image_id="b",
            score=0.3,
            record=_record("b"),
            provenance_line="b",
            score_kind="dense",
        ),
    ]
    filtered = _filter_by_min_score(results, 0.4)
    assert [r.image_id for r in filtered] == ["a"]


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_uses_visual_score_not_rerank(mock_get_records):
    records = [_record("a"), _record("b"), _record("c")]
    mock_get_records.return_value = records

    visual_hits = [("a", 0.85), ("b", 0.55), ("c", 0.40)]
    text_ranked = [
        RankedResult(
            image_id="c",
            score=0.99,
            record=records[2],
            provenance_line="c",
            score_kind="rerank",
        ),
        RankedResult(
            image_id="b",
            score=0.88,
            record=records[1],
            provenance_line="b",
            score_kind="rerank",
        ),
    ]

    results = _fuse_and_rank(visual_hits, text_ranked, top_k=3, min_score=0.0)
    by_id = {r.image_id: r for r in results}
    assert by_id["a"].score == 0.85
    assert by_id["b"].score == 0.55
    assert by_id["c"].score == 0.40
    assert all(r.score_kind == "dense" for r in results)


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_filters_on_visual_dense_score(mock_get_records):
    records = [_record("a"), _record("b")]
    mock_get_records.return_value = records

    visual_hits = [("a", 0.85), ("b", 0.30)]
    text_ranked = [
        RankedResult(
            image_id="b",
            score=0.95,
            record=records[1],
            provenance_line="b",
            score_kind="rerank",
        ),
    ]

    results = _fuse_and_rank(visual_hits, text_ranked, top_k=2, min_score=0.5)
    assert [r.image_id for r in results] == ["a"]


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_uses_embed_image_for_upload(
    mock_active,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    mock_active.return_value = ["img-1", "img-2"]
    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    captioner.query_image.return_value = ImageQueryJSON(
        search_query="office photo",
        subject="desk",
        style="photo",
        layout="centered",
    )
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.8), ("img-1", 0.6)]
    mock_text_leg.return_value = []

    pil = Image.new("RGB", (64, 64), color=(10, 20, 30))
    with patch("imagecb.retrieval.similar.metadata_db.get_records", return_value=[]):
        outcome = search_similar(image=pil, top_k=5, min_match_percent=0)

    embedder.embed_image.assert_called_once()
    embedder.embed_image_with_context.assert_not_called()
    assert outcome.facets.search_query == "office photo"


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar._load_image_for_record")
@patch("imagecb.retrieval.similar.metadata_db.get_record")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_uses_embed_image_for_indexed_image(
    mock_active,
    mock_get_record,
    mock_load_image,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    rec = _record("ref-1")
    rec.image_name = "Hero slide"
    mock_get_record.return_value = rec
    mock_load_image.return_value = Image.new("RGB", (64, 64))
    mock_active.return_value = ["ref-1", "img-2"]

    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    captioner.query_image.return_value = ImageQueryJSON(search_query="hero banner")
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.75)]
    mock_text_leg.return_value = []

    with patch(
        "imagecb.retrieval.similar.metadata_db.get_records",
        return_value=[_record("img-2")],
    ):
        outcome = search_similar(image_id="ref-1", top_k=3)

    embedder.embed_image.assert_called_once()
    embedder.embed_image_with_context.assert_not_called()
    assert outcome.spec.semantic_query
