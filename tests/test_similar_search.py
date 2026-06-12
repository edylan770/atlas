"""Tests for image-to-image similar search pipeline."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.hybrid import normalize_rrf_score, rrf_merge
from imagecb.retrieval.image_query import SimilarityAxis
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.similar import _filter_by_min_match_percent, _fuse_and_rank, search_similar
from imagecb.formatting.match_display import display_match_percent
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


def test_filter_by_min_match_percent_uses_display_scale():
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
    filtered = _filter_by_min_match_percent(results, 40)
    assert [r.image_id for r in filtered] == ["a"]


def test_normalize_rrf_score_single_list():
    k = 60
    assert normalize_rrf_score(1.0 / (k + 1), k, weight_sum=1.0) == 1.0


def test_normalize_rrf_score_dual_list():
    k = 60
    rank_one_both = 2.0 / (k + 1)
    assert normalize_rrf_score(rank_one_both, k, weight_sum=2.0) == 1.0


def test_rrf_merge_weighted_prefers_heavier_lane():
    visual = [("v_only", 0.9)]
    text = [("t_only", 0.95)]
    k = 60
    subject = rrf_merge(visual, text, k, dense_weight=0.35, sparse_weight=1.65)
    layout = rrf_merge(visual, text, k, dense_weight=1.65, sparse_weight=0.35)
    assert subject[0].image_id == "t_only"
    assert layout[0].image_id == "v_only"


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_uses_normalized_fusion_score(mock_get_records):
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

    results = _fuse_and_rank(visual_hits, text_ranked, top_k=3, min_match_percent=0)
    assert all(r.score_kind == "fusion" for r in results)
    assert results[0].image_id == "c"
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert display_match_percent(results[i].score, "fusion") >= display_match_percent(
            results[i + 1].score, "fusion"
        )


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_text_lane_survives_min_score(mock_get_records):
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

    results = _fuse_and_rank(visual_hits, text_ranked, top_k=2, min_match_percent=50)
    ids = [r.image_id for r in results]
    assert "b" in ids
    by_id = {r.image_id: r for r in results}
    assert display_match_percent(by_id["b"].score, "fusion") >= 50


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_text_only_leg_normalizes_to_one(mock_get_records):
    records = [_record("d")]
    mock_get_records.return_value = records

    text_ranked = [
        RankedResult(
            image_id="d",
            score=0.95,
            record=records[0],
            provenance_line="d",
            score_kind="rerank",
        ),
    ]

    results = _fuse_and_rank(
        [],
        text_ranked,
        top_k=1,
        min_match_percent=0,
        axis=SimilarityAxis.SUBJECT,
    )
    assert len(results) == 1
    assert results[0].score == 1.0
    assert results[0].score_kind == "fusion"


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_subject_prefers_text_lane(mock_get_records):
    records = [_record("v_only"), _record("t_only")]
    mock_get_records.return_value = records

    visual_hits = [("v_only", 0.9)]
    text_ranked = [
        RankedResult(
            image_id="t_only",
            score=0.95,
            record=records[1],
            provenance_line="t",
            score_kind="rerank",
        ),
    ]

    subject = _fuse_and_rank(
        visual_hits,
        text_ranked,
        top_k=2,
        min_match_percent=0,
        axis=SimilarityAxis.SUBJECT,
    )
    layout = _fuse_and_rank(
        visual_hits,
        text_ranked,
        top_k=2,
        min_match_percent=0,
        axis=SimilarityAxis.LAYOUT,
    )
    assert subject[0].image_id == "t_only"
    assert layout[0].image_id == "v_only"


@patch("imagecb.retrieval.similar.metadata_db.get_records")
def test_fuse_and_rank_layout_prefers_visual_lane(mock_get_records):
    records = [_record("v_only"), _record("t_only")]
    mock_get_records.return_value = records

    visual_hits = [("v_only", 0.9)]
    text_ranked = [
        RankedResult(
            image_id="t_only",
            score=0.95,
            record=records[1],
            provenance_line="t",
            score_kind="rerank",
        ),
    ]

    results = _fuse_and_rank(
        visual_hits,
        text_ranked,
        top_k=2,
        min_match_percent=0,
        axis=SimilarityAxis.STYLE,
    )
    assert results[0].image_id == "v_only"


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
    captioner.query_image.assert_not_called()
    assert outcome.spec.semantic_query


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_upload_still_calls_query_image(
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
    captioner.query_image.return_value = ImageQueryJSON(search_query="upload query")
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.8)]
    mock_text_leg.return_value = []

    pil = Image.new("RGB", (64, 64))
    with patch("imagecb.retrieval.similar.metadata_db.get_records", return_value=[]):
        search_similar(image=pil, top_k=5)

    captioner.query_image.assert_called_once()
    mock_get_captioner.assert_called_once()


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar._load_image_for_record")
@patch("imagecb.retrieval.similar.metadata_db.get_record")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_excludes_reference_from_text_leg(
    mock_active,
    mock_get_record,
    mock_load_image,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    rec_ref = _record("ref-1")
    rec_ref.caption_short = "hero banner"
    rec_img2 = _record("img-2")
    mock_get_record.return_value = rec_ref
    mock_load_image.return_value = Image.new("RGB", (64, 64))
    mock_active.return_value = ["ref-1", "img-2"]

    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.75)]
    mock_text_leg.return_value = [
        RankedResult(
            image_id="ref-1",
            score=0.99,
            record=rec_ref,
            provenance_line="ref",
            score_kind="rerank",
        ),
        RankedResult(
            image_id="img-2",
            score=0.80,
            record=rec_img2,
            provenance_line="img2",
            score_kind="rerank",
        ),
    ]

    with patch(
        "imagecb.retrieval.similar.metadata_db.get_records",
        return_value=[rec_img2],
    ):
        outcome = search_similar(image_id="ref-1", top_k=3, exclude_image_id="ref-1")

    mock_text_leg.assert_called_once()
    assert mock_text_leg.call_args.kwargs["exclude_image_id"] == "ref-1"
    captioner.query_image.assert_not_called()
    assert all(r.image_id != "ref-1" for r in outcome.results)
    assert [r.image_id for r in outcome.results] == ["img-2"]


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar._load_image_for_record")
@patch("imagecb.retrieval.similar.metadata_db.get_record")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_vlm_failure_uses_visual_only(
    mock_active,
    mock_get_record,
    mock_load_image,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    rec_ref = _record("ref-1")
    rec_ref.caption_quality = "failed"
    rec_img2 = _record("img-2")
    mock_get_record.return_value = rec_ref
    mock_load_image.return_value = Image.new("RGB", (64, 64))
    mock_active.return_value = ["ref-1", "img-2"]

    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.75)]

    with patch(
        "imagecb.retrieval.similar.metadata_db.get_records",
        return_value=[rec_img2],
    ):
        outcome = search_similar(image_id="ref-1", top_k=3, exclude_image_id="ref-1")

    mock_text_leg.assert_not_called()
    captioner.query_image.assert_not_called()
    assert [r.image_id for r in outcome.results] == ["img-2"]


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar._load_image_for_record")
@patch("imagecb.retrieval.similar.metadata_db.get_record")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_empty_stored_caption_uses_visual_only(
    mock_active,
    mock_get_record,
    mock_load_image,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    rec_ref = _record("ref-1")
    rec_ref.caption_short = ""
    rec_ref.caption_detailed = None
    rec_ref.scene = None
    rec_ref.use_case = None
    rec_ref.theme = None
    rec_ref.text_overlay_summary = None
    rec_ref.ocr_text = None
    rec_ref.recommended_cases_json = None
    rec_ref.objects_json = None
    rec_img2 = _record("img-2")
    mock_get_record.return_value = rec_ref
    mock_load_image.return_value = Image.new("RGB", (64, 64))
    mock_active.return_value = ["ref-1", "img-2"]

    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.75)]

    with patch(
        "imagecb.retrieval.similar.metadata_db.get_records",
        return_value=[rec_img2],
    ):
        outcome = search_similar(image_id="ref-1", top_k=3, exclude_image_id="ref-1")

    mock_text_leg.assert_not_called()
    captioner.query_image.assert_not_called()
    assert [r.image_id for r in outcome.results] == ["img-2"]


@patch("imagecb.retrieval.similar.run_text_similar_leg")
@patch("imagecb.retrieval.similar.vector_store.query")
@patch("imagecb.retrieval.similar.get_captioner")
@patch("imagecb.retrieval.similar.get_embedder")
@patch("imagecb.retrieval.similar.metadata_db.get_active_image_ids")
def test_search_similar_upload_vlm_failure_uses_visual_only(
    mock_active,
    mock_get_embedder,
    mock_get_captioner,
    mock_query,
    mock_text_leg,
):
    mock_active.return_value = ["img-2"]
    embedder = MagicMock()
    embedder.embed_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_get_embedder.return_value = embedder

    captioner = MagicMock()
    captioner.query_image.return_value = ImageQueryJSON.failed("timeout")
    mock_get_captioner.return_value = captioner

    mock_query.return_value = [("img-2", 0.75)]

    with patch(
        "imagecb.retrieval.similar.metadata_db.get_records",
        return_value=[_record("img-2")],
    ):
        outcome = search_similar(image=Image.new("RGB", (64, 64)), top_k=3)

    mock_text_leg.assert_not_called()
    captioner.query_image.assert_called_once()
    assert [r.image_id for r in outcome.results] == ["img-2"]
