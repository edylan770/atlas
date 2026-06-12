"""Tests for the caption-text dense lane in hybrid search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imagecb.retrieval.hybrid import rrf_merge_lanes, search
from imagecb.retrieval.query_parser import QuerySpec


def test_rrf_merge_lanes_fuses_three_lanes():
    dense = [("a", 0.9), ("b", 0.8)]
    text = [("b", 0.85), ("c", 0.7)]
    sparse = [("c", 5.0)]

    merged = rrf_merge_lanes(dense, text, sparse, k=60)
    by_id = {c.image_id: c for c in merged}

    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].dense_score == 0.9
    assert by_id["b"].text_score == 0.85
    assert by_id["c"].sparse_score == 5.0
    # b appears at rank 2 + rank 1, c at rank 2 + rank 1, a only at rank 1.
    assert merged[0].image_id in ("b", "c")
    assert by_id["b"].fused_score > by_id["a"].fused_score


def test_rrf_merge_lanes_single_lane_keeps_order():
    merged = rrf_merge_lanes([("a", 0.9), ("b", 0.5)], [], [], k=60)
    assert [c.image_id for c in merged] == ["a", "b"]


@patch("imagecb.retrieval.hybrid.metadata_db.get_active_image_ids", return_value=["a", "b"])
@patch("imagecb.retrieval.hybrid.vector_store")
@patch("imagecb.retrieval.hybrid.bm25_index")
@patch("imagecb.retrieval.hybrid.get_text_embedder")
@patch("imagecb.retrieval.hybrid.get_embedder")
def test_search_includes_caption_text_lane(
    mock_embedder,
    mock_text_embedder,
    mock_bm25,
    mock_vs,
    _active,
):
    mock_embedder.return_value.embed_text.return_value = [MagicMock()]
    mock_text_embedder.return_value.embed_query.return_value = MagicMock()
    mock_vs.query.return_value = [("a", 0.9)]
    mock_vs.query_text.return_value = [("b", 0.8)]
    mock_bm25.get_index.return_value.query.return_value = []

    spec = QuerySpec(semantic_query="tank convoy", raw_text="tank convoy")
    outcome = search(spec)

    ids = {c.image_id for c in outcome.candidates}
    assert ids == {"a", "b"}
    assert outcome.dense_failed is False


@patch("imagecb.retrieval.hybrid.metadata_db.get_active_image_ids", return_value=["a"])
@patch("imagecb.retrieval.hybrid.vector_store")
@patch("imagecb.retrieval.hybrid.bm25_index")
@patch("imagecb.retrieval.hybrid.get_text_embedder")
@patch("imagecb.retrieval.hybrid.get_embedder")
def test_search_survives_text_lane_failure(
    mock_embedder,
    mock_text_embedder,
    mock_bm25,
    mock_vs,
    _active,
):
    mock_embedder.return_value.embed_text.return_value = [MagicMock()]
    mock_text_embedder.return_value.embed_query.side_effect = RuntimeError("no model access")
    mock_vs.query.return_value = [("a", 0.9)]
    mock_bm25.get_index.return_value.query.return_value = []

    spec = QuerySpec(semantic_query="tank convoy", raw_text="tank convoy")
    outcome = search(spec)

    assert [c.image_id for c in outcome.candidates] == ["a"]
    assert outcome.dense_failed is False


@patch("imagecb.retrieval.hybrid.metadata_db.get_active_image_ids", return_value=["a"])
@patch("imagecb.retrieval.hybrid.vector_store")
@patch("imagecb.retrieval.hybrid.bm25_index")
@patch("imagecb.retrieval.hybrid.get_text_embedder")
@patch("imagecb.retrieval.hybrid.get_embedder")
def test_search_reports_dense_failed_when_both_dense_lanes_fail(
    mock_embedder,
    mock_text_embedder,
    mock_bm25,
    mock_vs,
    _active,
):
    mock_embedder.return_value.embed_text.side_effect = RuntimeError("down")
    mock_text_embedder.return_value.embed_query.side_effect = RuntimeError("down")
    mock_bm25.get_index.return_value.query.return_value = [("a", 3.0)]

    spec = QuerySpec(semantic_query="tank convoy", raw_text="tank convoy")
    outcome = search(spec)

    assert outcome.dense_failed is True
    assert [c.image_id for c in outcome.candidates] == ["a"]
