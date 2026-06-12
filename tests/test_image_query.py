"""Tests for image-query facet mapping and axis parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.image_query import (
    SimilarityAxis,
    axis_label,
    axis_lane_weights,
    image_query_from_record,
    query_spec_from_image_query,
    run_text_similar_leg,
)
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _facets() -> ImageQueryJSON:
    return ImageQueryJSON(
        search_query="dashboard screenshot with charts",
        subject="business analytics dashboard",
        style="flat UI screenshot",
        layout="grid of metric cards",
        salient_objects=["bar chart", "KPI tiles"],
        visible_text="Q3 Revenue",
        colors_mood="blue and white, professional",
    )


def test_similarity_axis_parse_valid():
    assert SimilarityAxis.parse("balanced") == SimilarityAxis.BALANCED
    assert SimilarityAxis.parse("SUBJECT") == SimilarityAxis.SUBJECT


def test_similarity_axis_parse_invalid():
    with pytest.raises(ValueError, match="Unknown similarity_axis"):
        SimilarityAxis.parse("color")


def test_query_spec_balanced_uses_search_query_and_facets():
    spec = query_spec_from_image_query(_facets(), SimilarityAxis.BALANCED, top_k=8)
    assert "dashboard screenshot" in spec.semantic_query
    assert "business analytics dashboard" in spec.must_have_keywords
    assert "flat UI screenshot" in spec.must_have_keywords
    assert "Q3 Revenue" in spec.must_have_keywords
    assert spec.top_k == 8


def test_query_spec_subject_axis():
    spec = query_spec_from_image_query(_facets(), SimilarityAxis.SUBJECT, top_k=5)
    assert "business analytics dashboard" in spec.semantic_query
    assert "bar chart" in spec.semantic_query


def test_query_spec_style_axis():
    spec = query_spec_from_image_query(_facets(), SimilarityAxis.STYLE, top_k=5)
    assert "flat UI screenshot" in spec.semantic_query
    assert "blue and white" in spec.semantic_query


def test_query_spec_layout_axis():
    spec = query_spec_from_image_query(_facets(), SimilarityAxis.LAYOUT, top_k=5)
    assert spec.semantic_query == "grid of metric cards"
    assert "bar chart" in spec.must_have_keywords


def test_query_spec_fallback_when_empty_semantic():
    facets = ImageQueryJSON(search_query="", subject="", style="", layout="")
    spec = query_spec_from_image_query(facets, SimilarityAxis.SUBJECT, top_k=3)
    assert spec.semantic_query == "[similar image search]"


def test_axis_label():
    assert axis_label(SimilarityAxis.STYLE) == "style"


def test_axis_lane_weights_balanced():
    assert axis_lane_weights(SimilarityAxis.BALANCED) == (1.0, 1.0)


def test_axis_lane_weights_subject():
    visual, text = axis_lane_weights(SimilarityAxis.SUBJECT)
    assert text > visual


def test_axis_lane_weights_style_layout():
    for axis in (SimilarityAxis.STYLE, SimilarityAxis.LAYOUT):
        visual, text = axis_lane_weights(axis)
        assert visual > text


@patch("imagecb.retrieval.image_query.rerank")
@patch("imagecb.retrieval.image_query.search")
def test_run_text_similar_leg_excludes_reference(mock_search, mock_rerank):
    spec = QuerySpec(semantic_query="hero banner", raw_text="[similar]", top_k=3)
    facets = ImageQueryJSON(search_query="hero banner")
    mock_search.return_value = SearchOutcome(
        candidates=[
            Candidate(image_id="ref-1", fused_score=0.9),
            Candidate(image_id="img-2", fused_score=0.8),
        ]
    )
    mock_rerank.return_value = [
        RankedResult(
            image_id="img-2",
            score=0.85,
            record=MagicMock(),
            provenance_line="img2",
        )
    ]

    results = run_text_similar_leg(spec, facets, top_k=3, exclude_image_id="ref-1")

    mock_search.assert_called_once_with(spec, restrict_to=None)
    rerank_candidates = mock_rerank.call_args[0][1]
    assert [c.image_id for c in rerank_candidates] == ["img-2"]
    assert all(r.image_id != "ref-1" for r in results)


@patch("imagecb.retrieval.image_query.rerank")
@patch("imagecb.retrieval.image_query.search")
def test_run_text_similar_leg_excludes_reference_from_restrict_to(mock_search, mock_rerank):
    spec = QuerySpec(semantic_query="hero banner", raw_text="[similar]", top_k=3)
    facets = ImageQueryJSON(search_query="hero banner")
    mock_search.return_value = SearchOutcome(candidates=[Candidate(image_id="img-2", fused_score=0.8)])
    mock_rerank.return_value = []

    run_text_similar_leg(
        spec,
        facets,
        restrict_to=["ref-1", "img-2"],
        top_k=3,
        exclude_image_id="ref-1",
    )

    mock_search.assert_called_once_with(spec, restrict_to=["img-2"])


def _sample_record() -> ImageRecord:
    return ImageRecord(
        image_id="img-1",
        content_hash="hash-1",
        image_path="data/images/img-1.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        caption_short="Bar chart of quarterly revenue",
        caption_detailed="Colorful bars show revenue increasing each quarter.",
        scene="presentation slide",
        use_case="quarterly business review",
        theme="revenue growth",
        text_overlay_summary="Q3 2024",
        ocr_text="Extra OCR",
        objects_json=serialize_list(["bar chart", "KPI tiles"]),
        recommended_cases_json=serialize_list(["quarterly revenue chart", "sales by region"]),
        caption_quality="ok",
    )


def test_image_query_from_record_maps_caption_fields():
    facets = image_query_from_record(_sample_record())
    assert facets.is_usable()
    assert facets.search_query == "quarterly revenue chart"
    assert facets.subject == "presentation slide"
    assert facets.style == "revenue growth"
    assert facets.layout == ""
    assert facets.salient_objects == ["bar chart", "KPI tiles"]
    assert facets.visible_text == "Q3 2024"
    assert facets.colors_mood == "revenue growth"


def test_image_query_from_record_failed_caption_not_usable():
    record = _sample_record()
    record.caption_quality = "failed"
    facets = image_query_from_record(record)
    assert not facets.is_usable()
    assert facets.query_quality == "failed"
