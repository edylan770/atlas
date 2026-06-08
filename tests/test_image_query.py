"""Tests for image-query facet mapping and axis parsing."""

from __future__ import annotations

import pytest

from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.image_query import (
    SimilarityAxis,
    axis_label,
    query_spec_from_image_query,
)


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
