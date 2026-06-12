"""Tests for post-parse query filter sanitization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.query_parser import (
    QuerySpec,
    SourceFilters,
    _build_spec,
    sanitize_query_spec,
)
from imagecb.retrieval.session import ChatSession


def _spec_with_filters(
    *,
    raw_text: str,
    asset_types: list[str] | None = None,
    file_types: list[str] | None = None,
    semantic_query: str | None = None,
) -> QuerySpec:
    return QuerySpec(
        semantic_query=semantic_query or raw_text,
        raw_text=raw_text,
        source_filters=SourceFilters(
            asset_types=asset_types or [],
            file_types=file_types or [],
        ),
    )


def test_sanitize_strips_inferred_asset_types_for_bare_diagram():
    spec = _spec_with_filters(raw_text="diagram", asset_types=["diagram"])
    sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.asset_types == []
    assert sanitized.semantic_query == "diagram"
    assert sanitized.sanitization_notes


def test_sanitize_strips_inferred_asset_types_for_bare_flowchart():
    spec = _spec_with_filters(raw_text="flowchart", asset_types=["diagram"])
    sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.asset_types == []
    assert sanitized.semantic_query == "flowchart"


def test_sanitize_strips_inferred_pptx_for_bare_presentation():
    spec = _spec_with_filters(raw_text="presentation", file_types=["pptx"])
    sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.file_types == []
    assert sanitized.semantic_query == "presentation"
    assert sanitized.sanitization_notes


def test_sanitize_preserves_explicit_file_type_filter():
    spec = _spec_with_filters(
        raw_text="only pptx files with charts",
        file_types=["pptx"],
        semantic_query="charts",
    )
    with patch(
        "imagecb.retrieval.query_parser._file_type_filter_has_index_coverage",
        return_value=True,
    ):
        sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.file_types == ["pptx"]


def test_sanitize_preserves_explicit_asset_type_filter():
    spec = _spec_with_filters(
        raw_text="only diagrams from last quarter",
        asset_types=["diagram"],
        semantic_query="last quarter",
    )
    with patch(
        "imagecb.retrieval.query_parser._corpus_asset_type_unclassified_rate",
        return_value=0.0,
    ):
        sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.asset_types == ["diagram"]


def test_build_spec_then_sanitize_strips_llm_style_filters():
    spec = _build_spec(
        {
            "semantic_query": "presentation",
            "source_filters": {"file_types": ["pptx"]},
        },
        "presentation",
    )
    sanitized = sanitize_query_spec(spec)
    assert sanitized.source_filters.file_types == []
    assert sanitized.source_filters.asset_types == []


@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_session_ask_nonzero_candidates_after_sanitized_spec(
    mock_parse,
    mock_search,
    mock_rerank,
):
    from imagecb.retrieval.rerank import RankedResult

    bad_spec = _spec_with_filters(raw_text="diagram", asset_types=["diagram"])
    good_spec = sanitize_query_spec(bad_spec)
    assert good_spec.source_filters.asset_types == []

    mock_parse.return_value = good_spec
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rerank.return_value = [
        RankedResult(
            image_id="img-1",
            score=0.8,
            record=MagicMock(),
            provenance_line="Slide 1",
        )
    ]

    result = ChatSession().ask("diagram")

    assert result.candidate_count == 1
    assert len(result.results) == 1
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs.get("restrict_to") is None
