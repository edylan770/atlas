"""Tests for multi-turn session behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.query_parser import QuerySpec, SourceFilters
from imagecb.retrieval.session import ChatSession


def _spec(
    *,
    semantic_query: str = "",
    is_refinement: bool = False,
    filename_contains: list[str] | None = None,
) -> QuerySpec:
    return QuerySpec(
        semantic_query=semantic_query,
        raw_text=semantic_query,
        is_refinement=is_refinement,
        source_filters=SourceFilters(filename_contains=filename_contains or []),
    )


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_fresh_turn_searches_full_corpus(mock_parse, mock_search, mock_rank):
    session = ChatSession()
    session.last_spec = _spec(filename_contains=["Q3_Review.pptx"])
    session.last_candidate_ids = ["img-1", "img-2"]

    mock_parse.return_value = _spec(semantic_query="cybersecurity", is_refinement=False)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-9", fused_score=0.5)]
    )
    mock_rank.return_value = []

    session.ask("cybersecurity")

    mock_search.assert_called_once()


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_refinement_turn_does_not_carry_previous_filters(mock_parse, mock_search, mock_rank):
    session = ChatSession()
    session.last_spec = _spec(filename_contains=["Q3_Review.pptx"])
    session.last_candidate_ids = ["img-1", "img-2"]

    mock_parse.return_value = _spec(semantic_query="only charts", is_refinement=True)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-9", fused_score=0.5)]
    )
    mock_rank.return_value = []

    result = session.ask("only charts")

    assert result.spec.source_filters.filename_contains == []


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_ask_applies_min_match_percent_to_fused_results(mock_parse, mock_search, mock_rank):
    from imagecb.retrieval.rerank import RankedResult

    session = ChatSession()
    mock_parse.return_value = _spec(semantic_query="cybersecurity")
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rank.return_value = [
        RankedResult(
            image_id="img-1",
            score=0.9,
            record=MagicMock(),
            provenance_line="Slide 1",
        )
    ]

    result = session.ask("cybersecurity", min_match_percent=80)

    assert mock_rank.call_count == 1
    assert len(result.results) == 1
    assert result.relaxed_min_score is False


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_ask_uses_no_score_floor_when_min_match_zero(mock_parse, mock_search, mock_rank):
    from imagecb.retrieval.rerank import RankedResult

    session = ChatSession()
    mock_parse.return_value = _spec(semantic_query="cybersecurity")
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rank.return_value = [
        RankedResult(
            image_id="img-1",
            score=0.9,
            record=MagicMock(),
            provenance_line="Slide 1",
        )
    ]

    result = session.ask("cybersecurity")

    assert mock_rank.call_count == 1
    assert len(result.results) == 1
    assert result.relaxed_min_score is False


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_ask_marks_relaxed_when_below_threshold(mock_parse, mock_search, mock_rank):
    from imagecb.retrieval.rerank import RankedResult

    session = ChatSession()
    mock_parse.return_value = _spec(semantic_query="charts")
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rank.return_value = [
        RankedResult(
            image_id="img-1",
            score=0.5,
            record=MagicMock(),
            provenance_line="Slide 1",
        )
    ]

    result = session.ask("charts", min_match_percent=80)

    assert mock_rank.call_count == 1
    assert result.relaxed_min_score is True


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_ask_retains_results_when_all_are_below_threshold(mock_parse, mock_search, mock_rank):
    from imagecb.retrieval.rerank import RankedResult

    session = ChatSession()
    mock_parse.return_value = _spec(semantic_query="charts")
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    ranked = RankedResult(
        image_id="img-1",
        score=0.5,
        record=MagicMock(),
        provenance_line="Slide 1",
    )
    mock_rank.return_value = [ranked]

    result = session.ask("charts", min_match_percent=80)

    assert mock_rank.call_count == 1
    assert result.relaxed_min_score is True
    assert len(result.results) == 1


def test_apply_similar_results_not_refinement():
    from imagecb.retrieval.rerank import RankedResult

    session = ChatSession()
    spec = _spec(semantic_query="hero banner", is_refinement=True)
    results = [
        RankedResult(
            image_id="img-1",
            score=0.9,
            record=MagicMock(),
            provenance_line="Slide 1",
        ),
        RankedResult(
            image_id="img-2",
            score=0.8,
            record=MagicMock(),
            provenance_line="Slide 2",
        ),
    ]

    session.apply_similar_results(results, spec=spec)

    assert session.last_candidate_ids == ["img-1", "img-2"]
    assert session.last_spec is not None
    assert session.last_spec.is_refinement is False
    assert session.last_spec.semantic_query == "hero banner"
