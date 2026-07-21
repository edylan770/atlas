"""Tests for the current fused-score ranking path in ChatSession."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import ChatSession


def _spec(query: str) -> QuerySpec:
    return QuerySpec(semantic_query=query, raw_text=query)


def _fused_result(score: float) -> RankedResult:
    return RankedResult(
        image_id="img-fused",
        score=score,
        record=MagicMock(),
        provenance_line="Slide 1",
        score_kind="fusion",
    )


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_short_query_uses_fused_ranking(mock_parse, mock_search, mock_rank):
    mock_parse.return_value = _spec("gpus")
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rank.return_value = [_fused_result(0.6)]

    result = ChatSession().ask("gpus")

    mock_rank.assert_called_once()
    assert result.results[0].score_kind == "fusion"
    assert result.visual_fallback is False
    assert result.low_confidence_visual is False


@patch("imagecb.retrieval.session._rank_by_fused_score")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_long_query_uses_same_fused_ranking(mock_parse, mock_search, mock_rank):
    query = "operational dashboards with charts"
    mock_parse.return_value = _spec(query)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", fused_score=0.5)]
    )
    mock_rank.return_value = [_fused_result(0.7)]

    result = ChatSession().ask(query)

    mock_rank.assert_called_once()
    assert result.results[0].score_kind == "fusion"
    assert result.visual_fallback is False
