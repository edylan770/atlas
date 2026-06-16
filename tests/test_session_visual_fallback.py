"""Tests for visual-only ranking in ChatSession.ask.

Covers the proactive short-query route (skip Cohere) and the reactive
weak-rerank fallback, both ranking by pure Titan visual similarity
(``score_kind="dense"``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from imagecb.config import SETTINGS
from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.session import ChatSession


def _spec(semantic_query: str) -> QuerySpec:
    return QuerySpec(semantic_query=semantic_query, raw_text=semantic_query)


SHORT_QUERY = "gpus"  # 1 token (<= SHORT_QUERY_MAX_TOKENS)
LONG_QUERY = "operational dashboards with charts"  # 4 tokens (> threshold)


def _rerank_result(score: float):
    from imagecb.retrieval.rerank import RankedResult

    return RankedResult(
        image_id="img-rerank",
        score=score,
        record=MagicMock(),
        provenance_line="Slide 1",
        score_kind="rerank",
    )


def _dense_result(score: float = 0.6):
    from imagecb.retrieval.rerank import RankedResult

    return RankedResult(
        image_id="img-dense",
        score=score,
        record=MagicMock(),
        provenance_line="Slide 2",
        score_kind="dense",
    )


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=False)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_short_query_skips_cohere_and_ranks_by_dense(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(SHORT_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.6)],
    )
    mock_visual.return_value = [_dense_result(0.6)]

    result = session.ask(SHORT_QUERY)

    assert mock_rerank.call_count == 0
    assert mock_visual.call_count == 1
    assert result.visual_fallback is True
    assert result.results[0].score_kind == "dense"


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=True)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_short_query_with_lexical_hit_uses_rerank(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(SHORT_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.6, sparse_score=8.0)],
    )
    mock_rerank.return_value = [_rerank_result(0.8)]

    result = session.ask(SHORT_QUERY)

    # A confident lexical hit keeps the fused/rerank path even for a short query.
    assert mock_visual.call_count == 0
    assert mock_rerank.call_count == 1
    assert result.visual_fallback is False
    assert result.results[0].score_kind == "rerank"


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=False)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_short_query_weak_visual_flags_low_confidence(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(SHORT_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.3)],
    )
    # Top adjusted score 0.3 is below the default confidence floor (0.5).
    mock_visual.return_value = [_dense_result(0.3)]

    result = session.ask(SHORT_QUERY)

    assert result.visual_fallback is True
    assert result.low_confidence_visual is True


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=False)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_long_query_weak_cohere_triggers_dense_fallback(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(LONG_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.6)],
    )
    # raw 0.04 maps to ~3% on the rerank display scale (<= 30%).
    mock_rerank.return_value = [_rerank_result(0.04)]
    mock_visual.return_value = [_dense_result(0.6)]

    result = session.ask(LONG_QUERY)

    assert mock_rerank.call_count == 1
    assert mock_visual.call_count == 1
    assert result.visual_fallback is True
    assert result.results[0].score_kind == "dense"


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=False)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_long_query_strong_cohere_keeps_rerank(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(LONG_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.6)],
    )
    # raw 0.7 maps to ~80% on the rerank display scale (> 30%).
    mock_rerank.return_value = [_rerank_result(0.7)]

    result = session.ask(LONG_QUERY)

    assert mock_visual.call_count == 0
    assert result.visual_fallback is False
    assert result.results[0].score_kind == "rerank"


@patch("imagecb.retrieval.session.has_high_confidence_lexical_hit", return_value=False)
@patch("imagecb.retrieval.session.visual_only_rank")
@patch("imagecb.retrieval.session.rerank")
@patch("imagecb.retrieval.session.search")
@patch("imagecb.retrieval.session.parse_query")
def test_both_visual_paths_disabled_behaves_as_before(
    mock_parse, mock_search, mock_rerank, mock_visual, _mock_lexical
):
    session = ChatSession()
    mock_parse.return_value = _spec(SHORT_QUERY)
    mock_search.return_value = SearchOutcome(
        candidates=[Candidate(image_id="img-1", dense_score=0.6)],
    )
    mock_rerank.return_value = [_rerank_result(0.04)]

    # Settings is a frozen dataclass; swap the module reference for a stand-in
    # with both visual routes disabled.
    disabled = SimpleNamespace(
        visual_fallback_enabled=False,
        visual_short_query_enabled=False,
        visual_fallback_max_display_percent=SETTINGS.visual_fallback_max_display_percent,
        rrf_k=SETTINGS.rrf_k,
    )
    with patch("imagecb.retrieval.session.SETTINGS", disabled):
        result = session.ask(SHORT_QUERY)

    assert mock_visual.call_count == 0
    assert result.visual_fallback is False
    assert result.results[0].score_kind == "rerank"
