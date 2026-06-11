"""Tests for context-aware follow-up query suggestions."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import AskResult
from imagecb.storage.metadata_db import ImageRecord, serialize_list
from imagecb.suggestions.corpus_summary import CorpusContext
from imagecb.suggestions.follow_up import (
    _follow_up_heuristic_suggestions,
    _is_duplicate_query,
    generate_follow_up_suggestions,
)


def _record(
    *,
    image_id: str = "id-1",
    caption_short: str = "Bar chart of revenue",
    tags: list[str] | None = None,
    recommended_cases: list[str] | None = None,
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"data/images/{image_id}.png",
        source_file="/docs/Q3.pptx",
        source_type="pptx",
        source_modified_at=datetime(2024, 9, 15),
        source_created_at=None,
        author="Alice",
        slide_index=1,
        page_index=None,
        slide_title=None,
        slide_notes=None,
        ocr_text=None,
        caption_short=caption_short,
        caption_detailed=None,
        objects_json=None,
        tags_json=serialize_list(tags or ["chart", "finance"]),
        scene=None,
        text_overlay_summary=None,
        created_at=datetime.utcnow(),
        recommended_cases_json=serialize_list(recommended_cases or ["quarterly revenue charts"]),
    )


def _ranked(record: ImageRecord) -> RankedResult:
    return RankedResult(
        image_id=record.image_id,
        score=0.9,
        record=record,
        provenance_line="Slide 1",
    )


def _ask_result(
    *,
    user_query: str = "revenue charts",
    results: list[RankedResult] | None = None,
) -> AskResult:
    return AskResult(
        spec=QuerySpec(semantic_query=user_query, raw_text=user_query),
        results=results or [],
    )


def _ctx(**kwargs) -> CorpusContext:
    defaults = dict(
        indexed_count=10,
        fingerprint="fp",
        top_tags=("chart", "finance"),
        sample_recommended_cases=("quarterly revenue charts",),
        authors=("Alice",),
    )
    defaults.update(kwargs)
    return CorpusContext(**defaults)


def test_is_duplicate_query_detects_user_and_semantic():
    assert _is_duplicate_query("revenue charts", "revenue charts", "revenue charts")
    assert _is_duplicate_query("Revenue Charts", "revenue charts", "")
    assert not _is_duplicate_query("finance dashboards", "revenue charts", "revenue charts")


def test_heuristic_excludes_current_query():
    record = _record(recommended_cases=["revenue charts", "finance dashboards"])
    ask = _ask_result(user_query="revenue charts", results=[_ranked(record)])
    ctx = _ctx()
    items = _follow_up_heuristic_suggestions("revenue charts", ask, ctx, limit=3)
    lowered = [s.lower() for s in items]
    assert "revenue charts" not in lowered
    assert len(items) >= 2


def test_heuristic_uses_result_tags_and_corpus():
    record = _record(tags=["healthcare"], recommended_cases=["patient care visuals"])
    ask = _ask_result(user_query="medical imagery", results=[_ranked(record)])
    ctx = _ctx(top_tags=("healthcare",), sample_recommended_cases=("patient care visuals",))
    items = _follow_up_heuristic_suggestions("medical imagery", ask, ctx, limit=3)
    assert len(items) >= 2
    joined = " ".join(items).lower()
    assert "healthcare" in joined or "patient" in joined


def test_heuristic_strips_filename_filters():
    record = _record()
    ask = _ask_result(results=[_ranked(record)])
    ctx = _ctx()
    with patch(
        "imagecb.suggestions.follow_up._collect_result_candidates",
        return_value=["images from report.pptx", "finance dashboards"],
    ):
        items = _follow_up_heuristic_suggestions("charts", ask, ctx, limit=3)
    assert "images from report.pptx" not in items


def test_empty_corpus_returns_onboarding_when_disabled_llm():
    ask = _ask_result()
    ctx = CorpusContext(indexed_count=0, fingerprint="empty")
    with patch("imagecb.suggestions.follow_up.get_suggestion_llm") as mock_llm:
        mock_llm.side_effect = AssertionError("LLM should not be called")
        items = _follow_up_heuristic_suggestions("test", ask, ctx, limit=3)
    assert len(items) >= 2


@patch("imagecb.suggestions.follow_up.get_suggestion_llm")
@patch("imagecb.suggestions.follow_up.SETTINGS")
def test_generate_returns_empty_when_disabled(mock_settings, mock_llm):
    mock_settings.enable_follow_up_suggestions = False
    ask = _ask_result()
    ctx = _ctx()
    items = generate_follow_up_suggestions("charts", ask, [], corpus=ctx)
    mock_llm.assert_not_called()
    assert items == []


@patch("imagecb.suggestions.follow_up.get_suggestion_llm")
def test_generate_llm_success(mock_llm):
    ask = _ask_result(results=[_ranked(_record())])
    ctx = _ctx()
    mock_llm.return_value.generate.return_value = (
        '{"suggestions": ["finance dashboards", "quarterly reports"]}'
    )
    items = generate_follow_up_suggestions("revenue charts", ask, [], corpus=ctx, limit=3)
    mock_llm.return_value.generate.assert_called_once()
    assert "finance dashboards" in items


@patch("imagecb.suggestions.follow_up.get_suggestion_llm")
def test_generate_llm_failure_falls_back_to_heuristics(mock_llm):
    ask = _ask_result(results=[_ranked(_record())])
    ctx = _ctx()
    mock_llm.return_value.generate.side_effect = RuntimeError("LLM down")
    items = generate_follow_up_suggestions("revenue charts", ask, [], corpus=ctx, limit=3)
    assert len(items) >= 2
