"""Tests for suggestion generation and cache (heuristics on request path)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from imagecb.suggestions import generate as gen_mod
from imagecb.suggestions.corpus_summary import CorpusContext, SourceFileStat
from imagecb.suggestions.generate import (
    ONBOARDING_SUGGESTIONS,
    _blend_suggestions,
    _corpus_heuristic_suggestions,
    _is_filename_filter_suggestion,
    generate_suggestions,
    _coerce_suggestions_json,
)


@pytest.fixture(autouse=True)
def clear_cache():
    gen_mod._cache.clear()
    yield
    gen_mod._cache.clear()


def _ctx(**kwargs) -> CorpusContext:
    defaults = dict(indexed_count=5, fingerprint="fp")
    defaults.update(kwargs)
    return CorpusContext(**defaults)


def test_coerce_suggestions_json_from_object():
    raw = '{"suggestions": ["Find charts", "Logos only"]}'
    assert _coerce_suggestions_json(raw) == ["Find charts", "Logos only"]


def test_coerce_suggestions_json_strips_fences():
    raw = '```json\n{"suggestions": ["A", "B"]}\n```'
    assert _coerce_suggestions_json(raw) == ["A", "B"]


def test_is_filename_filter_suggestion_detects_patterns():
    assert _is_filename_filter_suggestion("images from report.pptx")
    assert _is_filename_filter_suggestion("Images from deck.pptx")
    assert _is_filename_filter_suggestion("slides from annual.pdf")
    assert not _is_filename_filter_suggestion("holographic data analytics")
    assert not _is_filename_filter_suggestion("cybersecurity alerts and digital threats")


def test_blend_suggestions_strips_filename_filters():
    ctx = _ctx(
        sample_recommended_cases=("Healthcare technology visuals",),
        top_tags=("healthcare",),
    )
    blended = _blend_suggestions(
        ["topic A", "images from report.pptx", "topic B"],
        ctx,
        4,
    )
    assert "images from report.pptx" not in blended
    assert "report.pptx" not in " ".join(blended).lower()
    assert len(blended) == 4


def test_empty_corpus_returns_onboarding_without_llm():
    ctx = CorpusContext(indexed_count=0, fingerprint="empty")
    with patch.object(gen_mod, "get_suggestion_llm") as mock_llm:
        result = generate_suggestions(limit=4, ctx=ctx)
    mock_llm.assert_not_called()
    assert result.suggestions == ONBOARDING_SUGGESTIONS[:4]
    assert result.cached is False


def test_indexed_corpus_uses_heuristics_without_llm():
    ctx = _ctx(
        indexed_count=10,
        fingerprint="abc123",
        sample_recommended_cases=("Healthcare technology", "Cybersecurity alerts"),
        top_tags=("healthcare", "cybersecurity"),
        source_files=(SourceFileStat(name="deck.pptx", source_type="pptx", count=5),),
    )
    with patch.object(gen_mod, "get_suggestion_llm") as mock_llm:
        result = generate_suggestions(limit=4, ctx=ctx)
    mock_llm.assert_not_called()
    assert len(result.suggestions) == 4
    assert "Healthcare technology" in result.suggestions
    assert "images from" not in " ".join(result.suggestions).lower()
    assert result.cached is False


def test_cache_hit_on_second_call():
    ctx = _ctx(
        indexed_count=5,
        fingerprint="fp1",
        sample_recommended_cases=("One", "Two", "Three", "Four"),
    )
    with patch.object(gen_mod, "get_suggestion_llm") as mock_llm:
        r1 = generate_suggestions(limit=4, ctx=ctx)
        r2 = generate_suggestions(limit=4, ctx=ctx)
    mock_llm.assert_not_called()
    assert r1.cached is False
    assert r2.cached is True
    assert r2.suggestions == r1.suggestions


def test_cache_key_differs_by_limit():
    ctx = _ctx(fingerprint="fp1")
    key_a = gen_mod._cache_key(ctx, 4)
    key_b = gen_mod._cache_key(ctx, 6)
    assert key_a != key_b


def test_heuristic_uses_recommended_cases_not_filename_filters():
    ctx = _ctx(
        sample_recommended_cases=("Photos of team meetings",),
        top_tags=("meeting",),
        source_files=(SourceFileStat(name="report.pptx", source_type="pptx", count=3),),
    )
    items = _corpus_heuristic_suggestions(ctx, 4)
    assert "Q3_Review.pptx" not in items
    assert "Photos of team meetings" in items
    assert sum(1 for s in items if _is_filename_filter_suggestion(s)) == 0
    assert "report.pptx" not in " ".join(items).lower()


def test_sparse_indexed_corpus_does_not_use_onboarding():
    """Indexed rows with thin caption/tag metadata must not show empty-corpus chips."""
    ctx = _ctx(
        indexed_count=50,
        fingerprint="sparse",
        source_files=(SourceFileStat(name="deck.pptx", source_type="pptx", count=50),),
        file_type_counts=(("pptx", 50),),
        top_asset_types=(("photo", 30), ("diagram", 20)),
        sample_image_names=("Developer Code Review", "System Architecture"),
    )
    with patch.object(gen_mod, "get_suggestion_llm") as mock_llm:
        result = generate_suggestions(limit=4, ctx=ctx)
    mock_llm.assert_not_called()
    assert len(result.suggestions) == 4
    assert result.suggestions != ONBOARDING_SUGGESTIONS[:4]
    assert "Upload slides or PDFs" not in result.suggestions
    joined = " ".join(result.suggestions).lower()
    assert "photo" in joined or "developer" in joined or "diagram" in joined


def test_cache_expires_after_ttl():
    ctx = _ctx(
        indexed_count=2,
        fingerprint="fp3",
        sample_recommended_cases=("X", "Y", "Z", "W"),
    )
    key = gen_mod._cache_key(ctx, 4)
    with patch.object(gen_mod, "get_suggestion_llm") as mock_llm:
        r1 = generate_suggestions(limit=4, ctx=ctx)
        gen_mod._cache[key] = (time.monotonic() - 10_000, list(r1.suggestions))
        r2 = generate_suggestions(limit=4, ctx=ctx)
    mock_llm.assert_not_called()
    assert r1.cached is False
    assert r2.cached is False
    assert r2.suggestions == r1.suggestions
