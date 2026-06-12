"""Tests for corpus summary aggregation."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from imagecb.suggestions.corpus_summary import (
    _summarize_records,
    build_corpus_context,
    context_to_prompt_text,
)


def _record(
    *,
    source_file: str = "/data/deck.pptx",
    source_type: str = "pptx",
    author: str = "Alice",
    caption_short: str = "A bar chart",
    tags_json: str | None = '["chart", "bar"]',
    recommended_cases_json: str | None = '["Find bar charts"]',
    modified: datetime | None = None,
):
    return SimpleNamespace(
        source_file=source_file,
        source_type=source_type,
        author=author,
        caption_short=caption_short,
        tags_json=tags_json,
        recommended_cases_json=recommended_cases_json,
        source_modified_at=modified or datetime(2024, 6, 1),
    )


def test_summarize_records_aggregates_sources_and_authors():
    records = [
        _record(source_file="/a/Q3.pptx", author="Alice"),
        _record(source_file="/a/Q3.pptx", author="Alice"),
        _record(source_file="/b/report.pdf", source_type="pdf", author="Bob"),
    ]
    ctx = _summarize_records(records)
    assert ctx.indexed_count == 3
    assert ctx.source_files[0].name == "Q3.pptx"
    assert ctx.source_files[0].count == 2
    assert "Alice" in ctx.authors
    assert ctx.fingerprint


def test_fingerprint_stable_for_same_data():
    records = [_record(), _record(source_file="/b/x.pdf", source_type="pdf")]
    a = _summarize_records(records)
    b = _summarize_records(records)
    assert a.fingerprint == b.fingerprint


def test_empty_records_zero_count():
    ctx = _summarize_records([])
    assert ctx.indexed_count == 0
    assert ctx.fingerprint


def test_context_to_prompt_text_includes_sources():
    ctx = _summarize_records([_record(source_file="/data/deck.pptx")])
    text = context_to_prompt_text(ctx)
    assert "deck.pptx" in text
    assert "Indexed images: 1" in text


@patch("imagecb.suggestions.corpus_summary.metadata_db.get_all_records")
def test_build_corpus_context_calls_db(mock_get):
    mock_get.return_value = [_record()]
    ctx = build_corpus_context()
    assert ctx.indexed_count == 1
    mock_get.assert_called_once()


def test_summarize_records_includes_tags_and_recommended_cases():
    records = [
        _record(tags_json='["chart", "revenue"]', recommended_cases_json='["Show revenue charts"]'),
        _record(tags_json='["chart", "sales"]', recommended_cases_json='["Quarterly sales"]'),
    ]
    ctx = _summarize_records(records)
    assert "chart" in ctx.top_tags
    assert "Show revenue charts" in ctx.sample_recommended_cases


def test_context_to_prompt_text_includes_tags_and_cases():
    ctx = _summarize_records([_record()])
    text = context_to_prompt_text(ctx)
    assert "chart" in text
    assert "Find bar charts" in text


def test_context_to_prompt_text_deemphasizes_filenames():
    ctx = _summarize_records([_record(source_file="/data/report.pptx")])
    text = context_to_prompt_text(ctx)
    tags_pos = text.find("Common tags")
    sources_pos = text.find("Top source files")
    assert tags_pos != -1
    assert sources_pos != -1
    assert tags_pos < sources_pos
    assert "do not suggest filename-filter" in text
