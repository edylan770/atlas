"""Tests for tag normalization."""

from __future__ import annotations

from imagecb.caption.normalize import normalize_tag, normalize_tags, tokenize_text


def test_normalize_tag_lowercase_and_plural():
    assert normalize_tag("Charts") == "chart"
    assert normalize_tag("  Revenue  ") == "revenue"


def test_normalize_tag_plural_exceptions():
    assert normalize_tag("sales") == "sales"
    assert normalize_tag("business") == "business"


def test_normalize_tags_dedupes():
    vocab: set[str] = set()
    result = normalize_tags(["chart", "charts", "Chart"], vocab)
    assert result == ["chart"]


def test_normalize_tags_drops_short_and_empty():
    result = normalize_tags(["", "  ", "a", "chart"], set())
    assert result == ["chart"]


def test_normalize_tags_keeps_vocab_canonical_form():
    vocab = {"chart", "dashboard"}
    result = normalize_tags(["Charts", "Dashboards", "logo"], vocab)
    assert result == ["chart", "dashboard", "logo"]


def test_tokenize_text():
    assert tokenize_text("Sales Dashboard, Q3!") == ["sales", "dashboard", "q3"]
    assert tokenize_text("") == []
