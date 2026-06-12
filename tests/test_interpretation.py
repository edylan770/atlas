"""Tests for query interpretation notes."""

from __future__ import annotations

from imagecb.api.interpretation import build_interpretation_notes
from imagecb.retrieval.query_parser import QuerySpec


def test_notes_must_have():
    spec = QuerySpec(
        semantic_query="screenshots",
        must_have_keywords=["revenue"],
        must_avoid_keywords=["logo"],
    )
    notes = build_interpretation_notes(spec)
    assert any("Must include" in n for n in notes)
    assert any("Excluding" in n for n in notes)


def test_notes_min_match_percent():
    spec = QuerySpec(semantic_query="cybersecurity")
    notes = build_interpretation_notes(spec, min_match_percent=80)
    assert any("80%" in n for n in notes)


def test_notes_relaxed_min_score():
    spec = QuerySpec(semantic_query="charts")
    notes = build_interpretation_notes(
        spec,
        min_match_percent=80,
        relaxed_min_score=True,
    )
    assert any("80%" in n and "fill results" in n for n in notes)


def test_notes_relaxed_ignored_when_min_match_zero():
    spec = QuerySpec(semantic_query="charts")
    notes = build_interpretation_notes(spec, relaxed_min_score=True)
    assert notes == []


def test_notes_retrieval_failures():
    spec = QuerySpec(semantic_query="charts")
    notes = build_interpretation_notes(
        spec,
        dense_failed=True,
        sparse_failed=True,
    )
    assert any("Dense and sparse" in n for n in notes)


def test_notes_empty_for_plain_query():
    spec = QuerySpec(semantic_query="cybersecurity")
    assert build_interpretation_notes(spec) == []


def test_notes_include_sanitization():
    spec = QuerySpec(
        semantic_query="diagrams",
        sanitization_notes=["Removed asset type filter (inferred, not explicit)."],
    )
    notes = build_interpretation_notes(spec)
    assert any("Removed asset type filter" in n for n in notes)
