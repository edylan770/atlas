"""Tests for shared query text and retrieval-tuning heuristics."""

from __future__ import annotations

from imagecb.retrieval.query_build import (
    dense_query_text,
    is_short_query,
    rerank_query_text,
    resolve_rerank_top_n,
    resolve_retrieval_top_k,
)
from imagecb.retrieval.query_parser import QuerySpec


def test_dense_query_includes_must_have():
    spec = QuerySpec(
        semantic_query="dashboard screenshots",
        must_have_keywords=["revenue", "chart"],
        raw_text="dashboards with revenue charts",
    )
    assert "dashboard screenshots" in dense_query_text(spec)
    assert "revenue" in dense_query_text(spec)
    assert "chart" in dense_query_text(spec)


def test_dense_query_dedupes_repeated_terms():
    spec = QuerySpec(
        semantic_query="sales dashboard",
        must_have_keywords=["sales dashboard", "chart"],
        raw_text="sales dashboard chart",
    )
    text = dense_query_text(spec)
    assert text == "sales dashboard chart"


def test_rerank_query_falls_back():
    spec = QuerySpec(raw_text="")
    assert rerank_query_text(spec, "user asked") == "user asked"


def test_short_query_uses_wider_retrieval_pool():
    spec = QuerySpec(semantic_query="diagram", raw_text="diagram")
    assert is_short_query(spec)
    dense_k, sparse_k = resolve_retrieval_top_k(spec)
    assert dense_k == 100
    assert sparse_k == 100
    assert resolve_rerank_top_n(spec) == 100


def test_rerank_top_n_scales_with_top_k():
    spec = QuerySpec(
        semantic_query="office worker giving a presentation",
        raw_text="office worker giving a presentation",
    )
    assert resolve_rerank_top_n(spec, top_k=50) >= 150


def test_long_query_uses_default_retrieval_pool():
    spec = QuerySpec(
        semantic_query="office worker giving a presentation",
        raw_text="office worker giving a presentation",
    )
    assert not is_short_query(spec)
    dense_k, sparse_k = resolve_retrieval_top_k(spec)
    assert dense_k == 50
    assert sparse_k == 50
