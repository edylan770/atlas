"""Shared query text and retrieval-tuning heuristics for hybrid search."""

from __future__ import annotations

from typing import List, Tuple

from imagecb.caption.normalize import tokenize_text
from imagecb.config import SETTINGS
from imagecb.retrieval.query_parser import QuerySpec


def _dedupe_join(parts: List[str]) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out).strip()


def query_token_count(spec: QuerySpec) -> int:
    """Approximate query length in tokens for retrieval tuning."""
    text = (spec.semantic_query or spec.raw_text or "").strip()
    return len(tokenize_text(text)) + len(spec.must_have_keywords)


def is_short_query(spec: QuerySpec) -> bool:
    return query_token_count(spec) <= SETTINGS.short_query_max_tokens


def dense_query_text(spec: QuerySpec) -> str:
    """Text used for dense embedding and sparse BM25 (semantic + must-have only)."""
    parts: List[str] = []
    semantic = (spec.semantic_query or spec.raw_text or "").strip()
    if semantic:
        parts.append(semantic)
    parts.extend(k for k in spec.must_have_keywords if k)
    return _dedupe_join(parts)


def rerank_query_text(spec: QuerySpec, fallback: str = "") -> str:
    """Text passed to the cross-encoder reranker."""
    text = dense_query_text(spec)
    if text:
        return text
    return (fallback or "").strip()


def resolve_rerank_top_n(spec: QuerySpec, top_k: int = 10) -> int:
    if is_short_query(spec):
        base = SETTINGS.short_query_rerank_top_n
    else:
        base = SETTINGS.rerank_top_n
    return max(base, top_k * 3)


def resolve_retrieval_top_k(spec: QuerySpec) -> Tuple[int, int]:
    if is_short_query(spec):
        k = SETTINGS.short_query_retrieval_top_k
        return k, k
    return SETTINGS.dense_top_k, SETTINGS.sparse_top_k
