"""Multi-turn session state.

Tracks:

- (user, assistant) chat history (only short text summaries; results are
  rendered separately by the UI).
- The last `QuerySpec` (sticky source/time filters carry forward on
  explicit refinement turns only).
- The last ranked candidate pool, so refinement turns can restrict to it
  instead of searching the whole index again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from imagecb.config import SETTINGS
from imagecb.retrieval.hybrid import search
from imagecb.retrieval.query_build import rerank_query_text, should_restrict_to_previous
from imagecb.retrieval.query_parser import (
    QuerySpec,
    SourceFilters,
    TimeFilter,
    build_session_context,
    parse_query,
    summarize_history,
)
from imagecb.retrieval.rerank import RankedResult, rerank


def _merge_filters(prev: QuerySpec, new: QuerySpec) -> QuerySpec:
    """Carry forward sticky filters from `prev` if `new` doesn't set them."""
    sf_new = new.source_filters
    sf_prev = prev.source_filters
    merged_sf = SourceFilters(
        file_types=sf_new.file_types or list(sf_prev.file_types),
        filename_contains=sf_new.filename_contains or list(sf_prev.filename_contains),
        authors=sf_new.authors or list(sf_prev.authors),
    )
    tf_new = new.time_filter
    tf_prev = prev.time_filter
    merged_tf = TimeFilter(
        before=tf_new.before or tf_prev.before,
        after=tf_new.after or tf_prev.after,
    )
    new.source_filters = merged_sf
    new.time_filter = merged_tf
    return new


def _filters_were_merged(prev: QuerySpec, merged: QuerySpec) -> bool:
    """True if sticky carry-forward changed source/time filters."""
    sf_prev, sf = prev.source_filters, merged.source_filters
    tf_prev, tf = prev.time_filter, merged.time_filter
    if sf.file_types != sf_prev.file_types and sf_prev.file_types:
        return True
    if sf.filename_contains != sf_prev.filename_contains and sf_prev.filename_contains:
        return True
    if sf.authors != sf_prev.authors and sf_prev.authors:
        return True
    if tf.after != tf_prev.after and tf_prev.after:
        return True
    if tf.before != tf_prev.before and tf_prev.before:
        return True
    return False


def has_metadata_filters(spec: QuerySpec) -> bool:
    sf = spec.source_filters
    tf = spec.time_filter
    return bool(sf.file_types or sf.filename_contains or sf.authors or tf.before or tf.after)


@dataclass
class AskResult:
    spec: QuerySpec
    results: List[RankedResult]
    applied_refinement_pool: bool = False
    pool_size: int = 0
    sticky_merged: bool = False
    min_match_percent: int = 0
    candidate_count: int = 0
    filtered_by_min_score: bool = False
    relaxed_min_score: bool = False
    dense_failed: bool = False
    sparse_failed: bool = False
    indexed_count: int = 0


@dataclass
class ChatSession:
    history: List[Tuple[str, str]] = field(default_factory=list)
    last_spec: Optional[QuerySpec] = None
    last_candidate_ids: List[str] = field(default_factory=list)
    last_results: List[RankedResult] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()
        self.last_spec = None
        self.last_candidate_ids = []
        self.last_results = []

    def ask(
        self,
        text: str,
        *,
        top_k: Optional[int] = None,
        min_match_percent: int = 0,
    ) -> AskResult:
        history_summary = summarize_history(self.history)
        session_ctx = build_session_context(self.last_spec, self.last_results)
        spec = parse_query(text, history_summary, session_context=session_ctx)

        sticky_merged = False
        if self.last_spec is not None and spec.is_refinement:
            spec = _merge_filters(self.last_spec, spec)
            sticky_merged = _filters_were_merged(self.last_spec, spec)

        if top_k is not None:
            spec.top_k = max(1, min(int(top_k), 50))

        min_score = max(0.0, min(float(min_match_percent) / 100.0, 1.0))

        pool_size = len(self.last_candidate_ids)
        applied_pool = should_restrict_to_previous(spec, text, pool_size)
        restrict_to: Optional[List[str]] = None
        if applied_pool:
            restrict_to = list(self.last_candidate_ids)

        outcome = search(spec, restrict_to=restrict_to)
        candidates = outcome.candidates
        query_for_rerank = rerank_query_text(spec, text)

        results = rerank(
            query_for_rerank,
            candidates,
            top_k=spec.top_k,
            min_score=min_score,
        )
        relaxed_min_score = False
        filtered_by_min_score = False
        if not results and candidates and min_score > 0:
            filtered_by_min_score = True
            results = rerank(
                query_for_rerank,
                candidates,
                top_k=spec.top_k,
                min_score=0.0,
            )
            relaxed_min_score = bool(results)

        self.last_spec = spec
        self.last_results = list(results)
        self.last_candidate_ids = [r.image_id for r in results] or [
            c.image_id for c in candidates
        ]

        return AskResult(
            spec=spec,
            results=results,
            applied_refinement_pool=applied_pool,
            pool_size=pool_size,
            sticky_merged=sticky_merged,
            min_match_percent=min_match_percent,
            candidate_count=len(candidates),
            filtered_by_min_score=filtered_by_min_score,
            relaxed_min_score=relaxed_min_score,
            dense_failed=outcome.dense_failed,
            sparse_failed=outcome.sparse_failed,
        )

    def record_turn(self, user_text: str, assistant_message: str) -> None:
        """Append a turn using the full assistant reply for better follow-up context."""
        summary = assistant_message.strip() or _summarize_results(self.last_results)
        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."
        self.history.append((user_text, summary))

    def apply_similar_results(
        self,
        results: List[RankedResult],
        *,
        spec: QuerySpec,
    ) -> None:
        """Update candidate pool after a similar-image search."""
        self.last_results = list(results)
        self.last_candidate_ids = [r.image_id for r in results]
        merged = QuerySpec(
            semantic_query=spec.semantic_query,
            raw_text=spec.raw_text,
            must_have_keywords=list(spec.must_have_keywords),
            must_avoid_keywords=list(spec.must_avoid_keywords),
            source_filters=spec.source_filters,
            time_filter=spec.time_filter,
            top_k=spec.top_k,
            is_refinement=True,
        )
        self.last_spec = merged


def _summarize_results(results: List[RankedResult]) -> str:
    if not results:
        return "No results."
    bits = [f"{len(results)} results."]
    for r in results[:3]:
        bits.append(f"- {r.provenance_line}: {r.record.caption_short or ''}".strip())
    return "\n".join(bits)
