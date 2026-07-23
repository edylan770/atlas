"""Multi-turn session state.

Tracks chat history, the last QuerySpec, and the last ranked results for
follow-up query parsing context. Each search runs parse_query -> hybrid -> rank
over the full active corpus.

Ranking uses 2-lane RRF fused score (visual dense + caption-text dense).
BM25 is retrieved but excluded from fusion (sparse_weight=0.0 in hybrid.py).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from imagecb.retrieval.hybrid import Candidate, normalize_rrf_score, search
from imagecb.retrieval.query_parser import (
    QuerySpec,
    build_session_context,
    parse_query,
    summarize_history,
)
from imagecb.config import SETTINGS
from imagecb.formatting.match_display import meets_min_match_percent
from imagecb.retrieval.rerank import RankedResult, _format_provenance
from imagecb.storage import metadata_db

if TYPE_CHECKING:
    from imagecb.query_timing import QueryTimingSession


@dataclass
class AskResult:
    spec: QuerySpec
    results: List[RankedResult]
    min_match_percent: int = 0
    candidate_count: int = 0
    relaxed_min_score: bool = False
    dense_failed: bool = False
    sparse_failed: bool = False
    visual_fallback: bool = False
    low_confidence_visual: bool = False
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
        sort: Optional[str] = None,
        timing: Optional["QueryTimingSession"] = None,
    ) -> AskResult:
        def _timed(step: str):
            if timing is not None:
                return timing.timed(step)
            return nullcontext()

        with _timed("ask_total"):
            history_summary = summarize_history(self.history)
            session_ctx = build_session_context(self.last_spec, self.last_results)
            with _timed("parse_query"):
                spec = parse_query(text, history_summary, session_context=session_ctx)

            if top_k is not None:
                spec.top_k = max(1, min(int(top_k), 50))

            outcome = search(spec, timing=timing)
            candidates = outcome.candidates

            relaxed_min_score = False

            # Rank by 2-lane RRF fused score (visual dense + caption-text dense).
            with _timed("rrf_rank"):
                ranked = _rank_by_fused_score(candidates, spec.top_k)
                results, relaxed_min_score = _apply_min_match(
                    ranked, candidates, spec.top_k, min_match_percent
                )

                from imagecb.retrieval.sort import resolve_sort, sort_ranked_results

                resolved_sort = resolve_sort(sort, is_search=True)
                results = sort_ranked_results(results, resolved_sort)

            self.last_spec = spec
            self.last_results = list(results)
            self.last_candidate_ids = [r.image_id for r in results] or [
                c.image_id for c in candidates
            ]

            return AskResult(
                spec=spec,
                results=results,
                min_match_percent=min_match_percent,
                candidate_count=len(candidates),
                relaxed_min_score=relaxed_min_score,
                dense_failed=outcome.dense_failed,
                sparse_failed=outcome.sparse_failed,
                visual_fallback=False,
                low_confidence_visual=False,
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
        """Update candidate pool after similar-image search.

        Similar search is a new anchor, not a refinement of the prior result set.
        """
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
            is_refinement=False,
        )
        self.last_spec = merged


def _rank_by_fused_score(candidates: List[Candidate], top_k: int) -> List[RankedResult]:
    """Rank candidates by 2-lane RRF fused score (visual + caption-text dense)."""
    from imagecb.retrieval.dedupe import dedupe_results

    ids = [c.image_id for c in candidates]
    records = {r.image_id: r for r in metadata_db.get_records(ids)}
    built = [
        RankedResult(
            image_id=c.image_id,
            score=normalize_rrf_score(c.fused_score, SETTINGS.rrf_k, weight_sum=2.0),
            record=records[c.image_id],
            provenance_line=_format_provenance(records[c.image_id]),
            score_kind="fusion",
        )
        for c in candidates
        if c.image_id in records
    ]
    built.sort(key=lambda r: r.score, reverse=True)
    return dedupe_results(built, top_k=top_k)


def _apply_min_match(
    results: List[RankedResult],
    candidates: List[Candidate],
    top_k: int,
    min_match_percent: int,
) -> Tuple[List[RankedResult], bool]:
    """Filter results by min-match %, relaxing when nothing qualifies.

    Returns ``(results, relaxed_min_score)``: the kept results when any meet the
    threshold, otherwise the full unfiltered list with ``relaxed_min_score=True``.
    """
    if min_match_percent <= 0:
        return results, False
    kept = [
        r
        for r in results
        if meets_min_match_percent(r.score, r.score_kind, min_match_percent)
    ]
    if kept:
        return kept, False
    return results, True


def _summarize_results(results: List[RankedResult]) -> str:
    if not results:
        return "No results."
    bits = [f"{len(results)} results."]
    for r in results[:3]:
        bits.append(f"- {r.provenance_line}: {r.record.caption_short or ''}".strip())
    return "\n".join(bits)
