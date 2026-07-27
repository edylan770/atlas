"""Hybrid retrieval with metadata pre-filtering.

Two lanes fused with Reciprocal Rank Fusion:
- visual dense: query text -> multimodal embedding -> image vectors
- caption-text dense: query text -> text embedding -> caption-document vectors

BM25 sparse scores are still retrieved and stored on each Candidate for
diagnostic use (pipeline lab), but contribute sparse_weight=0.0 to fusion.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.embedder import get_embedder, get_text_embedder
from imagecb.retrieval.query_build import dense_query_text, resolve_retrieval_top_k
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.storage import bm25_index, metadata_db, vector_store

if TYPE_CHECKING:
    from imagecb.query_timing import QueryTimingSession

logger = logging.getLogger(__name__)

# The visual and caption-text lanes are independent network calls; running
# them concurrently roughly halves per-query embedding latency.
_lane_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval-lane")


@dataclass
class SpeculativeEmbeds:
    """Query embeddings started while the query-parse LLM runs.

    Usable only when the parsed dense query text equals the text that was
    speculatively embedded; otherwise search re-embeds normally.
    """

    text: str
    visual: "Future"
    caption_text: Optional["Future"]


def start_speculative_embeds(query_text: str) -> Optional[SpeculativeEmbeds]:
    """Kick off query embeddings concurrently (e.g. during parse_query)."""
    text = (query_text or "").strip()
    if not text or not SETTINGS.speculative_query_embed:
        return None
    visual = _lane_executor.submit(lambda: get_embedder().embed_text([text])[0])
    caption: Optional[Future] = None
    if SETTINGS.caption_text_lane_enabled:
        caption = _lane_executor.submit(lambda: get_text_embedder().embed_query(text))
    return SpeculativeEmbeds(text=text, visual=visual, caption_text=caption)


@dataclass
class Candidate:
    image_id: str
    dense_score: float = 0.0
    text_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class SearchOutcome:
    candidates: List[Candidate]
    dense_failed: bool = False
    sparse_failed: bool = False
    # Sum of RRF weights for the lanes that actually contributed this query.
    # The theoretical max fused score is weight_sum/(rrf_k+1); normalization
    # must use this rather than assuming both dense lanes ran.
    weight_sum: float = 2.0


def _apply_metadata_filter(spec: QuerySpec, restrict_to: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the allowed `image_id` set after applying filters.

    Returns None when there are no filters at all and no restriction
    (caller can search the whole index). Returns an empty list when the
    filters resolve to no images.
    """
    sf = spec.source_filters
    tf = spec.time_filter
    has_filters = bool(
        sf.file_types
        or sf.asset_types
        or sf.filename_contains
        or sf.authors
        or tf.before
        or tf.after
        or restrict_to
    )
    if not has_filters:
        return None

    ids = metadata_db.filter_image_ids(
        file_types=sf.file_types or None,
        asset_types=sf.asset_types or None,
        filename_contains=sf.filename_contains or None,
        authors=sf.authors or None,
        modified_after=tf.after,
        modified_before=tf.before,
    )
    if restrict_to is not None:
        allowed = set(restrict_to)
        ids = [i for i in ids if i in allowed]
    return ids


def _rrf_accumulate(
    cands: Dict[str, Candidate],
    hits: List[tuple[str, float]],
    k: int,
    weight: float,
    score_attr: str,
) -> None:
    for rank, (image_id, score) in enumerate(hits, start=1):
        c = cands.setdefault(image_id, Candidate(image_id=image_id))
        setattr(c, score_attr, score)
        c.fused_score += weight / (k + rank)


def rrf_merge(
    dense: List[tuple[str, float]],
    sparse: List[tuple[str, float]],
    k: int,
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Candidate]:
    """Two-lane Reciprocal Rank Fusion, sorted by fused score desc."""
    cands: Dict[str, Candidate] = {}
    _rrf_accumulate(cands, dense, k, dense_weight, "dense_score")
    _rrf_accumulate(cands, sparse, k, sparse_weight, "sparse_score")
    return sorted(cands.values(), key=lambda c: c.fused_score, reverse=True)


def rrf_merge_lanes(
    dense: List[tuple[str, float]],
    text: List[tuple[str, float]],
    sparse: List[tuple[str, float]],
    k: int,
    *,
    dense_weight: float = 1.0,
    text_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Candidate]:
    """Three-lane (visual, caption-text, sparse) RRF with configurable per-lane weights."""
    cands: Dict[str, Candidate] = {}
    _rrf_accumulate(cands, dense, k, dense_weight, "dense_score")
    _rrf_accumulate(cands, text, k, text_weight, "text_score")
    _rrf_accumulate(cands, sparse, k, sparse_weight, "sparse_score")
    return sorted(cands.values(), key=lambda c: c.fused_score, reverse=True)


def normalize_rrf_score(
    fused_score: float,
    k: int,
    *,
    weight_sum: float,
) -> float:
    """Map raw RRF sum to [0, 1] using theoretical max for active lane weights."""
    if weight_sum <= 0 or fused_score <= 0:
        return 0.0
    max_score = weight_sum / (k + 1)
    return min(1.0, fused_score / max_score)


def search(
    spec: QuerySpec,
    *,
    restrict_to: Optional[Sequence[str]] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    timing: Optional["QueryTimingSession"] = None,
    speculative: Optional[SpeculativeEmbeds] = None,
) -> SearchOutcome:
    """Run dense + sparse search and merge with RRF."""
    default_dense, default_sparse = resolve_retrieval_top_k(spec)
    dense_k = dense_top_k if dense_top_k is not None else default_dense
    sparse_k = sparse_top_k if sparse_top_k is not None else default_sparse
    rrf = rrf_k or SETTINGS.rrf_k

    def _timed(step: str):
        if timing is not None:
            return timing.timed(step)
        return nullcontext()

    with _timed("metadata_filter"):
        filtered = _apply_metadata_filter(spec, restrict_to)
        active_ids = set(metadata_db.get_active_image_ids())
        if filtered is None:
            # Unfiltered search: let Chroma scan its whole index instead of
            # evaluating a corpus-sized $in filter, and post-filter the hits
            # against active ids (guards stale/soft-deleted vectors).
            allowed = None
        else:
            allowed = [i for i in filtered if i in active_ids]
            if not allowed:
                return SearchOutcome(candidates=[])
    if not active_ids:
        return SearchOutcome(candidates=[])

    query_text = dense_query_text(spec)
    if not query_text:
        return SearchOutcome(candidates=[])

    visual_failed = False
    text_failed = False
    sparse_failed = False

    spec_ok = speculative is not None and speculative.text == query_text

    def _visual_lane() -> List[tuple[str, float]]:
        with _timed("embed_visual"):
            query_emb = None
            if spec_ok:
                try:
                    query_emb = speculative.visual.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Speculative visual embed failed: %s", exc)
            if query_emb is None:
                query_emb = get_embedder().embed_text([query_text])[0]
        with _timed("chroma_visual"):
            return vector_store.query(query_emb, top_k=dense_k, allowed_ids=allowed)

    def _text_lane() -> List[tuple[str, float]]:
        with _timed("embed_text"):
            text_query_emb = None
            if spec_ok and speculative.caption_text is not None:
                try:
                    text_query_emb = speculative.caption_text.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Speculative text embed failed: %s", exc)
            if text_query_emb is None:
                text_query_emb = get_text_embedder().embed_query(query_text)
        with _timed("chroma_text"):
            return vector_store.query_text(
                text_query_emb, top_k=dense_k, allowed_ids=allowed
            )

    # Both dense lanes are independent Bedrock+Chroma round-trips: overlap them.
    visual_future = _lane_executor.submit(_visual_lane)
    text_future = (
        _lane_executor.submit(_text_lane)
        if SETTINGS.caption_text_lane_enabled
        else None
    )

    try:
        dense_hits = visual_future.result()
    except Exception as exc:  # noqa: BLE001
        visual_failed = True
        logger.warning("Visual dense search failed (%s): %s", type(exc).__name__, exc)
        dense_hits = []

    text_hits: List[tuple[str, float]] = []
    if text_future is not None:
        try:
            text_hits = text_future.result()
        except Exception as exc:  # noqa: BLE001
            text_failed = True
            logger.warning("Caption-text search failed (%s): %s", type(exc).__name__, exc)

    # Sparse via BM25
    try:
        with _timed("bm25"):
            sparse_hits = bm25_index.get_index().query(
                query_text, top_k=sparse_k, allowed_ids=allowed
            )
    except Exception as exc:  # noqa: BLE001
        sparse_failed = True
        logger.warning("Sparse search failed (%s): %s", type(exc).__name__, exc)
        sparse_hits = []

    if allowed is None:
        # Unfiltered path: drop hits for stale/soft-deleted vectors.
        dense_hits = [(i, sc) for i, sc in dense_hits if i in active_ids]
        text_hits = [(i, sc) for i, sc in text_hits if i in active_ids]
        sparse_hits = [(i, sc) for i, sc in sparse_hits if i in active_ids]

    # Report dense failure only when no dense lane produced results.
    dense_failed = visual_failed and (
        text_failed or not SETTINGS.caption_text_lane_enabled
    )

    with _timed("rrf_rank"):
        merged = rrf_merge_lanes(dense_hits, text_hits, sparse_hits, rrf, sparse_weight=0.0)

        # must_avoid_keywords post-filter: drop any candidate whose text contains
        # an avoided keyword. We look it up from SQLite to keep memory bounded.
        if spec.must_avoid_keywords and merged:
            ids = [c.image_id for c in merged]
            records = {r.image_id: r for r in metadata_db.get_records(ids)}
            avoid = [k.lower() for k in spec.must_avoid_keywords if k]
            kept: List[Candidate] = []
            for c in merged:
                r = records.get(c.image_id)
                if r is None:
                    kept.append(c)
                    continue
                blob = " ".join(
                    filter(
                        None,
                        [
                            r.caption_short,
                            r.caption_detailed,
                            r.scene,
                            r.text_overlay_summary,
                            r.ocr_text,
                            r.slide_title,
                            r.slide_notes,
                        ],
                    )
                ).lower()
                if any(a in blob for a in avoid):
                    continue
                kept.append(c)
            merged = kept

    active_weight_sum = (0.0 if visual_failed else 1.0) + (
        1.0
        if SETTINGS.caption_text_lane_enabled and not text_failed
        else 0.0
    )

    return SearchOutcome(
        candidates=merged,
        dense_failed=dense_failed,
        sparse_failed=sparse_failed,
        weight_sum=active_weight_sum,
    )
