"""Hybrid retrieval with metadata pre-filtering.

Three lanes fused with Reciprocal Rank Fusion:
- visual dense: query text -> multimodal embedding -> image vectors
- caption-text dense: query text -> text embedding -> caption-document vectors
- sparse: BM25 over the same caption documents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.embedder import get_embedder, get_text_embedder
from imagecb.retrieval.query_build import dense_query_text, resolve_retrieval_top_k
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.storage import bm25_index, metadata_db, vector_store

logger = logging.getLogger(__name__)


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
) -> List[Candidate]:
    """Three-lane (visual, caption-text, sparse) RRF with equal weights."""
    cands: Dict[str, Candidate] = {}
    _rrf_accumulate(cands, dense, k, 1.0, "dense_score")
    _rrf_accumulate(cands, text, k, 1.0, "text_score")
    _rrf_accumulate(cands, sparse, k, 1.0, "sparse_score")
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
) -> SearchOutcome:
    """Run dense + sparse search and merge with RRF."""
    default_dense, default_sparse = resolve_retrieval_top_k(spec)
    dense_k = dense_top_k if dense_top_k is not None else default_dense
    sparse_k = sparse_top_k if sparse_top_k is not None else default_sparse
    rrf = rrf_k or SETTINGS.rrf_k

    allowed = _apply_metadata_filter(spec, restrict_to)
    active_ids = set(metadata_db.get_active_image_ids())
    if allowed is None:
        allowed = list(active_ids)
    else:
        allowed = [i for i in allowed if i in active_ids]
    if not allowed:
        return SearchOutcome(candidates=[])

    query_text = dense_query_text(spec)
    if not query_text:
        return SearchOutcome(candidates=[])

    visual_failed = False
    text_failed = False
    sparse_failed = False

    # Visual dense: cross-modal query embedding -> image vectors
    try:
        query_emb = get_embedder().embed_text([query_text])[0]
        dense_hits = vector_store.query(query_emb, top_k=dense_k, allowed_ids=allowed)
    except Exception as exc:  # noqa: BLE001
        visual_failed = True
        logger.warning("Visual dense search failed (%s): %s", type(exc).__name__, exc)
        dense_hits = []

    # Caption-text dense: text query embedding -> caption-document vectors
    text_hits: List[tuple[str, float]] = []
    if SETTINGS.caption_text_lane_enabled:
        try:
            text_query_emb = get_text_embedder().embed_query(query_text)
            text_hits = vector_store.query_text(
                text_query_emb, top_k=dense_k, allowed_ids=allowed
            )
        except Exception as exc:  # noqa: BLE001
            text_failed = True
            logger.warning("Caption-text search failed (%s): %s", type(exc).__name__, exc)

    # Sparse via BM25
    try:
        sparse_hits = bm25_index.get_index().query(
            query_text, top_k=sparse_k, allowed_ids=allowed
        )
    except Exception as exc:  # noqa: BLE001
        sparse_failed = True
        logger.warning("Sparse search failed (%s): %s", type(exc).__name__, exc)
        sparse_hits = []

    # Report dense failure only when no dense lane produced results.
    dense_failed = visual_failed and (
        text_failed or not SETTINGS.caption_text_lane_enabled
    )

    merged = rrf_merge_lanes(dense_hits, text_hits, sparse_hits, rrf)

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

    return SearchOutcome(
        candidates=merged,
        dense_failed=dense_failed,
        sparse_failed=sparse_failed,
    )
