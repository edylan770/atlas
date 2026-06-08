"""Hybrid (dense + sparse) retrieval with metadata pre-filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.embedder import get_embedder
from imagecb.retrieval.query_build import dense_query_text
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.storage import bm25_index, metadata_db, vector_store

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    image_id: str
    dense_score: float = 0.0
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
        sf.file_types or sf.filename_contains or sf.authors or tf.before or tf.after or restrict_to
    )
    if not has_filters:
        return None

    ids = metadata_db.filter_image_ids(
        file_types=sf.file_types or None,
        filename_contains=sf.filename_contains or None,
        authors=sf.authors or None,
        modified_after=tf.after,
        modified_before=tf.before,
    )
    if restrict_to is not None:
        allowed = set(restrict_to)
        ids = [i for i in ids if i in allowed]
    return ids


def rrf_merge(
    dense: List[tuple[str, float]],
    sparse: List[tuple[str, float]],
    k: int,
) -> List[Candidate]:
    """Reciprocal Rank Fusion. Returns candidates sorted by fused score desc."""
    cands: Dict[str, Candidate] = {}
    for rank, (image_id, score) in enumerate(dense, start=1):
        c = cands.setdefault(image_id, Candidate(image_id=image_id))
        c.dense_score = score
        c.fused_score += 1.0 / (k + rank)
    for rank, (image_id, score) in enumerate(sparse, start=1):
        c = cands.setdefault(image_id, Candidate(image_id=image_id))
        c.sparse_score = score
        c.fused_score += 1.0 / (k + rank)
    return sorted(cands.values(), key=lambda c: c.fused_score, reverse=True)


def search(
    spec: QuerySpec,
    *,
    restrict_to: Optional[Sequence[str]] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
) -> SearchOutcome:
    """Run dense + sparse search and merge with RRF."""
    dense_k = dense_top_k or SETTINGS.dense_top_k
    sparse_k = sparse_top_k or SETTINGS.sparse_top_k
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

    dense_failed = False
    sparse_failed = False

    # Dense via Titan text embedding -> Chroma
    try:
        text_emb = get_embedder().embed_text([query_text])[0]
        dense_hits = vector_store.query(text_emb, top_k=dense_k, allowed_ids=allowed)
    except Exception as exc:  # noqa: BLE001
        dense_failed = True
        logger.warning("Dense search failed (%s): %s", type(exc).__name__, exc)
        dense_hits = []

    # Sparse via BM25
    sparse_query = query_text
    try:
        sparse_hits = bm25_index.get_index().query(
            sparse_query, top_k=sparse_k, allowed_ids=allowed
        )
    except Exception as exc:  # noqa: BLE001
        sparse_failed = True
        logger.warning("Sparse search failed (%s): %s", type(exc).__name__, exc)
        sparse_hits = []

    merged = rrf_merge(dense_hits, sparse_hits, rrf)

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


def search_candidates(
    spec: QuerySpec,
    *,
    restrict_to: Optional[Sequence[str]] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
) -> List[Candidate]:
    """Backward-compatible wrapper returning only the candidate list."""
    return search(
        spec,
        restrict_to=restrict_to,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        rrf_k=rrf_k,
    ).candidates
