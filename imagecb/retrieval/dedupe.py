"""De-duplicate near-identical images in ranked result lists."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from imagecb.config import SETTINGS
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage import vector_store


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _is_near_duplicate(
    embedding: np.ndarray,
    kept_embeddings: Sequence[np.ndarray],
    *,
    threshold: float,
) -> bool:
    for kept in kept_embeddings:
        if cosine_similarity(embedding, kept) >= threshold:
            return True
    return False


def dedupe_results(
    results: Sequence[RankedResult],
    *,
    top_k: int,
    pool: Optional[Sequence[RankedResult]] = None,
) -> List[RankedResult]:
    """Return up to top_k unique results, walking score-ordered candidates.

    Exact duplicates are collapsed by content_hash. Near-duplicates are
    collapsed when stored embedding cosine similarity meets the configured
    threshold (default 0.98). Missing embeddings skip near-dup checks.
    """
    if not results or top_k <= 0:
        return []
    if not SETTINGS.result_deduplicate_enabled:
        return list(results[:top_k])

    threshold = SETTINGS.result_deduplicate_similarity_threshold
    seen_ids: set[str] = set()
    ordered: List[RankedResult] = []
    for r in results:
        if r.image_id not in seen_ids:
            seen_ids.add(r.image_id)
            ordered.append(r)
    if pool:
        for r in pool:
            if r.image_id not in seen_ids:
                seen_ids.add(r.image_id)
                ordered.append(r)

    embed_ids = [r.image_id for r in ordered]
    embeddings = vector_store.get_embeddings(embed_ids)

    kept: List[RankedResult] = []
    kept_embeddings: List[np.ndarray] = []
    kept_hashes: set[str] = set()

    for r in ordered:
        if len(kept) >= top_k:
            break

        content_hash = (r.record.content_hash or "").strip()
        if content_hash and content_hash in kept_hashes:
            continue

        emb = embeddings.get(r.image_id)
        if emb is not None and _is_near_duplicate(emb, kept_embeddings, threshold=threshold):
            continue

        kept.append(r)
        if content_hash:
            kept_hashes.add(content_hash)
        if emb is not None:
            kept_embeddings.append(emb)

    return kept
