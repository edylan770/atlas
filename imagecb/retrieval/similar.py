"""Image-to-image similarity search with visual + text dual-signal fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.formatting.match_display import meets_min_match_percent
from imagecb.models.embedder import get_embedder
from imagecb.models.vlm import ImageQueryJSON, get_captioner
from imagecb.paths import open_record_image
from imagecb.retrieval.hybrid import normalize_rrf_score, rrf_merge
from imagecb.retrieval.image_query import (
    SimilarityAxis,
    axis_lane_weights,
    image_query_from_record,
    query_spec_from_image_query,
    run_text_similar_leg,
)
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult, _format_provenance
from imagecb.storage import metadata_db, vector_store
from imagecb.storage.metadata_db import ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class SimilarSearchOutcome:
    results: List[RankedResult]
    facets: ImageQueryJSON
    spec: QuerySpec


def _load_image_for_record(record: ImageRecord) -> Optional[Image.Image]:
    try:
        return open_record_image(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load image %s: %s", record.image_path, exc)
        return None


def _filter_by_min_match_percent(
    results: List[RankedResult],
    min_match_percent: int,
) -> List[RankedResult]:
    if min_match_percent <= 0:
        return results
    return [
        r
        for r in results
        if meets_min_match_percent(r.score, r.score_kind, min_match_percent)
    ]


def _resolve_reference_image(
    *,
    image_id: Optional[str],
    image: Optional[Image.Image],
) -> Tuple[Optional[Image.Image], Optional[ImageRecord], Optional[str]]:
    """Return (pil, record, exclude_id) for the reference image."""
    if image_id:
        record = metadata_db.get_record(image_id)
        if record is None:
            return None, None, None
        pil = _load_image_for_record(record)
        if pil is None:
            return None, record, image_id
        return pil, record, image_id
    if image is not None:
        return image, None, None
    return None, None, None


def _visual_hits(
    query_emb,
    *,
    dense_k: int,
    exclude_image_id: Optional[str],
) -> List[tuple[str, float]]:
    active_ids = metadata_db.get_active_image_ids()
    try:
        hits = vector_store.query(
            query_emb,
            top_k=dense_k,
            allowed_ids=active_ids if active_ids else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Similar search failed: %s", exc)
        return []

    if exclude_image_id:
        hits = [(i, s) for i, s in hits if i != exclude_image_id]
    return hits


def _fuse_and_rank(
    visual_hits: List[tuple[str, float]],
    text_ranked: List[RankedResult],
    *,
    top_k: int,
    min_match_percent: int,
    axis: SimilarityAxis = SimilarityAxis.BALANCED,
    exclude_image_id: Optional[str] = None,
) -> List[RankedResult]:
    text_hits = [(r.image_id, r.score) for r in text_ranked]
    visual_w, text_w = axis_lane_weights(axis)
    merged = rrf_merge(
        visual_hits,
        text_hits,
        SETTINGS.rrf_k,
        dense_weight=visual_w,
        sparse_weight=text_w,
    )
    weight_sum = (visual_w if visual_hits else 0.0) + (text_w if text_hits else 0.0)

    head = merged[: max(top_k * 5, top_k)]
    ids = [c.image_id for c in head]
    records = {r.image_id: r for r in metadata_db.get_records(ids)}

    built: List[RankedResult] = []
    for c in head:
        if exclude_image_id and c.image_id == exclude_image_id:
            continue
        rec = records.get(c.image_id)
        if rec is None:
            continue
        built.append(
            RankedResult(
                image_id=c.image_id,
                score=normalize_rrf_score(
                    c.fused_score,
                    SETTINGS.rrf_k,
                    weight_sum=weight_sum,
                ),
                record=rec,
                provenance_line=_format_provenance(rec),
                score_kind="fusion",
            )
        )

    from imagecb.retrieval.dedupe import dedupe_results

    results = dedupe_results(built, top_k=top_k)
    return _filter_by_min_match_percent(results, min_match_percent)


def search_similar(
    *,
    image_id: Optional[str] = None,
    image: Optional[Image.Image] = None,
    top_k: int = 10,
    exclude_image_id: Optional[str] = None,
    min_match_percent: int = 0,
    similarity_axis: str = "balanced",
    restrict_to: Optional[Sequence[str]] = None,
    sort: Optional[str] = None,
) -> SimilarSearchOutcome:
    """Find images similar to a reference via visual embedding + VLM text query fusion."""
    top_k = max(1, min(int(top_k), 50))
    axis = SimilarityAxis.parse(similarity_axis)

    pil, record, resolved_exclude = _resolve_reference_image(image_id=image_id, image=image)
    if pil is None:
        empty_spec = QuerySpec(raw_text="[similar image search]", top_k=top_k)
        return SimilarSearchOutcome(results=[], facets=ImageQueryJSON.empty(), spec=empty_spec)

    exclude_image_id = exclude_image_id or resolved_exclude
    dense_k = min(SETTINGS.dense_top_k, max(top_k * 5, 25))

    query_emb = get_embedder().embed_image(pil)
    if record is not None:
        facets = image_query_from_record(record)
    else:
        facets = get_captioner().query_image(pil)

    raw_text = "[similar image search]"
    if record and record.image_name:
        raw_text = f"[Find similar] {record.image_name}"

    spec = query_spec_from_image_query(facets, axis, top_k=top_k, raw_text=raw_text)

    visual_hits = _visual_hits(query_emb, dense_k=dense_k, exclude_image_id=exclude_image_id)
    text_ranked: List[RankedResult] = []
    if facets.is_usable():
        text_ranked = run_text_similar_leg(
            spec,
            facets,
            restrict_to=restrict_to,
            top_k=top_k,
            exclude_image_id=exclude_image_id,
        )
    else:
        logger.warning("VLM image query unavailable; using visual-only similar search")

    if not visual_hits and not text_ranked:
        return SimilarSearchOutcome(results=[], facets=facets, spec=spec)

    results = _fuse_and_rank(
        visual_hits,
        text_ranked,
        top_k=top_k,
        min_match_percent=min_match_percent,
        axis=axis,
        exclude_image_id=exclude_image_id,
    )
    if exclude_image_id:
        results = [r for r in results if r.image_id != exclude_image_id]

    from imagecb.retrieval.sort import resolve_sort, sort_ranked_results

    resolved_sort = resolve_sort(sort, is_search=True)
    results = sort_ranked_results(results, resolved_sort)

    return SimilarSearchOutcome(results=results, facets=facets, spec=spec)
