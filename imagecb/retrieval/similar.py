"""Image-to-image similarity search with visual + text dual-signal fusion."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.models.embedder import get_embedder
from imagecb.models.vlm import ImageQueryJSON, get_captioner
from imagecb.paths import resolve_image_file
from imagecb.retrieval.hybrid import rrf_merge
from imagecb.retrieval.image_query import (
    SimilarityAxis,
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
    path = resolve_image_file(record)
    if path is None:
        return None
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load image %s: %s", path, exc)
        return None


def _filter_by_min_score(results: List[RankedResult], min_score: float) -> List[RankedResult]:
    if min_score <= 0:
        return results
    return [r for r in results if r.score >= min_score]


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
    min_score: float,
) -> List[RankedResult]:
    text_hits = [(r.image_id, r.score) for r in text_ranked]
    merged = rrf_merge(visual_hits, text_hits, SETTINGS.rrf_k)

    dense_by_id = {i: s for i, s in visual_hits}
    head = merged[: max(top_k * 3, top_k)]
    ids = [c.image_id for c in head]
    records = {r.image_id: r for r in metadata_db.get_records(ids)}

    results: List[RankedResult] = []
    for c in head:
        rec = records.get(c.image_id)
        if rec is None:
            continue
        results.append(
            RankedResult(
                image_id=c.image_id,
                score=dense_by_id.get(c.image_id, 0.0),
                record=rec,
                provenance_line=_format_provenance(rec),
                score_kind="dense",
            )
        )
        if len(results) >= top_k:
            break

    return _filter_by_min_score(results, min_score)


def search_similar(
    *,
    image_id: Optional[str] = None,
    image: Optional[Image.Image] = None,
    top_k: int = 10,
    exclude_image_id: Optional[str] = None,
    min_match_percent: int = 0,
    similarity_axis: str = "balanced",
    restrict_to: Optional[Sequence[str]] = None,
) -> SimilarSearchOutcome:
    """Find images similar to a reference via visual embedding + VLM text query fusion."""
    top_k = max(1, min(int(top_k), 50))
    min_score = max(0.0, min(float(min_match_percent) / 100.0, 1.0))
    axis = SimilarityAxis.parse(similarity_axis)

    pil, record, resolved_exclude = _resolve_reference_image(image_id=image_id, image=image)
    if pil is None:
        empty_spec = QuerySpec(raw_text="[similar image search]", top_k=top_k)
        return SimilarSearchOutcome(results=[], facets=ImageQueryJSON.empty(), spec=empty_spec)

    exclude_image_id = exclude_image_id or resolved_exclude
    dense_k = min(SETTINGS.dense_top_k, max(top_k * 5, 25))

    def _embed() -> object:
        return get_embedder().embed_image(pil)

    def _query_vlm() -> ImageQueryJSON:
        return get_captioner().query_image(pil)

    with ThreadPoolExecutor(max_workers=2) as pool:
        emb_future = pool.submit(_embed)
        vlm_future = pool.submit(_query_vlm)
        query_emb = emb_future.result()
        facets = vlm_future.result()

    raw_text = "[similar image search]"
    if record and record.image_name:
        raw_text = f"[Find similar] {record.image_name}"

    spec = query_spec_from_image_query(facets, axis, top_k=top_k, raw_text=raw_text)

    visual_hits = _visual_hits(query_emb, dense_k=dense_k, exclude_image_id=exclude_image_id)
    text_ranked = run_text_similar_leg(spec, facets, restrict_to=restrict_to, top_k=top_k)

    if not visual_hits and not text_ranked:
        return SimilarSearchOutcome(results=[], facets=facets, spec=spec)

    results = _fuse_and_rank(visual_hits, text_ranked, top_k=top_k, min_score=min_score)
    return SimilarSearchOutcome(results=results, facets=facets, spec=spec)
