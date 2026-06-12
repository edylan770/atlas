"""Image-query facets and axis-weighted text search for similar-image retrieval."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.hybrid import search
from imagecb.retrieval.query_build import rerank_query_text
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult, rerank
from imagecb.storage.metadata_db import ImageRecord


class SimilarityAxis(str, Enum):
    BALANCED = "balanced"
    SUBJECT = "subject"
    STYLE = "style"
    LAYOUT = "layout"

    @classmethod
    def parse(cls, value: str) -> "SimilarityAxis":
        try:
            return cls(value.lower().strip())
        except ValueError as exc:
            raise ValueError(
                f"Unknown similarity_axis: {value!r}. "
                f"Expected one of: {', '.join(a.value for a in cls)}"
            ) from exc


def axis_lane_weights(axis: SimilarityAxis) -> tuple[float, float]:
    """Return (visual_weight, text_weight) for similar-search RRF fusion."""
    if axis == SimilarityAxis.SUBJECT:
        return (0.35, 1.65)
    if axis in (SimilarityAxis.STYLE, SimilarityAxis.LAYOUT):
        return (1.65, 0.35)
    return (1.0, 1.0)


def image_query_from_record(record: ImageRecord) -> ImageQueryJSON:
    """Build similar-search facets from ingest-time caption stored on the record."""
    from imagecb.caption.quality import caption_json_from_record

    cap = caption_json_from_record(record)
    if cap.caption_quality == "failed":
        return ImageQueryJSON.failed("stored caption failed")

    if cap.recommended_cases:
        search_query = cap.recommended_cases[0]
    elif cap.short_caption:
        search_query = cap.short_caption
    else:
        search_query = cap.detailed_description

    subject = cap.scene or cap.use_case
    theme = cap.theme
    visible_text = cap.readable_text or (record.ocr_text or "").strip()

    return ImageQueryJSON(
        search_query=search_query,
        subject=subject,
        style=theme,
        layout="",
        salient_objects=cap.objects[:6],
        visible_text=visible_text,
        colors_mood=theme,
    )


def query_spec_from_image_query(
    facets: ImageQueryJSON,
    axis: SimilarityAxis,
    *,
    top_k: int,
    raw_text: str = "[similar image search]",
) -> QuerySpec:
    """Map VLM facets to a QuerySpec for the hybrid text search leg."""
    must_have: List[str] = []

    if axis == SimilarityAxis.SUBJECT:
        semantic = " ".join(
            filter(None, [facets.subject.strip(), " ".join(facets.salient_objects[:6])])
        ).strip()
    elif axis == SimilarityAxis.STYLE:
        semantic = " ".join(filter(None, [facets.style.strip(), facets.colors_mood.strip()])).strip()
    elif axis == SimilarityAxis.LAYOUT:
        semantic = facets.layout.strip()
        if facets.salient_objects:
            must_have.extend(facets.salient_objects[:3])
    else:
        semantic = facets.search_query.strip()
        for kw in (facets.subject, facets.style):
            if kw:
                must_have.append(kw)

    if facets.visible_text.strip():
        must_have.append(facets.visible_text.strip())

    if not semantic:
        semantic = facets.search_query.strip() or raw_text

    return QuerySpec(
        semantic_query=semantic,
        raw_text=raw_text,
        must_have_keywords=must_have,
        top_k=top_k,
    )


def run_text_similar_leg(
    spec: QuerySpec,
    facets: ImageQueryJSON,
    *,
    restrict_to: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    exclude_image_id: Optional[str] = None,
) -> List[RankedResult]:
    """Run hybrid dense+BM25 RRF then rerank; no min-score filter on this leg."""
    k = top_k or spec.top_k
    effective_restrict = restrict_to
    if exclude_image_id and restrict_to is not None:
        effective_restrict = [i for i in restrict_to if i != exclude_image_id]
    outcome = search(spec, restrict_to=effective_restrict)
    candidates = outcome.candidates
    if exclude_image_id:
        candidates = [c for c in candidates if c.image_id != exclude_image_id]
    if not candidates:
        return []

    query_for_rerank = rerank_query_text(spec, facets.search_query)
    return rerank(
        query_for_rerank,
        candidates,
        top_k=min(k, SETTINGS.rerank_top_n),
        min_match_percent=0,
    )


def axis_label(axis: SimilarityAxis) -> str:
    return {
        SimilarityAxis.BALANCED: "balanced visual + text",
        SimilarityAxis.SUBJECT: "subject",
        SimilarityAxis.STYLE: "style",
        SimilarityAxis.LAYOUT: "layout",
    }[axis]
