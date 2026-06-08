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
) -> List[RankedResult]:
    """Run hybrid dense+BM25 RRF then rerank; no min-score filter on this leg."""
    k = top_k or spec.top_k
    outcome = search(spec, restrict_to=restrict_to)
    candidates = outcome.candidates
    if not candidates:
        return []

    query_for_rerank = rerank_query_text(spec, facets.search_query)
    return rerank(
        query_for_rerank,
        candidates,
        top_k=min(k, SETTINGS.rerank_top_n),
        min_score=0.0,
    )


def axis_label(axis: SimilarityAxis) -> str:
    return {
        SimilarityAxis.BALANCED: "balanced visual + text",
        SimilarityAxis.SUBJECT: "subject",
        SimilarityAxis.STYLE: "style",
        SimilarityAxis.LAYOUT: "layout",
    }[axis]
