"""Human-readable notes about how a query was interpreted."""

from __future__ import annotations

from typing import List

from imagecb.retrieval.query_parser import QuerySpec


def build_interpretation_notes(
    spec: QuerySpec,
    *,
    min_match_percent: int = 0,
    relaxed_min_score: bool = False,
    dense_failed: bool = False,
    sparse_failed: bool = False,
    visual_fallback: bool = False,
    low_confidence_visual: bool = False,
) -> List[str]:
    notes: List[str] = []
    if low_confidence_visual:
        notes.append(
            "No image strongly matches your search; showing the closest visual matches."
        )
    elif visual_fallback:
        notes.append("Results are ranked by visual similarity.")
    if min_match_percent > 0:
        notes.append(f"Showing matches at or above {min_match_percent}%.")
    if relaxed_min_score and min_match_percent > 0:
        notes.append(
            f"Fewer than {min_match_percent}% matches were available; "
            "including the closest matches to fill results."
        )
    if dense_failed and sparse_failed:
        notes.append("Dense and sparse retrieval both failed (check Bedrock / index).")
    elif dense_failed:
        notes.append("Dense embedding search failed (check Bedrock embedding access).")
    elif sparse_failed:
        notes.append("Sparse BM25 search failed (index may need re-ingest).")

    if spec.must_have_keywords:
        notes.append("Must include: " + ", ".join(spec.must_have_keywords) + ".")
    if spec.must_avoid_keywords:
        notes.append("Excluding: " + ", ".join(spec.must_avoid_keywords) + ".")

    for note in spec.sanitization_notes:
        notes.append(note)

    return notes
