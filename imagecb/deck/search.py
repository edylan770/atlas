"""Direct semantic search for deck slide descriptions (no query LLM)."""

from __future__ import annotations

from typing import List, Optional

from imagecb.formatting.assistant_reply import ResultCard, build_result_cards
from imagecb.retrieval.hybrid import search
from imagecb.retrieval.query_build import rerank_query_text, resolve_rerank_top_n
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult, rerank


def search_for_description(
    description: str,
    *,
    top_k: int = 10,
    min_match_percent: int = 0,
    sort: Optional[str] = None,
    image_url_prefix: str = "/api/images",
) -> tuple[List[ResultCard], List[RankedResult]]:
    """Run hybrid search + rerank using description as semantic_query."""
    spec = QuerySpec(
        semantic_query=description,
        raw_text=description,
        top_k=max(1, min(int(top_k), 50)),
    )
    outcome = search(spec)
    candidates = outcome.candidates
    query_for_rerank = rerank_query_text(spec, description)

    rerank_top_n = resolve_rerank_top_n(spec, spec.top_k)
    results = rerank(
        query_for_rerank,
        candidates,
        top_k=spec.top_k,
        top_n=rerank_top_n,
        min_match_percent=min_match_percent,
        spec=spec,
    )
    if not results and candidates and min_match_percent > 0:
        results = rerank(
            query_for_rerank,
            candidates,
            top_k=spec.top_k,
            top_n=rerank_top_n,
            min_match_percent=0,
            spec=spec,
        )

    from imagecb.retrieval.sort import resolve_sort, sort_ranked_results

    resolved_sort = resolve_sort(sort, is_search=True)
    results = sort_ranked_results(results, resolved_sort)

    cards = build_result_cards(results, image_url_prefix=image_url_prefix)
    return cards, results


def result_cards_to_dicts(cards: List[ResultCard]) -> List[dict]:
    """Serialize result cards for disk cache (API layer rehydrates)."""
    out: List[dict] = []
    for c in cards:
        prov = c.provenance
        out.append(
            {
                "rank": c.rank,
                "image_id": c.image_id,
                "image_url": c.image_url,
                "thumb_url": c.thumb_url or f"/api/images/{c.image_id}/thumb",
                "provenance": {
                    "source_name": prov.source_name,
                    "source_type": prov.source_type,
                    "slide_index": prov.slide_index,
                    "page_index": prov.page_index,
                    "modified": prov.modified,
                    "author": prov.author,
                    "chips": prov.chips(),
                },
                "caption": c.caption,
                "match_hint": c.match_hint,
                "match_percent": c.match_percent,
                "has_image_file": c.has_image_file,
                "image_name": c.image_name,
                "use_case": c.use_case,
                "tags": list(c.tags),
                "recommended_cases": list(c.recommended_cases),
                "source_url": c.source_url,
                "source_location": c.source_location,
                "source_path": c.source_path,
                "created_at": c.created_at,
            }
        )
    return out
