"""Query-time pipeline variants for the isolated Pipeline Lab.

Runs ONE text query through several ranking strategies against the existing
index so they can be compared side-by-side. A single ``parse_query`` +
``search`` produces one candidate pool (each ``Candidate`` already carries
per-lane scores), and every variant is a different ranking/rerank over that
same pool. Only the baseline / Cohere-filter variants make extra Bedrock calls.

This module reuses public core functions read-only and does not modify any
core behavior. Safe to delete with the rest of ``imagecb/experiments``.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterator, List, Sequence

from imagecb.config import SETTINGS
from imagecb.formatting.assistant_reply import build_result_cards
from imagecb.retrieval.hybrid import (
    Candidate,
    normalize_rrf_score,
    rrf_merge_lanes,
    search,
)
from imagecb.retrieval.query_build import rerank_query_text, resolve_rerank_top_n
from imagecb.retrieval.query_parser import QuerySpec, parse_query
from imagecb.retrieval.rerank import (
    RankedResult,
    _format_provenance,
    _hubness_adjuster,
    rerank,
    visual_only_rank,
)
from imagecb.retrieval.session import ChatSession
from imagecb.storage import metadata_db
from imagecb.storage.metadata_db import ImageRecord

logger = logging.getLogger(__name__)


# Ordered variant catalog: (key, label, description). The frontend renders one
# column per entry in this order.
VARIANTS: List[tuple[str, str, str]] = [
    (
        "baseline",
        "Baseline (production)",
        "3-lane RRF then Cohere rerank, with the short-query / weak-rerank visual fallbacks. This is exactly what the live chat returns.",
    ),
    (
        "visual_text",
        "Visual + caption-text RRF",
        "2-lane RRF over the visual and caption-text dense lanes only. No BM25, no Cohere rerank.",
    ),
    (
        "rrf_fusion",
        "3-lane RRF only",
        "Reciprocal Rank Fusion of all three lanes (visual, caption-text, BM25). No Cohere rerank.",
    ),
    (
        "visual_raw",
        "Titan visual (raw)",
        "Pure Titan multimodal cosine similarity, no hubness correction.",
    ),
    (
        "caption_text",
        "Caption-text dense only",
        "Titan text-embedding lane: query vs caption documents. Cosine similarity only.",
    ),
    (
        "visual_hubness",
        "Titan visual + hubness",
        "Pure Titan multimodal cosine similarity with CSLS hubness correction (demotes hub images).",
    ),
    (
        "visual_text_visual_heavy",
        "Visual+caption RRF (2x visual)",
        "2-lane RRF with dense_weight=2.0; visual lane counts double.",
    ),
    (
        "visual_text_text_heavy",
        "Visual+caption RRF (2x text)",
        "2-lane RRF with text_weight=2.0; caption-text lane counts double.",
    ),
    (
        "visual_text_hubness",
        "Visual+caption RRF + hubness",
        "2-lane RRF with CSLS hubness-adjusted visual scores before fusion.",
    ),
    (
        "visual_text_k10",
        "Visual+caption RRF k=10",
        "2-lane RRF with k=10: heavily rewards rank-1 hits, large position gradient.",
    ),
    (
        "visual_text_k30",
        "Visual+caption RRF k=30",
        "2-lane RRF with k=30: moderate position gradient. Compare with default k=60.",
    ),
    (
        "visual_text_k120",
        "Visual+caption RRF k=120",
        "2-lane RRF with k=120: very gradual decay, scores are flatter across positions.",
    ),
    (
        "cohere_filter_rrf_3lane",
        "Cohere filter then 3-lane RRF",
        "Cohere scores the pool and drops candidates below 0.25; survivors are re-fused with 3-lane RRF.",
    ),
    (
        "cohere_filter_visual_text",
        "Cohere filter then visual+caption RRF",
        "Cohere scores the pool and drops candidates below 0.25; survivors are re-fused with visual+caption RRF.",
    ),
]


def variant_catalog() -> List[Dict[str, str]]:
    return [{"key": k, "label": label, "description": desc} for k, label, desc in VARIANTS]


def _records_for(candidates: Sequence[Candidate]) -> Dict[str, ImageRecord]:
    ids = [c.image_id for c in candidates]
    return {r.image_id: r for r in metadata_db.get_records(ids)}


def _lane_hits(candidates: Sequence[Candidate], attr: str) -> List[tuple[str, float]]:
    hits = [
        (c.image_id, float(getattr(c, attr)))
        for c in candidates
        if float(getattr(c, attr)) > 0.0
    ]
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def _fuse_to_ranked_results(
    fused: Sequence[Candidate],
    records: Dict[str, ImageRecord],
    top_k: int,
    *,
    k: int,
    weight_sum: float,
) -> List[RankedResult]:
    out: List[RankedResult] = []
    for c in fused[:top_k]:
        rec = records.get(c.image_id)
        if rec is None:
            continue
        out.append(
            RankedResult(
                image_id=c.image_id,
                score=normalize_rrf_score(c.fused_score, k, weight_sum=weight_sum),
                record=rec,
                provenance_line=_format_provenance(rec),
                score_kind="fusion",
            )
        )
    return out


def _visual_text_fuse(
    candidates: Sequence[Candidate],
    records: Dict[str, ImageRecord],
    top_k: int,
    *,
    k: int = SETTINGS.rrf_k,
    dense_weight: float = 1.0,
    text_weight: float = 1.0,
    hubness: bool = False,
) -> List[RankedResult]:
    """2-lane RRF over visual + caption-text dense lanes."""
    dense_hits = _lane_hits(candidates, "dense_score")
    if hubness:
        adjust = _hubness_adjuster()
        dense_hits = [(image_id, adjust(image_id, score)) for image_id, score in dense_hits]
    text_hits = _lane_hits(candidates, "text_score")
    weight_sum = dense_weight + text_weight
    fused = rrf_merge_lanes(
        dense_hits,
        text_hits,
        [],
        k,
        dense_weight=dense_weight,
        text_weight=text_weight,
        sparse_weight=0.0,
    )
    return _fuse_to_ranked_results(fused, records, top_k, k=k, weight_sum=weight_sum)


def _rrf_3lane_fuse(
    candidates: Sequence[Candidate],
    records: Dict[str, ImageRecord],
    top_k: int,
    *,
    k: int = SETTINGS.rrf_k,
) -> List[RankedResult]:
    """3-lane RRF over visual + caption-text + BM25."""
    dense_hits = _lane_hits(candidates, "dense_score")
    text_hits = _lane_hits(candidates, "text_score")
    sparse_hits = _lane_hits(candidates, "sparse_score")
    weight_sum = 3.0 if SETTINGS.caption_text_lane_enabled else 2.0
    fused = rrf_merge_lanes(dense_hits, text_hits, sparse_hits, k)
    return _fuse_to_ranked_results(fused, records, top_k, k=k, weight_sum=weight_sum)


def _cohere_filter_candidates(
    query: str,
    spec: QuerySpec,
    candidates: Sequence[Candidate],
    top_k: int,
) -> List[Candidate]:
    """Score candidates with Cohere and keep those above weak_result_score_threshold."""
    if not candidates:
        return []
    ranked = rerank(
        rerank_query_text(spec, query),
        candidates,
        top_k=top_k,
        top_n=resolve_rerank_top_n(spec, top_k),
        min_match_percent=0,
        spec=spec,
    )
    threshold = SETTINGS.weak_result_score_threshold
    kept_ids = {r.image_id for r in ranked if r.score >= threshold}
    if not kept_ids:
        kept_ids = {r.image_id for r in ranked[:top_k]}
    by_id = {c.image_id: c for c in candidates}
    return [by_id[i] for i in (r.image_id for r in ranked) if i in kept_ids and i in by_id]


def _rank_by_attr(
    candidates: Sequence[Candidate],
    records: Dict[str, ImageRecord],
    attr: str,
    *,
    score_kind: str,
    top_k: int,
) -> List[RankedResult]:
    """Rank candidates by a single per-lane score attribute (descending)."""
    scored = [
        (c, float(getattr(c, attr)))
        for c in candidates
        if c.image_id in records and float(getattr(c, attr)) > 0.0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: List[RankedResult] = []
    for c, s in scored[:top_k]:
        rec = records[c.image_id]
        out.append(
            RankedResult(
                image_id=c.image_id,
                score=float(s),
                record=rec,
                provenance_line=_format_provenance(rec),
                score_kind=score_kind,  # type: ignore[arg-type]
            )
        )
    return out


# --- Individual variant builders -------------------------------------------------
# Each takes (query, spec, candidates, records, top_k) and returns RankedResults.


def _v_baseline(query: str, spec, candidates, records, top_k) -> List[RankedResult]:
    result = ChatSession().ask(query, top_k=top_k, min_match_percent=0)
    return list(result.results)


def _v_visual_text(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k)


def _v_rrf_fusion(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _rrf_3lane_fuse(candidates, records, top_k)


def _v_visual_raw(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _rank_by_attr(candidates, records, "dense_score", score_kind="dense", top_k=top_k)


def _v_caption_text(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _rank_by_attr(candidates, records, "text_score", score_kind="dense", top_k=top_k)


def _v_visual_hubness(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return visual_only_rank(candidates, top_k=top_k)


def _v_visual_text_visual_heavy(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, dense_weight=2.0)


def _v_visual_text_text_heavy(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, text_weight=2.0)


def _v_visual_text_hubness(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, hubness=True)


def _v_visual_text_k10(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, k=10)


def _v_visual_text_k30(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, k=30)


def _v_visual_text_k120(query, spec, candidates, records, top_k) -> List[RankedResult]:
    return _visual_text_fuse(candidates, records, top_k, k=120)


def _v_cohere_filter_rrf_3lane(query, spec, candidates, records, top_k) -> List[RankedResult]:
    filtered = _cohere_filter_candidates(query, spec, candidates, top_k)
    if not filtered:
        return []
    return _rrf_3lane_fuse(filtered, records, top_k)


def _v_cohere_filter_visual_text(query, spec, candidates, records, top_k) -> List[RankedResult]:
    filtered = _cohere_filter_candidates(query, spec, candidates, top_k)
    if not filtered:
        return []
    return _visual_text_fuse(filtered, records, top_k)


_BUILDERS: Dict[str, Callable[..., List[RankedResult]]] = {
    "baseline": _v_baseline,
    "visual_text": _v_visual_text,
    "rrf_fusion": _v_rrf_fusion,
    "visual_raw": _v_visual_raw,
    "caption_text": _v_caption_text,
    "visual_hubness": _v_visual_hubness,
    "visual_text_visual_heavy": _v_visual_text_visual_heavy,
    "visual_text_text_heavy": _v_visual_text_text_heavy,
    "visual_text_hubness": _v_visual_text_hubness,
    "visual_text_k10": _v_visual_text_k10,
    "visual_text_k30": _v_visual_text_k30,
    "visual_text_k120": _v_visual_text_k120,
    "cohere_filter_rrf_3lane": _v_cohere_filter_rrf_3lane,
    "cohere_filter_visual_text": _v_cohere_filter_visual_text,
}


def _serialize(results: Sequence[RankedResult]) -> List[Dict]:
    cards = build_result_cards(results)
    out: List[Dict] = []
    for card in cards:
        out.append(
            {
                "rank": card.rank,
                "image_id": card.image_id,
                "image_url": card.image_url,
                "match_percent": card.match_percent,
                "image_name": card.image_name or card.provenance.source_name,
                "caption": card.caption,
                "provenance": " · ".join(card.provenance.chips()),
                "has_image_file": card.has_image_file,
            }
        )
    return out


def _spec_summary(spec: QuerySpec) -> Dict:
    return {
        "semantic_query": spec.semantic_query,
        "must_have_keywords": list(spec.must_have_keywords),
        "must_avoid_keywords": list(spec.must_avoid_keywords),
        "top_k": spec.top_k,
    }


def run_comparison(query: str, top_k: int = 10) -> Dict:
    """Run every variant for one query and return serialized columns."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    top_k = max(1, min(int(top_k), 30))

    spec = parse_query(query)
    spec.top_k = top_k
    outcome = search(spec)
    candidates = outcome.candidates
    records = _records_for(candidates)

    variants_out: List[Dict] = []
    for key, label, description in VARIANTS:
        entry: Dict = {"key": key, "label": label, "description": description}
        try:
            results = _BUILDERS[key](query, spec, candidates, records, top_k)
            entry["results"] = _serialize(results)
            entry["count"] = len(entry["results"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lab variant %s failed: %s", key, exc)
            entry["results"] = []
            entry["count"] = 0
            entry["error"] = str(exc)
        variants_out.append(entry)

    return {
        "query": query,
        "top_k": top_k,
        "parsed_query": _spec_summary(spec),
        "candidate_count": len(candidates),
        "variants": variants_out,
    }


def iter_comparison(query: str, top_k: int = 10) -> Iterator[Dict]:
    """Yield events for one query: a ``meta`` event, then one ``variant`` event
    per pipeline as it finishes, then a ``done`` event.

    Parse + search run once and the candidate pool is shared across variants, so
    columns can render progressively without paying for repeated retrieval.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    top_k = max(1, min(int(top_k), 30))

    spec = parse_query(query)
    spec.top_k = top_k
    outcome = search(spec)
    candidates = outcome.candidates
    records = _records_for(candidates)

    yield {
        "type": "meta",
        "query": query,
        "top_k": top_k,
        "parsed_query": _spec_summary(spec),
        "candidate_count": len(candidates),
        "variants": variant_catalog(),
    }

    for key, label, description in VARIANTS:
        entry: Dict = {"key": key, "label": label, "description": description}
        try:
            results = _BUILDERS[key](query, spec, candidates, records, top_k)
            entry["results"] = _serialize(results)
            entry["count"] = len(entry["results"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lab variant %s failed: %s", key, exc)
            entry["results"] = []
            entry["count"] = 0
            entry["error"] = str(exc)
        yield {"type": "variant", "variant": entry}

    yield {"type": "done"}


def run_single_variant(key: str, query: str, top_k: int = 10) -> Dict:
    """Run a single variant (used for progressive/independent column loading)."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    if key not in _BUILDERS:
        raise ValueError(f"unknown variant: {key}")
    top_k = max(1, min(int(top_k), 30))

    spec = parse_query(query)
    spec.top_k = top_k
    outcome = search(spec)
    candidates = outcome.candidates
    records = _records_for(candidates)

    label, description = next((l, d) for k, l, d in VARIANTS if k == key)
    entry: Dict = {"key": key, "label": label, "description": description}
    try:
        results = _BUILDERS[key](query, spec, candidates, records, top_k)
        entry["results"] = _serialize(results)
        entry["count"] = len(entry["results"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lab variant %s failed: %s", key, exc)
        entry["results"] = []
        entry["count"] = 0
        entry["error"] = str(exc)

    return {
        "query": query,
        "top_k": top_k,
        "parsed_query": _spec_summary(spec),
        "candidate_count": len(candidates),
        "variant": entry,
    }
