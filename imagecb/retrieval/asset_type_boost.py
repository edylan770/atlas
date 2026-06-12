"""Soft rerank boost when a short query maps to the asset_type taxonomy."""

from __future__ import annotations

from typing import List, Set

from imagecb.caption.asset_type import ASSET_TYPE_SET, normalize_asset_type, resolve_query_asset_type
from imagecb.caption.normalize import tokenize_text
from imagecb.config import SETTINGS
from imagecb.retrieval.query_build import is_short_query
from imagecb.retrieval.query_parser import QuerySpec


def _corpus_asset_type_unclassified_rate() -> float:
    from imagecb.retrieval.query_parser import _corpus_asset_type_unclassified_rate as _rate

    return _rate()


def infer_asset_types_from_short_query(spec: QuerySpec) -> Set[str]:
    """Return canonical asset types implied by a short query, if any."""
    if not is_short_query(spec):
        return set()

    text = (spec.semantic_query or spec.raw_text or "").strip()
    if not text:
        return set()

    tokens = tokenize_text(text)
    if not tokens:
        return set()

    matched: Set[str] = set()
    for token in tokens:
        resolved = resolve_query_asset_type(token)
        if resolved and resolved in ASSET_TYPE_SET and resolved != "other":
            matched.add(resolved)

    if len(matched) != 1:
        return set()
    return matched


def asset_type_rerank_multiplier(spec: QuerySpec, record_asset_type: str | None) -> float:
    """Multiplier for rerank scores when query and record share one asset type."""
    if _corpus_asset_type_unclassified_rate() >= 0.5:
        return 1.0

    target_types = infer_asset_types_from_short_query(spec)
    if not target_types:
        return 1.0

    record_type = normalize_asset_type(record_asset_type)
    if record_type not in target_types:
        return 1.0

    return SETTINGS.asset_type_rerank_boost


def apply_asset_type_boost(
    spec: QuerySpec,
    ranked: List,
) -> List:
    """Re-sort ranked results after applying a soft asset-type multiplier."""
    if not ranked:
        return ranked

    multiplier = None
    for item in ranked:
        mult = asset_type_rerank_multiplier(spec, item.record.asset_type)
        if mult != 1.0:
            multiplier = mult
            break
    if multiplier is None or multiplier == 1.0:
        return ranked

    boosted = sorted(
        ranked,
        key=lambda r: r.score * asset_type_rerank_multiplier(spec, r.record.asset_type),
        reverse=True,
    )
    return boosted
