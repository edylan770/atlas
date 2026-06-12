"""End-to-end caption generation with normalization and quality gating."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from imagecb.caption.asset_type import ASSET_TYPE_SET, normalize_asset_type
from imagecb.caption.context import caption_context_from_provenance
from imagecb.caption.normalize import normalize_tags
from imagecb.caption.quality import assess_caption
from imagecb.caption.vocab import load_tag_vocab
from imagecb.models.vlm import CaptionJSON, VLMCaptioner

if TYPE_CHECKING:
    from imagecb.extractors.types import ExtractedImage

_MAX_ALIASES = 8
_MAX_RECOMMENDED_CASES = 5


def _dedupe_lower(items: List[str], *, limit: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        s = (item or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _clean_search_terms(caption: CaptionJSON) -> CaptionJSON:
    """Dedupe aliases; dedupe recommended_cases and drop bare asset-type phrases."""
    caption.search.aliases = _dedupe_lower(caption.search.aliases, limit=_MAX_ALIASES)
    cases = _dedupe_lower(caption.search.recommended_cases, limit=_MAX_RECOMMENDED_CASES)
    caption.search.recommended_cases = [c for c in cases if c not in ASSET_TYPE_SET]
    return caption


def enrich_caption_search_terms(caption: CaptionJSON) -> CaptionJSON:
    """Public helper for repair-search-terms (no VLM re-call)."""
    return _clean_search_terms(caption)


def generate_caption(
    extracted: "ExtractedImage",
    captioner: VLMCaptioner,
    *,
    max_side: int,
    vocab: Optional[Set[str]] = None,
) -> CaptionJSON:
    """Generate caption with context, tag normalization, quality gate, and one retry."""
    if vocab is None:
        vocab = set(load_tag_vocab())

    ctx = caption_context_from_provenance(extracted.provenance)
    source_file = extracted.provenance.source_file

    def _run() -> CaptionJSON:
        cap = captioner.caption_image(
            extracted.image,
            max_side=max_side,
            context=ctx,
            source_file=source_file,
        )
        if cap.short_caption == "[caption failed]":
            return cap
        cap.grounded.asset_type = normalize_asset_type(cap.grounded.asset_type)
        cap.search.tags = normalize_tags(list(cap.search.tags), vocab)
        cap = _clean_search_terms(cap)
        quality = assess_caption(cap)
        cap.caption_quality = quality
        return cap

    caption = _run()
    if caption.caption_quality in ("weak", "failed"):
        retry = _run()
        if retry.caption_quality == "ok" or (
            retry.caption_quality == "weak" and caption.caption_quality == "failed"
        ):
            caption = retry
        elif retry.caption_quality == "weak" and caption.caption_quality == "weak":
            caption = retry

    return caption


def refresh_vocab_cache() -> Set[str]:
    """Rebuild in-memory tag vocab after an ingest batch."""
    from imagecb.caption import vocab as vocab_mod

    vocab_mod._vocab_cache = None  # noqa: SLF001
    return set(load_tag_vocab(refresh=True))
