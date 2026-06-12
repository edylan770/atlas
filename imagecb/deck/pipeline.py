"""Orchestrate deck upload: extract -> LLM -> search -> cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from imagecb.deck import cache as deck_cache
from imagecb.deck.extract import SlideContent, deck_hash, extract_slides_from_bytes
from imagecb.deck.llm import SlideLLMOutput, describe_slides_batched, get_slide_description_llm
from imagecb.deck.search import result_cards_to_dicts, search_for_description
from imagecb.config import SETTINGS
from imagecb.storage import vector_store

logger = logging.getLogger(__name__)


@dataclass
class SlideSuggestion:
    slide_index: int
    title: Optional[str]
    body_preview: str
    notes_preview: str
    content_hash: str
    status: str
    description: str = ""
    reason: str = ""
    results: List[dict] = field(default_factory=list)
    llm_cached: bool = False
    search_cached: bool = False


@dataclass
class DeckSuggestResult:
    deck_hash: str
    filename: str
    slides: List[SlideSuggestion]
    deck_cached: bool = False
    llm_batches: int = 0


def _slide_hashes(slides: List[SlideContent]) -> List[str]:
    return [s.content_hash for s in slides]


def _build_suggestion_from_manifest_entry(
    entry: dict,
    *,
    llm_cached: bool = True,
    search_cached: bool | None = None,
) -> SlideSuggestion:
    return SlideSuggestion(
        slide_index=int(entry.get("slide_index", 0)),
        title=entry.get("title"),
        body_preview=str(entry.get("body_preview", "")),
        notes_preview=str(entry.get("notes_preview", "")),
        content_hash=str(entry.get("content_hash", "")),
        status=str(entry.get("status", "no_image_needed")),
        description=str(entry.get("description", "") or ""),
        reason=str(entry.get("reason", "") or ""),
        results=list(entry.get("results") or []),
        llm_cached=llm_cached,
        search_cached=(
            bool(entry.get("search_cached"))
            if search_cached is None
            else search_cached
        ),
    )


def _slide_stub_from_manifest_entry(entry: dict) -> SlideContent:
    return SlideContent(
        slide_index=int(entry.get("slide_index", 0)),
        title=entry.get("title"),
        body=str(entry.get("body") or entry.get("body_preview") or ""),
        notes=entry.get("notes") or entry.get("notes_preview"),
        content_hash=str(entry.get("content_hash", "")),
    )


def _refresh_suggestions_from_manifest(
    manifest_entries: List[dict],
    *,
    top_k: int,
    min_match_percent: int,
    sort: str = "relevance",
) -> List[SlideSuggestion]:
    """Re-run search for manifest slides when request params or corpus changed."""
    suggestions: List[SlideSuggestion] = []
    for entry in manifest_entries:
        llm_out = SlideLLMOutput(
            slide_index=int(entry.get("slide_index", 0)),
            status=str(entry.get("status", "no_image_needed")),
            description=str(entry.get("description", "") or ""),
            reason=str(entry.get("reason", "") or ""),
        )
        results: List[dict] = []
        search_cached = False
        if llm_out.status == "image_needed" and llm_out.description:
            slide_stub = _slide_stub_from_manifest_entry(entry)
            results, search_cached = _run_search_for_slide(
                slide_stub,
                llm_out,
                top_k=top_k,
                min_match_percent=min_match_percent,
                sort=sort,
            )
        suggestions.append(
            _build_suggestion_from_manifest_entry(
                {
                    **entry,
                    "results": results,
                    "search_cached": search_cached,
                },
                llm_cached=True,
                search_cached=search_cached,
            )
        )
    return suggestions


def _manifest_entries_from_slides(
    suggestions: List[SlideSuggestion],
    *,
    source_slides: Optional[List[SlideContent]] = None,
) -> List[dict]:
    by_index = {s.slide_index: s for s in (source_slides or [])}
    entries: List[dict] = []
    for s in suggestions:
        src = by_index.get(s.slide_index)
        entry = {
            "slide_index": s.slide_index,
            "title": s.title,
            "body_preview": s.body_preview,
            "notes_preview": s.notes_preview,
            "body": src.body if src else s.body_preview,
            "notes": (src.notes or "") if src else s.notes_preview,
            "content_hash": s.content_hash,
            "status": s.status,
            "description": s.description,
            "reason": s.reason,
            "results": s.results,
            "search_cached": s.search_cached,
        }
        entries.append(entry)
    return entries


def _resolve_llm_for_slide(
    slide: SlideContent,
    llm_by_index: dict[int, SlideLLMOutput],
) -> tuple[SlideLLMOutput, bool]:
    cached = deck_cache.get_slide_llm_cache(slide.content_hash)
    if cached is not None:
        return deck_cache.llm_output_from_cache(cached), True
    if slide.slide_index in llm_by_index:
        out = llm_by_index[slide.slide_index]
        deck_cache.put_slide_llm_cache(slide.content_hash, slide.slide_index, out)
        return out, False
    return (
        SlideLLMOutput(
            slide_index=slide.slide_index,
            status="no_image_needed",
            reason="No LLM output available.",
        ),
        False,
    )


def _run_search_for_slide(
    slide: SlideContent,
    llm_out: SlideLLMOutput,
    *,
    top_k: int,
    min_match_percent: int,
    sort: str = "relevance",
) -> tuple[List[dict], bool]:
    if llm_out.status != "image_needed" or not llm_out.description:
        return [], False

    search_fp = deck_cache.search_fingerprint(
        llm_out.description,
        top_k=top_k,
        min_match_percent=min_match_percent,
        sort=sort,
    )
    cached_results = deck_cache.get_slide_search_cache(slide.content_hash, search_fp)
    if cached_results is not None:
        return cached_results, True

    cards, _ranked = search_for_description(
        llm_out.description,
        top_k=top_k,
        min_match_percent=min_match_percent,
        sort=sort,
    )
    result_dicts = result_cards_to_dicts(cards)
    deck_cache.put_slide_search_cache(
        slide.content_hash,
        slide.slide_index,
        status=llm_out.status,
        description=llm_out.description,
        reason=llm_out.reason,
        search_fp=search_fp,
        results=result_dicts,
    )
    return result_dicts, False


def process_deck_upload(
    data: bytes,
    filename: str,
    *,
    top_k: int = 10,
    min_match_percent: int = 0,
    sort: Optional[str] = None,
) -> DeckSuggestResult:
    from imagecb.retrieval.sort import resolve_sort

    resolved_sort = resolve_sort(sort, is_search=True)
    indexed = vector_store.count()
    if indexed == 0:
        raise ValueError(
            "Image corpus is empty. Add images via the Corpus panel before using deck suggest."
        )

    if len(data) > SETTINGS.deck_max_upload_bytes:
        raise ValueError(
            f"File exceeds maximum size ({SETTINGS.deck_max_upload_bytes // (1024 * 1024)} MB)"
        )

    slides = extract_slides_from_bytes(data)
    hashes = _slide_hashes(slides)
    d_hash = deck_hash(hashes)

    req_fp = deck_cache.request_fingerprint(
        top_k=top_k,
        min_match_percent=min_match_percent,
        sort=resolved_sort,
    )
    manifest = deck_cache.get_deck_manifest(d_hash)
    if manifest is not None and manifest.slides:
        if manifest.request_fingerprint == req_fp:
            return DeckSuggestResult(
                deck_hash=d_hash,
                filename=filename,
                slides=[
                    _build_suggestion_from_manifest_entry(e) for e in manifest.slides
                ],
                deck_cached=True,
                llm_batches=0,
            )
        suggestions = _refresh_suggestions_from_manifest(
            manifest.slides,
            top_k=top_k,
            min_match_percent=min_match_percent,
            sort=resolved_sort,
        )
        refreshed_entries = _manifest_entries_from_slides(suggestions)
        orig_by_index = {int(e.get("slide_index", 0)): e for e in manifest.slides}
        for entry in refreshed_entries:
            orig = orig_by_index.get(int(entry["slide_index"]))
            if orig:
                if orig.get("body"):
                    entry["body"] = orig["body"]
                if orig.get("notes") is not None:
                    entry["notes"] = orig["notes"]
        deck_cache.put_deck_manifest(
            d_hash,
            filename,
            hashes,
            refreshed_entries,
            request_fingerprint=req_fp,
        )
        return DeckSuggestResult(
            deck_hash=d_hash,
            filename=filename,
            slides=suggestions,
            deck_cached=False,
            llm_batches=0,
        )

    llm_by_index: dict[int, SlideLLMOutput] = {}
    to_llm: List[dict] = []
    llm_batches = 0

    for slide in slides:
        cached = deck_cache.get_slide_llm_cache(slide.content_hash)
        if cached is not None:
            llm_by_index[slide.slide_index] = deck_cache.llm_output_from_cache(cached)
        else:
            to_llm.append(slide.for_llm())

    if to_llm:
        batch_size = max(1, SETTINGS.deck_llm_batch_size)
        llm_batches = (len(to_llm) + batch_size - 1) // batch_size
        for out in describe_slides_batched(to_llm):
            llm_by_index[out.slide_index] = out
            for s in slides:
                if s.slide_index == out.slide_index:
                    deck_cache.put_slide_llm_cache(s.content_hash, s.slide_index, out)
                    break

    suggestions: List[SlideSuggestion] = []
    for slide in slides:
        body_prev, notes_prev = slide.preview()
        llm_out, llm_cached = _resolve_llm_for_slide(slide, llm_by_index)
        results: List[dict] = []
        search_cached = False
        if llm_out.status == "image_needed" and llm_out.description:
            results, search_cached = _run_search_for_slide(
                slide,
                llm_out,
                top_k=top_k,
                min_match_percent=min_match_percent,
                sort=resolved_sort,
            )

        suggestions.append(
            SlideSuggestion(
                slide_index=slide.slide_index,
                title=slide.title,
                body_preview=body_prev,
                notes_preview=notes_prev,
                content_hash=slide.content_hash,
                status=llm_out.status,
                description=llm_out.description,
                reason=llm_out.reason,
                results=results,
                llm_cached=llm_cached,
                search_cached=search_cached,
            )
        )

    deck_cache.put_deck_manifest(
        d_hash,
        filename,
        hashes,
        _manifest_entries_from_slides(suggestions, source_slides=slides),
        request_fingerprint=req_fp,
    )

    return DeckSuggestResult(
        deck_hash=d_hash,
        filename=filename,
        slides=suggestions,
        deck_cached=False,
        llm_batches=llm_batches,
    )


def force_slide_image(
    deck_hash_value: str,
    slide_index: int,
    *,
    top_k: int = 10,
    min_match_percent: int = 0,
    sort: Optional[str] = None,
) -> SlideSuggestion:
    """Force image suggestion for one slide (re-LLM + search)."""
    from imagecb.retrieval.sort import resolve_sort

    resolved_sort = resolve_sort(sort, is_search=True)
    manifest = deck_cache.get_deck_manifest(deck_hash_value)
    if manifest is None:
        raise ValueError("Deck not found in cache; re-upload the presentation.")

    entry = next(
        (e for e in manifest.slides if int(e.get("slide_index", 0)) == slide_index),
        None,
    )
    if entry is None:
        raise ValueError(f"Slide {slide_index} not found in cached deck.")

    content_hash = str(entry.get("content_hash", ""))
    llm_payload = {
        "slide_index": slide_index,
        "title": entry.get("title") or "",
        "body": entry.get("body") or entry.get("body_preview") or "",
        "notes": entry.get("notes") or entry.get("notes_preview") or "",
    }
    llm_out = get_slide_description_llm().describe_force(llm_payload)
    deck_cache.put_slide_llm_cache(content_hash, slide_index, llm_out)

    slide_stub = SlideContent(
        slide_index=slide_index,
        title=entry.get("title"),
        body=str(entry.get("body") or entry.get("body_preview") or ""),
        notes=entry.get("notes") or entry.get("notes_preview"),
        content_hash=content_hash,
    )
    results, search_cached = _run_search_for_slide(
        slide_stub,
        llm_out,
        top_k=top_k,
        min_match_percent=min_match_percent,
        sort=resolved_sort,
    )

    suggestion = SlideSuggestion(
        slide_index=slide_index,
        title=entry.get("title"),
        body_preview=str(entry.get("body_preview", "")),
        notes_preview=str(entry.get("notes_preview", "")),
        content_hash=content_hash,
        status=llm_out.status,
        description=llm_out.description,
        reason=llm_out.reason,
        results=results,
        llm_cached=False,
        search_cached=search_cached,
    )

    updated_slides = []
    for e in manifest.slides:
        if int(e.get("slide_index", 0)) == slide_index:
            updated_slides.append(_manifest_entries_from_slides([suggestion])[0])
        else:
            updated_slides.append(e)

    deck_cache.put_deck_manifest(
        deck_hash_value,
        manifest.filename,
        manifest.slide_hashes,
        updated_slides,
        request_fingerprint=deck_cache.request_fingerprint(
            top_k=top_k,
            min_match_percent=min_match_percent,
            sort=resolved_sort,
        ),
    )

    return suggestion
