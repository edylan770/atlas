"""Disk cache for deck LLM outputs and per-slide search results."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.deck.llm import SlideLLMOutput
from imagecb.suggestions.corpus_summary import build_corpus_context

logger = logging.getLogger(__name__)


@dataclass
class CachedSlideLLM:
    slide_index: int
    content_hash: str
    status: str
    description: str = ""
    reason: str = ""
    cached: bool = True


@dataclass
class CachedSlideSearch:
    slide_index: int
    content_hash: str
    description: str
    search_fingerprint: str
    results: List[dict] = field(default_factory=list)
    cached: bool = True


@dataclass
class DeckManifest:
    deck_hash: str
    filename: str
    slide_hashes: List[str]
    slides: List[dict]
    request_fingerprint: str = ""
    created_at: float = field(default_factory=time.time)


def _slide_cache_path(content_hash: str) -> Path:
    return SETTINGS.deck_cache_dir / "slides" / f"{content_hash}.json"


def _deck_cache_path(deck_hash: str) -> Path:
    return SETTINGS.deck_cache_dir / "decks" / f"{deck_hash}.json"


def corpus_fingerprint() -> str:
    return build_corpus_context().fingerprint


def request_fingerprint(*, top_k: int, min_match_percent: int, sort: str = "relevance") -> str:
    """Fingerprint for deck-level manifest cache (search params + corpus)."""
    raw = json.dumps(
        {
            "top_k": top_k,
            "min_match_percent": min_match_percent,
            "sort": sort,
            "corpus": corpus_fingerprint(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def search_fingerprint(
    description: str,
    *,
    top_k: int,
    min_match_percent: int,
    sort: str = "relevance",
) -> str:
    raw = json.dumps(
        {
            "description": description.strip(),
            "top_k": top_k,
            "min_match_percent": min_match_percent,
            "sort": sort,
            "corpus": corpus_fingerprint(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read cache %s: %s", path, exc)
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")


def get_slide_llm_cache(content_hash: str) -> Optional[CachedSlideLLM]:
    if not SETTINGS.deck_cache_enabled:
        return None
    data = _read_json(_slide_cache_path(content_hash))
    if not data or data.get("content_hash") != content_hash:
        return None
    return CachedSlideLLM(
        slide_index=int(data.get("slide_index", 0)),
        content_hash=content_hash,
        status=str(data.get("status", "no_image_needed")),
        description=str(data.get("description", "") or ""),
        reason=str(data.get("reason", "") or ""),
        cached=True,
    )


def put_slide_llm_cache(
    content_hash: str,
    slide_index: int,
    output: SlideLLMOutput,
) -> None:
    if not SETTINGS.deck_cache_enabled:
        return
    _write_json(
        _slide_cache_path(content_hash),
        {
            "slide_index": slide_index,
            "content_hash": content_hash,
            "status": output.status,
            "description": output.description,
            "reason": output.reason,
            "updated_at": time.time(),
        },
    )


def get_slide_search_cache(
    content_hash: str,
    search_fp: str,
) -> Optional[List[dict]]:
    if not SETTINGS.deck_cache_enabled:
        return None
    data = _read_json(_slide_cache_path(content_hash))
    if not data:
        return None
    search_block = data.get("search")
    if not isinstance(search_block, dict):
        return None
    if search_block.get("fingerprint") != search_fp:
        return None
    results = search_block.get("results")
    if isinstance(results, list):
        return results
    return None


def put_slide_search_cache(
    content_hash: str,
    slide_index: int,
    *,
    status: str,
    description: str,
    reason: str,
    search_fp: str,
    results: List[dict],
) -> None:
    if not SETTINGS.deck_cache_enabled:
        return
    path = _slide_cache_path(content_hash)
    existing = _read_json(path) or {
        "slide_index": slide_index,
        "content_hash": content_hash,
    }
    existing.update(
        {
            "slide_index": slide_index,
            "content_hash": content_hash,
            "status": status,
            "description": description,
            "reason": reason,
            "updated_at": time.time(),
            "search": {
                "fingerprint": search_fp,
                "results": results,
                "updated_at": time.time(),
            },
        }
    )
    _write_json(path, existing)


def get_deck_manifest(deck_hash: str) -> Optional[DeckManifest]:
    if not SETTINGS.deck_cache_enabled:
        return None
    data = _read_json(_deck_cache_path(deck_hash))
    if not data or data.get("deck_hash") != deck_hash:
        return None
    return DeckManifest(
        deck_hash=deck_hash,
        filename=str(data.get("filename", "")),
        slide_hashes=list(data.get("slide_hashes") or []),
        slides=list(data.get("slides") or []),
        request_fingerprint=str(data.get("request_fingerprint", "") or ""),
        created_at=float(data.get("created_at", 0)),
    )


def put_deck_manifest(
    deck_hash: str,
    filename: str,
    slide_hashes: Sequence[str],
    slides: List[dict],
    *,
    request_fingerprint: str = "",
) -> None:
    if not SETTINGS.deck_cache_enabled:
        return
    _write_json(
        _deck_cache_path(deck_hash),
        {
            "deck_hash": deck_hash,
            "filename": filename,
            "slide_hashes": list(slide_hashes),
            "slides": slides,
            "request_fingerprint": request_fingerprint,
            "created_at": time.time(),
        },
    )


def llm_output_from_cache(cached: CachedSlideLLM) -> SlideLLMOutput:
    return SlideLLMOutput(
        slide_index=cached.slide_index,
        status=cached.status,
        description=cached.description,
        reason=cached.reason,
    )
