"""Build short text context for Titan multimodal embeddings at ingest."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from imagecb.config import SETTINGS

if TYPE_CHECKING:
    from imagecb.extractors.types import Provenance
    from imagecb.models.vlm import CaptionJSON
    from imagecb.storage.metadata_db import ImageRecord


def _max_context_chars() -> int:
    return max(200, SETTINGS.embed_context_max_chars)


def embed_context_text(
    *,
    slide_title: Optional[str] = None,
    slide_notes: Optional[str] = None,
    page_text: Optional[str] = None,
    source_type: Optional[str] = None,
    asset_type: Optional[str] = None,
    theme: Optional[str] = None,
    short_caption: Optional[str] = None,
    use_case: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Join slide/PDF + interpretive caption context for embedding."""
    from imagecb.caption.asset_type import format_asset_type_label

    parts: list[str] = []
    asset_label = format_asset_type_label(asset_type)
    if asset_label:
        parts.append(f"asset_type: {asset_label}")
    has_interpretive = bool(
        (theme and theme.strip()) or (short_caption and short_caption.strip())
    )
    if theme and theme.strip():
        parts.append(theme.strip())
    if short_caption and short_caption.strip():
        parts.append(short_caption.strip())
    if use_case and use_case.strip():
        parts.append(use_case.strip())
    if tags:
        tag_line = ", ".join(t.strip() for t in tags[:8] if t and t.strip())
        if tag_line:
            parts.append(f"tags: {tag_line}")
    if slide_title and slide_title.strip():
        parts.append(slide_title.strip())
    if slide_notes and slide_notes.strip() and not has_interpretive:
        parts.append(slide_notes.strip())
    if page_text and page_text.strip() and len(parts) < 3:
        first_line = page_text.strip().splitlines()[0].strip()
        if first_line:
            parts.append(first_line)
    if not parts:
        return ""
    text = " | ".join(parts)
    max_chars = _max_context_chars()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def embed_context_from_provenance(provenance: "Provenance") -> str:
    page_text = None
    if provenance.extra:
        page_text = provenance.extra.get("page_text")
    return embed_context_text(
        slide_title=provenance.slide_title,
        slide_notes=provenance.slide_notes,
        page_text=page_text,
        source_type=provenance.source_type,
    )


def embed_context_from_caption_and_provenance(
    caption: "CaptionJSON",
    provenance: "Provenance",
) -> str:
    page_text = None
    if provenance.extra:
        page_text = provenance.extra.get("page_text")
    return embed_context_text(
        slide_title=provenance.slide_title,
        slide_notes=provenance.slide_notes,
        page_text=page_text,
        source_type=provenance.source_type,
        asset_type=caption.asset_type,
        theme=caption.theme,
        short_caption=caption.short_caption,
        use_case=caption.use_case,
        tags=list(caption.tags or []),
    )


def embed_context_from_record(record: "ImageRecord") -> str:
    from imagecb.storage.metadata_db import deserialize_list

    return embed_context_text(
        slide_title=record.slide_title,
        slide_notes=record.slide_notes,
        source_type=record.source_type,
        asset_type=record.asset_type,
        theme=record.theme,
        short_caption=record.caption_short,
        use_case=record.use_case,
        tags=deserialize_list(record.tags_json),
    )
