"""Build surrounding-context text blocks for VLM caption prompts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from imagecb.extractors.types import Provenance
    from imagecb.storage.metadata_db import ImageRecord

_MAX_CONTEXT_CHARS = 2500


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def slide_body_from_provenance(provenance: "Provenance") -> str:
    extra = provenance.extra or {}
    if provenance.source_type == "pptx":
        return (extra.get("slide_body_text") or "").strip()
    if provenance.source_type == "pdf":
        return (extra.get("nearby_text") or "").strip()
    return ""


def _build_sections(
    *,
    source_file: Optional[str],
    author: Optional[str],
    slide_index: Optional[int],
    page_index: Optional[int],
    slide_title: Optional[str],
    slide_notes: Optional[str],
    nearby_text: Optional[str],
) -> list[tuple[str, str, int]]:
    """Return (label, text, priority) sections; lower priority = kept when truncating."""
    sections: list[tuple[str, str, int]] = []
    if slide_title and slide_title.strip():
        sections.append(("title", slide_title.strip(), 1))
    if slide_notes and slide_notes.strip():
        sections.append(("notes", slide_notes.strip(), 2))
    if nearby_text and nearby_text.strip():
        sections.append(("nearby_text", nearby_text.strip(), 3))
    if source_file:
        sections.append(("filename", Path(source_file).name, 4))
    if author and author.strip():
        sections.append(("author", author.strip(), 5))
    if slide_index is not None:
        sections.append(("slide_index", str(slide_index), 6))
    if page_index is not None:
        sections.append(("page_index", str(page_index), 6))
    return sections


def _format_context(sections: list[tuple[str, str, int]]) -> str:
    if not sections:
        return ""
    lines = ["Surrounding context (use only to disambiguate; do not invent visible facts):"]
    for label, text, _prio in sorted(sections, key=lambda x: x[2]):
        lines.append(f"- {label}: {text}")
    full = "\n".join(lines)
    if len(full) <= _MAX_CONTEXT_CHARS:
        return full
    # Drop lowest-priority sections until we fit.
    ordered = sorted(sections, key=lambda x: -x[2])
    kept = list(sections)
    while len(_format_context_raw(kept)) > _MAX_CONTEXT_CHARS and ordered:
        drop = ordered.pop(0)
        kept = [s for s in kept if s[0] != drop[0] or s[1] != drop[1]]
    return _format_context_raw(kept)


def _format_context_raw(sections: list[tuple[str, str, int]]) -> str:
    if not sections:
        return ""
    lines = ["Surrounding context (use only to disambiguate; do not invent visible facts):"]
    for label, text, _prio in sorted(sections, key=lambda x: x[2]):
        lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def caption_context_from_provenance(provenance: "Provenance") -> str:
    nearby = slide_body_from_provenance(provenance)
    sections = _build_sections(
        source_file=provenance.source_file,
        author=provenance.author,
        slide_index=provenance.slide_index,
        page_index=provenance.page_index,
        slide_title=provenance.slide_title,
        slide_notes=provenance.slide_notes,
        nearby_text=nearby or None,
    )
    return _format_context(sections)


def caption_context_from_record(record: "ImageRecord") -> str:
    sections = _build_sections(
        source_file=record.source_file,
        author=record.author,
        slide_index=record.slide_index,
        page_index=record.page_index,
        slide_title=record.slide_title,
        slide_notes=record.slide_notes,
        nearby_text=(record.slide_body_text or "").strip() or None,
    )
    return _format_context(sections)
