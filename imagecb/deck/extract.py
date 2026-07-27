"""Extract per-slide text from uploaded PowerPoint decks."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from imagecb.config import SETTINGS
from imagecb.extractors.pptx_text import slide_text_parts

logger = logging.getLogger(__name__)

_TRUNC_MARKER = "\n[...truncated]"


@dataclass(frozen=True)
class SlideContent:
    slide_index: int
    title: Optional[str]
    body: str
    notes: Optional[str]
    content_hash: str

    def normalized_for_hash(self) -> str:
        parts = [
            (self.title or "").strip(),
            (self.body or "").strip(),
            (self.notes or "").strip(),
        ]
        return "\n".join(parts)

    def preview(self, *, max_len: int = 400) -> tuple[str, str]:
        body_prev = _truncate_preview(self.body, max_len=max_len)
        notes_prev = _truncate_preview(self.notes or "", max_len=max_len)
        return body_prev, notes_prev

    def for_llm(self) -> dict:
        """Payload slice sent to the description LLM (may truncate)."""
        max_chars = SETTINGS.deck_max_chars_per_slide
        return {
            "slide_index": self.slide_index,
            "title": self.title or "",
            "body": _truncate_field(self.body, max_chars),
            "notes": _truncate_field(self.notes or "", max_chars),
        }


def normalize_text_for_hash(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\r\n?", "\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slide_content_hash(title: Optional[str], body: str, notes: Optional[str]) -> str:
    normalized = normalize_text_for_hash(
        "\n".join([(title or "").strip(), (body or "").strip(), (notes or "").strip()])
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deck_hash(slide_hashes: List[str]) -> str:
    raw = "|".join(slide_hashes)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _truncate_field(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(_TRUNC_MARKER)] + _TRUNC_MARKER


def _truncate_preview(text: str, *, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def extract_slides_from_bytes(data: bytes) -> List[SlideContent]:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-pptx is required for deck extraction") from exc

    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid PowerPoint file: {exc}") from exc

    slides: List[SlideContent] = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        title, body, notes = slide_text_parts(slide)
        body = body or ""
        ch = slide_content_hash(title, body, notes)
        slides.append(
            SlideContent(
                slide_index=slide_idx,
                title=title,
                body=body,
                notes=notes,
                content_hash=ch,
            )
        )

    max_slides = SETTINGS.deck_max_slides
    if len(slides) > max_slides:
        raise ValueError(f"Deck has {len(slides)} slides; maximum is {max_slides}")

    if not slides:
        raise ValueError("PowerPoint file contains no slides")

    return slides


