"""Extract images plus page context from PDFs via PyMuPDF."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from .types import ExtractedImage, Provenance

logger = logging.getLogger(__name__)

_MAX_NEARBY_CHARS = 1500


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_pdf_date(raw: Optional[str]) -> Optional[datetime]:
    """PDF date strings look like 'D:20240115093045+00'00''. Parse leniently."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Take first 14 chars (YYYYMMDDHHMMSS) if available.
    digits = s[:14]
    try:
        if len(digits) >= 14:
            return datetime.strptime(digits, "%Y%m%d%H%M%S")
        if len(digits) >= 8:
            return datetime.strptime(digits[:8], "%Y%m%d")
    except ValueError:
        return None
    return None


def _page_text(page) -> str:
    try:
        return (page.get_text("text") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _first_nonempty_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def extract(path: Path) -> Iterator[ExtractedImage]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        logger.error("PyMuPDF not installed: %s", exc)
        return

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to open pdf %s: %s", path, exc)
        return

    meta = doc.metadata or {}
    author = meta.get("author") or None
    created_at = _parse_pdf_date(meta.get("creationDate"))
    modified_at = _parse_pdf_date(meta.get("modDate")) or datetime.fromtimestamp(path.stat().st_mtime)

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_text = _page_text(page)
            page_title = _first_nonempty_line(page_text)
            try:
                images = page.get_images(full=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not enumerate images on page %s: %s", page_index + 1, exc)
                images = []
            for img_info in images:
                xref = img_info[0]
                try:
                    raw = doc.extract_image(xref)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to extract image xref %s: %s", xref, exc)
                    continue
                image_bytes = raw.get("image")
                if not image_bytes:
                    continue
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    img.load()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Pillow could not open extracted image: %s", exc)
                    continue
                nearby = _truncate(page_text, _MAX_NEARBY_CHARS) if page_text else ""
                provenance = Provenance(
                    source_file=str(path),
                    source_type="pdf",
                    source_modified_at=modified_at,
                    source_created_at=created_at,
                    author=author,
                    page_index=page_index + 1,
                    slide_title=page_title,
                    slide_notes=page_text if page_text else None,
                    extra={"nearby_text": nearby} if nearby else {},
                )
                yield ExtractedImage(image=img, provenance=provenance)
    finally:
        doc.close()
