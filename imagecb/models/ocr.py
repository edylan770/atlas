"""Tesseract-backed OCR.

Returns the empty string on any failure (missing binary, unreadable
image, etc.) so ingestion never breaks on OCR alone.
"""

from __future__ import annotations

import logging

from PIL import Image

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)

_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    if SETTINGS.tesseract_cmd:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = SETTINGS.tesseract_cmd
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set tesseract_cmd: %s", exc)
    _configured = True


def extract_text(image: Image.Image) -> str:
    _configure_once()
    try:
        import pytesseract

        text = pytesseract.image_to_string(image.convert("RGB"))
        return " ".join(text.split())
    except Exception as exc:  # noqa: BLE001
        logger.debug("OCR failed: %s", exc)
        return ""
