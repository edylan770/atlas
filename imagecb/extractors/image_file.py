"""Extractor for standalone image files on disk."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from PIL import ExifTags, Image, ImageOps

from .types import ExtractedImage, Provenance

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _exif_datetime(img: Image.Image) -> Optional[datetime]:
    try:
        raw = img.getexif()
        if not raw:
            return None
        tag_id = next(
            (k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"),
            None,
        )
        if tag_id is None:
            return None
        value = raw.get(tag_id)
        if not value:
            return None
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return None


def _exif_artist(img: Image.Image) -> Optional[str]:
    try:
        raw = img.getexif()
        if not raw:
            return None
        for tag_id, name in ExifTags.TAGS.items():
            if name == "Artist":
                value = raw.get(tag_id)
                if value:
                    return str(value)
        return None
    except Exception:  # noqa: BLE001
        return None


def extract(path: Path) -> Iterator[ExtractedImage]:
    try:
        img = Image.open(path)
        img.load()
        # Bake EXIF orientation into the pixels: the cached PNG (which strips
        # EXIF) becomes the record of truth for captioning/embedding/thumbs.
        img = ImageOps.exif_transpose(img)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open %s: %s", path, exc)
        return
    stat = path.stat()
    provenance = Provenance(
        source_file=str(path),
        source_type="image",
        source_modified_at=datetime.fromtimestamp(stat.st_mtime),
        source_created_at=_exif_datetime(img) or datetime.fromtimestamp(stat.st_ctime),
        author=_exif_artist(img),
    )
    yield ExtractedImage(image=img, provenance=provenance)
