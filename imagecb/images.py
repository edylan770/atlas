"""Image utilities for ingest and model input."""

from __future__ import annotations

import io

from PIL import Image

from imagecb.config import SETTINGS


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest edge is at most ``max_side`` (RGB)."""
    if max_side <= 0:
        return image.convert("RGB")
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def make_thumbnail(
    image: Image.Image,
    *,
    max_side: int | None = None,
    quality: int | None = None,
) -> bytes:
    """Encode a small JPEG thumbnail for UI display grids."""
    side = SETTINGS.thumb_max_side if max_side is None else max_side
    q = SETTINGS.thumb_jpeg_quality if quality is None else quality
    thumb = resize_for_model(image, side)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=max(1, min(int(q), 95)), optimize=True)
    return buf.getvalue()
