"""Image utilities for ingest and model input."""

from __future__ import annotations

import io

from PIL import Image

from imagecb.config import SETTINGS


def _to_rgb(image: Image.Image) -> Image.Image:
    """RGB with transparency composited onto white, not black."""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base
    return image.convert("RGB")


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest edge is at most ``max_side`` (RGB)."""
    img = _to_rgb(image)
    if max_side <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    # Extreme aspect ratios (e.g. 2000x3 divider strips) must not round to 0.
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
    )


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
