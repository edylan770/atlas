"""Nano Banana 2 (Gemini) image editing."""

from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.models.providers import get_genai_client
from imagecb.models.secrets import get_gemini_api_key

logger = logging.getLogger(__name__)


def _resize_for_edit(img: Image.Image, max_side: int) -> Image.Image:
    max_side = max(64, int(max_side))
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def edit_image(
    image_bytes: bytes,
    prompt: str,
    *,
    model: Optional[str] = None,
    max_side: Optional[int] = None,
) -> bytes:
    """Edit ``image_bytes`` with Nano Banana 2; return PNG bytes of the result."""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt is required")
    if not image_bytes:
        raise ValueError("image_bytes is required")

    # Ensure key resolves before constructing the client (clearer errors).
    get_gemini_api_key()

    max_side = max_side if max_side is not None else SETTINGS.ingest_max_image_side
    model_id = model or SETTINGS.nano_banana_model

    src = Image.open(io.BytesIO(image_bytes))
    src = _resize_for_edit(src.convert("RGB"), max_side)
    png_in = _image_to_png_bytes(src)

    from google.genai import types

    client = get_genai_client()
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=png_in, mime_type="image/png"),
                    types.Part.from_text(text=text),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    parts = []
    if getattr(response, "candidates", None):
        for cand in response.candidates or []:
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                parts.extend(content.parts)
    if not parts and getattr(response, "parts", None):
        parts = list(response.parts)

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if not data:
            continue
        if isinstance(data, str):
            import base64

            data = base64.b64decode(data)
        # Normalize to PNG for consistent session/pending storage.
        out = Image.open(io.BytesIO(data)).convert("RGB")
        return _image_to_png_bytes(out)

    raise RuntimeError("Nano Banana response did not include an image")
