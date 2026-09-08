"""Nano Banana 2 (Gemini) image editing and text-to-image creation."""

from __future__ import annotations

import io
import logging
import re
import time
from typing import Optional

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.models.providers import get_genai_client
from imagecb.models.secrets import get_gemini_vertex_config

logger = logging.getLogger(__name__)


class ImageEditError(RuntimeError):
    """A Gemini failure with a stable, safe error code for the API/UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _display_enum(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _short_text(value: object, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _response_diagnostic(response: object, parts: list[object]) -> str:
    details: list[str] = []
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = _display_enum(getattr(feedback, "block_reason", None))
    if block_reason:
        details.append(f"prompt blocked: {block_reason}")

    finish_reasons: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        reason = _display_enum(getattr(candidate, "finish_reason", None))
        if reason and reason not in finish_reasons:
            finish_reasons.append(reason)
    if finish_reasons:
        details.append(f"finish reason: {', '.join(finish_reasons)}")

    model_text = next(
        (
            _short_text(getattr(part, "text", None))
            for part in parts
            if getattr(part, "text", None)
        ),
        "",
    )
    if model_text:
        details.append(f"model response: {model_text}")
    return "; ".join(details)


def _provider_error(exc: BaseException) -> ImageEditError:
    text = str(exc).lower()
    if "401" in text or "unauth" in text or "api key" in text:
        return ImageEditError(
            "authentication_failed",
            "Gemini rejected the API key. Rotate or verify the Vertex Express key.",
        )
    if "403" in text or "permission" in text or "forbidden" in text:
        return ImageEditError(
            "permission_denied",
            "Gemini denied access. Verify the key's Vertex AI and image-model access.",
        )
    if "404" in text or "not found" in text or "unsupported model" in text:
        return ImageEditError(
            "model_unavailable",
            "The configured Gemini image model is unavailable for this account.",
        )
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return ImageEditError(
            "quota_exceeded",
            "Gemini quota or rate limit was exceeded. Try again later.",
        )
    if "timeout" in text or "timed out" in text or "deadline" in text:
        return ImageEditError(
            "provider_timeout",
            "Gemini did not finish the image edit before the provider timeout.",
        )
    return ImageEditError(
        "provider_error",
        f"Gemini image generation failed ({type(exc).__name__}). Check server logs.",
    )


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


def _collect_response_parts(response: object) -> list[object]:
    parts: list[object] = []
    if getattr(response, "candidates", None):
        for cand in response.candidates or []:
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                parts.extend(content.parts)
    if not parts and getattr(response, "parts", None):
        parts = list(response.parts)
    return parts


def _png_from_response(response: object) -> bytes:
    parts = _collect_response_parts(response)
    found_inline = False
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if not data:
            continue
        found_inline = True
        try:
            if isinstance(data, str):
                import base64

                data = base64.b64decode(data, validate=True)
            # Normalize to PNG for consistent session/pending storage.
            out = Image.open(io.BytesIO(data)).convert("RGB")
            return _image_to_png_bytes(out)
        except Exception:  # noqa: BLE001
            continue

    if found_inline:
        raise ImageEditError(
            "invalid_image",
            "Gemini returned image data that could not be decoded.",
        )
    diagnostic = _response_diagnostic(response, parts)
    message = "Gemini completed the request but did not return an image."
    if diagnostic:
        message = f"{message} {diagnostic}"
    raise ImageEditError("no_image", message)


def _run_gemini_image(
    *,
    contents: list[object],
    model_id: str,
    log_event: str,
) -> bytes:
    from google.genai import types

    client = get_genai_client()
    config = get_gemini_vertex_config()
    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        result = _png_from_response(response)
        logger.info(
            "%s status=success model=%s backend=%s latency_ms=%.1f",
            log_event,
            model_id,
            config.backend,
            (time.perf_counter() - started) * 1000,
        )
        return result
    except ImageEditError as exc:
        logger.warning(
            "%s status=failure model=%s backend=%s latency_ms=%.1f error_code=%s",
            log_event,
            model_id,
            config.backend,
            (time.perf_counter() - started) * 1000,
            exc.code,
        )
        raise
    except Exception as exc:
        public_error = _provider_error(exc)
        logger.exception(
            "%s status=failure model=%s backend=%s latency_ms=%.1f error_code=%s",
            log_event,
            model_id,
            config.backend,
            (time.perf_counter() - started) * 1000,
            public_error.code,
        )
        raise public_error from exc


def generate_image(
    prompt: str,
    *,
    model: Optional[str] = None,
) -> bytes:
    """Create an image from text with Nano Banana 2; return PNG bytes."""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt is required")

    # Ensure key resolves before constructing the client (clearer errors).
    get_gemini_vertex_config()
    model_id = model or SETTINGS.nano_banana_model

    from google.genai import types

    return _run_gemini_image(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            )
        ],
        model_id=model_id,
        log_event="gemini_create",
    )


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
    get_gemini_vertex_config()

    max_side = max_side if max_side is not None else SETTINGS.ingest_max_image_side
    model_id = model or SETTINGS.nano_banana_model

    src = Image.open(io.BytesIO(image_bytes))
    src = _resize_for_edit(src.convert("RGB"), max_side)
    png_in = _image_to_png_bytes(src)

    from google.genai import types

    return _run_gemini_image(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=png_in, mime_type="image/png"),
                    types.Part.from_text(text=text),
                ],
            )
        ],
        model_id=model_id,
        log_event="gemini_edit",
    )
