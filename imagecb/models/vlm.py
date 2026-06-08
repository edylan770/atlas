"""Vision-language model wrapper that produces a structured caption.

The captioner returns a fixed schema regardless of provider so the rest
of the pipeline can rely on it. We ask for strict JSON; if a provider
returns malformed JSON we recover gracefully and store a stub so the
ingest run never aborts on a single bad image.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.images import resize_for_model


CAPTION_SYSTEM_PROMPT = (
    "You are an image-cataloging assistant. For each image, return a STRICT JSON "
    "object describing it for downstream retrieval. Be concrete, mention visible "
    "objects, scene, and any readable text. Do not include markdown fences."
)

CAPTION_USER_PROMPT = (
    "Describe this image and return JSON with EXACTLY these keys:\n"
    "- image_name: short human-friendly title for this image (<= 8 words)\n"
    "- short_caption: <= 20 words, single sentence\n"
    "- detailed_description: 1-3 sentences\n"
    "- use_case: one sentence describing the most likely business or creative use case\n"
    "- objects: list of salient object/entity names\n"
    "- scene: short phrase for the overall scene/setting\n"
    "- text_overlay_summary: any text visible in the image, or empty string\n"
    "- tags: list of 3-10 lowercase keywords useful for search\n"
    "- recommended_cases: list of 2-4 example natural-language search queries a user "
    "might type to find this image later"
)

QUERY_SYSTEM_PROMPT = (
    "You are a visual search assistant. Given a reference image, produce a STRICT JSON "
    "object optimized for finding similar images in a corpus — not for cataloging. "
    "Focus on what a user would want to match when searching. Do not include markdown fences."
)

QUERY_USER_PROMPT = (
    "Analyze this image for visual search and return JSON with EXACTLY these keys:\n"
    "- search_query: one natural-language query (1-2 sentences) to find similar images\n"
    "- subject: short phrase for the main subject or scene focus\n"
    "- style: short phrase for visual style (e.g. flat illustration, photo, diagram)\n"
    "- layout: short phrase for composition/layout (e.g. centered hero, grid, split panel)\n"
    "- salient_objects: list of prominent visible objects or entities\n"
    "- visible_text: any readable text in the image, or empty string\n"
    "- colors_mood: dominant colors and overall mood/atmosphere"
)


@dataclass
class CaptionJSON:
    image_name: str = ""
    short_caption: str = ""
    detailed_description: str = ""
    use_case: str = ""
    objects: List[str] = field(default_factory=list)
    scene: str = ""
    text_overlay_summary: str = ""
    tags: List[str] = field(default_factory=list)
    recommended_cases: List[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "CaptionJSON":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "CaptionJSON":
        def _slist(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            if isinstance(v, str) and v:
                return [v]
            return []

        return cls(
            image_name=str(d.get("image_name", "") or ""),
            short_caption=str(d.get("short_caption", "") or ""),
            detailed_description=str(d.get("detailed_description", "") or ""),
            use_case=str(d.get("use_case", "") or ""),
            objects=_slist(d.get("objects")),
            scene=str(d.get("scene", "") or ""),
            text_overlay_summary=str(d.get("text_overlay_summary", "") or ""),
            tags=_slist(d.get("tags")),
            recommended_cases=_slist(d.get("recommended_cases")),
        )


@dataclass
class ImageQueryJSON:
    search_query: str = ""
    subject: str = ""
    style: str = ""
    layout: str = ""
    salient_objects: List[str] = field(default_factory=list)
    visible_text: str = ""
    colors_mood: str = ""

    @classmethod
    def empty(cls) -> "ImageQueryJSON":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "ImageQueryJSON":
        def _slist(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            if isinstance(v, str) and v:
                return [v]
            return []

        return cls(
            search_query=str(d.get("search_query", "") or ""),
            subject=str(d.get("subject", "") or ""),
            style=str(d.get("style", "") or ""),
            layout=str(d.get("layout", "") or ""),
            salient_objects=_slist(d.get("salient_objects")),
            visible_text=str(d.get("visible_text", "") or ""),
            colors_mood=str(d.get("colors_mood", "") or ""),
        )


def _pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _parse_json_lenient(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown fences.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class VLMCaptioner:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or SETTINGS.vlm_provider).lower()
        self.model = model or SETTINGS.vlm_model

    def caption_image(self, image: Image.Image, *, max_side: Optional[int] = None) -> CaptionJSON:
        side = max_side if max_side is not None else SETTINGS.ingest_max_image_side
        if side and side > 0:
            image = resize_for_model(image, side)
        try:
            if self.provider == "bedrock":
                raw = self._caption_bedrock(image)
            elif self.provider == "openai":
                raw = self._caption_openai(image)
            elif self.provider == "anthropic":
                raw = self._caption_anthropic(image)
            else:
                raise ValueError(f"Unknown VLM provider: {self.provider}")
        except Exception as exc:  # noqa: BLE001
            return CaptionJSON(
                short_caption="[caption failed]",
                detailed_description=f"VLM error: {exc}",
            )

        data = _parse_json_lenient(raw) or {}
        return CaptionJSON.from_dict(data)

    def query_image(self, image: Image.Image, *, max_side: Optional[int] = None) -> ImageQueryJSON:
        side = max_side if max_side is not None else SETTINGS.ingest_max_image_side
        if side and side > 0:
            image = resize_for_model(image, side)
        try:
            if self.provider == "bedrock":
                raw = self._query_bedrock(image)
            elif self.provider == "openai":
                raw = self._query_openai(image)
            elif self.provider == "anthropic":
                raw = self._query_anthropic(image)
            else:
                raise ValueError(f"Unknown VLM provider: {self.provider}")
        except Exception as exc:  # noqa: BLE001
            return ImageQueryJSON(search_query=f"[query failed: {exc}]")

        data = _parse_json_lenient(raw) or {}
        return ImageQueryJSON.from_dict(data)

    def _invoke_vlm(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self.provider == "bedrock":
            return self._converse_bedrock(image, system_prompt, user_prompt)
        if self.provider == "openai":
            return self._converse_openai(image, system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._converse_anthropic(image, system_prompt, user_prompt)
        raise ValueError(f"Unknown VLM provider: {self.provider}")

    def _caption_bedrock(self, image: Image.Image) -> str:
        return self._converse_bedrock(image, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT)

    def _query_bedrock(self, image: Image.Image) -> str:
        return self._converse_bedrock(image, QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT)

    def _converse_bedrock(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        from imagecb.models.bedrock_client import get_bedrock_runtime

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        image_bytes = buf.getvalue()

        response = get_bedrock_runtime().converse(
            modelId=self.model,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                        {"text": user_prompt},
                    ],
                }
            ],
            inferenceConfig={"temperature": 0.2, "maxTokens": 1200},
        )
        parts = [
            block.get("text", "")
            for block in response["output"]["message"]["content"]
            if "text" in block
        ]
        return "".join(parts) or "{}"

    def _caption_openai(self, image: Image.Image) -> str:
        return self._converse_openai(image, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT)

    def _query_openai(self, image: Image.Image) -> str:
        return self._converse_openai(image, QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT)

    def _converse_openai(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        data_url = _pil_to_data_url(image)
        resp = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or "{}"

    def _caption_anthropic(self, image: Image.Image) -> str:
        return self._converse_anthropic(image, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT)

    def _query_anthropic(self, image: Image.Image) -> str:
        return self._converse_anthropic(image, QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT)

    def _converse_anthropic(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        msg = client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts) or "{}"


_captioner: Optional[VLMCaptioner] = None


def get_captioner() -> VLMCaptioner:
    global _captioner
    if _captioner is None:
        _captioner = VLMCaptioner()
    return _captioner
