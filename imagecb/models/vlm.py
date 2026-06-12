"""Vision-language model wrapper that produces a structured caption.

The captioner returns a fixed schema regardless of provider. Structured
output (tool calling / JSON schema) is enforced per provider; malformed
responses become a failed-caption stub so ingest never aborts.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from PIL import Image

from imagecb.caption.asset_type import (
    DEFAULT_ASSET_TYPE,
    format_taxonomy_for_prompt,
    normalize_asset_type,
)
from imagecb.caption.examples import format_few_shot_for_prompt
from imagecb.caption.schema import (
    CAPTION_JSON_SCHEMA,
    CAPTION_TOOL_NAME,
    validate_caption_dict,
)
from imagecb.caption.vocab import format_vocab_for_prompt, vocab_for_prompt
from imagecb.config import SETTINGS
from imagecb.images import resize_for_model

_CAPTION_TEMPERATURE = 0.1
_CAPTION_MAX_TOKENS = 2000

CAPTION_SYSTEM_PROMPT = (
    "You are an image-cataloging assistant for a semantic image search system. "
    "Return structured data via the required tool. Keep literally visible facts "
    "(objects, scene, readable text) separate from interpretation. Use surrounding "
    "context only to disambiguate what is visible — never invent objects, text, or "
    "scenes from context alone. Report only text you can actually read; flag "
    "uncertainty instead of guessing brands or identities. Keep every field "
    "specific to this image."
)

CAPTION_USER_PROMPT_TEMPLATE = (
    "Describe this image for retrieval.\n\n"
    "{context_block}"
    "Existing corpus tags (reuse one only when it matches this image exactly; "
    "otherwise invent a concise new tag):\n"
    "{vocab_block}\n\n"
    "Return via the emit_caption tool with:\n"
    "- image_name: short title (<= 8 words)\n"
    "- grounded: objects (visible only), scene, readable_text (legible text only), "
    "text_read_uncertain (true if text is partial/illegible), asset_type (exactly one "
    "value from the taxonomy below)\n"
    "Asset type taxonomy:\n{asset_type_block}\n"
    "- interpretive: theme, use_case, short_caption (<= 20 words), detailed_description "
    "(1-3 sentences of catalog prose)\n"
    "- search: tags (3-10 lowercase singular words), recommended_cases (3-4 specific "
    "queries a user would type to find THIS image — never a bare format word like "
    "'diagram' alone), aliases (alternate names or acronym expansions for what is shown)\n\n"
    "Match the granularity of these examples:\n"
    "{examples_block}"
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
class GroundedCaption:
    objects: List[str] = field(default_factory=list)
    scene: str = ""
    readable_text: str = ""
    text_read_uncertain: bool = False
    asset_type: str = DEFAULT_ASSET_TYPE


@dataclass
class InterpretiveCaption:
    theme: str = ""
    use_case: str = ""
    short_caption: str = ""
    detailed_description: str = ""


@dataclass
class SearchTerms:
    tags: List[str] = field(default_factory=list)
    recommended_cases: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


@dataclass
class CaptionJSON:
    image_name: str = ""
    grounded: GroundedCaption = field(default_factory=GroundedCaption)
    interpretive: InterpretiveCaption = field(default_factory=InterpretiveCaption)
    search: SearchTerms = field(default_factory=SearchTerms)
    caption_quality: str = "ok"

    # Flat accessors for backward compatibility
    @property
    def short_caption(self) -> str:
        return self.interpretive.short_caption

    @property
    def detailed_description(self) -> str:
        return self.interpretive.detailed_description

    @property
    def use_case(self) -> str:
        return self.interpretive.use_case

    @property
    def theme(self) -> str:
        return self.interpretive.theme

    @property
    def objects(self) -> List[str]:
        return self.grounded.objects

    @property
    def scene(self) -> str:
        return self.grounded.scene

    @property
    def readable_text(self) -> str:
        return self.grounded.readable_text

    @property
    def text_read_uncertain(self) -> bool:
        return self.grounded.text_read_uncertain

    @property
    def asset_type(self) -> str:
        return self.grounded.asset_type

    @property
    def text_overlay_summary(self) -> str:
        return self.grounded.readable_text

    @property
    def tags(self) -> List[str]:
        return self.search.tags

    @property
    def recommended_cases(self) -> List[str]:
        return self.search.recommended_cases

    @property
    def aliases(self) -> List[str]:
        return self.search.aliases

    @classmethod
    def empty(cls) -> "CaptionJSON":
        return cls()

    @classmethod
    def failed(cls, message: str) -> "CaptionJSON":
        return cls(
            interpretive=InterpretiveCaption(
                short_caption="[caption failed]",
                detailed_description=f"VLM error: {message}",
            ),
            caption_quality="failed",
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CaptionJSON":
        def _slist(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x is not None and str(x).strip()]
            if isinstance(v, str) and v:
                return [v]
            return []

        # New nested schema
        if "grounded" in d or "interpretive" in d or "search" in d:
            g = d.get("grounded") or {}
            i = d.get("interpretive") or {}
            s = d.get("search") or {}
            return cls(
                image_name=str(d.get("image_name", "") or ""),
                grounded=GroundedCaption(
                    objects=_slist(g.get("objects")),
                    scene=str(g.get("scene", "") or ""),
                    readable_text=str(g.get("readable_text", "") or ""),
                    text_read_uncertain=bool(g.get("text_read_uncertain", False)),
                    asset_type=normalize_asset_type(g.get("asset_type")),
                ),
                interpretive=InterpretiveCaption(
                    theme=str(i.get("theme", "") or ""),
                    use_case=str(i.get("use_case", "") or ""),
                    short_caption=str(i.get("short_caption", "") or ""),
                    detailed_description=str(i.get("detailed_description", "") or ""),
                ),
                search=SearchTerms(
                    tags=_slist(s.get("tags")),
                    recommended_cases=_slist(s.get("recommended_cases")),
                    aliases=_slist(s.get("aliases")),
                ),
                caption_quality=str(d.get("caption_quality", "ok") or "ok"),
            )

        # Legacy flat schema
        return cls(
            image_name=str(d.get("image_name", "") or ""),
            grounded=GroundedCaption(
                objects=_slist(d.get("objects")),
                scene=str(d.get("scene", "") or ""),
                readable_text=str(d.get("text_overlay_summary", "") or d.get("readable_text", "") or ""),
                text_read_uncertain=bool(d.get("text_read_uncertain", False)),
                asset_type=normalize_asset_type(d.get("asset_type")),
            ),
            interpretive=InterpretiveCaption(
                theme=str(d.get("theme", "") or ""),
                use_case=str(d.get("use_case", "") or ""),
                short_caption=str(d.get("short_caption", "") or ""),
                detailed_description=str(d.get("detailed_description", "") or ""),
            ),
            search=SearchTerms(
                tags=_slist(d.get("tags")),
                recommended_cases=_slist(d.get("recommended_cases")),
                aliases=_slist(d.get("aliases")),
            ),
            caption_quality=str(d.get("caption_quality", "ok") or "ok"),
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
    query_quality: str = "ok"
    error_message: str = ""

    @classmethod
    def empty(cls) -> "ImageQueryJSON":
        return cls()

    @classmethod
    def failed(cls, message: str) -> "ImageQueryJSON":
        return cls(query_quality="failed", error_message=message)

    def is_usable(self) -> bool:
        if self.query_quality != "ok":
            return False
        return bool(
            self.search_query.strip()
            or self.subject.strip()
            or self.style.strip()
            or self.layout.strip()
            or self.salient_objects
            or self.visible_text.strip()
        )

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


def _accept_caption_payload(data: Any) -> Optional[dict]:
    """Accept only complete caption payloads from structured provider output."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if validate_caption_dict(data):
        return data
    return None


def _build_caption_user_prompt(
    *,
    context: Optional[str],
    source_file: Optional[str],
) -> str:
    vocab = vocab_for_prompt(source_file=source_file)
    vocab_block = format_vocab_for_prompt(vocab)
    context_block = ""
    if context and context.strip():
        context_block = context.strip() + "\n\n"
    return CAPTION_USER_PROMPT_TEMPLATE.format(
        context_block=context_block,
        vocab_block=vocab_block,
        asset_type_block=format_taxonomy_for_prompt(),
        examples_block=format_few_shot_for_prompt(),
    )


def _extract_tool_input(response_content: list, tool_name: str) -> Optional[dict]:
    for block in response_content:
        if not isinstance(block, dict):
            continue
        if block.get("toolUse", {}).get("name") == tool_name:
            inp = block["toolUse"].get("input")
            return _accept_caption_payload(inp)
    return None


class VLMCaptioner:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or SETTINGS.vlm_provider).lower()
        self.model = model or SETTINGS.vlm_model

    def caption_image(
        self,
        image: Image.Image,
        *,
        max_side: Optional[int] = None,
        context: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> CaptionJSON:
        side = max_side if max_side is not None else SETTINGS.ingest_max_image_side
        if side and side > 0:
            image = resize_for_model(image, side)
        user_prompt = _build_caption_user_prompt(context=context, source_file=source_file)
        try:
            if self.provider == "bedrock":
                data = self._caption_bedrock_structured(image, user_prompt)
            elif self.provider == "openai":
                data = self._caption_openai_structured(image, user_prompt)
            elif self.provider == "anthropic":
                data = self._caption_anthropic_structured(image, user_prompt)
            else:
                raise ValueError(f"Unknown VLM provider: {self.provider}")
        except Exception as exc:  # noqa: BLE001
            return CaptionJSON.failed(str(exc))

        if not data or not validate_caption_dict(data):
            return CaptionJSON.failed("invalid structured caption response")
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
            return ImageQueryJSON.failed(str(exc))

        data = _parse_json_lenient(raw) or {}
        facets = ImageQueryJSON.from_dict(data)
        if not facets.is_usable():
            return ImageQueryJSON.failed("invalid or empty query response")
        return facets

    def _caption_bedrock_structured(self, image: Image.Image, user_prompt: str) -> Optional[dict]:
        from imagecb.models.bedrock_client import bedrock_converse

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        image_bytes = buf.getvalue()

        response = bedrock_converse(
            modelId=self.model,
            system=[{"text": CAPTION_SYSTEM_PROMPT}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                        {"text": user_prompt},
                    ],
                }
            ],
            inferenceConfig={"temperature": _CAPTION_TEMPERATURE, "maxTokens": _CAPTION_MAX_TOKENS},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": CAPTION_TOOL_NAME,
                            "description": "Emit structured image caption for search indexing.",
                            "inputSchema": {"json": CAPTION_JSON_SCHEMA},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": CAPTION_TOOL_NAME}},
            },
        )
        content = response["output"]["message"]["content"]
        return _extract_tool_input(content, CAPTION_TOOL_NAME)

    def _caption_openai_structured(self, image: Image.Image, user_prompt: str) -> Optional[dict]:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        data_url = _pil_to_data_url(image)
        resp = client.chat.completions.create(
            model=self.model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": CAPTION_TOOL_NAME,
                    "strict": True,
                    "schema": CAPTION_JSON_SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=_CAPTION_TEMPERATURE,
            max_tokens=_CAPTION_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content or "{}"
        return _accept_caption_payload(raw)

    def _caption_anthropic_structured(self, image: Image.Image, user_prompt: str) -> Optional[dict]:
        import anthropic

        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        msg = client.messages.create(
            model=self.model,
            max_tokens=_CAPTION_MAX_TOKENS,
            temperature=_CAPTION_TEMPERATURE,
            system=CAPTION_SYSTEM_PROMPT,
            tools=[
                {
                    "name": CAPTION_TOOL_NAME,
                    "description": "Emit structured image caption for search indexing.",
                    "input_schema": CAPTION_JSON_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": CAPTION_TOOL_NAME},
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
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == CAPTION_TOOL_NAME:
                return _accept_caption_payload(block.input)
        return None

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

    def _query_bedrock(self, image: Image.Image) -> str:
        return self._converse_bedrock(image, QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT)

    def _converse_bedrock(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        from imagecb.models.bedrock_client import bedrock_converse

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        image_bytes = buf.getvalue()

        response = bedrock_converse(
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
