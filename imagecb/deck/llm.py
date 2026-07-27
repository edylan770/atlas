"""Batched LLM domain translation: slide text -> image search descriptions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.llm import _coerce_json

logger = logging.getLogger(__name__)

SLIDE_DESCRIPTION_SYSTEM_PROMPT = """You translate PowerPoint slide text into concise, concrete, \
caption-style image descriptions for a semantic image search system.

Return ONLY a JSON object:
{"slides": [{"slide_index": integer, "status": "image_needed" | "no_image_needed", \
"description": string (required when image_needed), "reason": string (required when no_image_needed)}]}

Rules:
- Ground every output STRICTLY in the provided title, body, and notes. Do not invent entities, \
scenes, or details not supported by the slide text.
- No creative embellishment. Rewrite terse or abstract slide language into concrete visual \
descriptions suitable as search queries.
- Use status "no_image_needed" for slides that should not be illustrated: data tables, agendas, \
section dividers, pure bullet lists with no visual subject, thank-you/closing slides, or text-only \
administrative content. Provide a brief reason.
- Use status "image_needed" with a single concise description (under 120 words) when the slide \
benefits from a supporting stock or corpus image.
- Output exactly one entry per input slide_index; indices must match the input.
- Do not include markdown, code fences, or text outside the JSON object."""

FORCE_SLIDE_USER_PREFIX = (
    "The user explicitly requires an image suggestion for this slide. "
    "You must return status \"image_needed\" with a minimal concrete description "
    "grounded strictly in the slide text below.\n\n"
)


@dataclass(frozen=True)
class SlideLLMOutput:
    slide_index: int
    status: str  # image_needed | no_image_needed
    description: str = ""
    reason: str = ""


def _batch_user_payload(slides: Sequence[dict]) -> str:
    payload = {"slides": list(slides)}
    return (
        "Translate each slide below into the JSON format described.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _force_user_payload(slide: dict) -> str:
    return FORCE_SLIDE_USER_PREFIX + json.dumps({"slides": [slide]}, ensure_ascii=False)


def _coerce_slides_json(raw: str, expected_indices: Sequence[int]) -> List[SlideLLMOutput]:
    data = _coerce_json(raw)
    items = data.get("slides", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        items = []

    by_index: Dict[int, SlideLLMOutput] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("slide_index", 0))
        except (TypeError, ValueError):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status not in ("image_needed", "no_image_needed"):
            continue
        desc = str(item.get("description", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if status == "image_needed" and not desc:
            continue
        if status == "no_image_needed" and not reason:
            reason = "No illustration needed for this slide."
        by_index[idx] = SlideLLMOutput(
            slide_index=idx,
            status=status,
            description=desc,
            reason=reason,
        )

    out: List[SlideLLMOutput] = []
    for idx in expected_indices:
        if idx in by_index:
            out.append(by_index[idx])
        else:
            out.append(
                SlideLLMOutput(
                    slide_index=idx,
                    status="no_image_needed",
                    reason="LLM did not return a valid entry for this slide.",
                )
            )
    return out


class SlideDescriptionLLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or SETTINGS.llm_provider).lower()
        self.model = model or SETTINGS.llm_model

    def describe_batch(self, slides: Sequence[dict]) -> List[SlideLLMOutput]:
        if not slides:
            return []
        indices = [int(s["slide_index"]) for s in slides]
        payload = _batch_user_payload(slides)
        raw = self._call(payload, max_tokens=2000)
        return _coerce_slides_json(raw, indices)

    def describe_force(self, slide: dict) -> SlideLLMOutput:
        idx = int(slide["slide_index"])
        payload = _force_user_payload(slide)
        raw = self._call(payload, max_tokens=400)
        results = _coerce_slides_json(raw, [idx])
        result = results[0]
        if result.status != "image_needed" or not result.description:
            body = slide.get("body") or slide.get("title") or ""
            fallback = (body.strip()[:200] or "presentation slide content").strip()
            return SlideLLMOutput(
                slide_index=idx,
                status="image_needed",
                description=fallback,
                reason="",
            )
        return result

    def _call(self, user_payload: str, *, max_tokens: int) -> str:
        if self.provider == "bedrock":
            return self._bedrock(user_payload, max_tokens=max_tokens)
        if self.provider == "openai":
            return self._openai(user_payload, max_tokens=max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(user_payload, max_tokens=max_tokens)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _bedrock(self, user_payload: str, *, max_tokens: int) -> str:
        from imagecb.models.bedrock_client import bedrock_converse

        response = bedrock_converse(
            modelId=self.model,
            system=[{"text": SLIDE_DESCRIPTION_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_payload}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": max_tokens},
        )
        parts = [
            block.get("text", "")
            for block in response["output"]["message"]["content"]
            if "text" in block
        ]
        return "".join(parts) or "{}"

    def _openai(self, user_payload: str, *, max_tokens: int) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SLIDE_DESCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or "{}"

    def _anthropic(self, user_payload: str, *, max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SLIDE_DESCRIPTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts) or "{}"


_llm: Optional[SlideDescriptionLLM] = None


def get_slide_description_llm() -> SlideDescriptionLLM:
    global _llm
    if _llm is None:
        _llm = SlideDescriptionLLM()
    return _llm


def describe_slides_batched(slides: Sequence[dict]) -> List[SlideLLMOutput]:
    """Run LLM in batches of SETTINGS.deck_llm_batch_size."""
    batch_size = max(1, SETTINGS.deck_llm_batch_size)
    llm = get_slide_description_llm()
    all_out: List[SlideLLMOutput] = []
    slide_list = list(slides)
    for start in range(0, len(slide_list), batch_size):
        chunk = slide_list[start : start + batch_size]
        all_out.extend(llm.describe_batch(chunk))
    return all_out
