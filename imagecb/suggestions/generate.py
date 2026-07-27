"""LLM-generated empty-state query suggestions with in-process cache."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from imagecb.config import SETTINGS
from imagecb.suggestions.corpus_summary import (
    CorpusContext,
    build_corpus_context,
    context_to_prompt_text,
)

SUGGESTIONS_SYSTEM_PROMPT = """You suggest starter search queries for an image search app over \
ingested slides, PDFs, and standalone images.

Return ONLY a JSON object:
{"suggestions": ["...", "..."]}

Rules:
- Each suggestion is a short natural-language phrase the user can click to search (under 80 chars).
- Use ONLY topics, tags, captions, and recommended search phrases present in the corpus context. \
Never invent assets or topics.
- Never reference source filenames ("images from X.pptx"); filenames in the context are for \
grounding only.
- Do not include markdown, code fences, or explanation outside the JSON object."""

ONBOARDING_SUGGESTIONS = [
    "Upload slides or PDFs in the Corpus panel, then search here",
    "Charts and diagrams",
    "Screenshots and UI mockups",
    "Logos and icons on plain backgrounds",
]

# Used only when the corpus has indexed rows but thin caption/tag metadata.
_INDEXED_FALLBACK_SUGGESTIONS = [
    "Photos of people collaborating",
    "Charts and diagrams",
    "Screenshots and UI mockups",
    "Icons and logos",
]

_ASSET_TYPE_SUGGESTIONS = {
    "photo": "Photos and real-world scenes",
    "illustration": "Illustrations and conceptual art",
    "diagram": "Diagrams and process visuals",
    "icon": "Icons and simple symbols",
    "chart": "Charts and data visualizations",
    "screenshot": "Screenshots and product UI",
    "logo": "Logos and brand marks",
}

_FILENAME_FILTER_RE = re.compile(
    r"^(?:images?|slides?|photos?|pictures?|content|visuals?|graphics?)\s+from\s+",
    re.IGNORECASE,
)
_FILENAME_EXT_RE = re.compile(r"\.(?:pptx?|pdf|docx?|xlsx?|png|jpe?g|gif|webp|bmp|tiff?)\b", re.IGNORECASE)

_cache: dict[str, Tuple[float, List[str]]] = {}


@dataclass(frozen=True)
class SuggestionsResult:
    suggestions: List[str]
    cached: bool


def _cache_key(ctx: CorpusContext, limit: int) -> str:
    return f"{ctx.fingerprint}|limit:{limit}"


def _coerce_suggestions_json(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        items = data.get("suggestions", [])
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s:
            out.append(s)
    return out


def _trim_suggestions(items: Sequence[str], limit: int) -> List[str]:
    return list(items[:limit]) if items else []


def _is_filename_filter_suggestion(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if _FILENAME_FILTER_RE.match(s):
        return True
    if _FILENAME_EXT_RE.search(s) and re.search(r"\bfrom\b", s, re.IGNORECASE):
        return True
    return False


def _indexed_fallback_pool(ctx: CorpusContext) -> List[str]:
    """Corpus-grounded fillers when caption/tag metadata is sparse."""
    pool: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        s = text.strip()
        if not s or _is_filename_filter_suggestion(s):
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        pool.append(s)

    for asset_type, _count in ctx.top_asset_types[:4]:
        suggestion = _ASSET_TYPE_SUGGESTIONS.get(asset_type)
        if suggestion:
            add(suggestion)
        else:
            add(f"{asset_type} visuals")

    for name in ctx.sample_image_names[:4]:
        add(name)

    for use in ctx.sample_use_cases[:3]:
        add(use)

    for alias in ctx.sample_aliases[:3]:
        add(alias)

    for source_type, _count in ctx.file_type_counts[:3]:
        if source_type == "pptx":
            add("Slide deck visuals and graphics")
        elif source_type == "pdf":
            add("PDF figures and document images")
        elif source_type == "image":
            add("Standalone uploaded images")

    for item in _INDEXED_FALLBACK_SUGGESTIONS:
        add(item)

    return pool


def _corpus_semantic_pool(ctx: CorpusContext) -> List[str]:
    pool: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        s = text.strip()
        if not s or _is_filename_filter_suggestion(s):
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        pool.append(s)

    for case in ctx.sample_recommended_cases:
        add(case)

    for tag in ctx.top_tags[:6]:
        add(f"{tag} and related visuals")

    for cap in ctx.sample_captions[:6]:
        if len(cap) <= 60:
            add(cap)
        else:
            add(cap[:57] + "...")

    for name in ctx.sample_image_names[:4]:
        add(name)

    for use in ctx.sample_use_cases[:3]:
        add(use)

    for alias in ctx.sample_aliases[:3]:
        add(alias)

    for asset_type, _count in ctx.top_asset_types[:3]:
        suggestion = _ASSET_TYPE_SUGGESTIONS.get(asset_type)
        if suggestion:
            add(suggestion)

    return pool


def _blend_suggestions(
    candidates: Sequence[str],
    ctx: CorpusContext,
    limit: int,
) -> List[str]:
    """Merge corpus semantic seeds and filtered candidates."""
    out: List[str] = []
    seen: set[str] = set()
    backfill = _corpus_semantic_pool(ctx)

    def add(text: str) -> None:
        s = text.strip()
        if not s or _is_filename_filter_suggestion(s):
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    for c in candidates:
        add(c)

    for item in backfill:
        if len(out) >= limit:
            break
        add(item)

    if len(out) < limit:
        # Never show empty-corpus onboarding when images are already indexed.
        fillers = (
            _indexed_fallback_pool(ctx)
            if ctx.indexed_count > 0
            else ONBOARDING_SUGGESTIONS
        )
        for item in fillers:
            if len(out) >= limit:
                break
            add(item)

    return _trim_suggestions(out, limit)


def _user_payload(ctx: CorpusContext, limit: int) -> str:
    blocks: List[str] = []
    if ctx.sample_recommended_cases:
        blocks.append(
            "Recommended search phrases from corpus:\n"
            + "\n".join(f"- {c}" for c in ctx.sample_recommended_cases)
        )
    topic_lines: List[str] = []
    if ctx.top_tags:
        topic_lines.append(f"Tags: {', '.join(ctx.top_tags)}")
    if ctx.top_asset_types:
        assets = ", ".join(f"{t} ({n})" for t, n in ctx.top_asset_types[:6])
        topic_lines.append(f"Asset types: {assets}")
    if ctx.sample_image_names:
        topic_lines.append("Image names:")
        topic_lines.extend(f"  - {n}" for n in ctx.sample_image_names[:6])
    if ctx.sample_use_cases:
        topic_lines.append("Use cases:")
        topic_lines.extend(f"  - {u}" for u in ctx.sample_use_cases[:4])
    if ctx.sample_aliases:
        topic_lines.append("Search aliases:")
        topic_lines.extend(f"  - {a}" for a in ctx.sample_aliases[:4])
    if ctx.sample_captions:
        topic_lines.append("Captions:")
        topic_lines.extend(f"  - {c}" for c in ctx.sample_captions[:6])
    if topic_lines:
        blocks.append("Corpus topics:\n" + "\n".join(topic_lines))
    blocks.append("Corpus context:\n" + context_to_prompt_text(ctx))
    blocks.append(
        f"Generate exactly {limit} semantic, topic-based suggestions. "
        "Never use filename-filter phrasing."
    )
    return "\n\n".join(blocks)


def _corpus_heuristic_suggestions(ctx: CorpusContext, limit: int) -> List[str]:
    """Build suggestions from corpus metadata only."""
    candidates: List[str] = []

    for case in ctx.sample_recommended_cases:
        candidates.append(case)

    for cap in ctx.sample_captions[:4]:
        if len(cap) <= 60:
            candidates.append(cap)
        else:
            candidates.append(cap[:57] + "...")

    for tag in ctx.top_tags[:4]:
        candidates.append(f"{tag} and related visuals")

    for name in ctx.sample_image_names[:3]:
        candidates.append(name)

    for use in ctx.sample_use_cases[:2]:
        candidates.append(use)

    return _blend_suggestions(candidates, ctx, limit)


class SuggestionLLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or SETTINGS.llm_provider).lower()
        self.model = model or SETTINGS.llm_model

    def generate(self, user_payload: str, *, system_prompt: Optional[str] = None) -> str:
        system = system_prompt or SUGGESTIONS_SYSTEM_PROMPT
        if self.provider == "bedrock":
            return self._bedrock(user_payload, system)
        if self.provider == "openai":
            return self._openai(user_payload, system)
        if self.provider == "anthropic":
            return self._anthropic(user_payload, system)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _bedrock(self, user_payload: str, system: str) -> str:
        from imagecb.models.bedrock_client import bedrock_converse

        response = bedrock_converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_payload}]}],
            inferenceConfig={"temperature": 0.3, "maxTokens": 400},
        )
        parts = [
            block.get("text", "")
            for block in response["output"]["message"]["content"]
            if "text" in block
        ]
        return "".join(parts) or "{}"

    def _openai(self, user_payload: str, system: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or "{}"

    def _anthropic(self, user_payload: str, system: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user_payload}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts) or "{}"


_llm: Optional[SuggestionLLM] = None


def get_suggestion_llm() -> SuggestionLLM:
    global _llm
    if _llm is None:
        _llm = SuggestionLLM()
    return _llm


def _onboarding_list(limit: int) -> List[str]:
    return _trim_suggestions(ONBOARDING_SUGGESTIONS, limit)


def _llm_suggestions(ctx: CorpusContext, limit: int) -> List[str]:
    payload = _user_payload(ctx, limit)
    raw = get_suggestion_llm().generate(payload)
    parsed = _coerce_suggestions_json(raw)
    if len(parsed) < 2:
        return _corpus_heuristic_suggestions(ctx, limit)
    return _blend_suggestions(parsed, ctx, limit)


def generate_suggestions(
    *,
    limit: Optional[int] = None,
    ctx: Optional[CorpusContext] = None,
) -> SuggestionsResult:
    """Return suggested queries; uses cache keyed by corpus fingerprint."""
    n = limit if limit is not None else SETTINGS.suggestions_limit
    n = max(2, min(8, n))
    corpus = ctx if ctx is not None else build_corpus_context()

    if corpus.indexed_count == 0:
        return SuggestionsResult(
            suggestions=_onboarding_list(n),
            cached=False,
        )

    key = _cache_key(corpus, n)
    now = time.monotonic()
    ttl = SETTINGS.suggestions_cache_ttl_sec
    cached_entry = _cache.get(key)
    if cached_entry is not None:
        ts, items = cached_entry
        if now - ts < ttl and len(items) >= 2:
            return SuggestionsResult(
                suggestions=_trim_suggestions(items, n),
                cached=True,
            )

    # Heuristics only on the request path — avoid blocking on Bedrock/OpenAI.
    items = _corpus_heuristic_suggestions(corpus, n)

    _cache[key] = (now, items)
    return SuggestionsResult(suggestions=items, cached=False)
