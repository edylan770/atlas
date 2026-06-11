"""Context-aware follow-up query suggestions after a chat search turn."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.formatting.assistant_reply import provenance_from_record
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import AskResult
from imagecb.storage.metadata_db import deserialize_list
from imagecb.suggestions.corpus_summary import (
    CorpusContext,
    build_corpus_context,
    context_to_prompt_text,
)
from imagecb.suggestions.generate import (
    ONBOARDING_SUGGESTIONS,
    SuggestionLLM,
    _blend_suggestions,
    _coerce_suggestions_json,
    _corpus_semantic_pool,
    _is_filename_filter_suggestion,
    _trim_suggestions,
    get_suggestion_llm,
)

logger = logging.getLogger(__name__)

FOLLOW_UP_SYSTEM_PROMPT = """You suggest follow-up search queries for an image search app over \
ingested slides, PDFs, and standalone images.

Return ONLY a JSON object:
{"suggestions": ["...", "..."]}

Rules:
- Each suggestion is a short natural-language phrase the user can click to search (under 80 chars).
- Ground suggestions in the corpus context AND the current search results shown below.
- Prefer refinements when many or weak matches (narrow by topic, asset type, author, or date).
- When few results, suggest related corpus topics the user could explore next.
- Never repeat the user's current query verbatim.
- Use ONLY topics, tags, captions, and recommended search phrases present in the context. \
Never invent assets or topics.
- Never reference source filenames ("images from X.pptx"); filenames are for grounding only.
- Do not include markdown, code fences, or explanation outside the JSON object."""

_RESULT_LINES = 5
_CAPTION_TRUNC = 100


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_key(text: str) -> str:
    return text.strip().lower()


def _is_duplicate_query(suggestion: str, user_message: str, semantic_query: str) -> bool:
    key = _normalize_key(suggestion)
    if not key:
        return True
    for q in (user_message, semantic_query):
        if q and key == _normalize_key(q):
            return True
    return False


def _results_context_block(results: Sequence[RankedResult]) -> str:
    if not results:
        return "(no results)"
    lines: List[str] = []
    for i, r in enumerate(results[:_RESULT_LINES], start=1):
        prov = provenance_from_record(r.record)
        cap = _truncate(
            (r.record.caption_short or r.record.caption_detailed or "").strip(),
            _CAPTION_TRUNC,
        )
        tags = deserialize_list(r.record.tags_json)
        tag_part = f" — tags: {', '.join(tags[:4])}" if tags else ""
        lines.append(f"{i}. {prov.location_label()} ({prov.source_name}) — {cap or 'no caption'}{tag_part}")
    if len(results) > _RESULT_LINES:
        lines.append(f"... and {len(results) - _RESULT_LINES} more")
    return "\n".join(lines)


def _collect_result_candidates(results: Sequence[RankedResult]) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        s = text.strip()
        if not s or _is_filename_filter_suggestion(s):
            return
        key = _normalize_key(s)
        if key in seen:
            return
        seen.add(key)
        candidates.append(s)

    for r in results[:5]:
        for case in deserialize_list(r.record.recommended_cases_json):
            add(case)
        for tag in deserialize_list(r.record.tags_json):
            add(f"{tag} and related visuals")
        cap = (r.record.caption_short or "").strip()
        if cap and cap != "[caption failed]":
            if len(cap) <= 60:
                add(cap)
            else:
                add(cap[:57] + "...")

    return candidates


def _refinement_candidates(spec: QuerySpec, ctx: CorpusContext) -> List[str]:
    candidates: List[str] = []
    sf = spec.source_filters

    if not sf.asset_types:
        for tag in ctx.top_tags[:3]:
            if "chart" in tag or "diagram" in tag:
                candidates.append("only charts and diagrams")
                break
        else:
            candidates.append("only charts and diagrams")

    if not sf.authors and ctx.authors:
        candidates.append(f"by {ctx.authors[0]}")

    if spec.is_refinement:
        candidates.append("broaden the search")
    elif spec.semantic_query:
        candidates.append(f"more like {spec.semantic_query}")

    return candidates


def _follow_up_heuristic_suggestions(
    user_message: str,
    ask_result: AskResult,
    ctx: CorpusContext,
    limit: int,
) -> List[str]:
    spec = ask_result.spec
    results = ask_result.results
    candidates = _collect_result_candidates(results)
    candidates.extend(_refinement_candidates(spec, ctx))

    pool = _corpus_semantic_pool(ctx)
    for item in pool:
        if len(candidates) >= limit * 2:
            break
        candidates.append(item)

    if ctx.indexed_count == 0:
        for item in ONBOARDING_SUGGESTIONS:
            candidates.append(item)

    out: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        s = text.strip()
        if not s or _is_filename_filter_suggestion(s):
            return
        if _is_duplicate_query(s, user_message, spec.semantic_query):
            return
        key = _normalize_key(s)
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    for c in candidates:
        if len(out) >= limit:
            break
        add(c)

    for item in pool:
        if len(out) >= limit:
            break
        add(item)

    if len(out) < 2 and ctx.indexed_count == 0:
        for item in ONBOARDING_SUGGESTIONS:
            if len(out) >= limit:
                break
            add(item)

    return _trim_suggestions(out, limit)


def _build_follow_up_payload(
    user_message: str,
    ask_result: AskResult,
    interpretation_notes: Sequence[str],
    ctx: CorpusContext,
    limit: int,
) -> str:
    spec = ask_result.spec
    notes = "\n".join(f"- {n}" for n in interpretation_notes) if interpretation_notes else "(none)"
    blocks = [
        f"User's current query: {user_message}",
        f"Semantic query: {spec.semantic_query}",
        f"Is refinement of prior results: {spec.is_refinement}",
        f"Result count: {len(ask_result.results)}",
        f"Indexed corpus size: {ctx.indexed_count}",
        f"\nInterpretation notes:\n{notes}",
        f"\nTop results:\n{_results_context_block(ask_result.results)}",
        f"\nCorpus context:\n{context_to_prompt_text(ctx)}",
        f"\nGenerate exactly {limit} follow-up search suggestions. "
        "Never repeat the current query. Never use filename-filter phrasing.",
    ]
    return "\n\n".join(blocks)


def _llm_follow_up_suggestions(
    user_message: str,
    ask_result: AskResult,
    interpretation_notes: Sequence[str],
    ctx: CorpusContext,
    limit: int,
    llm: SuggestionLLM,
) -> List[str]:
    payload = _build_follow_up_payload(
        user_message, ask_result, interpretation_notes, ctx, limit
    )
    raw = llm.generate(payload, system_prompt=FOLLOW_UP_SYSTEM_PROMPT)
    parsed = _coerce_suggestions_json(raw)
    filtered = [
        s
        for s in parsed
        if not _is_duplicate_query(s, user_message, ask_result.spec.semantic_query)
    ]
    if len(filtered) < 2:
        return _follow_up_heuristic_suggestions(user_message, ask_result, ctx, limit)
    return _blend_suggestions(filtered, ctx, limit)


def generate_follow_up_suggestions(
    user_message: str,
    ask_result: AskResult,
    interpretation_notes: Sequence[str],
    *,
    corpus: Optional[CorpusContext] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Return follow-up queries grounded in corpus and current search context."""
    if not SETTINGS.enable_follow_up_suggestions:
        return []

    n = limit if limit is not None else SETTINGS.follow_up_suggestions_limit
    n = max(2, min(6, n))
    ctx = corpus if corpus is not None else build_corpus_context()

    try:
        llm = get_suggestion_llm()
        return _llm_follow_up_suggestions(
            user_message, ask_result, interpretation_notes, ctx, n, llm
        )
    except Exception:  # noqa: BLE001
        logger.exception("Follow-up suggestion LLM failed; using heuristics")
        return _follow_up_heuristic_suggestions(user_message, ask_result, ctx, n)
