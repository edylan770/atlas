"""Build assistant replies: LLM when enabled, template fallback otherwise."""

from __future__ import annotations

import logging
from typing import Iterator, List, Sequence

from imagecb.config import SETTINGS
from imagecb.formatting.match_display import display_match_percent
from imagecb.formatting.assistant_reply import (
    AssistantReply,
    build_assistant_reply,
    build_result_cards,
    provenance_from_record,
)
from imagecb.models.conversation_llm import get_conversation_llm
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import AskResult

logger = logging.getLogger(__name__)

_RESULT_LINES = 5
_CAPTION_TRUNC = 100


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _results_block(results: Sequence[RankedResult]) -> str:
    if not results:
        return "(no results)"
    lines: List[str] = []
    for i, r in enumerate(results[:_RESULT_LINES], start=1):
        prov = provenance_from_record(r.record)
        cap = _truncate(
            (r.record.caption_short or r.record.caption_detailed or "").strip(),
            _CAPTION_TRUNC,
        )
        pct = display_match_percent(r.score, r.score_kind)
        lines.append(
            f"{i}. {prov.location_label()} ({prov.source_name}) — {pct}% match — {cap or 'no caption'}"
        )
    if len(results) > _RESULT_LINES:
        lines.append(f"... and {len(results) - _RESULT_LINES} more")
    return "\n".join(lines)


def _build_llm_payload(
    user_message: str,
    spec: QuerySpec,
    results: Sequence[RankedResult],
    *,
    interpretation_notes: List[str],
    indexed_count: int,
) -> str:
    notes = "\n".join(f"- {n}" for n in interpretation_notes) if interpretation_notes else "(none)"
    return (
        f"User message: {user_message}\n\n"
        f"Semantic query: {spec.semantic_query}\n"
        f"Is refinement of prior results: {spec.is_refinement}\n"
        f"Result count: {len(results)}\n"
        f"Indexed corpus size: {indexed_count}\n\n"
        f"Interpretation notes:\n{notes}\n\n"
        f"Top results:\n{_results_block(results)}"
    )


def build_conversational_reply(
    user_message: str,
    ask_result: AskResult,
    interpretation_notes: List[str],
    *,
    indexed_count: int = 0,
    image_url_prefix: str = "/api/images",
) -> AssistantReply:
    """Return structured reply; uses LLM prose when enabled, else templates."""
    spec = ask_result.spec
    results = ask_result.results
    template_reply = build_assistant_reply(results, spec, image_url_prefix=image_url_prefix)

    if not SETTINGS.enable_conversational_llm:
        return template_reply

    try:
        llm = get_conversation_llm()
        payload = _build_llm_payload(
            user_message,
            spec,
            results,
            interpretation_notes=interpretation_notes,
            indexed_count=indexed_count,
        )
        message = llm.reply(payload)
        if not message:
            return template_reply
        return AssistantReply(
            message=message,
            results=build_result_cards(results, image_url_prefix=image_url_prefix),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Conversational LLM failed; using template reply")
        return template_reply


def iter_conversational_reply_text(
    user_message: str,
    ask_result: AskResult,
    interpretation_notes: List[str],
    *,
    indexed_count: int = 0,
    image_url_prefix: str = "/api/images",
) -> Iterator[str]:
    """Yield assistant prose chunks; uses LLM stream when enabled, else template."""
    spec = ask_result.spec
    results = ask_result.results
    template_reply = build_assistant_reply(results, spec, image_url_prefix=image_url_prefix)

    if not SETTINGS.enable_conversational_llm:
        yield template_reply.message
        return

    try:
        llm = get_conversation_llm()
        payload = _build_llm_payload(
            user_message,
            spec,
            results,
            interpretation_notes=interpretation_notes,
            indexed_count=indexed_count,
        )
        had_output = False
        for chunk in llm.reply_stream(payload):
            if chunk:
                had_output = True
                yield chunk
        if not had_output:
            yield template_reply.message
    except Exception:  # noqa: BLE001
        logger.exception("Conversational LLM stream failed; using template reply")
        yield template_reply.message
