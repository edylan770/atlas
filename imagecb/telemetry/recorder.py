"""Record search and interaction telemetry events."""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Literal, Optional, Sequence

from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import session_scope
from imagecb.telemetry.models import InteractionEvent, SearchEvent
from imagecb.telemetry.schema import ensure_telemetry_schema

SearchKind = Literal["chat", "similar"]
InteractionType = Literal["view", "download", "similar"]


def _new_id() -> str:
    return str(uuid.uuid4())


def record_search_from_results(
    *,
    query_text: str,
    user_id: str,
    session_id: Optional[str],
    search_kind: SearchKind,
    results: Sequence[RankedResult],
    spec: Optional[QuerySpec] = None,
    total_ms: Optional[float] = None,
    ask_ms: Optional[float] = None,
    reply_ms: Optional[float] = None,
    timings_ms: Optional[Dict[str, float]] = None,
    timing_log: Optional[str] = None,
) -> str:
    """Persist a search event and return its id."""
    ensure_telemetry_schema()
    served = [r.image_id for r in results]
    top_score: Optional[float] = None
    top_score_kind: Optional[str] = None
    if results:
        top_score = float(results[0].score)
        top_score_kind = results[0].score_kind

    semantic = spec.semantic_query if spec else None
    stored_query = query_text
    if spec and (spec.raw_text or "").strip():
        stored_query = spec.raw_text.strip()
    event_id = _new_id()
    with session_scope() as s:
        s.add(
            SearchEvent(
                id=event_id,
                query_text=stored_query,
                user_id=user_id or "anonymous",
                session_id=session_id,
                search_kind=search_kind,
                served_image_ids_json=json.dumps(served),
                result_count=len(served),
                top_score=top_score,
                top_score_kind=top_score_kind,
                parsed_semantic_query=semantic,
                total_ms=total_ms,
                ask_ms=ask_ms,
                reply_ms=reply_ms,
                timings_json=json.dumps(timings_ms) if timings_ms is not None else None,
                timing_log=timing_log,
            )
        )
    return event_id


def attach_search_timings(
    search_event_id: str,
    *,
    total_ms: Optional[float] = None,
    ask_ms: Optional[float] = None,
    reply_ms: Optional[float] = None,
    timings_ms: Optional[Dict[str, float]] = None,
    timing_log: Optional[str] = None,
) -> None:
    """Update an existing search event with latency fields (e.g. after stream completes)."""
    ensure_telemetry_schema()
    from sqlalchemy import select

    with session_scope() as s:
        row = s.execute(
            select(SearchEvent).where(SearchEvent.id == search_event_id)
        ).scalar_one_or_none()
        if row is None:
            return
        if total_ms is not None:
            row.total_ms = total_ms
        if ask_ms is not None:
            row.ask_ms = ask_ms
        if reply_ms is not None:
            row.reply_ms = reply_ms
        if timings_ms is not None:
            row.timings_json = json.dumps(timings_ms)
        if timing_log is not None:
            row.timing_log = timing_log


def get_served_image_ids(search_event_id: str) -> List[str]:
    ensure_telemetry_schema()
    from sqlalchemy import select

    with session_scope() as s:
        row = s.execute(
            select(SearchEvent.served_image_ids_json).where(SearchEvent.id == search_event_id)
        ).scalar_one_or_none()
        if row is None:
            return []
        try:
            loaded = json.loads(row)
            if isinstance(loaded, list):
                return [str(x) for x in loaded]
        except json.JSONDecodeError:
            pass
        return []


def record_interaction(
    *,
    search_event_id: str,
    image_id: str,
    interaction_type: InteractionType,
    user_id: str = "anonymous",
    rank: Optional[int] = None,
) -> str:
    """Record a user interaction linked to a search event. Raises ValueError if invalid."""
    ensure_telemetry_schema()
    served = get_served_image_ids(search_event_id)
    if image_id not in served:
        raise ValueError("image_id was not in the originating search results")

    from sqlalchemy import select

    with session_scope() as s:
        exists = s.execute(
            select(SearchEvent.id).where(SearchEvent.id == search_event_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ValueError("search_event_id not found")

    interaction_id = _new_id()
    with session_scope() as s:
        s.add(
            InteractionEvent(
                id=interaction_id,
                search_event_id=search_event_id,
                image_id=image_id,
                interaction_type=interaction_type,
                user_id=user_id or "anonymous",
                rank=rank,
            )
        )
    return interaction_id


