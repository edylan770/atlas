"""Record search and interaction telemetry events."""

from __future__ import annotations

import json
import uuid
from typing import List, Literal, Optional, Sequence

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
            )
        )
    return event_id


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


def get_recent_user_queries(user_id: str, *, limit: int = 8) -> List[str]:
    """Return deduped recent chat query texts for a user, most recent first."""
    if not user_id or user_id == "anonymous":
        return []
    ensure_telemetry_schema()
    from sqlalchemy import select

    seen: set[str] = set()
    out: List[str] = []
    with session_scope() as s:
        rows = s.execute(
            select(SearchEvent.query_text)
            .where(SearchEvent.user_id == user_id)
            .where(SearchEvent.search_kind == "chat")
            .order_by(SearchEvent.created_at.desc())
            .limit(limit * 3)
        ).scalars().all()
    for raw in rows:
        q = (raw or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out
