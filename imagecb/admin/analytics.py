"""Search quality analytics over telemetry."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from imagecb.config import SETTINGS
from imagecb.storage.metadata_db import session_scope
from imagecb.telemetry.models import InteractionEvent, SearchEvent
from imagecb.telemetry.schema import ensure_telemetry_schema


def _display_query(row: SearchEvent) -> str:
    """Primary label for admin tables: semantic intent for chat, raw label for similar."""
    user = (row.query_text or "").strip()
    semantic = (row.parsed_semantic_query or "").strip()
    if row.search_kind == "similar":
        return user or "[similar image search]"
    if semantic:
        return semantic
    return user or "—"


def _parse_since(since: Optional[str]) -> Optional[datetime]:
    if not since:
        return None
    try:
        return datetime.fromisoformat(since.replace("Z", "+00:00").replace("+00:00", ""))
    except ValueError:
        return None


def _event_dict(row: SearchEvent, *, category: str) -> dict:
    try:
        served = json.loads(row.served_image_ids_json or "[]")
    except json.JSONDecodeError:
        served = []
    user_message = (row.query_text or "").strip()
    return {
        "search_event_id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "query_text": row.query_text,
        "user_message": user_message,
        "display_query": _display_query(row),
        "user_id": row.user_id,
        "session_id": row.session_id,
        "search_kind": row.search_kind,
        "served_image_ids": served,
        "result_count": row.result_count,
        "top_score": row.top_score,
        "top_score_kind": row.top_score_kind,
        "parsed_semantic_query": row.parsed_semantic_query,
        "category": category,
    }


def search_quality_lists(
    *,
    since: Optional[str] = None,
    limit: int = 50,
    weak_score_threshold: Optional[float] = None,
) -> Dict[str, List[dict]]:
    ensure_telemetry_schema()
    threshold = (
        weak_score_threshold
        if weak_score_threshold is not None
        else SETTINGS.weak_result_score_threshold
    )
    since_dt = _parse_since(since)

    with session_scope() as s:
        base = select(SearchEvent)
        if since_dt:
            base = base.where(SearchEvent.created_at >= since_dt)

        zero_rows = s.execute(
            base.where(SearchEvent.result_count == 0)
            .order_by(SearchEvent.created_at.desc())
            .limit(limit)
        ).scalars().all()

        weak_rows = s.execute(
            base.where(SearchEvent.result_count > 0)
            .where(SearchEvent.top_score.is_not(None))
            .where(SearchEvent.top_score < threshold)
            .order_by(SearchEvent.created_at.desc())
            .limit(limit)
        ).scalars().all()

        interacted = (
            select(InteractionEvent.search_event_id)
            .distinct()
            .scalar_subquery()
        )
        no_ix_rows = s.execute(
            base.where(SearchEvent.result_count > 0)
            .where(SearchEvent.id.not_in(interacted))
            .order_by(SearchEvent.created_at.desc())
            .limit(limit)
        ).scalars().all()

    return {
        "zero_result": [_event_dict(r, category="zero_result") for r in zero_rows],
        "weak_result": [_event_dict(r, category="weak_result") for r in weak_rows],
        "no_interaction": [_event_dict(r, category="no_interaction") for r in no_ix_rows],
        "weak_score_threshold": threshold,
    }


def funnel_detail(search_event_id: str) -> Optional[dict]:
    ensure_telemetry_schema()
    with session_scope() as s:
        row = s.execute(
            select(SearchEvent).where(SearchEvent.id == search_event_id)
        ).scalar_one_or_none()
        if row is None:
            return None
        interactions = s.execute(
            select(InteractionEvent)
            .where(InteractionEvent.search_event_id == search_event_id)
            .order_by(InteractionEvent.created_at)
        ).scalars().all()

    try:
        served = json.loads(row.served_image_ids_json or "[]")
    except json.JSONDecodeError:
        served = []

    return {
        "search": _event_dict(row, category="search"),
        "interactions": [
            {
                "id": i.id,
                "image_id": i.image_id,
                "interaction_type": i.interaction_type,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "user_id": i.user_id,
                "rank": i.rank,
            }
            for i in interactions
        ],
        "served_image_ids": served,
    }


def analytics_summary(
    *,
    since: Optional[str] = None,
    days: int = 7,
) -> dict[str, Any]:
    ensure_telemetry_schema()
    since_dt = _parse_since(since)
    if since_dt is None:
        since_dt = datetime.utcnow() - timedelta(days=days)

    with session_scope() as s:
        total = s.execute(
            select(func.count()).select_from(SearchEvent).where(SearchEvent.created_at >= since_dt)
        ).scalar() or 0

        zero = s.execute(
            select(func.count())
            .select_from(SearchEvent)
            .where(SearchEvent.created_at >= since_dt)
            .where(SearchEvent.result_count == 0)
        ).scalar() or 0

        threshold = SETTINGS.weak_result_score_threshold
        weak = s.execute(
            select(func.count())
            .select_from(SearchEvent)
            .where(SearchEvent.created_at >= since_dt)
            .where(SearchEvent.result_count > 0)
            .where(SearchEvent.top_score.is_not(None))
            .where(SearchEvent.top_score < threshold)
        ).scalar() or 0

        with_results = s.execute(
            select(func.count())
            .select_from(SearchEvent)
            .where(SearchEvent.created_at >= since_dt)
            .where(SearchEvent.result_count > 0)
        ).scalar() or 0

        interacted_subq = (
            select(InteractionEvent.search_event_id)
            .distinct()
            .scalar_subquery()
        )
        no_ix = s.execute(
            select(func.count())
            .select_from(SearchEvent)
            .where(SearchEvent.created_at >= since_dt)
            .where(SearchEvent.result_count > 0)
            .where(SearchEvent.id.not_in(interacted_subq))
        ).scalar() or 0

        interaction_count = s.execute(
            select(func.count())
            .select_from(InteractionEvent)
            .where(InteractionEvent.created_at >= since_dt)
        ).scalar() or 0

    ctr = (interaction_count / with_results) if with_results else 0.0
    return {
        "since": since_dt.isoformat(),
        "total_searches": total,
        "zero_result_count": zero,
        "weak_result_count": weak,
        "no_interaction_count": no_ix,
        "searches_with_results": with_results,
        "interaction_count": interaction_count,
        "interaction_rate": round(ctr, 4),
        "zero_result_rate": round(zero / total, 4) if total else 0.0,
        "weak_result_rate": round(weak / total, 4) if total else 0.0,
        "no_interaction_rate": round(no_ix / with_results, 4) if with_results else 0.0,
        "weak_score_threshold": SETTINGS.weak_result_score_threshold,
    }
