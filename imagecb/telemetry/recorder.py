"""Record search and interaction telemetry events (blob/S3 store)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.telemetry import s3_store

SearchKind = Literal["chat", "similar"]
InteractionType = Literal["view", "download", "similar"]


def _new_id() -> str:
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.utcnow()


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
    created = _utc_now()
    result_count = len(served)
    threshold = SETTINGS.weak_result_score_threshold
    is_weak = (
        result_count > 0
        and top_score is not None
        and top_score < threshold
    )

    event = {
        "id": event_id,
        "created_at": created.isoformat(),
        "query_text": stored_query,
        "user_id": user_id or "anonymous",
        "session_id": session_id,
        "search_kind": search_kind,
        "served_image_ids": served,
        "result_count": result_count,
        "top_score": top_score,
        "top_score_kind": top_score_kind,
        "parsed_semantic_query": semantic,
        "total_ms": total_ms,
        "ask_ms": ask_ms,
        "reply_ms": reply_ms,
        "timings": timings_ms,
        "timing_log": timing_log,
        "has_interaction": False,
    }
    s3_store.put_search_event(event)

    deltas = {
        "total_searches": 1,
        "zero_result_count": 1 if result_count == 0 else 0,
        "weak_result_count": 1 if is_weak else 0,
        "searches_with_results": 1 if result_count > 0 else 0,
        "no_interaction_count": 1 if result_count > 0 else 0,
        "interaction_count": 0,
    }
    s3_store.bump_daily_rollup(s3_store._dt_str(created), deltas)
    s3_store.invalidate_quality_cache()
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
    row = s3_store.get_search_event(search_event_id)
    if row is None:
        return
    if total_ms is not None:
        row["total_ms"] = total_ms
    if ask_ms is not None:
        row["ask_ms"] = ask_ms
    if reply_ms is not None:
        row["reply_ms"] = reply_ms
    if timings_ms is not None:
        row["timings"] = timings_ms
    if timing_log is not None:
        row["timing_log"] = timing_log
    s3_store.put_search_event(row)
    s3_store.invalidate_quality_cache()


def get_served_image_ids(search_event_id: str) -> List[str]:
    row = s3_store.get_search_event(search_event_id)
    if row is None:
        return []
    loaded = row.get("served_image_ids") or []
    if isinstance(loaded, list):
        return [str(x) for x in loaded]
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
    search = s3_store.get_search_event(search_event_id)
    if search is None:
        raise ValueError("search_event_id not found")

    served = search.get("served_image_ids") or []
    if not isinstance(served, list):
        served = []
    served_ids = [str(x) for x in served]
    if image_id not in served_ids:
        raise ValueError("image_id was not in the originating search results")

    interaction_id = _new_id()
    created = _utc_now()
    s3_store.put_interaction_event(
        {
            "id": interaction_id,
            "search_event_id": search_event_id,
            "image_id": image_id,
            "interaction_type": interaction_type,
            "created_at": created.isoformat(),
            "user_id": user_id or "anonymous",
            "rank": rank,
        }
    )

    first_interaction = not bool(search.get("has_interaction"))
    if first_interaction:
        search["has_interaction"] = True
        s3_store.put_search_event(search)

    search_created = s3_store._parse_created_at(search.get("created_at")) or created
    search_dt = s3_store._dt_str(search_created)
    interaction_dt = s3_store._dt_str(created)

    s3_store.bump_daily_rollup(interaction_dt, {"interaction_count": 1})
    if first_interaction:
        s3_store.bump_daily_rollup(search_dt, {"no_interaction_count": -1})

    s3_store.invalidate_quality_cache()
    return interaction_id
