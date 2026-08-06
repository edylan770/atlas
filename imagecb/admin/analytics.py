"""Search quality analytics over blob/S3 telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from imagecb.config import SETTINGS
from imagecb.telemetry import s3_store


def _display_query(row: dict) -> str:
    """Primary label for admin tables: semantic intent for chat, raw label for similar."""
    user = (row.get("query_text") or "").strip()
    semantic = (row.get("parsed_semantic_query") or "").strip()
    if row.get("search_kind") == "similar":
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


def _event_dict(row: dict, *, category: str) -> dict:
    served = row.get("served_image_ids") or []
    if not isinstance(served, list):
        served = []
    timings = row.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    user_message = (row.get("query_text") or "").strip()
    created = row.get("created_at")
    if isinstance(created, datetime):
        created = created.isoformat()
    return {
        "search_event_id": row.get("id"),
        "created_at": created,
        "query_text": row.get("query_text"),
        "user_message": user_message,
        "display_query": _display_query(row),
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id"),
        "search_kind": row.get("search_kind"),
        "served_image_ids": [str(x) for x in served],
        "result_count": int(row.get("result_count") or 0),
        "top_score": row.get("top_score"),
        "top_score_kind": row.get("top_score_kind"),
        "parsed_semantic_query": row.get("parsed_semantic_query"),
        "total_ms": row.get("total_ms"),
        "ask_ms": row.get("ask_ms"),
        "reply_ms": row.get("reply_ms"),
        "timings": timings,
        "timing_log": row.get("timing_log"),
        "category": category,
    }


def search_quality_lists(
    *,
    since: Optional[str] = None,
    limit: int = 50,
    weak_score_threshold: Optional[float] = None,
) -> Dict[str, List[dict]]:
    threshold = (
        weak_score_threshold
        if weak_score_threshold is not None
        else SETTINGS.weak_result_score_threshold
    )
    since_dt = _parse_since(since)
    if since_dt is None:
        since_dt = s3_store.retention_cutoff()

    cache_key = f"{since_dt.isoformat()}|{limit}|{threshold}"
    cached = s3_store.get_quality_cache(cache_key)
    if cached is not None:
        return cached

    events = sorted(
        s3_store.iter_search_events(since=since_dt),
        key=lambda e: e.get("created_at") or "",
        reverse=True,
    )
    interacted = {
        str(e.get("id"))
        for e in events
        if e.get("has_interaction")
    }
    # Interactions may exist without has_interaction if flag write failed — merge
    interacted |= {
        str(i["search_event_id"])
        for i in s3_store.iter_interaction_events(since=since_dt)
        if i.get("search_event_id")
    }

    zero_rows: list[dict] = []
    weak_rows: list[dict] = []
    no_ix_rows: list[dict] = []
    recent_rows: list[dict] = []

    for row in events:
        result_count = int(row.get("result_count") or 0)
        top_score = row.get("top_score")
        eid = str(row.get("id") or "")

        if len(recent_rows) < limit:
            recent_rows.append(_event_dict(row, category="recent"))
        if result_count == 0 and len(zero_rows) < limit:
            zero_rows.append(_event_dict(row, category="zero_result"))
        if (
            result_count > 0
            and top_score is not None
            and float(top_score) < threshold
            and len(weak_rows) < limit
        ):
            weak_rows.append(_event_dict(row, category="weak_result"))
        if (
            result_count > 0
            and eid not in interacted
            and len(no_ix_rows) < limit
        ):
            no_ix_rows.append(_event_dict(row, category="no_interaction"))

        if (
            len(recent_rows) >= limit
            and len(zero_rows) >= limit
            and len(weak_rows) >= limit
            and len(no_ix_rows) >= limit
        ):
            break

    payload = {
        "recent": recent_rows,
        "zero_result": zero_rows,
        "weak_result": weak_rows,
        "no_interaction": no_ix_rows,
        "weak_score_threshold": threshold,
    }
    s3_store.set_quality_cache(cache_key, payload)
    return payload


def funnel_detail(search_event_id: str) -> Optional[dict]:
    row = s3_store.get_search_event(search_event_id)
    if row is None:
        return None

    created = s3_store._parse_created_at(row.get("created_at"))
    since = created or s3_store.retention_cutoff()
    interactions = [
        i
        for i in s3_store.iter_interaction_events(since=since - timedelta(days=1))
        if str(i.get("search_event_id")) == search_event_id
    ]
    interactions.sort(key=lambda i: i.get("created_at") or "")

    served = row.get("served_image_ids") or []
    if not isinstance(served, list):
        served = []

    return {
        "search": _event_dict(row, category="search"),
        "interactions": [
            {
                "id": i.get("id"),
                "image_id": i.get("image_id"),
                "interaction_type": i.get("interaction_type"),
                "created_at": i.get("created_at"),
                "user_id": i.get("user_id"),
                "rank": i.get("rank"),
            }
            for i in interactions
        ],
        "served_image_ids": [str(x) for x in served],
    }


def analytics_summary(
    *,
    since: Optional[str] = None,
    days: Optional[int] = None,
) -> dict[str, Any]:
    if days is None:
        days = SETTINGS.telemetry_default_window_days
    days = max(1, min(int(days), SETTINGS.telemetry_retention_days))

    since_dt = _parse_since(since)
    if since_dt is None:
        since_dt = datetime.utcnow() - timedelta(days=days)

    # Clamp to retention window
    cutoff = s3_store.retention_cutoff()
    if since_dt < cutoff:
        since_dt = cutoff

    totals = s3_store.load_rollups(since=since_dt)
    total = int(totals["total_searches"])
    zero = int(totals["zero_result_count"])
    weak = int(totals["weak_result_count"])
    with_results = int(totals["searches_with_results"])
    no_ix = int(totals["no_interaction_count"])
    interaction_count = int(totals["interaction_count"])

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
        "retention_days": SETTINGS.telemetry_retention_days,
        "window_days": days,
    }
