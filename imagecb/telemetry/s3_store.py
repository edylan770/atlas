"""Durable search/interaction telemetry on S3 (or local DATA_DIR mirror).

Layout (Athena-friendly partitions):
  {prefix}/telemetry/v1/searches/dt=YYYY-MM-DD/{event_id}.json.gz
  {prefix}/telemetry/v1/searches/id/{event_id}.json          # {"dt": "..."}
  {prefix}/telemetry/v1/interactions/dt=YYYY-MM-DD/{id}.json.gz
  {prefix}/telemetry/v1/rollups/daily/dt=YYYY-MM-DD.json
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional

from imagecb.config import SETTINGS
from imagecb.storage import blob_store as blobs

logger = logging.getLogger(__name__)

_VERSION = "v1"
_rollup_lock = threading.Lock()
_quality_cache_lock = threading.Lock()
_quality_cache: dict[str, Any] = {"key": None, "expires": 0.0, "payload": None}

EMPTY_ROLLUP: dict[str, int] = {
    "total_searches": 0,
    "zero_result_count": 0,
    "weak_result_count": 0,
    "searches_with_results": 0,
    "no_interaction_count": 0,
    "interaction_count": 0,
}


def _prefix() -> str:
    return blobs._key(SETTINGS.s3_prefix, "telemetry", _VERSION)


def search_event_key(dt: str, event_id: str) -> str:
    return blobs._key(_prefix(), "searches", f"dt={dt}", f"{event_id}.json.gz")


def search_id_pointer_key(event_id: str) -> str:
    return blobs._key(_prefix(), "searches", "id", f"{event_id}.json")


def interaction_event_key(dt: str, interaction_id: str) -> str:
    return blobs._key(_prefix(), "interactions", f"dt={dt}", f"{interaction_id}.json.gz")


def daily_rollup_key(dt: str) -> str:
    return blobs._key(_prefix(), "rollups", "daily", f"dt={dt}.json")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _dt_str(when: datetime) -> str:
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when.date().isoformat()


def retention_cutoff() -> datetime:
    days = max(1, int(SETTINGS.telemetry_retention_days))
    return _utc_now() - timedelta(days=days)


def _within_retention(created_at: datetime) -> bool:
    return created_at >= retention_cutoff()


def _put_raw(key: str, data: bytes, *, content_type: str) -> None:
    blobs.put_bytes(data, key, content_type=content_type)


def _get_raw(key: str) -> Optional[bytes]:
    if SETTINGS.blob_storage_backend == "s3" and SETTINGS.s3_bucket:
        try:
            return blobs.read_bytes(blobs.s3_uri(key))
        except Exception as exc:  # noqa: BLE001
            if blobs.is_missing_blob_error(exc):
                return None
            logger.warning("telemetry get failed for %s: %s", key, exc)
            return None
    path = SETTINGS.data_dir.joinpath(*PurePosixPath(key).parts)
    if path.is_file():
        return path.read_bytes()
    return None


def _list_under(prefix: str) -> List[str]:
    normalized = prefix if prefix.endswith("/") else prefix + "/"
    if SETTINGS.blob_storage_backend == "s3" and SETTINGS.s3_bucket:
        try:
            return blobs.list_keys(normalized)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telemetry list failed for %s: %s", normalized, exc)
            return []
    root = SETTINGS.data_dir.joinpath(*PurePosixPath(normalized.rstrip("/")).parts)
    if not root.is_dir():
        return []
    keys: list[str] = []
    base_parts = PurePosixPath(normalized.rstrip("/")).parts
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(SETTINGS.data_dir)
        keys.append(str(PurePosixPath(*rel.parts)))
    # Keep only keys that start with the prefix parts
    prefix_key = str(PurePosixPath(*base_parts))
    return [k for k in keys if k == prefix_key or k.startswith(prefix_key + "/")]


def put_gzip_json(key: str, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    _put_raw(key, gzip.compress(raw, compresslevel=6), content_type="application/gzip")


def get_gzip_json(key: str) -> Optional[dict]:
    data = _get_raw(key)
    if data is None:
        return None
    try:
        if key.endswith(".gz") or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        loaded = json.loads(data.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("telemetry decode failed for %s: %s", key, exc)
        return None


def put_json(key: str, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    _put_raw(key, raw, content_type="application/json")


def get_json(key: str) -> Optional[dict]:
    data = _get_raw(key)
    if data is None:
        return None
    try:
        loaded = json.loads(data.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("telemetry json decode failed for %s: %s", key, exc)
        return None


def _parse_created_at(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def resolve_search_dt(event_id: str) -> Optional[str]:
    pointer = get_json(search_id_pointer_key(event_id))
    if pointer and isinstance(pointer.get("dt"), str):
        return pointer["dt"]
    return None


def get_search_event(event_id: str) -> Optional[dict]:
    dt = resolve_search_dt(event_id)
    if not dt:
        return None
    return get_gzip_json(search_event_key(dt, event_id))


def put_search_event(event: dict) -> None:
    event_id = str(event["id"])
    created = _parse_created_at(event.get("created_at")) or _utc_now()
    dt = _dt_str(created)
    event = dict(event)
    event["created_at"] = created.isoformat()
    put_gzip_json(search_event_key(dt, event_id), event)
    put_json(search_id_pointer_key(event_id), {"dt": dt})


def put_interaction_event(event: dict) -> None:
    interaction_id = str(event["id"])
    created = _parse_created_at(event.get("created_at")) or _utc_now()
    dt = _dt_str(created)
    event = dict(event)
    event["created_at"] = created.isoformat()
    put_gzip_json(interaction_event_key(dt, interaction_id), event)


def load_daily_rollup(dt: str) -> dict[str, int]:
    loaded = get_json(daily_rollup_key(dt))
    if not loaded:
        return dict(EMPTY_ROLLUP)
    out = dict(EMPTY_ROLLUP)
    for key in EMPTY_ROLLUP:
        try:
            out[key] = int(loaded.get(key, 0) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


def bump_daily_rollup(dt: str, deltas: Dict[str, int]) -> dict[str, int]:
    with _rollup_lock:
        current = load_daily_rollup(dt)
        for key, delta in deltas.items():
            if key not in current:
                current[key] = 0
            current[key] = int(current[key]) + int(delta)
            if current[key] < 0:
                current[key] = 0
        payload = {
            **current,
            "dt": dt,
            "updated_at": _utc_now().isoformat(),
        }
        put_json(daily_rollup_key(dt), payload)
        return current


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _parse_dt_token(token: str) -> Optional[date]:
    """Parse ``dt=YYYY-MM-DD`` or ``dt=YYYY-MM-DD.json`` partition tokens."""
    if not token.startswith("dt="):
        return None
    raw = token[3:]
    if raw.endswith(".json"):
        raw = raw[: -len(".json")]
    if raw.endswith(".json.gz"):
        raw = raw[: -len(".json.gz")]
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _existing_dates_under(kind_prefix: str, *, start: date, end: date) -> list[date]:
    """Return partition dates that already have objects (avoids 90 empty GETs)."""
    found: set[date] = set()
    for key in _list_under(kind_prefix):
        for part in PurePosixPath(key).parts:
            parsed = _parse_dt_token(part)
            if parsed is None:
                continue
            if start <= parsed <= end:
                found.add(parsed)
    return sorted(found)


def load_rollups(*, since: datetime, until: Optional[datetime] = None) -> dict[str, int]:
    end = until or _utc_now()
    start = max(since, retention_cutoff())
    totals = dict(EMPTY_ROLLUP)
    if start > end:
        return totals
    rollup_prefix = blobs._key(_prefix(), "rollups", "daily")
    days = _existing_dates_under(rollup_prefix, start=start.date(), end=end.date())
    for day in days:
        rollup = load_daily_rollup(day.isoformat())
        for key in EMPTY_ROLLUP:
            totals[key] += int(rollup.get(key, 0) or 0)
    return totals


def _iter_partition_events(kind: str, day: date) -> Iterator[dict]:
    dt = day.isoformat()
    prefix = blobs._key(_prefix(), kind, f"dt={dt}")
    for key in _list_under(prefix):
        if not key.endswith(".json.gz"):
            continue
        event = get_gzip_json(key)
        if event:
            yield event


def iter_search_events(*, since: datetime, until: Optional[datetime] = None) -> Iterator[dict]:
    end = until or _utc_now()
    start = max(since, retention_cutoff())
    if start > end:
        return
    search_prefix = blobs._key(_prefix(), "searches")
    days = _existing_dates_under(search_prefix, start=start.date(), end=end.date())
    for day in days:
        for event in _iter_partition_events("searches", day):
            created = _parse_created_at(event.get("created_at"))
            if created is None or created < start or created > end:
                continue
            if not _within_retention(created):
                continue
            yield event


def iter_interaction_events(
    *, since: datetime, until: Optional[datetime] = None
) -> Iterator[dict]:
    end = until or _utc_now()
    start = max(since, retention_cutoff())
    if start > end:
        return
    interaction_prefix = blobs._key(_prefix(), "interactions")
    days = _existing_dates_under(interaction_prefix, start=start.date(), end=end.date())
    for day in days:
        for event in _iter_partition_events("interactions", day):
            created = _parse_created_at(event.get("created_at"))
            if created is None or created < start or created > end:
                continue
            yield event


def interacted_search_ids(*, since: Optional[datetime] = None) -> set[str]:
    """Search event ids that have at least one interaction (optionally since)."""
    if since is None:
        since = retention_cutoff()
    ids: set[str] = set()
    for event in iter_interaction_events(since=since):
        sid = event.get("search_event_id")
        if sid:
            ids.add(str(sid))
    # Also trust has_interaction on search docs inside the window
    for event in iter_search_events(since=since):
        if event.get("has_interaction"):
            ids.add(str(event["id"]))
    return ids


def all_served_image_ids() -> set[str]:
    served: set[str] = set()
    for event in iter_search_events(since=retention_cutoff()):
        ids = event.get("served_image_ids") or []
        if isinstance(ids, list):
            served.update(str(x) for x in ids)
    return served


def all_interacted_image_ids() -> set[str]:
    ids: set[str] = set()
    for event in iter_interaction_events(since=retention_cutoff()):
        image_id = event.get("image_id")
        if image_id:
            ids.add(str(image_id))
    return ids


def invalidate_quality_cache() -> None:
    with _quality_cache_lock:
        _quality_cache["key"] = None
        _quality_cache["expires"] = 0.0
        _quality_cache["payload"] = None


def get_quality_cache(cache_key: str) -> Optional[Any]:
    with _quality_cache_lock:
        if _quality_cache["key"] == cache_key and time.monotonic() < float(
            _quality_cache["expires"]
        ):
            return _quality_cache["payload"]
    return None


def set_quality_cache(cache_key: str, payload: Any, *, ttl_sec: float = 60.0) -> None:
    with _quality_cache_lock:
        _quality_cache["key"] = cache_key
        _quality_cache["expires"] = time.monotonic() + ttl_sec
        _quality_cache["payload"] = payload
