"""Per-step query timing collection and durable .txt reports."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional

from imagecb.config import SETTINGS
from imagecb.storage import blob_store

logger = logging.getLogger(__name__)

# Fixed vocabulary for Admin + reports (see plan).
QUERY_TIMING_STEPS = (
    "parse_query",
    "metadata_filter",
    "embed_visual",
    "embed_text",
    "chroma_visual",
    "chroma_text",
    "bm25",
    "rrf_rank",
    "ask_total",
    "conversational_reply",
    "follow_ups",
    "request_total",
    "image_fetch",
)


class QueryTimingSession:
    """Collector for one chat/search request.

    Nested stages that share a name (e.g. hybrid fuse + session rank both use
    ``rrf_rank``) accumulate into a single total.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        persist: Optional[bool] = None,
        meta: Optional[dict] = None,
    ) -> None:
        self.enabled = SETTINGS.query_timing_log if enabled is None else enabled
        self.persist_enabled = (
            SETTINGS.query_timing_persist if persist is None else persist
        )
        self.run_id = uuid.uuid4().hex[:8]
        self.meta: dict = dict(meta or {})
        self.wall_start = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._steps: Dict[str, float] = {}

    @contextmanager
    def timed(self, step: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(step, time.perf_counter() - t0)

    def record(self, step: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._steps[step] = self._steps.get(step, 0.0) + float(seconds)

    def wall_elapsed(self) -> float:
        return time.perf_counter() - self.wall_start

    def timings_sec(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._steps)

    def timings_ms(self) -> Dict[str, float]:
        return {k: round(v * 1000.0, 1) for k, v in self.timings_sec().items()}

    def ask_ms(self) -> Optional[float]:
        ms = self.timings_ms()
        if "ask_total" in ms:
            return ms["ask_total"]
        return None

    def reply_ms(self) -> Optional[float]:
        ms = self.timings_ms()
        if "conversational_reply" in ms:
            return ms["conversational_reply"]
        return None

    def total_ms(self) -> Optional[float]:
        ms = self.timings_ms()
        if "request_total" in ms:
            return ms["request_total"]
        wall_ms = round(self.wall_elapsed() * 1000.0, 1)
        return wall_ms if self.enabled else None

    def format_report(self, stats: Optional[dict] = None) -> str:
        stats = stats or {}
        steps = self.timings_ms()
        wall_ms = round(self.wall_elapsed() * 1000.0, 1)
        lines = [
            "ImageCB query timing report",
            "=" * 72,
            f"started_at_utc: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"run_id:         {self.run_id}",
            f"blob_backend:   {SETTINGS.blob_storage_backend}",
            f"s3_bucket:      {SETTINGS.s3_bucket or '-'}",
            f"s3_prefix:      {SETTINGS.s3_prefix}",
            f"wall_ms:        {wall_ms}",
        ]
        for key in (
            "search_event_id",
            "session_id",
            "query_text",
            "search_kind",
            "result_count",
        ):
            if key in stats or key in self.meta:
                lines.append(f"{key}: {(stats.get(key, self.meta.get(key, '-')))}")
        for key, value in sorted(self.meta.items()):
            if key in {
                "search_event_id",
                "session_id",
                "query_text",
                "search_kind",
                "result_count",
            }:
                continue
            lines.append(f"{key}: {value}")
        for key, value in sorted(stats.items()):
            if key in {
                "search_event_id",
                "session_id",
                "query_text",
                "search_kind",
                "result_count",
            }:
                continue
            lines.append(f"{key}: {value}")

        lines.extend(["", "STEPS (ms)", "-" * 72])
        header = f"{'step':<24} {'ms':>10} {'%wall':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        ordered = sorted(steps.keys(), key=lambda s: (-steps[s], s))
        for step in ordered:
            ms = steps[step]
            pct = (100.0 * ms / wall_ms) if wall_ms > 0 else 0.0
            lines.append(f"{step:<24} {ms:10.1f} {pct:7.1f}%")
        if not ordered:
            lines.append("(none)")

        lines.append("")
        return "\n".join(lines)

    def persist_report(self, stats: Optional[dict] = None) -> Optional[str]:
        """Format and upload the report. Never raises; returns URI/path or None."""
        if not self.enabled or not self.persist_enabled:
            return None
        text = self.format_report(stats)
        event_id = str((stats or {}).get("search_event_id") or self.meta.get("search_event_id") or self.run_id)
        try:
            key = blob_store.query_log_key(event_id, when=self.started_at)
            ref = blob_store.put_bytes(
                text.encode("utf-8"),
                key,
                content_type="text/plain; charset=utf-8",
            )
            logger.info("Query timing report written to %s", ref)
            return ref
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist query timing report: %s", exc)
            return None

    def log_summary(self, *, search_event_id: Optional[str] = None) -> None:
        if not self.enabled:
            return
        ms = self.timings_ms()
        payload = {
            "search_event_id": search_event_id or self.meta.get("search_event_id"),
            "total_ms": self.total_ms(),
            "ask_ms": self.ask_ms(),
            "reply_ms": self.reply_ms(),
            "steps_ms": ms,
        }
        logger.info("query_timing %s", json.dumps(payload, default=str))


def finalize_query_timing(
    timing: QueryTimingSession,
    *,
    search_event_id: str,
    stats: Optional[dict] = None,
) -> Optional[str]:
    """Record request_total if missing, log, persist, and return timing_log ref."""
    if not timing.enabled:
        return None
    if "request_total" not in timing.timings_sec():
        timing.record("request_total", timing.wall_elapsed())
    timing.meta["search_event_id"] = search_event_id
    merged = dict(timing.meta)
    if stats:
        merged.update(stats)
    merged["search_event_id"] = search_event_id
    timing.log_summary(search_event_id=search_event_id)
    return timing.persist_report(merged)
