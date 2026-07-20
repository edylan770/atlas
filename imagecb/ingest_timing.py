"""Per-step ingest timing collection and durable .txt reports."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from imagecb.config import SETTINGS
from imagecb.storage import blob_store

logger = logging.getLogger(__name__)

# Steps that count toward "% of worker time" in aggregate tables.
_WORKER_STEPS = (
    "hash_image",
    "cache_image",
    "ocr",
    "caption_vlm",
    "embed_image",
    "sqlite_write",
    "embed_text",
)


@dataclass
class ImageTimingDetail:
    image_id: str
    source_file: str
    outcome: str
    steps: Dict[str, float] = field(default_factory=dict)
    total_sec: float = 0.0
    error: Optional[str] = None


class IngestTimingSession:
    """Thread-safe collector for one ingest run."""

    def __init__(
        self,
        *,
        mode: str = "ingest",
        enabled: Optional[bool] = None,
        meta: Optional[dict] = None,
    ) -> None:
        self.enabled = SETTINGS.ingest_timing_log if enabled is None else enabled
        self.run_id = uuid.uuid4().hex[:8]
        self.mode = mode
        self.meta: dict = dict(meta or {})
        self.wall_start = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._step_samples: Dict[str, List[float]] = defaultdict(list)
        self._image_details: List[ImageTimingDetail] = []

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
            self._step_samples[step].append(float(seconds))

    def add_image_detail(self, detail: ImageTimingDetail) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._image_details.append(detail)
            for step, seconds in detail.steps.items():
                self._step_samples[step].append(float(seconds))
            if detail.total_sec > 0:
                self._step_samples["image_total"].append(float(detail.total_sec))

    def wall_elapsed(self) -> float:
        return time.perf_counter() - self.wall_start

    def format_report(self, stats: Optional[dict] = None) -> str:
        stats = stats or {}
        with self._lock:
            samples = {k: list(v) for k, v in self._step_samples.items()}
            details = list(self._image_details)

        wall = self.wall_elapsed()
        lines: List[str] = [
            "ImageCB ingest timing report",
            "=" * 72,
            f"started_at_utc: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"run_id:         {self.run_id}",
            f"mode:           {self.mode}",
            f"blob_backend:   {SETTINGS.blob_storage_backend}",
            f"s3_bucket:      {SETTINGS.s3_bucket or '-'}",
            f"s3_prefix:      {SETTINGS.s3_prefix}",
            f"workers:        {stats.get('workers', self.meta.get('workers', '-'))}",
            f"skip_caption:   {self.meta.get('skip_caption', '-')}",
            f"skip_ocr:       {self.meta.get('skip_ocr', '-')}",
            f"force:          {self.meta.get('force', '-')}",
            f"files:          {stats.get('files', 0)}",
            f"images_seen:    {stats.get('images_seen', 0)}",
            f"images_added:   {stats.get('images_added', 0)}",
            f"images_updated: {stats.get('images_updated', 0)}",
            f"duplicates:     {stats.get('skipped_duplicates', 0)}",
            f"errors:         {stats.get('errors', 0)}",
            f"wall_sec:       {wall:.3f}",
            f"elapsed_sec:    {stats.get('elapsed_sec', round(wall, 1))}",
        ]
        for key, value in sorted(self.meta.items()):
            if key in {"workers", "skip_caption", "skip_ocr", "force"}:
                continue
            lines.append(f"{key}: {value}")

        lines.extend(["", "AGGREGATE BY STEP", "-" * 72])
        header = (
            f"{'step':<22} {'count':>6} {'sum_s':>10} {'avg_s':>10} "
            f"{'p50_s':>10} {'p95_s':>10} {'%worker':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        worker_sum = sum(sum(samples.get(step, [])) for step in _WORKER_STEPS)
        ordered_steps = sorted(
            samples.keys(),
            key=lambda s: (-sum(samples[s]), s),
        )
        for step in ordered_steps:
            vals = samples[step]
            if not vals:
                continue
            total = sum(vals)
            avg = total / len(vals)
            p50 = _percentile(vals, 50)
            p95 = _percentile(vals, 95)
            pct = (100.0 * total / worker_sum) if worker_sum > 0 and step in _WORKER_STEPS else 0.0
            pct_s = f"{pct:7.1f}%" if step in _WORKER_STEPS else "       -"
            lines.append(
                f"{step:<22} {len(vals):6d} {total:10.3f} {avg:10.3f} "
                f"{p50:10.3f} {p95:10.3f} {pct_s}"
            )

        lines.extend(["", "PER-IMAGE DETAIL", "-" * 72])
        if not details:
            lines.append("(none)")
        else:
            for detail in details:
                step_bits = " ".join(
                    f"{name}={secs * 1000:.0f}ms" for name, secs in sorted(detail.steps.items())
                )
                err = f" err={detail.error}" if detail.error else ""
                lines.append(
                    f"image_id={detail.image_id} outcome={detail.outcome} "
                    f"total={detail.total_sec * 1000:.0f}ms "
                    f"source={detail.source_file} {step_bits}{err}"
                )

        lines.append("")
        return "\n".join(lines)

    def persist_report(self, stats: Optional[dict] = None) -> Optional[str]:
        """Format and upload the report. Never raises; returns URI/path or None."""
        if not self.enabled:
            return None
        self.record("wall_total", self.wall_elapsed())
        text = self.format_report(stats)
        try:
            key = blob_store.ingest_log_key(self.run_id, when=self.started_at)
            ref = blob_store.put_bytes(
                text.encode("utf-8"),
                key,
                content_type="text/plain; charset=utf-8",
            )
            logger.info("Ingest timing report written to %s", ref)
            return ref
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist ingest timing report: %s", exc)
            return None


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac
