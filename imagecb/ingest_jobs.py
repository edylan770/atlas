"""Durable, single-worker ingest job management."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from imagecb.config import SETTINGS
from imagecb.storage.metadata_db import IngestJob, get_engine, session_scope

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"queued", "running", "cancel_requested"}
TERMINAL_STATUSES = {"cancelled", "succeeded", "failed"}


def _loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def job_to_dict(job: IngestJob) -> dict:
    files = _loads(job.files_json, [])
    return {
        "job_id": job.job_id,
        "status": job.status,
        "files": [Path(str(path)).name for path in files],
        "files_total": job.files_total,
        "files_done": job.files_done,
        "images_seen": job.images_seen,
        "images_processed": job.images_processed,
        "options": _loads(job.options_json, {}),
        "stats": _loads(job.stats_json, {}),
        "stage_errors": _loads(job.stage_errors_json, []),
        "error": job.error,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "heartbeat_at": _iso(job.heartbeat_at),
        "cancel_requested_at": _iso(job.cancel_requested_at),
        "cancellable": job.status in ACTIVE_STATUSES,
    }


def new_job_id() -> str:
    return str(uuid.uuid4())


def job_stage_dir(job_id: str) -> Path:
    return SETTINGS.data_dir / "ingest_jobs" / job_id


def ensure_job_schema() -> None:
    # IngestJob is registered on the shared metadata before create_all runs.
    engine = get_engine()
    IngestJob.__table__.create(engine, checkfirst=True)


def create_job(
    job_id: str,
    files: list[Path],
    options: dict,
    *,
    stage_errors: Optional[list[str]] = None,
) -> dict:
    ensure_job_schema()
    now = datetime.utcnow()
    record = IngestJob(
        job_id=job_id,
        status="queued",
        files_json=json.dumps([str(path.resolve()) for path in files]),
        options_json=json.dumps(options),
        stats_json="{}",
        stage_errors_json=json.dumps(stage_errors or []),
        files_total=len(files),
        created_at=now,
        heartbeat_at=now,
    )
    with session_scope() as session:
        session.add(record)
    wake_worker()
    return get_job(job_id) or {}


def get_job(job_id: str) -> Optional[dict]:
    ensure_job_schema()
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        return job_to_dict(row)


def list_jobs(*, limit: int = 100) -> list[dict]:
    ensure_job_schema()
    with session_scope() as session:
        rows = (
            session.execute(
                select(IngestJob)
                .order_by(IngestJob.created_at.desc())
                .limit(max(1, min(limit, 500)))
            )
            .scalars()
            .all()
        )
        return [job_to_dict(row) for row in rows]


def request_cancel(job_id: str) -> Optional[dict]:
    ensure_job_schema()
    now = datetime.utcnow()
    cancelled_while_queued = False
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        if row.status == "queued":
            cancelled_while_queued = True
            row.status = "cancelled"
            row.cancel_requested_at = now
            row.finished_at = now
            row.heartbeat_at = now
        elif row.status == "running":
            row.status = "cancel_requested"
            row.cancel_requested_at = now
            row.heartbeat_at = now
        result = job_to_dict(row)
    if cancelled_while_queued:
        shutil.rmtree(job_stage_dir(job_id), ignore_errors=True)
    wake_worker()
    return result


def _recover_interrupted_jobs() -> None:
    now = datetime.utcnow()
    with session_scope() as session:
        rows = (
            session.execute(
                select(IngestJob).where(
                    IngestJob.status.in_(["running", "cancel_requested"])
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            if row.status == "cancel_requested":
                row.status = "cancelled"
                row.finished_at = now
            else:
                row.status = "queued"
                row.error = "Recovered after server restart"
            row.heartbeat_at = now


def _claim_next_job() -> Optional[tuple[str, list[Path], dict]]:
    now = datetime.utcnow()
    with session_scope() as session:
        row = session.execute(
            select(IngestJob)
            .where(IngestJob.status == "queued")
            .order_by(IngestJob.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = "running"
        row.started_at = row.started_at or now
        row.heartbeat_at = now
        row.error = None
        return (
            row.job_id,
            [Path(path) for path in _loads(row.files_json, [])],
            _loads(row.options_json, {}),
        )


def _cancel_requested(job_id: str) -> bool:
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        return row is None or row.status == "cancel_requested"


def _update_progress(job_id: str, progress: dict) -> None:
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None or row.status not in {"running", "cancel_requested"}:
            return
        row.files_done = int(progress.get("files_done", row.files_done))
        row.images_seen = int(progress.get("images_seen", row.images_seen))
        row.images_processed = int(
            progress.get("images_processed", row.images_processed)
        )
        row.stats_json = json.dumps(progress.get("stats", {}))
        row.heartbeat_at = datetime.utcnow()


def _finish_job(
    job_id: str,
    *,
    status: str,
    stats: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    now = datetime.utcnow()
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return
        row.status = status
        row.stats_json = json.dumps(stats or {})
        row.error = error
        row.files_done = row.files_total if status == "succeeded" else row.files_done
        row.images_seen = int((stats or {}).get("images_seen", row.images_seen))
        row.images_processed = int(
            (stats or {}).get("images_added", 0)
            + (stats or {}).get("images_updated", 0)
            + (stats or {}).get("skipped_duplicates", 0)
            + (stats or {}).get("errors", 0)
        )
        row.finished_at = now
        row.heartbeat_at = now


def _requeue_job(job_id: str, error: str) -> None:
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return
        if row.status == "cancel_requested":
            row.status = "cancelled"
            row.finished_at = datetime.utcnow()
        else:
            row.status = "queued"
            row.error = error
        row.heartbeat_at = datetime.utcnow()


def _cleanup_staging(job_id: str, status: str) -> None:
    if status not in {"succeeded", "cancelled"}:
        return
    # Local provenance points at the staged source, so only S3-backed jobs can
    # safely remove their local staging directory after persist_source().
    if SETTINGS.blob_storage_backend == "s3":
        shutil.rmtree(job_stage_dir(job_id), ignore_errors=True)


class IngestJobRunner:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        ensure_job_schema()
        _recover_interrupted_jobs()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ingest-job-runner",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)

    def wake(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            claimed = _claim_next_job()
            if claimed is None:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            self._execute(*claimed)

    def _execute(self, job_id: str, files: list[Path], options: dict) -> None:
        from imagecb.ingest import IngestInProgressError, ingest_paths_batched

        try:
            stats = ingest_paths_batched(
                files,
                batch_size=max(1, int(options.get("batch_size", 25))),
                skip_caption=bool(options.get("skip_caption", False)),
                skip_ocr=bool(options.get("skip_ocr", False)),
                force=bool(options.get("force", False)),
                workers=int(options.get("workers") or SETTINGS.ingest_workers),
                should_cancel=lambda: _cancel_requested(job_id),
                progress_callback=lambda progress: _update_progress(job_id, progress),
            )
            status = "cancelled" if stats.get("cancelled") else "succeeded"
            _finish_job(job_id, status=status, stats=stats)
            _cleanup_staging(job_id, status)
        except IngestInProgressError:
            _requeue_job(job_id, "Waiting for the active ingest to finish")
            self._stop.wait(timeout=1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingest job %s failed", job_id)
            _finish_job(job_id, status="failed", error=str(exc))


_RUNNER = IngestJobRunner()


def start_job_runner() -> None:
    _RUNNER.start()


def stop_job_runner() -> None:
    _RUNNER.stop()


def wake_worker() -> None:
    _RUNNER.wake()
