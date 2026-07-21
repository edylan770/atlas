"""Durable, single-worker ingest job management."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from imagecb.config import SETTINGS
from imagecb.storage.metadata_db import IngestJob, get_engine, session_scope

logger = logging.getLogger(__name__)

# Worker / processing statuses (excludes staging — uploads still in progress).
ACTIVE_STATUSES = {"queued", "running", "cancel_requested"}
# Statuses the UI may cancel (includes staging uploads).
CANCELLABLE_STATUSES = {"staging", "queued", "running", "cancel_requested"}
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
        "phase": job.phase,
        "status_detail": job.status_detail,
        "runner_id": job.runner_id,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "heartbeat_at": _iso(job.heartbeat_at),
        "cancel_requested_at": _iso(job.cancel_requested_at),
        "cancellable": job.status in CANCELLABLE_STATUSES,
    }


def new_job_id() -> str:
    return str(uuid.uuid4())


def job_stage_dir(job_id: str) -> Path:
    return SETTINGS.data_dir / "ingest_jobs" / job_id


def ensure_job_schema() -> None:
    # IngestJob is registered on the shared metadata before create_all runs.
    engine = get_engine()
    IngestJob.__table__.create(engine, checkfirst=True)
    # create_all does not add columns to an existing deployment database.
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(ingest_jobs)"
            ).fetchall()
        }
        for name, ddl in (
            ("phase", "ALTER TABLE ingest_jobs ADD COLUMN phase VARCHAR"),
            ("status_detail", "ALTER TABLE ingest_jobs ADD COLUMN status_detail TEXT"),
            ("runner_id", "ALTER TABLE ingest_jobs ADD COLUMN runner_id VARCHAR"),
        ):
            if name not in columns:
                connection.exec_driver_sql(ddl)


def create_job(
    job_id: str,
    files: list[Path],
    options: dict,
    *,
    stage_errors: Optional[list[str]] = None,
    status: str = "queued",
) -> dict:
    if status not in {"queued", "staging"}:
        raise ValueError(f"invalid create_job status: {status}")
    ensure_job_schema()
    now = datetime.utcnow()
    if status == "staging":
        phase = "staging"
        status_detail = "Waiting for remaining file uploads"
    else:
        phase = "queued"
        status_detail = "Waiting for the ingest worker"
    record = IngestJob(
        job_id=job_id,
        status=status,
        files_json=json.dumps([str(path.resolve()) for path in files]),
        options_json=json.dumps(options),
        stats_json="{}",
        stage_errors_json=json.dumps(stage_errors or []),
        files_total=len(files),
        phase=phase,
        status_detail=status_detail,
        created_at=now,
        heartbeat_at=now,
    )
    with session_scope() as session:
        session.add(record)
    if status == "queued":
        wake_worker()
    return get_job(job_id) or {}


def append_job_files(
    job_id: str,
    files: list[Path],
    *,
    stage_errors: Optional[list[str]] = None,
) -> Optional[dict]:
    """Append staged paths to a staging job. Returns None if job is missing."""
    ensure_job_schema()
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        if row.status != "staging":
            raise ValueError(f"job {job_id} is not staging (status={row.status})")
        existing = _loads(row.files_json, [])
        existing.extend(str(path.resolve()) for path in files)
        row.files_json = json.dumps(existing)
        row.files_total = len(existing)
        errors = _loads(row.stage_errors_json, [])
        if stage_errors:
            errors.extend(stage_errors)
        row.stage_errors_json = json.dumps(errors)
        row.heartbeat_at = datetime.utcnow()
        row.status_detail = f"Staged {row.files_total} file(s); waiting for more uploads or start"
    return get_job(job_id)


def start_job(job_id: str) -> Optional[dict]:
    """Move a staging job to queued so the worker can claim it."""
    ensure_job_schema()
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        if row.status != "staging":
            raise ValueError(f"job {job_id} is not staging (status={row.status})")
        files = _loads(row.files_json, [])
        if not files:
            raise ValueError(f"job {job_id} has no staged files")
        row.status = "queued"
        row.phase = "queued"
        row.status_detail = "Waiting for the ingest worker"
        row.heartbeat_at = datetime.utcnow()
    wake_worker()
    return get_job(job_id)


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
    cleanup_staging_dir = False
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        if row.status in {"queued", "staging"}:
            cleanup_staging_dir = True
            detail = (
                "Cancelled during upload staging"
                if row.status == "staging"
                else "Cancelled before the worker started"
            )
            row.status = "cancelled"
            row.phase = "cancelled"
            row.status_detail = detail
            row.cancel_requested_at = now
            row.finished_at = now
            row.heartbeat_at = now
        elif row.status == "running":
            row.status = "cancel_requested"
            row.phase = "cancel_requested"
            row.status_detail = "Cancellation requested"
            row.cancel_requested_at = now
            row.heartbeat_at = now
        result = job_to_dict(row)
    if cleanup_staging_dir:
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
                row.phase = "cancelled"
                row.status_detail = "Cancelled during server restart"
                row.finished_at = now
            else:
                row.status = "queued"
                row.error = "Recovered after server restart"
                row.phase = "queued"
                row.status_detail = "Recovered after server restart"
            row.runner_id = None
            row.heartbeat_at = now


def _claim_next_job(runner_id: Optional[str] = None) -> Optional[tuple[str, list[Path], dict]]:
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
        row.phase = "claimed"
        row.status_detail = "Claimed by the ingest worker"
        row.runner_id = runner_id
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
        if progress.get("phase"):
            row.phase = str(progress["phase"])
        if progress.get("status_detail") is not None:
            row.status_detail = str(progress["status_detail"])
        row.heartbeat_at = datetime.utcnow()


def _set_phase(job_id: str, phase: str, detail: Optional[str] = None) -> None:
    logger.info(
        "Ingest job lifecycle job_id=%s phase=%s detail=%s",
        job_id,
        phase,
        detail or "",
    )
    try:
        _update_progress(
            job_id,
            {"phase": phase, "status_detail": detail or phase.replace("_", " ")},
        )
    except Exception:  # noqa: BLE001
        logger.exception("Could not persist ingest phase job_id=%s phase=%s", job_id, phase)


def _touch_heartbeat(job_id: str) -> None:
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is not None and row.status in {"running", "cancel_requested"}:
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
        row.phase = status
        row.status_detail = error or status.replace("_", " ")
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
            row.phase = "queued"
            row.status_detail = error
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
        self.runner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

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
            claimed = _claim_next_job(self.runner_id)
            if claimed is None:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            self._execute(*claimed)

    def _execute(self, job_id: str, files: list[Path], options: dict) -> None:
        from imagecb.ingest import IngestInProgressError, ingest_paths_batched

        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(timeout=5):
                _touch_heartbeat(job_id)

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"ingest-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            _set_phase(job_id, "preparing", "Preparing ingestion dependencies")
            stats = ingest_paths_batched(
                files,
                batch_size=max(1, int(options.get("batch_size", 25))),
                skip_caption=bool(options.get("skip_caption", False)),
                skip_ocr=bool(options.get("skip_ocr", False)),
                force=bool(options.get("force", False)),
                workers=int(options.get("workers") or SETTINGS.ingest_workers),
                should_cancel=lambda: _cancel_requested(job_id),
                progress_callback=lambda progress: _update_progress(job_id, progress),
                phase_callback=lambda phase, detail=None: _set_phase(job_id, phase, detail),
            )
            processed = (
                int(stats.get("images_added", 0))
                + int(stats.get("images_updated", 0))
                + int(stats.get("skipped_duplicates", 0))
            )
            errors = int(stats.get("errors", 0))
            if stats.get("cancelled"):
                status = "cancelled"
                error = None
            elif errors and processed == 0:
                status = "failed"
                error = str(
                    stats.get("last_error")
                    or (
                        f"All ingestion work failed ({errors} error(s)); "
                        "check the recorded phase and dependency preflight"
                    )
                )
            else:
                status = "succeeded"
                error = None
            _finish_job(job_id, status=status, stats=stats, error=error)
            _cleanup_staging(job_id, status)
        except IngestInProgressError:
            _requeue_job(job_id, "Waiting for the active ingest to finish")
            self._stop.wait(timeout=1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingest job %s failed", job_id)
            _finish_job(job_id, status="failed", error=str(exc))
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)


_RUNNER = IngestJobRunner()


def runner_health() -> dict:
    thread = _RUNNER._thread
    return {
        "runner_id": _RUNNER.runner_id,
        "alive": bool(thread and thread.is_alive()),
        "thread_name": thread.name if thread else None,
    }


def start_job_runner() -> None:
    _RUNNER.start()


def stop_job_runner() -> None:
    _RUNNER.stop()


def wake_worker() -> None:
    _RUNNER.wake()
