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

from sqlalchemy import select, update as sa_update

from imagecb.config import SETTINGS
from imagecb.storage.metadata_db import IngestJob, get_engine, session_scope

logger = logging.getLogger(__name__)

# Worker / processing statuses (excludes staging — uploads still in progress).
ACTIVE_STATUSES = {"queued", "running", "cancel_requested"}
# Statuses the UI may cancel (includes staging uploads).
CANCELLABLE_STATUSES = {"staging", "queued", "running", "cancel_requested"}
TERMINAL_STATUSES = {"cancelled", "succeeded", "failed"}
_job_update_lock = threading.RLock()


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
    upload_manifest = _loads(job.upload_manifest_json, [])
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
        "completed_batches": _loads(job.completed_batches_json, []),
        "uploads_total": len(upload_manifest),
        "upload_bytes_total": sum(int(item.get("size", 0)) for item in upload_manifest),
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
            (
                "completed_batches_json",
                "ALTER TABLE ingest_jobs ADD COLUMN completed_batches_json TEXT",
            ),
            (
                "upload_manifest_json",
                "ALTER TABLE ingest_jobs ADD COLUMN upload_manifest_json TEXT",
            ),
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
    upload_manifest: Optional[list[dict]] = None,
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
        completed_batches_json="[]",
        upload_manifest_json=json.dumps(upload_manifest or []),
        files_total=len(upload_manifest) if upload_manifest is not None else len(files),
        phase=phase,
        status_detail=status_detail,
        created_at=now,
        heartbeat_at=now,
    )
    with session_scope() as session:
        session.add(record)
        session.flush()
        created = job_to_dict(record)
    if status == "queued":
        wake_worker()
    return created


def finalize_s3_job(job_id: str, files: list[str]) -> Optional[dict]:
    """Atomically attach validated S3 sources and queue a staging job."""
    ensure_job_schema()
    with _job_update_lock:
        with session_scope() as session:
            row = session.get(IngestJob, job_id)
            if row is None:
                return None
            manifest = _loads(row.upload_manifest_json, [])
            if not manifest:
                raise ValueError("job does not have a direct-upload manifest")
            if row.status != "staging":
                if row.status in ACTIVE_STATUSES | TERMINAL_STATUSES:
                    return job_to_dict(row)
                raise ValueError(f"job {job_id} cannot be finalized (status={row.status})")
            if len(files) != len(manifest):
                raise ValueError("not all manifest files were uploaded")
            row.files_json = json.dumps(files)
            row.files_total = len(files)
            row.status = "queued"
            row.phase = "queued"
            row.status_detail = "Uploads verified; waiting for the ingest worker"
            row.heartbeat_at = datetime.utcnow()
            result = job_to_dict(row)
    wake_worker()
    return result


def append_job_files(
    job_id: str,
    files: list[Path],
    *,
    stage_errors: Optional[list[str]] = None,
) -> Optional[dict]:
    """Append staged paths to a staging job. Returns None if job is missing."""
    job, _ = append_job_batch(
        job_id,
        files,
        stage_errors=stage_errors,
        batch_id=None,
    )
    return job


def append_job_batch(
    job_id: str,
    files: list[Path],
    *,
    stage_errors: Optional[list[str]] = None,
    batch_id: Optional[str],
) -> tuple[Optional[dict], bool]:
    """Atomically append one upload batch.

    Returns ``(job, accepted)``. A repeated batch ID is successful but is not
    appended again, which makes client retries safe after an uncertain timeout.
    """
    ensure_job_schema()
    accepted = False
    with _job_update_lock:
        with session_scope() as session:
            row = session.get(IngestJob, job_id)
            if row is None:
                return None, False
            if row.status != "staging":
                raise ValueError(f"job {job_id} is not staging (status={row.status})")
            completed_batches = _loads(row.completed_batches_json, [])
            if batch_id and batch_id in completed_batches:
                return job_to_dict(row), False
            existing = _loads(row.files_json, [])
            existing.extend(str(path.resolve()) for path in files)
            row.files_json = json.dumps(existing)
            row.files_total = len(existing)
            errors = _loads(row.stage_errors_json, [])
            if stage_errors:
                errors.extend(stage_errors)
            row.stage_errors_json = json.dumps(errors)
            if batch_id:
                completed_batches.append(batch_id)
                row.completed_batches_json = json.dumps(completed_batches)
            row.heartbeat_at = datetime.utcnow()
            row.status_detail = (
                f"Staged {row.files_total} file(s); waiting for more uploads or start"
            )
            accepted = True
            result = job_to_dict(row)
    return result, accepted


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


def get_upload_manifest(job_id: str) -> Optional[list[dict]]:
    ensure_job_schema()
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return None
        return _loads(row.upload_manifest_json, [])


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


def list_busy_jobs() -> list[dict]:
    """Return ingest jobs that must block maintenance (walk-away uploads in flight)."""
    ensure_job_schema()
    with session_scope() as session:
        rows = (
            session.execute(
                select(IngestJob).where(IngestJob.status.in_(sorted(CANCELLABLE_STATUSES)))
            )
            .scalars()
            .all()
        )
        return [job_to_dict(row) for row in rows]


def cancel_stale_jobs_after_restore() -> int:
    """Cancel jobs resurrected from a restored snapshot's database.

    Their staged files no longer exist; requeueing them just burns a chunk of
    failures. Called by index restore after the archive is applied.
    """
    ensure_job_schema()
    now = datetime.utcnow()
    with session_scope() as session:
        rows = (
            session.execute(
                select(IngestJob).where(
                    IngestJob.status.in_(
                        ["staging", "queued", "running", "cancel_requested"]
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            row.status = "cancelled"
            row.phase = "cancelled"
            row.status_detail = "Cancelled: job predates the index restore"
            row.finished_at = now
        return len(rows)


def busy_staging_uris() -> set[str]:
    """Staging object URIs referenced by busy ingest jobs (must not be GC'd)."""
    uris: set[str] = set()
    for job in list_busy_jobs():
        for item in get_upload_manifest(job["job_id"]) or []:
            uri = (item.get("uri") or "").strip()
            if uri:
                uris.add(uri)
    return uris


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
        if SETTINGS.blob_storage_backend == "s3":
            _cleanup_staging(job_id, "cancelled")
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


def _claim_next_job(runner_id: Optional[str] = None) -> Optional[tuple[str, list[str], dict]]:
    now = datetime.utcnow()
    with session_scope() as session:
        candidate = session.execute(
            select(IngestJob.job_id)
            .where(IngestJob.status == "queued")
            .order_by(IngestJob.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
        if candidate is None:
            return None
        # Compare-and-swap: only the runner that flips queued->running owns the
        # job (safe if a second process ever runs a claimer).
        claimed = session.execute(
            sa_update(IngestJob)
            .where(IngestJob.job_id == candidate, IngestJob.status == "queued")
            .values(
                status="running",
                phase="claimed",
                status_detail="Claimed by the ingest worker",
                runner_id=runner_id,
                heartbeat_at=now,
                error=None,
            )
        )
        if claimed.rowcount != 1:
            return None
        row = session.get(IngestJob, candidate)
        if row.started_at is None:
            row.started_at = now
        return (
            row.job_id,
            [str(path) for path in _loads(row.files_json, [])],
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
        if stats is None:
            # Preserve cumulative multi-chunk progress: a generic failure used
            # to wipe stats_json and report 0 images processed.
            stats = _loads(row.stats_json, {}) or {}
        row.status = status
        row.phase = status
        row.status_detail = error or status.replace("_", " ")
        row.stats_json = json.dumps(stats)
        row.error = error
        row.files_done = row.files_total if status == "succeeded" else row.files_done
        row.images_seen = int(stats.get("images_seen", row.images_seen) or row.images_seen)
        computed = int(
            stats.get("images_added", 0)
            + stats.get("images_updated", 0)
            + stats.get("skipped_duplicates", 0)
            + stats.get("errors", 0)
        )
        row.images_processed = computed if stats else row.images_processed
        row.finished_at = now
        row.heartbeat_at = now


_CHUNK_STAT_KEYS = (
    "images_seen",
    "images_added",
    "images_updated",
    "skipped_duplicates",
    "errors",
    "timeouts",
    "captions_weak",
    "captions_failed",
    "checkpoint_errors",
)


def _merge_chunk_stats(prior: dict, chunk: dict) -> dict:
    """Accumulate numeric ingest stats across job chunks."""
    out = dict(prior or {})
    for key in _CHUNK_STAT_KEYS:
        out[key] = int(out.get(key, 0) or 0) + int((chunk or {}).get(key, 0) or 0)
    out["elapsed_sec"] = round(
        float(out.get("elapsed_sec", 0) or 0) + float((chunk or {}).get("elapsed_sec", 0) or 0),
        1,
    )
    out["chunks_completed"] = int(out.get("chunks_completed", 0) or 0) + 1
    if (chunk or {}).get("last_error"):
        out["last_error"] = chunk["last_error"]
    if (chunk or {}).get("last_checkpoint_id"):
        out["last_checkpoint_id"] = chunk["last_checkpoint_id"]
    if (chunk or {}).get("last_checkpoint_records") is not None:
        out["last_checkpoint_records"] = chunk["last_checkpoint_records"]
    if (chunk or {}).get("last_checkpoint_error"):
        out["last_checkpoint_error"] = chunk["last_checkpoint_error"]
    if (chunk or {}).get("post_repair") is not None:
        out["post_repair"] = chunk["post_repair"]
    return out


def _persist_job_options(job_id: str, options: dict, *, files_total: Optional[int] = None) -> None:
    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        if row is None:
            return
        row.options_json = json.dumps(options)
        if files_total is not None:
            row.files_total = int(files_total)


def _requeue_with_remaining(
    job_id: str,
    *,
    remaining: list[str],
    cumulative_stats: dict,
    original_files_total: int,
    options: dict,
) -> bool:
    """Requeue the same job with remaining files. Returns False if cancel won."""
    files_done = max(0, int(original_files_total) - len(remaining))
    now = datetime.utcnow()
    with _job_update_lock:
        with session_scope() as session:
            row = session.get(IngestJob, job_id)
            if row is None:
                return False
            if row.status == "cancel_requested":
                return False
            row.files_json = json.dumps(list(remaining))
            row.files_total = int(original_files_total)
            row.files_done = files_done
            row.options_json = json.dumps(options)
            row.stats_json = json.dumps(cumulative_stats)
            row.images_seen = int(cumulative_stats.get("images_seen", 0) or 0)
            row.images_processed = int(
                (cumulative_stats.get("images_added", 0) or 0)
                + (cumulative_stats.get("images_updated", 0) or 0)
                + (cumulative_stats.get("skipped_duplicates", 0) or 0)
                + (cumulative_stats.get("errors", 0) or 0)
            )
            row.status = "queued"
            row.phase = "queued"
            row.status_detail = f"Chunk complete; {len(remaining)} file(s) remaining"
            row.runner_id = None
            row.error = None
            row.finished_at = None
            row.heartbeat_at = now
    return True


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
    if SETTINGS.blob_storage_backend == "s3":
        job = get_job(job_id)
        if job is not None:
            with session_scope() as session:
                row = session.get(IngestJob, job_id)
                manifest = _loads(row.upload_manifest_json, []) if row else []
            from imagecb.storage import blob_store

            for item in manifest:
                try:
                    blob_store.delete(item.get("uri"))
                except Exception:  # noqa: BLE001
                    logger.warning("Could not clean staged S3 object for job %s", job_id)
        shutil.rmtree(job_stage_dir(job_id), ignore_errors=True)


class IngestJobRunner:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.runner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def start(self) -> None:
        # Clear the stop flag first: a prior stop() may have set it while a
        # claimed job was finishing; without this the still-alive thread exits
        # after that job and is never restarted.
        self._stop.clear()
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
            try:
                claimed = _claim_next_job(self.runner_id)
            except Exception:  # noqa: BLE001
                # A transient SQLite error (lock contention, engine freeze)
                # must not kill the daemon thread and strand queued jobs.
                logger.exception("Job claim failed; retrying shortly")
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            if claimed is None:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            try:
                self._execute(*claimed)
            except Exception:  # noqa: BLE001
                logger.exception("Job execution crashed; runner continues")

    def _execute(self, job_id: str, files: list[str], options: dict) -> None:
        from imagecb.ingest import IngestInProgressError, ingest_paths_batched
        from imagecb.storage.index_backup import maybe_checkpoint_progress

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
            options = dict(options or {})
            prior_stats: dict = {}
            with session_scope() as session:
                row = session.get(IngestJob, job_id)
                if row is not None:
                    prior_stats = _loads(row.stats_json, {}) or {}
                    stored_options = _loads(row.options_json, {}) or {}
                    for key, value in stored_options.items():
                        options.setdefault(key, value)

            original_total = int(options.get("original_files_total") or 0)
            if original_total <= 0:
                original_total = len(files)
                options["original_files_total"] = original_total
                _persist_job_options(job_id, options, files_total=original_total)
            else:
                _persist_job_options(job_id, options, files_total=original_total)

            chunk_size = max(1, int(SETTINGS.ingest_job_chunk_size))
            if options.get("chunk_size") is not None:
                chunk_size = max(1, int(options["chunk_size"]))
            current = list(files[:chunk_size])
            remaining = list(files[chunk_size:])
            already_done = max(0, original_total - len(files))

            _set_phase(
                job_id,
                "checkpointing_index",
                f"Saving durable index checkpoint before chunk "
                f"({already_done + 1}-{already_done + len(current)} of {original_total})",
            )
            maybe_checkpoint_progress(
                dict(prior_stats)
                if prior_stats
                else {"images_added": 0, "images_updated": 0, "_checkpoint_at": 0},
                job_id=job_id,
                force=True,
                label=f"chunk-start:{job_id}:{already_done}",
            )
            _set_phase(
                job_id,
                "preparing",
                f"Preparing chunk of {len(current)} file(s) "
                f"({already_done + len(current)}/{original_total})",
            )

            def progress_callback(progress: dict) -> None:
                adjusted = dict(progress)
                adjusted["files_done"] = already_done + int(progress.get("files_done", 0) or 0)
                chunk_stats = progress.get("stats") or {}
                merged = _merge_chunk_stats(prior_stats, chunk_stats)
                # progress merge uses a provisional +1 chunk; undo for live preview
                merged["chunks_completed"] = int(prior_stats.get("chunks_completed", 0) or 0)
                adjusted["stats"] = merged
                adjusted["images_seen"] = int(merged.get("images_seen", 0) or 0)
                adjusted["images_processed"] = int(
                    (merged.get("images_added", 0) or 0)
                    + (merged.get("images_updated", 0) or 0)
                    + (merged.get("skipped_duplicates", 0) or 0)
                    + (merged.get("errors", 0) or 0)
                )
                _update_progress(job_id, adjusted)

            stats = ingest_paths_batched(
                current,
                batch_size=max(1, int(options.get("batch_size", 25))),
                skip_caption=bool(options.get("skip_caption", False)),
                skip_ocr=bool(options.get("skip_ocr", False)),
                force=bool(options.get("force", False)),
                workers=int(options.get("workers") or SETTINGS.ingest_workers),
                should_cancel=lambda: _cancel_requested(job_id),
                progress_callback=progress_callback,
                phase_callback=lambda phase, detail=None: _set_phase(job_id, phase, detail),
                checkpoint_job_id=job_id,
            )
            cumulative = _merge_chunk_stats(prior_stats, stats)
            chunk_processed = (
                int(stats.get("images_added", 0))
                + int(stats.get("images_updated", 0))
                + int(stats.get("skipped_duplicates", 0))
            )
            chunk_errors = int(stats.get("errors", 0))
            if stats.get("cancelled"):
                status = "cancelled"
                error = None
            elif chunk_errors and chunk_processed == 0:
                status = "failed"
                error = str(
                    stats.get("last_error")
                    or (
                        f"All ingestion work failed ({chunk_errors} error(s)); "
                        "check the recorded phase and dependency preflight"
                    )
                )
            else:
                status = "succeeded"
                error = None

            if status in {"cancelled", "failed"}:
                _finish_job(job_id, status=status, stats=cumulative, error=error)
                if status == "cancelled":
                    _cleanup_staging(job_id, status)
                return

            # Chunk succeeded (possibly with some per-file errors).
            if remaining:
                if _cancel_requested(job_id):
                    _finish_job(job_id, status="cancelled", stats=cumulative)
                    _cleanup_staging(job_id, "cancelled")
                    return
                maybe_checkpoint_progress(
                    cumulative,
                    job_id=job_id,
                    force=True,
                    label=f"chunk-complete:{job_id}:{already_done + len(current)}",
                )
                requeued = _requeue_with_remaining(
                    job_id,
                    remaining=remaining,
                    cumulative_stats=cumulative,
                    original_files_total=original_total,
                    options=options,
                )
                if not requeued:
                    _finish_job(job_id, status="cancelled", stats=cumulative)
                    _cleanup_staging(job_id, "cancelled")
                    return
                logger.info(
                    "Ingest job %s chunk complete; requeued with %s file(s) remaining",
                    job_id,
                    len(remaining),
                )
                self.wake()
                return

            _finish_job(job_id, status="succeeded", stats=cumulative, error=None)
            maybe_checkpoint_progress(
                cumulative,
                job_id=job_id,
                force=True,
                label=f"job-success:{job_id}",
            )
            _cleanup_staging(job_id, "succeeded")
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
