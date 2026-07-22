"""S3 vault for consistent SQLite + Chroma + BM25 + hubness snapshots.

The EC2 host bind-mount remains the only live writable index. S3 stores
versioned archives under ``{s3_prefix}/index-backups/{id}/`` for disaster
recovery. ``manifest.json`` is written last so incomplete uploads are never listed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from imagecb.config import SETTINGS
from imagecb.storage import blob_store

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "archive.tar.gz"
_MANIFEST_NAME = "manifest.json"
_QUIESCE_TIMEOUT_SEC = 120
_QUIESCE_POLL_SEC = 0.5


class IndexBackupError(RuntimeError):
    """Raised when backup or restore cannot proceed safely."""


def _require_s3() -> None:
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        raise IndexBackupError(
            "Index backup/restore requires BLOB_STORAGE_BACKEND=s3 and S3_BUCKET"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_backup_id() -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sqlite(db_path: Path) -> None:
    if not db_path.is_file():
        return
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()
    for suffix in ("-shm", "-wal"):
        sidecar = Path(f"{db_path}{suffix}")
        try:
            sidecar.unlink(missing_ok=True)
        except OSError as exc:
            # Windows can keep WAL sidecars locked briefly after close; the
            # checkpointed main DB is still consistent for the archive.
            logger.warning("Could not remove SQLite sidecar %s: %s", sidecar, exc)


def _cancel_active_jobs() -> list[str]:
    from imagecb.ingest_jobs import (
        ACTIVE_STATUSES,
        CANCELLABLE_STATUSES,
        list_jobs,
        request_cancel,
    )

    cancelled: list[str] = []
    for job in list_jobs(limit=500):
        status = str(job.get("status") or "")
        job_id = str(job.get("job_id") or "")
        if not job_id:
            continue
        if status in CANCELLABLE_STATUSES or status in ACTIVE_STATUSES:
            request_cancel(job_id)
            cancelled.append(job_id)
    return cancelled


def _wait_for_idle(*, timeout_sec: float = _QUIESCE_TIMEOUT_SEC) -> None:
    from imagecb.ingest_jobs import ACTIVE_STATUSES, list_jobs

    deadline = time.monotonic() + timeout_sec
    while True:
        active = [
            job
            for job in list_jobs(limit=500)
            if str(job.get("status") or "") in ACTIVE_STATUSES
        ]
        if not active:
            return
        if time.monotonic() >= deadline:
            ids = ", ".join(str(job.get("job_id")) for job in active[:5])
            raise IndexBackupError(
                f"Timed out waiting for ingest jobs to stop (still active: {ids})"
            )
        time.sleep(_QUIESCE_POLL_SEC)


@contextmanager
def _quiesce_writers() -> Iterator[dict[str, Any]]:
    """Cancel ingest work, stop the runner, then restart it afterward."""
    from imagecb.ingest_jobs import start_job_runner, stop_job_runner

    cancelled = _cancel_active_jobs()
    _wait_for_idle()
    stop_job_runner()
    meta = {"cancelled_job_ids": cancelled, "quiesced": True}
    try:
        yield meta
    finally:
        start_job_runner()


def _dispose_live_stores() -> None:
    from imagecb.retrieval import hubness
    from imagecb.storage import bm25_index, metadata_db, vector_store

    metadata_db.dispose_engine()
    vector_store.reset_client()
    bm25_index.reset_cache()
    hubness.reset_cache()


def _reopen_live_stores() -> None:
    from imagecb.retrieval import hubness
    from imagecb.storage import bm25_index, metadata_db, vector_store

    metadata_db.reopen_engine()
    vector_store.reopen_client()
    bm25_index.reload_index()
    hubness.reload_index()


def _artifact_paths() -> dict[str, Path]:
    return {
        "sqlite": Path(SETTINGS.sqlite_path),
        "chroma": Path(SETTINGS.chroma_dir),
        "bm25": Path(SETTINGS.bm25_path),
        "hubness": Path(SETTINGS.hubness_path),
    }


def _build_archive(archive_path: Path) -> dict[str, Any]:
    paths = _artifact_paths()
    sqlite_path = paths["sqlite"]
    if not sqlite_path.is_file():
        raise IndexBackupError(f"SQLite database missing: {sqlite_path}")

    _checkpoint_sqlite(sqlite_path)

    artifacts: dict[str, Any] = {}
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(sqlite_path, arcname="imagecb.db")
        artifacts["sqlite"] = {
            "present": True,
            "bytes": sqlite_path.stat().st_size,
            "arcname": "imagecb.db",
        }

        chroma_dir = paths["chroma"]
        if chroma_dir.is_dir():
            tar.add(chroma_dir, arcname="chroma")
            total = sum(p.stat().st_size for p in chroma_dir.rglob("*") if p.is_file())
            artifacts["chroma"] = {
                "present": True,
                "bytes": total,
                "arcname": "chroma",
            }
        else:
            artifacts["chroma"] = {"present": False, "bytes": 0, "arcname": "chroma"}

        for name, arcname in (("bm25", "bm25.pkl"), ("hubness", "hubness.pkl")):
            path = paths[name]
            if path.is_file():
                tar.add(path, arcname=arcname)
                artifacts[name] = {
                    "present": True,
                    "bytes": path.stat().st_size,
                    "arcname": arcname,
                }
            else:
                artifacts[name] = {
                    "present": False,
                    "bytes": 0,
                    "arcname": arcname,
                }

    return artifacts


def _replace_path(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir():
        shutil.rmtree(dest)
    elif dest.exists():
        dest.unlink()
    src.replace(dest)


def _apply_archive(archive_path: Path) -> None:
    paths = _artifact_paths()
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SETTINGS.data_dir, prefix=".index-restore-") as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        extracted_db = tmp_dir / "imagecb.db"
        if not extracted_db.is_file():
            raise IndexBackupError("Snapshot archive is missing imagecb.db")

        staging_db = tmp_dir / "staging_imagecb.db"
        shutil.copy2(extracted_db, staging_db)

        extracted_chroma = tmp_dir / "chroma"
        staging_chroma = tmp_dir / "staging_chroma"
        if extracted_chroma.is_dir():
            shutil.copytree(extracted_chroma, staging_chroma)

        staging_bm25 = tmp_dir / "staging_bm25.pkl"
        extracted_bm25 = tmp_dir / "bm25.pkl"
        if extracted_bm25.is_file():
            shutil.copy2(extracted_bm25, staging_bm25)

        staging_hubness = tmp_dir / "staging_hubness.pkl"
        extracted_hubness = tmp_dir / "hubness.pkl"
        if extracted_hubness.is_file():
            shutil.copy2(extracted_hubness, staging_hubness)

        _replace_path(staging_db, paths["sqlite"])
        for suffix in ("-shm", "-wal"):
            Path(f"{paths['sqlite']}{suffix}").unlink(missing_ok=True)

        if staging_chroma.is_dir():
            _replace_path(staging_chroma, paths["chroma"])
        elif paths["chroma"].exists():
            if paths["chroma"].is_dir():
                shutil.rmtree(paths["chroma"])
            else:
                paths["chroma"].unlink()

        if staging_bm25.is_file():
            _replace_path(staging_bm25, paths["bm25"])
        elif paths["bm25"].exists():
            paths["bm25"].unlink()

        if staging_hubness.is_file():
            _replace_path(staging_hubness, paths["hubness"])
        elif paths["hubness"].exists():
            paths["hubness"].unlink()


def _read_manifest(backup_id: str) -> Optional[dict[str, Any]]:
    key = blob_store.index_backup_key(backup_id, _MANIFEST_NAME)
    uri = blob_store.s3_uri(key)
    if not blob_store.exists(uri):
        return None
    raw = blob_store.read_bytes(uri)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def list_backups() -> list[dict[str, Any]]:
    """Return completed backups (those with a readable manifest.json)."""
    _require_s3()
    prefix = blob_store.index_backup_prefix() + "/"
    keys = blob_store.list_keys(prefix)
    backup_ids: set[str] = set()
    for key in keys:
        relative = key[len(prefix) :] if key.startswith(prefix) else key
        parts = relative.split("/")
        if len(parts) >= 2 and parts[1] == _MANIFEST_NAME:
            backup_ids.add(parts[0])

    backups: list[dict[str, Any]] = []
    for backup_id in sorted(backup_ids, reverse=True):
        manifest = _read_manifest(backup_id)
        if not manifest:
            continue
        backups.append(
            {
                "id": backup_id,
                "created_at": manifest.get("created_at"),
                "label": manifest.get("label"),
                "size_bytes": manifest.get("archive_bytes"),
                "sha256": manifest.get("archive_sha256"),
                "prefix": blob_store.index_backup_prefix(backup_id),
                "s3_uri": blob_store.s3_uri(blob_store.index_backup_prefix(backup_id)),
                "artifacts": manifest.get("artifacts") or {},
                "build_id": manifest.get("build_id"),
            }
        )
    return backups


def create_backup(*, label: Optional[str] = None) -> dict[str, Any]:
    """Quiesce writers, package the live index, upload archive then manifest."""
    _require_s3()
    backup_id = _new_backup_id()
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)

    with _quiesce_writers() as quiesce_meta:
        _dispose_live_stores()
        try:
            with tempfile.TemporaryDirectory(
                dir=SETTINGS.data_dir, prefix=".index-backup-"
            ) as tmp:
                tmp_dir = Path(tmp)
                archive_path = tmp_dir / _ARCHIVE_NAME
                artifacts = _build_archive(archive_path)
                archive_sha = _sha256_file(archive_path)
                archive_bytes = archive_path.stat().st_size

                archive_key = blob_store.index_backup_key(backup_id, _ARCHIVE_NAME)
                archive_uri = blob_store.put_file(
                    archive_path,
                    archive_key,
                    content_type="application/gzip",
                )

                manifest = {
                    "backup_id": backup_id,
                    "created_at": _utc_now().isoformat(),
                    "label": (label or "").strip() or None,
                    "archive_bytes": archive_bytes,
                    "archive_sha256": archive_sha,
                    "archive_key": archive_key,
                    "artifacts": artifacts,
                    "build_id": os.environ.get("APP_BUILD_ID", "development"),
                    "sqlite_path": str(Path(SETTINGS.sqlite_path).resolve()),
                    "chroma_dir": str(Path(SETTINGS.chroma_dir).resolve()),
                }
                manifest_key = blob_store.index_backup_key(backup_id, _MANIFEST_NAME)
                manifest_uri = blob_store.put_bytes(
                    json.dumps(manifest, indent=2).encode("utf-8"),
                    manifest_key,
                    content_type="application/json",
                )
        finally:
            _reopen_live_stores()

    logger.info("Index backup complete backup_id=%s archive=%s", backup_id, archive_uri)
    return {
        "ok": True,
        "backup_id": backup_id,
        "s3_uri": blob_store.s3_uri(blob_store.index_backup_prefix(backup_id)),
        "archive_uri": archive_uri,
        "manifest_uri": manifest_uri,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "artifacts": artifacts,
        "label": manifest.get("label"),
        "quiesced": True,
        "cancelled_job_ids": quiesce_meta.get("cancelled_job_ids") or [],
    }


def restore_backup(backup_id: str, *, confirm: bool = False) -> dict[str, Any]:
    """Download a completed snapshot and replace the live index as a unit."""
    _require_s3()
    if not confirm:
        raise IndexBackupError("Restore requires confirm=true")

    safe_id = blob_store.safe_filename(backup_id)
    if safe_id != backup_id:
        raise IndexBackupError(f"Invalid backup_id: {backup_id}")

    manifest = _read_manifest(backup_id)
    if not manifest:
        raise IndexBackupError(f"Backup not found or incomplete: {backup_id}")

    archive_key = manifest.get("archive_key") or blob_store.index_backup_key(
        backup_id, _ARCHIVE_NAME
    )
    archive_uri = blob_store.s3_uri(str(archive_key))
    if not blob_store.exists(archive_uri):
        raise IndexBackupError(f"Backup archive missing: {archive_uri}")

    expected_sha = str(manifest.get("archive_sha256") or "")
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)

    with _quiesce_writers() as quiesce_meta:
        _dispose_live_stores()
        try:
            with tempfile.TemporaryDirectory(
                dir=SETTINGS.data_dir, prefix=".index-restore-dl-"
            ) as tmp:
                tmp_dir = Path(tmp)
                archive_path = tmp_dir / _ARCHIVE_NAME
                archive_path.write_bytes(blob_store.read_bytes(archive_uri))
                actual_sha = _sha256_file(archive_path)
                if expected_sha and actual_sha != expected_sha:
                    raise IndexBackupError(
                        f"Archive checksum mismatch for {backup_id}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
                _apply_archive(archive_path)
        finally:
            _reopen_live_stores()

    logger.info("Index restore complete backup_id=%s", backup_id)
    return {
        "ok": True,
        "backup_id": backup_id,
        "s3_uri": blob_store.s3_uri(blob_store.index_backup_prefix(backup_id)),
        "archive_sha256": expected_sha or actual_sha,
        "quiesced": True,
        "cancelled_job_ids": quiesce_meta.get("cancelled_job_ids") or [],
        "restart_required": False,
    }
