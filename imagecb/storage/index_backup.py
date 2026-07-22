"""S3 vault for consistent SQLite + Chroma + BM25 + hubness snapshots.

The EC2 host bind-mount remains the live writable index cache. S3 stores
versioned archives under ``{s3_prefix}/index-backups/{id}/`` for disaster
recovery, plus a rolling ``checkpoint-latest`` for crash recovery.
``manifest.json`` is written last so incomplete uploads are never listed.
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
import threading
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
CHECKPOINT_LATEST_ID = "checkpoint-latest"

_checkpoint_lock = threading.Lock()
_last_checkpoint_info: dict[str, Any] = {
    "backup_id": None,
    "total_records": None,
    "chroma_vectors": None,
    "label": None,
    "created_at": None,
    "error": None,
}
_startup_restore_info: dict[str, Any] = {
    "attempted": False,
    "restored": False,
    "backup_id": None,
    "total_records": None,
    "error": None,
}


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


def _online_copy_sqlite(src: Path, dest: Path) -> None:
    """Consistent online snapshot without closing live writers."""
    if not src.is_file():
        raise IndexBackupError(f"SQLite database missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    escaped = str(dest.resolve()).replace("'", "''")
    connection = sqlite3.connect(str(src))
    try:
        connection.execute(f"VACUUM INTO '{escaped}'")
        connection.commit()
    finally:
        connection.close()
    if not dest.is_file():
        raise IndexBackupError(f"VACUUM INTO failed to create {dest}")


def _index_counts() -> dict[str, int]:
    total_records = 0
    chroma_vectors = 0
    try:
        from imagecb.storage import metadata_db

        total_records = int(metadata_db.count_active_records())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read SQLite record count for manifest: %s", exc)
    try:
        from imagecb.storage import vector_store

        chroma_vectors = int(vector_store.count())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read Chroma vector count for manifest: %s", exc)
    return {"total_records": total_records, "chroma_vectors": chroma_vectors}


def _manifest_record_count(manifest: Optional[dict[str, Any]]) -> int:
    if not manifest:
        return 0
    raw = manifest.get("total_records")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


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


def _tar_artifacts(
    archive_path: Path,
    *,
    sqlite_path: Path,
    chroma_dir: Path,
    bm25_path: Path,
    hubness_path: Path,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(sqlite_path, arcname="imagecb.db")
        artifacts["sqlite"] = {
            "present": True,
            "bytes": sqlite_path.stat().st_size,
            "arcname": "imagecb.db",
        }

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

        for name, path, arcname in (
            ("bm25", bm25_path, "bm25.pkl"),
            ("hubness", hubness_path, "hubness.pkl"),
        ):
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


def _build_archive(archive_path: Path) -> dict[str, Any]:
    """Build archive after writers are disposed (manual backup path)."""
    paths = _artifact_paths()
    sqlite_path = paths["sqlite"]
    if not sqlite_path.is_file():
        raise IndexBackupError(f"SQLite database missing: {sqlite_path}")

    _checkpoint_sqlite(sqlite_path)
    return _tar_artifacts(
        archive_path,
        sqlite_path=sqlite_path,
        chroma_dir=paths["chroma"],
        bm25_path=paths["bm25"],
        hubness_path=paths["hubness"],
    )


def _build_archive_online(archive_path: Path) -> dict[str, Any]:
    """Build archive without cancelling ingest (online SQLite snapshot)."""
    paths = _artifact_paths()
    if not paths["sqlite"].is_file():
        raise IndexBackupError(f"SQLite database missing: {paths['sqlite']}")

    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=SETTINGS.data_dir, prefix=".index-checkpoint-snap-"
    ) as snap:
        snap_dir = Path(snap)
        snap_db = snap_dir / "imagecb.db"
        _online_copy_sqlite(paths["sqlite"], snap_db)

        snap_chroma = snap_dir / "chroma"
        if paths["chroma"].is_dir():
            shutil.copytree(paths["chroma"], snap_chroma)

        snap_bm25 = snap_dir / "bm25.pkl"
        if paths["bm25"].is_file():
            shutil.copy2(paths["bm25"], snap_bm25)

        snap_hubness = snap_dir / "hubness.pkl"
        if paths["hubness"].is_file():
            shutil.copy2(paths["hubness"], snap_hubness)

        return _tar_artifacts(
            archive_path,
            sqlite_path=snap_db,
            chroma_dir=snap_chroma,
            bm25_path=snap_bm25,
            hubness_path=snap_hubness,
        )


def _replace_path(src: Path, dest: Path) -> None:
    """Replace dest with src without delete-first (crash-safe rename aside)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        os.replace(src, dest)
        return

    aside = dest.with_name(f".{dest.name}.old-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(dest, aside)
        os.replace(src, dest)
    except Exception:
        if aside.exists() and not dest.exists():
            os.replace(aside, dest)
        raise
    if aside.is_dir():
        shutil.rmtree(aside, ignore_errors=True)
    elif aside.exists():
        aside.unlink(missing_ok=True)


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
            aside = paths["chroma"].with_name(
                f".{paths['chroma'].name}.clear-{uuid.uuid4().hex[:8]}"
            )
            os.replace(paths["chroma"], aside)
            if aside.is_dir():
                shutil.rmtree(aside, ignore_errors=True)
            elif aside.exists():
                aside.unlink(missing_ok=True)

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


def _upload_snapshot(
    *,
    backup_id: str,
    archive_path: Path,
    artifacts: dict[str, Any],
    label: Optional[str],
    job_id: Optional[str],
    kind: str,
    counts: dict[str, int],
) -> dict[str, Any]:
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
        "kind": kind,
        "job_id": job_id,
        "total_records": counts.get("total_records", 0),
        "chroma_vectors": counts.get("chroma_vectors", 0),
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
    return {
        "backup_id": backup_id,
        "archive_uri": archive_uri,
        "manifest_uri": manifest_uri,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "artifacts": artifacts,
        "manifest": manifest,
    }


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
                "kind": manifest.get("kind"),
                "total_records": manifest.get("total_records"),
                "chroma_vectors": manifest.get("chroma_vectors"),
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
    counts = _index_counts()

    with _quiesce_writers() as quiesce_meta:
        _dispose_live_stores()
        try:
            with tempfile.TemporaryDirectory(
                dir=SETTINGS.data_dir, prefix=".index-backup-"
            ) as tmp:
                tmp_dir = Path(tmp)
                archive_path = tmp_dir / _ARCHIVE_NAME
                artifacts = _build_archive(archive_path)
                uploaded = _upload_snapshot(
                    backup_id=backup_id,
                    archive_path=archive_path,
                    artifacts=artifacts,
                    label=label,
                    job_id=None,
                    kind="backup",
                    counts=counts,
                )
                manifest = uploaded["manifest"]
        finally:
            _reopen_live_stores()

    logger.info(
        "Index backup complete backup_id=%s archive=%s records=%s",
        backup_id,
        uploaded["archive_uri"],
        counts.get("total_records"),
    )
    return {
        "ok": True,
        "backup_id": backup_id,
        "s3_uri": blob_store.s3_uri(blob_store.index_backup_prefix(backup_id)),
        "archive_uri": uploaded["archive_uri"],
        "manifest_uri": uploaded["manifest_uri"],
        "archive_bytes": uploaded["archive_bytes"],
        "archive_sha256": uploaded["archive_sha256"],
        "artifacts": uploaded["artifacts"],
        "label": manifest.get("label"),
        "total_records": counts.get("total_records", 0),
        "chroma_vectors": counts.get("chroma_vectors", 0),
        "quiesced": True,
        "cancelled_job_ids": quiesce_meta.get("cancelled_job_ids") or [],
    }


def create_checkpoint(
    *, label: Optional[str] = None, job_id: Optional[str] = None
) -> dict[str, Any]:
    """Online snapshot to S3 without cancelling ingest jobs.

    Uploads a timestamped copy and overwrites ``checkpoint-latest``.
    """
    if not SETTINGS.index_checkpoint_enabled:
        raise IndexBackupError("Index checkpointing is disabled")
    _require_s3()

    with _checkpoint_lock:
        SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
        counts = _index_counts()
        backup_id = _new_backup_id()
        try:
            with tempfile.TemporaryDirectory(
                dir=SETTINGS.data_dir, prefix=".index-checkpoint-"
            ) as tmp:
                tmp_dir = Path(tmp)
                archive_path = tmp_dir / _ARCHIVE_NAME
                artifacts = _build_archive_online(archive_path)
                uploaded = _upload_snapshot(
                    backup_id=backup_id,
                    archive_path=archive_path,
                    artifacts=artifacts,
                    label=label,
                    job_id=job_id,
                    kind="checkpoint",
                    counts=counts,
                )
                # Rolling pointer for auto-restore (archive then manifest).
                _upload_snapshot(
                    backup_id=CHECKPOINT_LATEST_ID,
                    archive_path=archive_path,
                    artifacts=artifacts,
                    label=label,
                    job_id=job_id,
                    kind="checkpoint",
                    counts=counts,
                )
        except Exception as exc:
            _last_checkpoint_info.update(
                {
                    "backup_id": None,
                    "total_records": None,
                    "chroma_vectors": None,
                    "label": (label or "").strip() or None,
                    "created_at": None,
                    "error": str(exc),
                }
            )
            raise

        manifest = uploaded["manifest"]
        _last_checkpoint_info.update(
            {
                "backup_id": backup_id,
                "total_records": counts.get("total_records", 0),
                "chroma_vectors": counts.get("chroma_vectors", 0),
                "label": manifest.get("label"),
                "created_at": manifest.get("created_at"),
                "error": None,
            }
        )
        logger.info(
            "Index checkpoint complete backup_id=%s latest=%s records=%s",
            backup_id,
            CHECKPOINT_LATEST_ID,
            counts.get("total_records"),
        )
        return {
            "ok": True,
            "backup_id": backup_id,
            "latest_id": CHECKPOINT_LATEST_ID,
            "s3_uri": blob_store.s3_uri(blob_store.index_backup_prefix(backup_id)),
            "archive_uri": uploaded["archive_uri"],
            "manifest_uri": uploaded["manifest_uri"],
            "archive_bytes": uploaded["archive_bytes"],
            "archive_sha256": uploaded["archive_sha256"],
            "artifacts": uploaded["artifacts"],
            "label": manifest.get("label"),
            "job_id": job_id,
            "total_records": counts.get("total_records", 0),
            "chroma_vectors": counts.get("chroma_vectors", 0),
            "quiesced": False,
            "cancelled_job_ids": [],
        }


def maybe_checkpoint_progress(
    stats: dict[str, Any],
    *,
    job_id: Optional[str] = None,
    force: bool = False,
    label: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Best-effort checkpoint during ingest; never raises into the caller."""
    if not SETTINGS.index_checkpoint_enabled:
        return None
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        return None

    committed = int(stats.get("images_added", 0)) + int(stats.get("images_updated", 0))
    last = int(stats.get("_checkpoint_at", 0) or 0)
    every = max(1, int(SETTINGS.index_checkpoint_every_n))
    if not force and (committed - last) < every:
        return None

    try:
        result = create_checkpoint(
            label=label or f"ingest:{job_id or 'manual'}:{committed}",
            job_id=job_id,
        )
        stats["_checkpoint_at"] = committed
        stats["last_checkpoint_id"] = result.get("backup_id")
        stats["last_checkpoint_records"] = result.get("total_records")
        return result
    except Exception as exc:  # noqa: BLE001
        stats["checkpoint_errors"] = int(stats.get("checkpoint_errors", 0) or 0) + 1
        stats["last_checkpoint_error"] = str(exc)
        logger.warning("Index checkpoint failed (ingest continues): %s", exc)
        return None


def get_preferred_restore_backup_id() -> Optional[str]:
    """Prefer checkpoint-latest when it has records; else newest non-empty backup."""
    _require_s3()
    latest = _read_manifest(CHECKPOINT_LATEST_ID)
    if _manifest_record_count(latest) > 0:
        return CHECKPOINT_LATEST_ID

    best_id: Optional[str] = None
    best_created = ""
    for item in list_backups():
        backup_id = str(item.get("id") or "")
        if not backup_id or backup_id == CHECKPOINT_LATEST_ID:
            continue
        records = item.get("total_records")
        try:
            count = int(records) if records is not None else 0
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        created = str(item.get("created_at") or "")
        if best_id is None or created > best_created:
            best_id = backup_id
            best_created = created
    return best_id


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

    if _manifest_record_count(manifest) <= 0 and manifest.get("total_records") is not None:
        raise IndexBackupError(
            f"Refusing to restore empty snapshot {backup_id} (total_records=0)"
        )

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
        "total_records": manifest.get("total_records"),
        "quiesced": True,
        "cancelled_job_ids": quiesce_meta.get("cancelled_job_ids") or [],
        "restart_required": False,
    }


def maybe_auto_restore_on_startup() -> dict[str, Any]:
    """Restore preferred S3 checkpoint when it has more records than local.

    Keeps the local index when it is empty only if no non-empty remote snapshot
    exists, and when local count is greater than or equal to the preferred
    remote snapshot (so a newer local ingest is not clobbered by a smaller
    checkpoint). A small local smoke/bootstrap index is replaced when S3 has a
    strictly larger snapshot.
    """
    info = {
        "attempted": False,
        "restored": False,
        "backup_id": None,
        "total_records": None,
        "local_count": None,
        "remote_count": None,
        "error": None,
        "skipped": None,
    }
    global _startup_restore_info

    if not SETTINGS.index_auto_restore_on_startup:
        info["skipped"] = "auto_restore_disabled"
        _startup_restore_info = dict(info)
        return info
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        info["skipped"] = "s3_not_configured"
        _startup_restore_info = dict(info)
        return info

    try:
        from imagecb.storage import metadata_db

        local_count = int(metadata_db.count_active_records())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"local_count_failed: {exc}"
        _startup_restore_info = dict(info)
        return info

    info["local_count"] = local_count
    info["total_records"] = local_count

    try:
        backup_id = get_preferred_restore_backup_id()
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"preferred_backup_failed: {exc}"
        _startup_restore_info = dict(info)
        return info

    if not backup_id:
        info["skipped"] = "no_nonempty_checkpoint"
        _startup_restore_info = dict(info)
        return info

    manifest = _read_manifest(backup_id)
    remote_count = _manifest_record_count(manifest)
    info["remote_count"] = remote_count
    info["backup_id"] = backup_id

    if remote_count <= local_count:
        info["skipped"] = f"local_records={local_count}_remote_records={remote_count}"
        _startup_restore_info = dict(info)
        return info

    info["attempted"] = True
    try:
        # Startup: runner is not up yet; restore without cancelling jobs.
        result = _restore_without_job_cancel(backup_id)
        info["restored"] = True
        info["total_records"] = result.get("total_records", remote_count)
        logger.info(
            "Startup auto-restore complete backup_id=%s local=%s remote=%s records=%s",
            backup_id,
            local_count,
            remote_count,
            info["total_records"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Startup auto-restore failed")
        info["error"] = str(exc)

    _startup_restore_info = dict(info)
    return info


def _restore_without_job_cancel(backup_id: str) -> dict[str, Any]:
    """Restore used at startup before the job runner is active."""
    manifest = _read_manifest(backup_id)
    if not manifest:
        raise IndexBackupError(f"Backup not found or incomplete: {backup_id}")
    if _manifest_record_count(manifest) <= 0 and manifest.get("total_records") is not None:
        raise IndexBackupError(
            f"Refusing to restore empty snapshot {backup_id} (total_records=0)"
        )

    archive_key = manifest.get("archive_key") or blob_store.index_backup_key(
        backup_id, _ARCHIVE_NAME
    )
    archive_uri = blob_store.s3_uri(str(archive_key))
    if not blob_store.exists(archive_uri):
        raise IndexBackupError(f"Backup archive missing: {archive_uri}")

    expected_sha = str(manifest.get("archive_sha256") or "")
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)

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

    return {
        "ok": True,
        "backup_id": backup_id,
        "archive_sha256": expected_sha or actual_sha,
        "total_records": manifest.get("total_records"),
    }


def last_checkpoint_info() -> dict[str, Any]:
    return dict(_last_checkpoint_info)


def startup_restore_info() -> dict[str, Any]:
    return dict(_startup_restore_info)
