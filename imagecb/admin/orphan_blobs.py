"""Idle-only S3 orphan blob assessment and purge (does not touch ingest)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from imagecb.config import SETTINGS
from imagecb.ingest_jobs import busy_staging_uris, list_busy_jobs
from imagecb.storage import blob_store
from imagecb.storage.blob_store import ListedObject
from imagecb.storage.metadata_db import get_all_records

logger = logging.getLogger(__name__)

_SAMPLE_LIMIT = 50
_DEFAULT_MIN_AGE_HOURS = 1.0


class OrphanBlobError(Exception):
    """Raised when orphan GC cannot run safely."""


@dataclass
class OrphanCandidate:
    kind: str  # images | thumbs | uploads | staging
    uri: str
    key: str
    image_id: Optional[str] = None
    last_modified: Optional[datetime] = None
    too_new: bool = False


@dataclass
class OrphanBlobReport:
    dry_run: bool = True
    min_age_hours: float = _DEFAULT_MIN_AGE_HOURS
    images: list[OrphanCandidate] = field(default_factory=list)
    thumbs: list[OrphanCandidate] = field(default_factory=list)
    uploads: list[OrphanCandidate] = field(default_factory=list)
    staging: list[OrphanCandidate] = field(default_factory=list)
    skipped_too_new: list[OrphanCandidate] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def purgeable(self) -> list[OrphanCandidate]:
        return [*self.images, *self.thumbs, *self.uploads, *self.staging]

    def to_dict(self) -> dict[str, Any]:
        def _sample(cands: list[OrphanCandidate]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for c in cands[:_SAMPLE_LIMIT]:
                row: dict[str, Any] = {
                    "kind": c.kind,
                    "uri": c.uri,
                    "key": c.key,
                }
                if c.image_id:
                    row["image_id"] = c.image_id
                if c.last_modified is not None:
                    row["last_modified"] = c.last_modified.isoformat()
                if c.too_new:
                    row["too_new"] = True
                out.append(row)
            return out

        return {
            "dry_run": self.dry_run,
            "min_age_hours": self.min_age_hours,
            "orphan_image_count": len(self.images),
            "orphan_thumb_count": len(self.thumbs),
            "orphan_upload_count": len(self.uploads),
            "orphan_staging_count": len(self.staging),
            "skipped_too_new_count": len(self.skipped_too_new),
            "purgeable_count": len(self.purgeable),
            "deleted_count": len(self.deleted),
            "failed_count": len(self.failed),
            "elapsed_sec": self.elapsed_sec,
            "samples": {
                "images": _sample(self.images),
                "thumbs": _sample(self.thumbs),
                "uploads": _sample(self.uploads),
                "staging": _sample(self.staging),
                "skipped_too_new": _sample(self.skipped_too_new),
            },
            "deleted_sample": self.deleted[:_SAMPLE_LIMIT],
            "failed": self.failed[:_SAMPLE_LIMIT],
        }


def _require_s3() -> None:
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        raise OrphanBlobError(
            "Orphan blob GC requires BLOB_STORAGE_BACKEND=s3 and S3_BUCKET"
        )


def _refuse_if_busy() -> None:
    busy = list_busy_jobs()
    if not busy:
        return
    ids = ", ".join(f"{j['job_id']}({j['status']})" for j in busy[:5])
    more = "" if len(busy) <= 5 else f" (+{len(busy) - 5} more)"
    raise OrphanBlobError(
        f"Refusing orphan blob GC while ingest jobs are active: {ids}{more}. "
        "Wait for upload/ingest to finish so walk-away jobs are not disrupted."
    )


def _cutoff(min_age_hours: float) -> datetime:
    hours = max(0.0, float(min_age_hours))
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _is_too_new(obj: ListedObject, cutoff: datetime) -> bool:
    if obj.last_modified is None:
        # Unknown age: treat as eligible so long-lived orphans without
        # LastModified still get cleaned; busy-job + refuse gate cover races.
        return False
    modified = obj.last_modified
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=timezone.utc)
    return modified > cutoff


def _keep_sets() -> tuple[set[str], set[str]]:
    """Return (image_ids, source_file URIs) for all rows including soft-deleted."""
    records = get_all_records(include_deleted=True)
    ids = {r.image_id for r in records if r.image_id}
    sources = {(r.source_file or "").strip() for r in records if (r.source_file or "").strip()}
    return ids, sources


def _list_prefix_objects(prefix: str) -> list[ListedObject]:
    try:
        return blob_store.list_objects(prefix.rstrip("/") + "/")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list objects under %s: %s", prefix, exc)
        raise OrphanBlobError(f"Failed to list S3 objects under {prefix}: {exc}") from exc


def assess_orphan_blobs(*, min_age_hours: float = _DEFAULT_MIN_AGE_HOURS) -> OrphanBlobReport:
    """List orphan S3 blobs relative to SQLite (including soft-deleted keep set)."""
    t0 = time.perf_counter()
    _require_s3()
    _refuse_if_busy()

    keep_ids, keep_sources = _keep_sets()
    protected_staging = busy_staging_uris()
    cutoff = _cutoff(min_age_hours)
    report = OrphanBlobReport(dry_run=True, min_age_hours=float(min_age_hours))

    for obj in _list_prefix_objects(blob_store.images_prefix()):
        name = obj.key.rsplit("/", 1)[-1]
        if not name.lower().endswith(".png"):
            continue
        image_id = name[:-4]
        if image_id in keep_ids:
            continue
        cand = OrphanCandidate(
            kind="images",
            uri=blob_store.s3_uri(obj.key),
            key=obj.key,
            image_id=image_id,
            last_modified=obj.last_modified,
            too_new=_is_too_new(obj, cutoff),
        )
        if cand.too_new:
            report.skipped_too_new.append(cand)
        else:
            report.images.append(cand)

    for obj in _list_prefix_objects(blob_store.thumbs_prefix()):
        name = obj.key.rsplit("/", 1)[-1]
        if not name.lower().endswith(".jpg"):
            continue
        image_id = name[:-4]
        if image_id in keep_ids:
            continue
        cand = OrphanCandidate(
            kind="thumbs",
            uri=blob_store.s3_uri(obj.key),
            key=obj.key,
            image_id=image_id,
            last_modified=obj.last_modified,
            too_new=_is_too_new(obj, cutoff),
        )
        if cand.too_new:
            report.skipped_too_new.append(cand)
        else:
            report.thumbs.append(cand)

    for obj in _list_prefix_objects(blob_store.uploads_prefix()):
        uri = blob_store.s3_uri(obj.key)
        if uri in keep_sources:
            continue
        cand = OrphanCandidate(
            kind="uploads",
            uri=uri,
            key=obj.key,
            last_modified=obj.last_modified,
            too_new=_is_too_new(obj, cutoff),
        )
        if cand.too_new:
            report.skipped_too_new.append(cand)
        else:
            report.uploads.append(cand)

    for obj in _list_prefix_objects(blob_store.staging_prefix()):
        uri = blob_store.s3_uri(obj.key)
        if uri in protected_staging:
            continue
        cand = OrphanCandidate(
            kind="staging",
            uri=uri,
            key=obj.key,
            last_modified=obj.last_modified,
            too_new=_is_too_new(obj, cutoff),
        )
        if cand.too_new:
            report.skipped_too_new.append(cand)
        else:
            report.staging.append(cand)

    report.elapsed_sec = round(time.perf_counter() - t0, 2)
    return report


def purge_orphan_blobs(
    *,
    dry_run: bool = True,
    min_age_hours: float = _DEFAULT_MIN_AGE_HOURS,
) -> dict[str, Any]:
    """Assess orphans; when dry_run is False, delete purgeable candidates."""
    report = assess_orphan_blobs(min_age_hours=min_age_hours)
    report.dry_run = dry_run
    if dry_run:
        return report.to_dict()

    t0 = time.perf_counter()
    # Re-check busy immediately before deletes (walk-away upload may have started).
    _refuse_if_busy()
    for cand in report.purgeable:
        try:
            blob_store.delete(cand.uri)
            report.deleted.append(cand.uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete orphan blob %s: %s", cand.uri, exc)
            if len(report.failed) < _SAMPLE_LIMIT:
                report.failed.append({"uri": cand.uri, "error": str(exc)})
    report.elapsed_sec = round(report.elapsed_sec + (time.perf_counter() - t0), 2)
    return report.to_dict()
