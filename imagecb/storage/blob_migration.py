"""Idempotent migration of legacy local corpus blobs to S3."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from imagecb.config import SETTINGS
from imagecb.paths import image_fallbacks, source_fallbacks
from imagecb.storage import blob_store
from imagecb.storage.metadata_db import ImageRecord, get_all_records, session_scope

logger = logging.getLogger(__name__)


def _local_file(ref: str, fallbacks: tuple[Path, ...]) -> Path | None:
    if ref and not blob_store.is_s3_uri(ref):
        path = Path(ref).expanduser()
        if path.is_file():
            return path.resolve()
    for path in fallbacks:
        if path.is_file():
            return path.resolve()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_local_blobs_to_s3(*, dry_run: bool = True) -> dict[str, Any]:
    """Upload local blobs and rewrite rows only after successful object writes."""
    if SETTINGS.blob_storage_backend != "s3":
        raise ValueError("Set BLOB_STORAGE_BACKEND=s3 before running blob migration")
    SETTINGS.validate_blob_storage()

    stats: dict[str, Any] = {
        "dry_run": dry_run,
        "records_scanned": 0,
        "image_candidates": 0,
        "source_candidates": 0,
        "images_migrated": 0,
        "sources_migrated": 0,
        "already_migrated": 0,
        "missing_local": 0,
        "errors": 0,
    }
    source_uri_cache: dict[Path, str] = {}

    for record in get_all_records(include_deleted=True):
        stats["records_scanned"] += 1
        image_uri: str | None = None
        source_uri: str | None = None

        if blob_store.is_s3_uri(record.image_path):
            stats["already_migrated"] += 1
        else:
            image_path = _local_file(record.image_path, image_fallbacks(record))
            if image_path is None:
                stats["missing_local"] += 1
            else:
                stats["image_candidates"] += 1
                if not dry_run:
                    try:
                        image_uri = blob_store.put_file(
                            image_path,
                            blob_store.image_key(record.image_id),
                            content_type="image/png",
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["errors"] += 1
                        logger.warning("Could not migrate image %s: %s", record.image_id, exc)

        if blob_store.is_s3_uri(record.source_file):
            stats["already_migrated"] += 1
        else:
            source_path = _local_file(record.source_file, source_fallbacks(record))
            if source_path is None:
                stats["missing_local"] += 1
            else:
                stats["source_candidates"] += 1
                if not dry_run:
                    try:
                        source_uri = source_uri_cache.get(source_path)
                        if source_uri is None:
                            key = blob_store.source_key(
                                source_path.name,
                                _sha256(source_path),
                            )
                            source_uri = blob_store.put_file(source_path, key)
                            source_uri_cache[source_path] = source_uri
                    except Exception as exc:  # noqa: BLE001
                        stats["errors"] += 1
                        logger.warning("Could not migrate source %s: %s", source_path, exc)

        if dry_run or (image_uri is None and source_uri is None):
            continue
        with session_scope() as session:
            row = session.get(ImageRecord, record.image_id)
            if row is None:
                stats["errors"] += 1
                continue
            if image_uri is not None:
                row.image_path = image_uri
                stats["images_migrated"] += 1
            if source_uri is not None:
                row.source_file = source_uri
                stats["sources_migrated"] += 1

    return stats
