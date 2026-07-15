"""Resolve local and S3-backed corpus blobs."""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Optional

from PIL import Image

from imagecb.config import SETTINGS
from imagecb.storage import blob_store
from imagecb.storage.metadata_db import ImageRecord


def image_fallbacks(record: ImageRecord) -> tuple[Path, ...]:
    return (SETTINGS.image_cache_dir / f"{record.image_id}.png",)


def source_fallbacks(record: ImageRecord) -> tuple[Path, ...]:
    if not record.source_file:
        return ()
    return (SETTINGS.uploads_dir / blob_store.safe_filename(record.source_file.rsplit("/", 1)[-1]),)


def image_exists(record: ImageRecord) -> bool:
    refs = [record.image_path, record.source_file]
    return any(
        blob_store.exists(ref, fallbacks=image_fallbacks(record))
        for ref in refs
        if ref
    )


def source_exists(record: ImageRecord) -> bool:
    return blob_store.exists(record.source_file, fallbacks=source_fallbacks(record))


def resolve_source_file(record: ImageRecord) -> Optional[Path]:
    """Return a local source path, including a legacy fallback, if available."""
    if not record.source_file:
        return None
    for raw in (record.source_file, *source_fallbacks(record)):
        if blob_store.is_s3_uri(raw):
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def resolve_image_file(record: ImageRecord) -> Optional[Path]:
    """Return a local display image, caching an S3 image locally when necessary."""
    candidates: list[str | Path] = []
    if record.image_path:
        candidates.append(record.image_path)
    if record.source_file and record.source_file not in candidates:
        candidates.append(record.source_file)
    for raw in candidates:
        if blob_store.is_s3_uri(raw):
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()
    if record.image_path and blob_store.is_s3_uri(record.image_path):
        target = SETTINGS.image_cache_dir / f"{record.image_id}.png"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob_store.read_bytes(record.image_path))
            return target.resolve()
        except Exception:
            return None
    return None


def open_record_image(record: ImageRecord) -> Optional[Image.Image]:
    refs = [record.image_path, record.source_file]
    for ref in refs:
        if not ref:
            continue
        try:
            return blob_store.open_image(ref, fallbacks=image_fallbacks(record))
        except Exception:
            continue
    return None


@contextmanager
def materialize_source(record: ImageRecord) -> Iterator[Optional[Path]]:
    if not record.source_file:
        yield None
        return
    try:
        with blob_store.materialize(
            record.source_file,
            fallbacks=source_fallbacks(record),
        ) as path:
            yield path
    except Exception:
        yield None
