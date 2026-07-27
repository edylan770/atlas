"""Save browser uploads into the corpus staging directory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    from fastapi import UploadFile

from imagecb.config import SETTINGS
from imagecb.extractors.dispatch import SUPPORTED_EXTS

logger = logging.getLogger(__name__)

UploadInput = Union[str, Path, dict]


def is_supported_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def unique_dest(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding path under dest_dir for filename."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).name
    candidate = dest_dir / base
    if not candidate.exists():
        return candidate
    stem = Path(base).stem
    suffix = Path(base).suffix
    n = 2
    while True:
        candidate = dest_dir / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def cleanup_staged_uploads(paths: Sequence[Path]) -> None:
    """Remove temporary upload staging files after S3-backed ingest."""
    if SETTINGS.blob_storage_backend != "s3":
        return
    root = SETTINGS.uploads_dir.resolve()
    for path in paths:
        try:
            resolved = path.resolve()
            if resolved.parent == root:
                resolved.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove staged upload %s: %s", path, exc)


async def save_uploads_from_files(
    files: Sequence["UploadFile"],
    *,
    dest_dir: Path | None = None,
) -> Tuple[List[Path], List[str]]:
    """Stage FastAPI UploadFile objects into the uploads directory."""
    saved: List[Path] = []
    errors: List[str] = []
    target_dir = dest_dir if dest_dir is not None else SETTINGS.uploads_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024

    for upload in files or []:
        name = upload.filename or "upload"
        try:
            if not is_supported_extension(Path(name)):
                raise ValueError(
                    f"Unsupported file type '{Path(name).suffix}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
                )
            dest = unique_dest(target_dir, name)
            with dest.open("wb") as out:
                while True:
                    chunk = await upload.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
            saved.append(dest)
            logger.info("Staged API upload %s -> %s", name, dest)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("Failed to stage API upload %s: %s", name, exc)
    return saved, errors
