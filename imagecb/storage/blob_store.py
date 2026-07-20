"""Durable corpus blob storage backed by local disk or private S3."""

from __future__ import annotations

import io
import hashlib
import logging
import mimetypes
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Optional, Sequence

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)
_S3_SCHEME = "s3://"
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass(frozen=True)
class BlobInfo:
    filename: str
    content_type: str
    content_length: Optional[int] = None


def is_s3_uri(value: str | Path | None) -> bool:
    return bool(value) and str(value).lower().startswith(_S3_SCHEME)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    remainder = uri[len(_S3_SCHEME) :]
    bucket, separator, key = remainder.partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def safe_filename(filename: str) -> str:
    base = Path(filename).name.strip() or "upload"
    cleaned = _SAFE_FILENAME.sub("_", base).strip(" .")
    return cleaned or "upload"


def _key(*parts: str) -> str:
    clean = [part.strip("/") for part in parts if part and part.strip("/")]
    key = str(PurePosixPath(*clean))
    if key.startswith("../") or "/../" in key:
        raise ValueError("S3 object key cannot contain parent traversal")
    return key


def source_key(filename: str, content_hash: Optional[str] = None) -> str:
    identity = content_hash or str(uuid.uuid4())
    return _key(SETTINGS.s3_prefix, "uploads", identity[:2], identity, safe_filename(filename))


def image_key(image_id: str) -> str:
    return _key(SETTINGS.s3_prefix, "images", f"{image_id}.png")


def ingest_log_key(run_id: str, when: Optional[datetime] = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    safe_id = safe_filename(run_id).replace(" ", "_")
    return _key(SETTINGS.s3_prefix, "ingest-logs", f"{stamp}_{safe_id}.txt")


def s3_uri(key: str) -> str:
    if not SETTINGS.s3_bucket:
        raise ValueError("S3_BUCKET is not configured")
    return f"s3://{SETTINGS.s3_bucket}/{key}"


@lru_cache(maxsize=4)
def _s3_client(region: str) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def get_s3_client() -> Any:
    return _s3_client(SETTINGS.s3_region)


def put_file(path: Path, key: str, *, content_type: Optional[str] = None) -> str:
    """Upload a local file and return its canonical S3 URI."""
    if SETTINGS.blob_storage_backend != "s3":
        return str(path.expanduser().resolve())
    extra = {}
    guessed = content_type or mimetypes.guess_type(path.name)[0]
    if guessed:
        extra["ContentType"] = guessed
    with path.open("rb") as handle:
        kwargs = {"ExtraArgs": extra} if extra else {}
        get_s3_client().upload_fileobj(handle, SETTINGS.s3_bucket, key, **kwargs)
    return s3_uri(key)


def put_bytes(
    data: bytes,
    key: str,
    *,
    content_type: Optional[str] = None,
) -> str:
    """Write bytes to the configured blob backend and return a durable reference."""
    if SETTINGS.blob_storage_backend == "s3":
        kwargs: dict[str, Any] = {
            "Bucket": SETTINGS.s3_bucket,
            "Key": key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        get_s3_client().put_object(**kwargs)
        return s3_uri(key)

    relative = PurePosixPath(key)
    namespace = relative.parts[-2] if len(relative.parts) >= 2 else ""
    if namespace == "images":
        path = SETTINGS.image_cache_dir / relative.name
    elif namespace == "uploads":
        path = SETTINGS.uploads_dir / relative.name
    else:
        path = SETTINGS.data_dir.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path.resolve())


def persist_source(path: Path) -> str:
    """Persist an ingest source and return the provenance reference to store."""
    resolved = path.expanduser().resolve()
    if SETTINGS.blob_storage_backend != "s3":
        return str(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return put_file(resolved, source_key(resolved.name, digest.hexdigest()))


def persist_image_png(image_id: str, data: bytes) -> str:
    if SETTINGS.blob_storage_backend == "s3":
        return put_bytes(data, image_key(image_id), content_type="image/png")
    out_path = SETTINGS.image_cache_dir / f"{image_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return str(out_path.resolve())


def _local_candidate(ref: str | Path) -> Optional[Path]:
    if is_s3_uri(ref):
        return None
    path = Path(ref).expanduser()
    return path.resolve() if path.is_file() else None


def _first_local(ref: str | Path, fallbacks: Sequence[Path]) -> Optional[Path]:
    direct = _local_candidate(ref)
    if direct is not None:
        return direct
    for candidate in fallbacks:
        path = candidate.expanduser()
        if path.is_file():
            return path.resolve()
    return None


def exists(ref: str | Path | None, *, fallbacks: Sequence[Path] = ()) -> bool:
    if not ref:
        return False
    if _first_local(ref, fallbacks) is not None:
        return True
    if not is_s3_uri(ref):
        return False
    bucket, key = parse_s3_uri(str(ref))
    try:
        get_s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except get_s3_client().exceptions.NoSuchKey:
        return False
    except Exception as exc:  # botocore maps HEAD 404 to ClientError
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        logger.warning("Could not check blob %s: %s", ref, exc)
        return False


def read_bytes(ref: str | Path, *, fallbacks: Sequence[Path] = ()) -> bytes:
    local = _first_local(ref, fallbacks)
    if local is not None:
        return local.read_bytes()
    bucket, key = parse_s3_uri(str(ref))
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def describe(ref: str | Path, *, fallbacks: Sequence[Path] = ()) -> BlobInfo:
    local = _first_local(ref, fallbacks)
    if local is not None:
        return BlobInfo(
            filename=local.name,
            content_type=mimetypes.guess_type(local.name)[0] or "application/octet-stream",
            content_length=local.stat().st_size,
        )
    bucket, key = parse_s3_uri(str(ref))
    response = get_s3_client().head_object(Bucket=bucket, Key=key)
    return BlobInfo(
        filename=PurePosixPath(key).name,
        content_type=response.get("ContentType")
        or mimetypes.guess_type(key)[0]
        or "application/octet-stream",
        content_length=response.get("ContentLength"),
    )


def iter_bytes(
    ref: str | Path,
    *,
    fallbacks: Sequence[Path] = (),
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    local = _first_local(ref, fallbacks)
    if local is not None:
        with local.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk
        return

    bucket, key = parse_s3_uri(str(ref))
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        while chunk := body.read(chunk_size):
            yield chunk
    finally:
        body.close()


@contextmanager
def materialize(
    ref: str | Path,
    *,
    fallbacks: Sequence[Path] = (),
) -> Iterator[Path]:
    """Yield a local path for libraries that cannot consume object bytes."""
    local = _first_local(ref, fallbacks)
    if local is not None:
        yield local
        return

    suffix = PurePosixPath(parse_s3_uri(str(ref))[1]).suffix
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(read_bytes(ref))
        yield temp_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def open_image(ref: str | Path, *, fallbacks: Sequence[Path] = ()) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(read_bytes(ref, fallbacks=fallbacks)))
    image.load()
    return image
