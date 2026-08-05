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


@dataclass(frozen=True)
class ListedObject:
    """S3 object listing row with optional LastModified for age guards."""

    key: str
    last_modified: Optional[datetime] = None


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


def staging_source_key(job_id: str, file_id: str, filename: str) -> str:
    return _key(
        SETTINGS.s3_prefix,
        "staging",
        safe_filename(job_id),
        safe_filename(file_id),
        safe_filename(filename),
    )


def image_key(image_id: str) -> str:
    return _key(SETTINGS.s3_prefix, "images", f"{image_id}.png")


def thumb_key(image_id: str) -> str:
    return _key(SETTINGS.s3_prefix, "thumbs", f"{image_id}.jpg")


def pending_edit_key(pending_id: str) -> str:
    return _key(SETTINGS.s3_prefix, "pending-edits", f"{safe_filename(pending_id)}.png")


def pending_edit_thumb_key(pending_id: str) -> str:
    return _key(
        SETTINGS.s3_prefix, "pending-edits", "thumbs", f"{safe_filename(pending_id)}.jpg"
    )


def pending_edit_ref(pending_id: str) -> str:
    if SETTINGS.blob_storage_backend == "s3":
        return s3_uri(pending_edit_key(pending_id))
    return str(
        (SETTINGS.data_dir / SETTINGS.s3_prefix / "pending-edits" / f"{pending_id}.png").resolve()
    )


def persist_pending_edit(pending_id: str, data: bytes) -> str:
    """Write staged pending-edit PNG; return durable reference."""
    return put_bytes(data, pending_edit_key(pending_id), content_type="image/png")


def persist_pending_edit_thumb(pending_id: str, data: bytes) -> str:
    return put_bytes(
        data, pending_edit_thumb_key(pending_id), content_type="image/jpeg"
    )


def thumb_ref(image_id: str) -> str:
    """Canonical durable reference for an image's display thumbnail."""
    if SETTINGS.blob_storage_backend == "s3":
        return s3_uri(thumb_key(image_id))
    return str((SETTINGS.image_cache_dir / "thumbs" / f"{image_id}.jpg").resolve())


def ingest_log_key(run_id: str, when: Optional[datetime] = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    safe_id = safe_filename(run_id).replace(" ", "_")
    return _key(SETTINGS.s3_prefix, "ingest-logs", f"{stamp}_{safe_id}.txt")


def query_log_key(run_id: str, when: Optional[datetime] = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    safe_id = safe_filename(run_id).replace(" ", "_")[:32]
    return _key(SETTINGS.s3_prefix, "query-logs", f"{stamp}_{safe_id}.txt")


def index_backup_prefix(backup_id: Optional[str] = None) -> str:
    """S3 prefix for index snapshot vault objects."""
    if backup_id:
        return _key(SETTINGS.s3_prefix, "index-backups", safe_filename(backup_id))
    return _key(SETTINGS.s3_prefix, "index-backups")


def index_backup_key(backup_id: str, filename: str) -> str:
    return _key(index_backup_prefix(backup_id), safe_filename(filename))


def list_keys(prefix: str, *, max_keys: Optional[int] = None) -> list[str]:
    """List object keys under a prefix in the configured S3 bucket."""
    return [obj.key for obj in list_objects(prefix, max_keys=max_keys)]


def list_objects(prefix: str, *, max_keys: Optional[int] = None) -> list[ListedObject]:
    """List objects under a prefix, including LastModified when present."""
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        raise ValueError("Listing objects requires BLOB_STORAGE_BACKEND=s3 and S3_BUCKET")
    client = get_s3_client()
    objects: list[ListedObject] = []
    token: Optional[str] = None
    while True:
        kwargs: dict[str, Any] = {
            "Bucket": SETTINGS.s3_bucket,
            "Prefix": prefix,
        }
        if token:
            kwargs["ContinuationToken"] = token
        page_size = 1000
        if max_keys is not None:
            remaining = max_keys - len(objects)
            if remaining <= 0:
                break
            page_size = min(1000, remaining)
        kwargs["MaxKeys"] = page_size
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents") or []:
            key = item.get("Key")
            if not key:
                continue
            modified = item.get("LastModified")
            if isinstance(modified, datetime) and modified.tzinfo is None:
                modified = modified.replace(tzinfo=timezone.utc)
            objects.append(ListedObject(key=str(key), last_modified=modified))
            if max_keys is not None and len(objects) >= max_keys:
                return objects
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            break
    return objects


def s3_uri(key: str) -> str:
    if not SETTINGS.s3_bucket:
        raise ValueError("S3_BUCKET is not configured")
    return f"s3://{SETTINGS.s3_bucket}/{key}"


@lru_cache(maxsize=8)
def _s3_client(region: str, endpoint_url: str = "") -> Any:
    import boto3
    from botocore.config import Config

    config_kwargs: dict[str, Any] = {
        "connect_timeout": SETTINGS.s3_connect_timeout,
        "read_timeout": SETTINGS.s3_read_timeout,
        "retries": {"max_attempts": SETTINGS.s3_max_retries, "mode": "adaptive"},
    }
    if endpoint_url:
        # MinIO and other path-style endpoints need this; AWS virtual-host is fine otherwise.
        config_kwargs["s3"] = {"addressing_style": "path"}
    kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": region,
        "config": Config(**config_kwargs),
    }
    if endpoint_url:
        # Explicit endpoint wins over AWS_ENDPOINT_URL_S3 (needed so browser
        # presigns can target localhost while the app still talks to minio:9000).
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client(**kwargs)


def get_s3_client() -> Any:
    return _s3_client(SETTINGS.s3_region, "")


def get_presign_s3_client() -> Any:
    """Client used only to sign browser-facing upload URLs."""
    endpoint = SETTINGS.s3_presign_endpoint_url or ""
    return _s3_client(SETTINGS.s3_region, endpoint)


def presign_upload(
    key: str,
    *,
    content_type: Optional[str] = None,
    expires_in: Optional[int] = None,
) -> tuple[str, dict[str, str]]:
    """Create a scoped PUT URL and the headers the browser must send."""
    if SETTINGS.blob_storage_backend != "s3" or not SETTINGS.s3_bucket:
        raise ValueError("Direct uploads require BLOB_STORAGE_BACKEND=s3")
    headers: dict[str, str] = {}
    params: dict[str, Any] = {"Bucket": SETTINGS.s3_bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
        headers["Content-Type"] = content_type
    expiry = max(60, min(expires_in or SETTINGS.s3_presign_expiry_sec, 86400))
    url = get_presign_s3_client().generate_presigned_url(
        "put_object",
        Params=params,
        ExpiresIn=expiry,
        HttpMethod="PUT",
    )
    return url, headers


def validate_uploaded_object(ref: str, *, expected_size: int) -> None:
    """Require a staged S3 object with the exact manifest byte count."""
    bucket, key = parse_s3_uri(ref)
    response = get_s3_client().head_object(Bucket=bucket, Key=key)
    actual = int(response.get("ContentLength", -1))
    if actual != int(expected_size):
        raise ValueError(f"size mismatch for {PurePosixPath(key).name}: expected {expected_size}, got {actual}")


def promote_staged_source(ref: str, local_path: Path) -> str:
    """Copy a staged source to its content-addressed durable S3 key."""
    if not is_s3_uri(ref):
        return persist_source(local_path)
    digest = hashlib.sha256()
    with local_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    source_bucket, source_object_key = parse_s3_uri(ref)
    # The staged key ends with the user's real filename; local_path is a
    # NamedTemporaryFile (tmpXXXX.pptx) whose name must not become provenance.
    original_name = PurePosixPath(source_object_key).name or local_path.name
    durable_key = source_key(original_name, digest.hexdigest())
    get_s3_client().copy_object(
        Bucket=SETTINGS.s3_bucket,
        Key=durable_key,
        CopySource={"Bucket": source_bucket, "Key": source_object_key},
        MetadataDirective="COPY",
    )
    return s3_uri(durable_key)


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
    # Pending-edit blobs (including their thumbs) always live under data_dir
    # using the full key path — do not collide with corpus image thumbs.
    if "pending-edits" in relative.parts:
        path = SETTINGS.data_dir.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path.resolve())

    namespace = relative.parts[-2] if len(relative.parts) >= 2 else ""
    if namespace == "images":
        path = SETTINGS.image_cache_dir / relative.name
    elif namespace == "thumbs":
        path = SETTINGS.image_cache_dir / "thumbs" / relative.name
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


def persist_image_thumb(image_id: str, data: bytes) -> str:
    """Write the display thumbnail (JPEG). Overwrites the same key — one thumb per id."""
    if SETTINGS.blob_storage_backend == "s3":
        return put_bytes(data, thumb_key(image_id), content_type="image/jpeg")
    out_path = SETTINGS.image_cache_dir / "thumbs" / f"{image_id}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return str(out_path.resolve())


def thumb_exists(image_id: str) -> bool:
    return exists(thumb_ref(image_id))


def thumbs_prefix() -> str:
    return _key(SETTINGS.s3_prefix, "thumbs")


def images_prefix() -> str:
    return _key(SETTINGS.s3_prefix, "images")


def uploads_prefix() -> str:
    return _key(SETTINGS.s3_prefix, "uploads")


def staging_prefix() -> str:
    return _key(SETTINGS.s3_prefix, "staging")


def list_thumb_ids() -> set[str]:
    """Return image_ids that have a display thumbnail object (one listing, no per-id HEAD)."""
    if SETTINGS.blob_storage_backend == "s3":
        prefix = thumbs_prefix().rstrip("/") + "/"
        try:
            keys = list_keys(prefix)
        except Exception as exc:  # noqa: BLE001 — health scans must not fail hard
            logger.warning("Could not list thumb keys under %s: %s", prefix, exc)
            return set()
        ids: set[str] = set()
        for key in keys:
            name = PurePosixPath(key).name
            if name.lower().endswith(".jpg"):
                ids.add(name[:-4])
        return ids

    thumbs_dir = SETTINGS.image_cache_dir / "thumbs"
    if not thumbs_dir.is_dir():
        return set()
    return {path.stem for path in thumbs_dir.glob("*.jpg") if path.is_file()}


def list_image_ids() -> set[str]:
    """Return image_ids that have a display PNG object (one listing, no per-id HEAD)."""
    if SETTINGS.blob_storage_backend == "s3":
        prefix = images_prefix().rstrip("/") + "/"
        try:
            keys = list_keys(prefix)
        except Exception as exc:  # noqa: BLE001 — health scans must not fail hard
            logger.warning("Could not list image keys under %s: %s", prefix, exc)
            return set()
        ids: set[str] = set()
        for key in keys:
            name = PurePosixPath(key).name
            if name.lower().endswith(".png"):
                ids.add(name[:-4])
        return ids

    images_dir = SETTINGS.image_cache_dir
    if not images_dir.is_dir():
        return set()
    return {path.stem for path in images_dir.glob("*.png") if path.is_file()}


def list_upload_uris() -> set[str]:
    """Return durable upload object URIs under the uploads prefix."""
    if SETTINGS.blob_storage_backend != "s3":
        return set()
    prefix = uploads_prefix().rstrip("/") + "/"
    try:
        keys = list_keys(prefix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list upload keys under %s: %s", prefix, exc)
        return set()
    return {s3_uri(key) for key in keys}


def list_staging_uris() -> set[str]:
    """Return staging object URIs under the staging prefix."""
    if SETTINGS.blob_storage_backend != "s3":
        return set()
    prefix = staging_prefix().rstrip("/") + "/"
    try:
        keys = list_keys(prefix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list staging keys under %s: %s", prefix, exc)
        return set()
    return {s3_uri(key) for key in keys}


def is_missing_blob_error(exc: BaseException) -> bool:
    """True when an S3/local error indicates the object is absent."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound", "404 Not Found"}:
            return True
    status = getattr(exc, "status", None) or getattr(exc, "response", None)
    if status == 404:
        return True
    msg = str(exc).lower()
    return "nosuchkey" in msg or "not found" in msg or "404" in msg


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
    except Exception as exc:  # botocore maps HEAD failures to ClientError
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        logger.error("Could not check blob %s: %s", ref, exc)
        raise


def delete(ref: str | Path | None, *, fallbacks: Sequence[Path] = ()) -> bool:
    """Delete a durable blob. Missing objects are success; operational S3 errors raise.

    Returns True when at least one local file was removed or an S3 delete was issued.
    """
    if not ref:
        return False

    removed = False
    if not is_s3_uri(ref):
        path = Path(ref).expanduser()
        if path.is_file():
            path.unlink()
            removed = True

    for candidate in fallbacks:
        path = Path(candidate).expanduser()
        if path.is_file():
            path.unlink()
            removed = True

    if is_s3_uri(ref):
        bucket, key = parse_s3_uri(str(ref))
        try:
            get_s3_client().delete_object(Bucket=bucket, Key=key)
            removed = True
        except Exception as exc:  # botocore maps delete failures to ClientError
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return removed
            logger.error("Could not delete blob %s: %s", ref, exc)
            raise
    return removed


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
