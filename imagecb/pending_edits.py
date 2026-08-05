"""Pending Nano Banana edits awaiting admin accept/decline."""

from __future__ import annotations

import hashlib
import io
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
from sqlalchemy import select

from imagecb.config import SETTINGS
from imagecb.images import make_thumbnail
from imagecb.storage import blob_store, metadata_db
from imagecb.storage.metadata_db import PendingEdit, get_engine, session_scope

logger = logging.getLogger(__name__)


def ensure_pending_edits_schema() -> None:
    engine = get_engine()
    PendingEdit.__table__.create(engine, checkfirst=True)


def _png_content_hash(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def create_pending_edit(
    *,
    source_image_id: str,
    image_bytes: bytes,
    last_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Stage edited PNG bytes and insert a pending row."""
    ensure_pending_edits_schema()
    pending_id = str(uuid.uuid4())
    staged_ref = blob_store.persist_pending_edit(pending_id, image_bytes)
    thumb_ref: Optional[str] = None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        thumb_bytes = make_thumbnail(img)
        thumb_ref = blob_store.persist_pending_edit_thumb(pending_id, thumb_bytes)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write pending-edit thumb for %s", pending_id, exc_info=True)

    with session_scope() as s:
        s.add(
            PendingEdit(
                pending_id=pending_id,
                source_image_id=source_image_id,
                staged_ref=staged_ref,
                thumb_ref=thumb_ref,
                last_prompt=(last_prompt or "").strip() or None,
                status="pending",
                created_at=datetime.utcnow(),
            )
        )
    return get_pending_edit(pending_id) or {"pending_id": pending_id}


def get_pending_edit(pending_id: str) -> Optional[dict[str, Any]]:
    ensure_pending_edits_schema()
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is None:
            return None
        return _to_dict(row)


def list_pending_edits(*, limit: int = 100) -> List[dict[str, Any]]:
    ensure_pending_edits_schema()
    with session_scope() as s:
        rows = (
            s.execute(
                select(PendingEdit)
                .where(PendingEdit.status == "pending")
                .order_by(PendingEdit.created_at.desc())
                .limit(max(1, min(limit, 500)))
            )
            .scalars()
            .all()
        )
        return [_to_dict(r) for r in rows]


def _to_dict(row: PendingEdit) -> dict[str, Any]:
    created = row.created_at.isoformat() + "Z" if row.created_at else None
    return {
        "pending_id": row.pending_id,
        "source_image_id": row.source_image_id,
        "staged_ref": row.staged_ref,
        "thumb_ref": row.thumb_ref,
        "last_prompt": row.last_prompt,
        "status": row.status,
        "created_at": created,
        "image_url": f"/api/edit/pending/{row.pending_id}/image",
        "thumb_url": f"/api/edit/pending/{row.pending_id}/thumb",
    }


def delete_pending_artifacts(row: PendingEdit) -> None:
    """Delete staged blobs for a pending edit (not the source corpus image)."""
    for ref in (row.staged_ref, row.thumb_ref):
        if not ref:
            continue
        try:
            blob_store.delete(ref)
        except Exception:  # noqa: BLE001
            logger.warning("Failed deleting pending artifact %s", ref, exc_info=True)
    # Also try canonical keys in case refs were absolute paths that moved.
    for key_fn in (blob_store.pending_edit_key, blob_store.pending_edit_thumb_key):
        try:
            key = key_fn(row.pending_id)
            if SETTINGS.blob_storage_backend == "s3":
                blob_store.delete(blob_store.s3_uri(key))
            else:
                path = SETTINGS.data_dir.joinpath(*Path(key).parts)
                blob_store.delete(path)
        except Exception:  # noqa: BLE001
            pass


def decline_pending_edit(pending_id: str) -> dict[str, Any]:
    ensure_pending_edits_schema()
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is None:
            raise KeyError(pending_id)
        if row.status != "pending":
            raise ValueError(f"pending edit {pending_id} is {row.status}")
        snapshot = _to_dict(row)
        delete_pending_artifacts(row)
        s.delete(row)
    return snapshot


def accept_pending_edit(pending_id: str) -> dict[str, Any]:
    """Full-ingest the staged image as a new corpus record; set parent_image_id."""
    from imagecb.ingest import ingest_paths

    ensure_pending_edits_schema()
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is None:
            raise KeyError(pending_id)
        if row.status != "pending":
            raise ValueError(f"pending edit {pending_id} is {row.status}")
        source_image_id = row.source_image_id
        staged_ref = row.staged_ref
        last_prompt = row.last_prompt

    data = blob_store.read_bytes(staged_ref)
    content_hash = _png_content_hash(data)

    SETTINGS.ensure_dirs()
    with tempfile.TemporaryDirectory(prefix="nano-banana-accept-") as tmp:
        filename = f"nano-banana-{source_image_id}-{pending_id}.png"
        path = Path(tmp) / filename
        path.write_bytes(data)
        stats = ingest_paths([path], auto_repair=True)

    record = metadata_db.get_record_by_hash(content_hash)
    new_image_id: Optional[str] = None
    if record is not None:
        new_image_id = record.image_id
        with session_scope() as s:
            rec = s.get(metadata_db.ImageRecord, new_image_id)
            if rec is not None:
                rec.parent_image_id = source_image_id

    # Remove pending row + staged blobs (corpus blobs remain).
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is not None:
            delete_pending_artifacts(row)
            s.delete(row)

    return {
        "pending_id": pending_id,
        "source_image_id": source_image_id,
        "new_image_id": new_image_id,
        "last_prompt": last_prompt,
        "ingest_stats": stats,
    }


def read_pending_image_bytes(pending_id: str) -> bytes:
    ensure_pending_edits_schema()
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is None:
            raise KeyError(pending_id)
        ref = row.staged_ref
    return blob_store.read_bytes(ref)


def read_pending_thumb_bytes(pending_id: str) -> Optional[bytes]:
    ensure_pending_edits_schema()
    with session_scope() as s:
        row = s.get(PendingEdit, pending_id)
        if row is None:
            raise KeyError(pending_id)
        ref = row.thumb_ref
    if not ref:
        return None
    try:
        return blob_store.read_bytes(ref)
    except Exception:  # noqa: BLE001
        return None
