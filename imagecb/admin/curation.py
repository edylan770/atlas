"""Corpus curation: soft delete, restore, orphans."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from sqlalchemy import select

from imagecb.config import SETTINGS
from imagecb.ingest import _chroma_metadata
from imagecb.models.embedder import get_embedder
from imagecb.images import resize_for_model
from imagecb.paths import open_record_image
from imagecb.storage import bm25_index, metadata_db, vector_store
from imagecb.storage.metadata_db import ImageRecord, get_all_records, session_scope
from imagecb.telemetry.models import InteractionEvent, SearchEvent
from imagecb.telemetry.schema import ensure_telemetry_schema
from imagecb.admin.audit import append_audit
from imagecb.caption.quality import needs_regeneration

_VALID_CAPTION_QUALITY_FILTERS = frozenset({"all", "ok", "weak", "failed"})


def rebuild_bm25_active() -> None:
    bm25_index.rebuild_from_records(get_all_records(include_deleted=False))


def soft_delete_image(*, image_id: str, actor: str) -> None:
    ensure_telemetry_schema()
    with session_scope() as s:
        row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == image_id)
        ).scalar_one_or_none()
        if row is None:
            raise ValueError("image not found")
        if row.deleted_at is not None:
            raise ValueError("image already soft-deleted")
        row.deleted_at = datetime.utcnow()
        row.deleted_by = actor

    vector_store.delete([image_id])
    vector_store.delete_text([image_id])
    rebuild_bm25_active()
    append_audit(
        actor=actor,
        action="soft_delete",
        target_type="image",
        target_id=image_id,
        details={},
    )


def soft_delete_unrecoverable(*, actor: str) -> dict:
    """Bulk soft-delete rows with missing cache PNG and missing source file."""
    from imagecb.repair import assess_index_health

    ensure_telemetry_schema()
    report = assess_index_health(include_weak=False)
    image_ids = [r.image_id for r in report.unrecoverable_records]
    if not image_ids:
        return {"deleted": 0, "image_ids": []}

    now = datetime.utcnow()
    id_set = set(image_ids)
    with session_scope() as s:
        rows = s.execute(
            select(ImageRecord).where(ImageRecord.image_id.in_(image_ids))
        ).scalars().all()
        for row in rows:
            if row.deleted_at is not None:
                continue
            if row.image_id not in id_set:
                continue
            row.deleted_at = now
            row.deleted_by = actor

    vector_store.delete(image_ids)
    vector_store.delete_text(image_ids)
    rebuild_bm25_active()
    append_audit(
        actor=actor,
        action="purge_unrecoverable",
        target_type="corpus",
        target_id="unrecoverable",
        details={
            "deleted": len(image_ids),
            "image_ids_sample": image_ids[:20],
        },
    )
    return {"deleted": len(image_ids), "image_ids": image_ids}


def restore_image(*, image_id: str, actor: str) -> None:
    ensure_telemetry_schema()
    with session_scope() as s:
        row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == image_id)
        ).scalar_one_or_none()
        if row is None:
            raise ValueError("image not found")
        if row.deleted_at is None:
            raise ValueError("image is not deleted")
        record = row
        s.expunge(record)

    img = open_record_image(record)
    if img is None:
        raise ValueError("cached image file missing; cannot restore embedding")

    img = resize_for_model(img, SETTINGS.ingest_max_image_side)
    embedder = get_embedder()
    emb = embedder.embed_image(img)
    if isinstance(emb, np.ndarray) and emb.ndim == 1:
        emb = emb.reshape(1, -1)

    vector_store.upsert(
        image_ids=[image_id],
        embeddings=emb,
        metadatas=[_chroma_metadata(record)],
    )
    from imagecb.repair import refresh_text_vector

    refresh_text_vector(record)
    with session_scope() as s:
        row = s.execute(
            select(ImageRecord).where(ImageRecord.image_id == image_id)
        ).scalar_one()
        row.deleted_at = None
        row.deleted_by = None
    rebuild_bm25_active()
    append_audit(
        actor=actor,
        action="restore",
        target_type="image",
        target_id=image_id,
        details={},
    )


def _all_served_image_ids() -> set[str]:
    ensure_telemetry_schema()
    served: set[str] = set()
    with session_scope() as s:
        rows = s.execute(select(SearchEvent.served_image_ids_json)).all()
        for (raw,) in rows:
            try:
                ids = json.loads(raw or "[]")
                if isinstance(ids, list):
                    served.update(str(x) for x in ids)
            except json.JSONDecodeError:
                continue
    return served


def _all_interacted_image_ids() -> set[str]:
    ensure_telemetry_schema()
    with session_scope() as s:
        rows = s.execute(select(InteractionEvent.image_id).distinct()).all()
        return {r[0] for r in rows}


def corpus_health_summary() -> dict:
    """Index health summary for admin dashboard and corpus toolbar."""
    from imagecb.repair import assess_index_health

    report = assess_index_health(include_weak=True)
    payload = report.to_dict()
    payload["total_images"] = report.total_records
    return payload


def list_corpus_images(
    *,
    sort: str = "newest",
    caption_quality: Optional[str] = None,
) -> List[dict]:
    """All active indexed images for admin corpus browser."""
    from imagecb.retrieval.sort import resolve_sort, sort_image_records

    quality_filter = (caption_quality or "all").lower()
    if quality_filter not in _VALID_CAPTION_QUALITY_FILTERS:
        raise ValueError(
            f"invalid caption_quality: {caption_quality!r}; "
            f"expected one of {sorted(_VALID_CAPTION_QUALITY_FILTERS)}"
        )

    resolved = resolve_sort(sort, is_search=False)
    active = sort_image_records(get_all_records(include_deleted=False), resolved)
    out: List[dict] = []
    for r in active:
        quality = (r.caption_quality or "ok").lower()
        if quality_filter != "all" and quality != quality_filter:
            continue
        image_name = (r.image_name or "").strip() or Path(r.source_file or "").name
        created_at = r.created_at.isoformat() if r.created_at else None
        out.append(
            {
                "image_id": r.image_id,
                "caption_short": r.caption_short,
                "image_name": image_name,
                "source_file": r.source_file or "",
                "source_type": r.source_type,
                "author": r.author,
                "image_url": f"/api/images/{r.image_id}",
                "caption_quality": quality,
                "needs_regeneration": needs_regeneration(quality),
                "created_at": created_at,
            }
        )
    return out


def list_orphans(*, never_interacted: bool = False) -> List[dict]:
    active = get_all_records(include_deleted=False)
    served = _all_served_image_ids()
    interacted = _all_interacted_image_ids() if never_interacted else set()

    out: List[dict] = []
    for r in active:
        if r.image_id in served:
            continue
        if never_interacted and r.image_id in interacted:
            continue
        out.append(
            {
                "image_id": r.image_id,
                "caption_short": r.caption_short,
                "source_file": Path(r.source_file or "").name,
                "source_type": r.source_type,
                "image_url": f"/api/images/{r.image_id}",
            }
        )
    return out


def list_soft_deleted() -> List[dict]:
    rows = metadata_db.get_deleted_records()
    return [
        {
            "image_id": r.image_id,
            "deleted_at": r.deleted_at.isoformat() if r.deleted_at else None,
            "deleted_by": r.deleted_by,
            "caption_short": r.caption_short,
            "source_file": Path(r.source_file or "").name,
        }
        for r in rows
    ]
