"""Corpus curation: soft delete, restore, orphans."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from sqlalchemy import delete, select

from imagecb.config import SETTINGS
from imagecb.ingest import _chroma_metadata
from imagecb.models.embedder import get_embedder
from imagecb.images import resize_for_model
from imagecb.paths import (
    image_exists,
    image_fallbacks,
    open_record_image,
    resolve_source_file,
    source_exists,
    source_fallbacks,
)
from imagecb.storage import blob_store, bm25_index, metadata_db, vector_store
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


def _is_unrecoverable(record: ImageRecord) -> bool:
    if image_exists(record):
        return False
    if resolve_source_file(record) is not None:
        return False
    return not source_exists(record)


def _delete_record_blobs(
    record: ImageRecord,
    *,
    purge_source: bool,
) -> tuple[int, int]:
    """Delete residual blobs for a purged row. Returns (deleted, skipped)."""
    deleted = 0
    skipped = 0

    if blob_store.delete(record.image_path, fallbacks=image_fallbacks(record)):
        deleted += 1
    else:
        skipped += 1

    if purge_source and record.source_file:
        if blob_store.delete(record.source_file, fallbacks=source_fallbacks(record)):
            deleted += 1
        else:
            skipped += 1
    elif record.source_file:
        skipped += 1

    return deleted, skipped


def hard_purge_unrecoverable(
    *,
    actor: str,
    image_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Permanently remove unrecoverable rows, vectors, and residual blobs."""
    from imagecb.repair import assess_index_health

    ensure_telemetry_schema()
    if image_ids is None:
        report = assess_index_health(include_weak=False)
        candidate_ids = list(
            dict.fromkeys(r.image_id for r in report.unrecoverable_records)
        )
    else:
        candidate_ids = list(dict.fromkeys(str(i) for i in image_ids if i))

    deleted_ids: list[str] = []
    files_deleted = 0
    files_skipped = 0
    purge_snapshots: list[tuple[ImageRecord, bool]] = []

    if candidate_ids:
        with session_scope() as s:
            rows = list(
                s.execute(
                    select(ImageRecord).where(
                        ImageRecord.image_id.in_(candidate_ids),
                        ImageRecord.deleted_at.is_(None),
                    )
                ).scalars().all()
            )
            for row in rows:
                s.expunge(row)

        eligible = [r for r in rows if _is_unrecoverable(r)]
        eligible_ids = [r.image_id for r in eligible]

        if eligible:
            sources = {
                (r.source_file or "").strip()
                for r in eligible
                if (r.source_file or "").strip()
            }
            source_counts: dict[str, int] = {}
            with session_scope() as s:
                for source in sources:
                    count = s.execute(
                        select(ImageRecord.image_id).where(
                            ImageRecord.source_file == source,
                            ImageRecord.deleted_at.is_(None),
                        )
                    ).all()
                    source_counts[source] = len(count)

            for record in eligible:
                source = (record.source_file or "").strip()
                purge_source = bool(source) and source_counts.get(source, 0) <= 1
                purge_snapshots.append((record, purge_source))

            with session_scope() as s:
                s.execute(
                    delete(ImageRecord).where(
                        ImageRecord.image_id.in_(eligible_ids)
                    )
                )
            deleted_ids = eligible_ids

    if deleted_ids:
        vector_store.delete(deleted_ids)
        vector_store.delete_text(deleted_ids)
        rebuild_bm25_active()

    for record, purge_source in purge_snapshots:
        d, sk = _delete_record_blobs(record, purge_source=purge_source)
        files_deleted += d
        files_skipped += sk

    stats = {
        "candidates": len(candidate_ids),
        "deleted": len(deleted_ids),
        "skipped": len(candidate_ids) - len(deleted_ids),
        "files_deleted": files_deleted,
        "files_skipped": files_skipped,
        "image_ids": deleted_ids,
    }
    append_audit(
        actor=actor,
        action="purge_unrecoverable",
        target_type="corpus",
        target_id="unrecoverable",
        details={
            "candidates": stats["candidates"],
            "deleted": stats["deleted"],
            "skipped": stats["skipped"],
            "files_deleted": files_deleted,
            "files_skipped": files_skipped,
            "image_ids_sample": deleted_ids[:20],
        },
    )
    return stats


# Backwards-compatible alias for older callers/tests.
soft_delete_unrecoverable = hard_purge_unrecoverable


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
    payload["unrecoverable_image_ids"] = [
        r.image_id for r in report.unrecoverable_records
    ]
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
