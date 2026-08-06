"""Admin API routes (API key protected)."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from imagecb.admin import analytics, audit, curation, duplicates
from imagecb.api.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class PurgeUnrecoverableRequest(BaseModel):
    image_ids: Optional[List[str]] = Field(default=None)


class PurgeOrphanBlobsRequest(BaseModel):
    dry_run: bool = True
    min_age_hours: float = Field(default=1.0, ge=0.25, le=24 * 30)


@router.get("/analytics/summary")
def admin_analytics_summary(
    since: Optional[str] = Query(None),
    days: int = Query(90, ge=1, le=365),
    _: str = Depends(require_admin),
):
    return analytics.analytics_summary(since=since, days=days)


@router.get("/analytics/search-quality")
def admin_search_quality(
    since: Optional[str] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
    weak_score_threshold: Optional[float] = Query(None),
    _: str = Depends(require_admin),
):
    resolved_since = since
    if resolved_since is None and days is not None:
        from datetime import datetime, timedelta

        resolved_since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    elif resolved_since is None:
        from datetime import datetime, timedelta

        from imagecb.config import SETTINGS

        resolved_since = (
            datetime.utcnow() - timedelta(days=SETTINGS.telemetry_default_window_days)
        ).isoformat()
    return analytics.search_quality_lists(
        since=resolved_since,
        limit=limit,
        weak_score_threshold=weak_score_threshold,
    )


@router.get("/audit")
def admin_audit_log(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_admin),
):
    return {"entries": audit.list_audit_entries(limit=limit, offset=offset)}


@router.get("/corpus/health")
def admin_corpus_health(
    response: Response,
    _: str = Depends(require_admin),
):
    response.headers["Cache-Control"] = "no-store"
    return curation.corpus_health_summary()


@router.get("/index/health")
def admin_index_health(
    id_limit: int = Query(20, ge=1, le=100),
    _: str = Depends(require_admin),
):
    from imagecb.repair import assess_index_health

    report = assess_index_health(include_weak=True)
    return report.to_dict(id_sample_limit=id_limit)


@router.post("/index/reconcile")
def admin_index_reconcile(
    actor: str = Depends(require_admin),
):
    from imagecb.repair import reconcile_index_safe

    stats = reconcile_index_safe()
    audit.append_audit(
        actor=actor,
        action="index_reconcile",
        target_type="index",
        target_id="corpus",
        details=stats,
    )
    return {"ok": True, **stats}


@router.post("/index/repair")
def admin_index_repair(
    include_weak: bool = Query(False),
    actor: str = Depends(require_admin),
):
    from imagecb.repair import repair_index_issues

    stats = repair_index_issues(include_weak_captions=include_weak)
    audit.append_audit(
        actor=actor,
        action="index_repair",
        target_type="index",
        target_id="corpus",
        details={
            "skipped": stats.get("skipped", False),
            "is_healthy": stats.get("is_healthy"),
            "elapsed_sec": stats.get("elapsed_sec"),
        },
    )
    return {"ok": True, **stats}


class IndexBackupRequest(BaseModel):
    label: Optional[str] = None


class IndexRestoreRequest(BaseModel):
    backup_id: str = Field(..., min_length=1)
    confirm: bool = False


@router.get("/index/backups")
def admin_index_backups(
    _: str = Depends(require_admin),
):
    from imagecb.storage.index_backup import IndexBackupError, list_backups

    try:
        return {"backups": list_backups()}
    except IndexBackupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/index/backup")
def admin_index_backup(
    body: IndexBackupRequest = IndexBackupRequest(),
    actor: str = Depends(require_admin),
):
    from imagecb.storage.index_backup import IndexBackupError, create_backup

    try:
        result = create_backup(label=body.label)
    except IndexBackupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Index backup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    audit.append_audit(
        actor=actor,
        action="index_backup",
        target_type="index",
        target_id=str(result.get("backup_id") or "unknown"),
        details={
            "archive_bytes": result.get("archive_bytes"),
            "label": result.get("label"),
            "s3_uri": result.get("s3_uri"),
        },
    )
    return result


@router.post("/index/restore")
def admin_index_restore(
    body: IndexRestoreRequest,
    actor: str = Depends(require_admin),
):
    from imagecb.storage.index_backup import IndexBackupError, restore_backup

    try:
        result = restore_backup(body.backup_id, confirm=body.confirm)
    except IndexBackupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Index restore failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    audit.append_audit(
        actor=actor,
        action="index_restore",
        target_type="index",
        target_id=body.backup_id,
        details={
            "archive_sha256": result.get("archive_sha256"),
            "s3_uri": result.get("s3_uri"),
        },
    )
    return result


@router.get("/corpus/images")
def admin_corpus_images(
    sort: Optional[str] = Query(None),
    caption_quality: Optional[str] = Query(None),
    _: str = Depends(require_admin),
):
    from imagecb.retrieval.sort import InvalidSortError, resolve_sort

    try:
        resolved = resolve_sort(sort, is_search=False)
    except InvalidSortError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        images = curation.list_corpus_images(sort=resolved, caption_quality=caption_quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"images": images}


@router.post("/corpus/repair-captions")
def admin_repair_captions(
    scope: Literal["failed", "weak"] = Query("failed"),
    actor: str = Depends(require_admin),
):
    from imagecb.repair import repair_failed_captions

    result = repair_failed_captions(scope=scope)
    audit.append_audit(
        actor=actor,
        action="repair_captions",
        target_type="corpus",
        target_id=scope,
        details={
            "scope": scope,
            "attempted": result.get("attempted", 0),
            "repaired": result.get("repaired", 0),
            "errors": result.get("errors", 0),
        },
    )
    return {"ok": True, **result}


@router.post("/corpus/regenerate-missing-thumbs")
def admin_regenerate_missing_thumbs(actor: str = Depends(require_admin)):
    from imagecb.repair import regenerate_missing_thumbs

    result = regenerate_missing_thumbs()
    audit.append_audit(
        actor=actor,
        action="regenerate_missing_thumbs",
        target_type="corpus",
        target_id="thumbs",
        details={
            "scanned": result.get("scanned", 0),
            "created": result.get("created", 0),
            "skipped": result.get("skipped", 0),
            "failed": result.get("failed", 0),
        },
    )
    return {"ok": True, **result}


@router.post("/corpus/purge-unrecoverable")
def admin_purge_unrecoverable(
    actor: str = Depends(require_admin),
    payload: PurgeUnrecoverableRequest = PurgeUnrecoverableRequest(),
):
    stats = curation.hard_purge_unrecoverable(
        actor=actor,
        image_ids=payload.image_ids,
    )
    return {"ok": True, **stats}


@router.get("/corpus/orphan-blobs")
def admin_orphan_blobs(
    min_age_hours: float = Query(1.0, ge=0.25, le=24 * 30),
    _: str = Depends(require_admin),
):
    from imagecb.admin.orphan_blobs import OrphanBlobError, assess_orphan_blobs

    try:
        report = assess_orphan_blobs(min_age_hours=min_age_hours)
    except OrphanBlobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, **report.to_dict()}


@router.post("/corpus/purge-orphan-blobs")
def admin_purge_orphan_blobs(
    actor: str = Depends(require_admin),
    payload: PurgeOrphanBlobsRequest = PurgeOrphanBlobsRequest(),
):
    from imagecb.admin.orphan_blobs import OrphanBlobError, purge_orphan_blobs

    try:
        result = purge_orphan_blobs(
            dry_run=payload.dry_run,
            min_age_hours=payload.min_age_hours,
        )
    except OrphanBlobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not payload.dry_run:
        audit.append_audit(
            actor=actor,
            action="purge_orphan_blobs",
            target_type="corpus",
            target_id="orphan-blobs",
            details={
                "min_age_hours": payload.min_age_hours,
                "deleted_count": result.get("deleted_count", 0),
                "orphan_image_count": result.get("orphan_image_count", 0),
                "orphan_thumb_count": result.get("orphan_thumb_count", 0),
                "orphan_upload_count": result.get("orphan_upload_count", 0),
                "orphan_staging_count": result.get("orphan_staging_count", 0),
                "failed_count": result.get("failed_count", 0),
            },
        )
    return {"ok": True, **result}


@router.get("/corpus/orphans")
def admin_orphans(
    never_interacted: bool = Query(False),
    _: str = Depends(require_admin),
):
    return {"orphans": curation.list_orphans(never_interacted=never_interacted)}


@router.get("/corpus/deleted")
def admin_deleted(
    _: str = Depends(require_admin),
):
    return {"deleted": curation.list_soft_deleted()}


@router.get("/corpus/duplicate-clusters")
def admin_duplicate_clusters(
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    _: str = Depends(require_admin),
):
    try:
        clusters = duplicates.find_duplicate_clusters(threshold=threshold)
        return {
            "clusters": clusters,
            "threshold": threshold if threshold is not None else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Duplicate cluster detection failed")
        return {
            "clusters": [],
            "threshold": threshold if threshold is not None else None,
            "error": str(exc),
        }


@router.post("/images/{image_id}/soft-delete")
def admin_soft_delete(
    image_id: str,
    actor: str = Depends(require_admin),
):
    try:
        curation.soft_delete_image(image_id=image_id, actor=actor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "image_id": image_id, "status": "soft_deleted"}


@router.post("/images/{image_id}/restore")
def admin_restore(
    image_id: str,
    actor: str = Depends(require_admin),
):
    try:
        curation.restore_image(image_id=image_id, actor=actor)
    except ValueError as exc:
        msg = str(exc)
        code = 404 if "not found" in msg or "missing" in msg else 400
        raise HTTPException(status_code=code, detail=msg) from exc
    return {"ok": True, "image_id": image_id, "status": "restored"}


@router.post("/images/{image_id}/regenerate-caption")
def admin_regenerate_caption(
    image_id: str,
    actor: str = Depends(require_admin),
):
    from imagecb.repair import regenerate_caption

    try:
        result = regenerate_caption(image_id)
    except ValueError as exc:
        msg = str(exc)
        code = 404 if "not found" in msg or "missing" in msg else 400
        raise HTTPException(status_code=code, detail=msg) from exc
    audit.append_audit(
        actor=actor,
        action="regenerate_caption",
        target_type="image",
        target_id=image_id,
        details={"caption_quality": result.get("caption_quality")},
    )
    return {"ok": True, **result}


@router.post("/images/{image_id}/reindex")
def admin_reindex_image(
    image_id: str,
    actor: str = Depends(require_admin),
):
    from imagecb.repair import reindex_image

    try:
        result = reindex_image(image_id)
    except ValueError as exc:
        msg = str(exc)
        code = 404 if "not found" in msg or "missing" in msg else 400
        raise HTTPException(status_code=code, detail=msg) from exc
    audit.append_audit(
        actor=actor,
        action="reindex",
        target_type="image",
        target_id=image_id,
        details={"caption_quality": result.get("caption_quality")},
    )
    return {"ok": True, **result}


@router.get("/pending-edits")
def admin_list_pending_edits(
    limit: int = Query(100, ge=1, le=500),
    _: str = Depends(require_admin),
):
    from imagecb.pending_edits import list_pending_edits

    return {"items": list_pending_edits(limit=limit)}


@router.get("/pending-edits/{pending_id}/image")
def admin_pending_edit_image(
    pending_id: str,
    _: str = Depends(require_admin),
):
    import io

    from fastapi.responses import StreamingResponse

    from imagecb.pending_edits import read_pending_image_bytes

    try:
        data = read_pending_image_bytes(pending_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="pending-{pending_id}.png"',
        },
    )


@router.get("/pending-edits/{pending_id}/thumb")
def admin_pending_edit_thumb(
    pending_id: str,
    _: str = Depends(require_admin),
):
    import io

    from fastapi.responses import StreamingResponse

    from imagecb.pending_edits import read_pending_image_bytes, read_pending_thumb_bytes

    try:
        data = read_pending_thumb_bytes(pending_id)
        if data is None:
            data = read_pending_image_bytes(pending_id)
            media = "image/png"
        else:
            media = "image/jpeg"
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/pending-edits/{pending_id}/accept")
def admin_accept_pending_edit(
    pending_id: str,
    actor: str = Depends(require_admin),
):
    from imagecb.ingest import IngestInProgressError
    from imagecb.pending_edits import accept_pending_edit

    try:
        result = accept_pending_edit(pending_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Accept pending edit failed: %s", pending_id)
        raise HTTPException(status_code=500, detail=f"accept failed: {exc}") from exc

    audit.append_audit(
        actor=actor,
        action="accept_pending_edit",
        target_type="pending_edit",
        target_id=pending_id,
        details={
            "source_image_id": result.get("source_image_id"),
            "new_image_id": result.get("new_image_id"),
        },
    )
    return {"ok": True, **result}


@router.post("/pending-edits/{pending_id}/decline")
def admin_decline_pending_edit(
    pending_id: str,
    actor: str = Depends(require_admin),
):
    from imagecb.pending_edits import decline_pending_edit

    try:
        snapshot = decline_pending_edit(pending_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    audit.append_audit(
        actor=actor,
        action="decline_pending_edit",
        target_type="pending_edit",
        target_id=pending_id,
        details={"source_image_id": snapshot.get("source_image_id")},
    )
    return {"ok": True, "pending_id": pending_id, "status": "declined"}
