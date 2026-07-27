"""Index health assessment and targeted repair without full corpus re-ingest."""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from imagecb.caption.context import caption_context_from_record
from imagecb.caption.document import caption_document_text
from imagecb.caption.normalize import normalize_tags
from imagecb.caption.pipeline import enrich_caption_search_terms
from imagecb.caption.quality import (
    CAPTION_FAILED,
    assess_caption,
    assess_caption_with_reasons,
    caption_json_from_record,
    needs_regeneration,
)
from imagecb.caption.vocab import load_tag_vocab
from imagecb.config import SETTINGS
from imagecb.ingest import _chroma_metadata, _flush_chroma_batch, embed_caption_document
from imagecb.ingest_context import embed_context_from_record
from imagecb.models.embedder import get_embedder
from imagecb.models.vlm import CaptionJSON, get_captioner
from imagecb.paths import (
    image_exists,
    materialize_source,
    open_record_image,
    resolve_source_file,
    source_exists,
)
from imagecb.images import make_thumbnail
from imagecb.storage import blob_store, bm25_index, metadata_db, vector_store
from imagecb.storage.metadata_db import ImageRecord, get_all_records, get_records, session_scope

logger = logging.getLogger(__name__)


@dataclass
class IndexHealthReport:
    total_records: int
    chroma_vectors: int
    missing_cache_count: int
    missing_cache_records: List[ImageRecord] = field(default_factory=list)
    failed_caption_count: int = 0
    failed_caption_records: List[ImageRecord] = field(default_factory=list)
    weak_caption_count: int = 0
    weak_caption_records: List[ImageRecord] = field(default_factory=list)
    needs_regeneration_count: int = 0
    missing_chroma_count: int = 0
    missing_chroma_ids: List[str] = field(default_factory=list)
    unrecoverable_source_missing_count: int = 0
    unrecoverable_records: List[ImageRecord] = field(default_factory=list)
    recoverable_source_files: List[str] = field(default_factory=list)
    missing_asset_type_count: int = 0
    missing_asset_type_ids: List[str] = field(default_factory=list)
    missing_text_vector_count: int = 0
    missing_text_vector_ids: List[str] = field(default_factory=list)
    orphan_chroma_count: int = 0
    orphan_chroma_ids: List[str] = field(default_factory=list)
    orphan_text_vector_count: int = 0
    orphan_text_vector_ids: List[str] = field(default_factory=list)
    stale_deleted_in_chroma_count: int = 0
    stale_deleted_in_chroma_ids: List[str] = field(default_factory=list)
    bm25_doc_count: int = 0
    bm25_missing_ids: List[str] = field(default_factory=list)
    bm25_orphan_ids: List[str] = field(default_factory=list)
    bm25_stale: bool = False
    text_vector_count: int = 0
    stores_in_sync: bool = True
    is_healthy: bool = True
    elapsed_sec: float = 0.0

    def to_dict(self, *, id_sample_limit: int = 20) -> Dict[str, Any]:
        """JSON-serializable summary for API responses."""

        def _sample(ids: List[str]) -> List[str]:
            return ids[:id_sample_limit]

        return {
            "total_records": self.total_records,
            "chroma_vectors": self.chroma_vectors,
            "text_vector_count": self.text_vector_count,
            "bm25_doc_count": self.bm25_doc_count,
            "missing_cache_count": self.missing_cache_count,
            "failed_caption_count": self.failed_caption_count,
            "weak_caption_count": self.weak_caption_count,
            "needs_regeneration_count": self.needs_regeneration_count,
            "missing_chroma_count": self.missing_chroma_count,
            "missing_text_vector_count": self.missing_text_vector_count,
            "orphan_chroma_count": self.orphan_chroma_count,
            "orphan_text_vector_count": self.orphan_text_vector_count,
            "stale_deleted_in_chroma_count": self.stale_deleted_in_chroma_count,
            "bm25_stale": self.bm25_stale,
            "bm25_missing_count": len(self.bm25_missing_ids),
            "bm25_orphan_count": len(self.bm25_orphan_ids),
            "missing_asset_type_count": self.missing_asset_type_count,
            "unrecoverable_source_missing_count": self.unrecoverable_source_missing_count,
            "stores_in_sync": self.stores_in_sync,
            "is_healthy": self.is_healthy,
            "elapsed_sec": self.elapsed_sec,
            "missing_chroma_ids": _sample(self.missing_chroma_ids),
            "missing_text_vector_ids": _sample(self.missing_text_vector_ids),
            "orphan_chroma_ids": _sample(self.orphan_chroma_ids),
            "orphan_text_vector_ids": _sample(self.orphan_text_vector_ids),
            "stale_deleted_in_chroma_ids": _sample(self.stale_deleted_in_chroma_ids),
            "bm25_missing_ids": _sample(self.bm25_missing_ids),
            "bm25_orphan_ids": _sample(self.bm25_orphan_ids),
            "recoverable_source_files": self.recoverable_source_files[:id_sample_limit],
        }


def _cache_missing(record: ImageRecord) -> bool:
    return not image_exists(record)


def _caption_failed(record: ImageRecord) -> bool:
    return (record.caption_short or "").strip() == CAPTION_FAILED or (
        (record.caption_quality or "").lower() == "failed"
    )


def _caption_weak(record: ImageRecord) -> bool:
    return (record.caption_quality or "").lower() == "weak"


def records_with_failed_captions() -> List[ImageRecord]:
    return [r for r in get_all_records() if _caption_failed(r)]


def records_with_weak_captions() -> List[ImageRecord]:
    return [r for r in get_all_records() if _caption_weak(r)]


def records_needing_regen(*, include_weak: bool = False) -> List[ImageRecord]:
    failed_ids = {r.image_id for r in records_with_failed_captions()}
    records = list(records_with_failed_captions())
    if include_weak:
        for r in records_with_weak_captions():
            if r.image_id not in failed_ids:
                records.append(r)
    return records


def records_missing_asset_type() -> List[ImageRecord]:
    """Active rows without a stored visual asset type."""
    return [
        r
        for r in get_all_records()
        if not (r.asset_type or "").strip()
    ]


def assess_index_health(*, include_weak: bool = False) -> IndexHealthReport:
    """Read-only scan of SQLite, cache files, and Chroma for known issue classes."""
    t0 = time.perf_counter()
    records = get_all_records(include_deleted=False)

    missing_cache_records = [r for r in records if _cache_missing(r)]
    failed_caption_records = [r for r in records if _caption_failed(r)]
    weak_caption_records = [r for r in records if _caption_weak(r)] if include_weak else []

    unrecoverable_records: List[ImageRecord] = []
    recoverable_source_files: Set[str] = set()
    for r in missing_cache_records:
        local_source = resolve_source_file(r)
        if local_source is not None:
            recoverable_source_files.add(str(local_source))
        elif source_exists(r):
            recoverable_source_files.add(r.source_file)
        else:
            unrecoverable_records.append(r)

    sqlite_ids = {r.image_id for r in records}
    try:
        chroma_vectors = vector_store.count()
        chroma_ids = vector_store.list_ids()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read Chroma collection: %s", exc)
        chroma_vectors = 0
        chroma_ids = set()

    missing_chroma_ids = sorted(sqlite_ids - chroma_ids)
    orphan_chroma_ids = sorted(chroma_ids - sqlite_ids)

    text_ids: Set[str] = set()
    text_vector_count = 0
    missing_text_vector_ids: List[str] = []
    orphan_text_vector_ids: List[str] = []
    if SETTINGS.caption_text_lane_enabled:
        try:
            text_ids = vector_store.list_text_ids()
            text_vector_count = len(text_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read caption-text collection: %s", exc)
            text_ids = set()
        embeddable = {r.image_id for r in records if caption_document_text(r).strip()}
        missing_text_vector_ids = sorted(embeddable - text_ids)
        orphan_text_vector_ids = sorted(text_ids - sqlite_ids)

    deleted_records = metadata_db.get_deleted_records()
    deleted_ids = {r.image_id for r in deleted_records}
    stale_deleted_in_chroma_ids = sorted(
        (chroma_ids | text_ids) & deleted_ids
    )

    bm25_ids: Set[str] = set()
    bm25_doc_count = 0
    bm25_stale = False
    bm25_missing_ids: List[str] = []
    bm25_orphan_ids: List[str] = []
    try:
        bm25_doc_count = bm25_index.count()
        bm25_ids = bm25_index.list_ids()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read BM25 index: %s", exc)
        bm25_stale = len(sqlite_ids) > 0

    if sqlite_ids or bm25_ids:
        bm25_missing_ids = sorted(sqlite_ids - bm25_ids)
        bm25_orphan_ids = sorted(bm25_ids - sqlite_ids)
        bm25_stale = bm25_stale or bool(bm25_missing_ids or bm25_orphan_ids)

    missing_cache_count = len(missing_cache_records)
    failed_caption_count = len(failed_caption_records)
    weak_caption_count = len(weak_caption_records)
    needs_regeneration_count = failed_caption_count + weak_caption_count
    missing_chroma_count = len(missing_chroma_ids)
    missing_text_vector_count = len(missing_text_vector_ids)
    orphan_chroma_count = len(orphan_chroma_ids)
    orphan_text_vector_count = len(orphan_text_vector_ids)
    stale_deleted_in_chroma_count = len(stale_deleted_in_chroma_ids)
    missing_asset_type_records = records_missing_asset_type()
    missing_asset_type_count = len(missing_asset_type_records)
    missing_asset_type_ids = [r.image_id for r in missing_asset_type_records[:50]]

    stores_in_sync = (
        missing_chroma_count == 0
        and missing_text_vector_count == 0
        and orphan_chroma_count == 0
        and orphan_text_vector_count == 0
        and stale_deleted_in_chroma_count == 0
        and not bm25_stale
    )
    is_healthy = (
        stores_in_sync
        and missing_cache_count == 0
        and failed_caption_count == 0
    )

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "Index health assess: records=%s healthy=%s in_sync=%s missing_cache=%s "
        "failed_captions=%s missing_chroma=%s orphans=%s bm25_stale=%s (%.2fs)",
        len(records),
        is_healthy,
        stores_in_sync,
        missing_cache_count,
        failed_caption_count,
        missing_chroma_count,
        orphan_chroma_count + orphan_text_vector_count,
        bm25_stale,
        elapsed,
    )

    return IndexHealthReport(
        total_records=len(records),
        chroma_vectors=chroma_vectors,
        missing_cache_count=missing_cache_count,
        missing_cache_records=missing_cache_records,
        failed_caption_count=failed_caption_count,
        failed_caption_records=failed_caption_records,
        weak_caption_count=weak_caption_count,
        weak_caption_records=weak_caption_records,
        needs_regeneration_count=needs_regeneration_count,
        missing_chroma_count=missing_chroma_count,
        missing_chroma_ids=missing_chroma_ids,
        unrecoverable_source_missing_count=len(unrecoverable_records),
        unrecoverable_records=unrecoverable_records,
        recoverable_source_files=sorted(recoverable_source_files),
        missing_asset_type_count=missing_asset_type_count,
        missing_asset_type_ids=missing_asset_type_ids,
        missing_text_vector_count=missing_text_vector_count,
        missing_text_vector_ids=missing_text_vector_ids,
        orphan_chroma_count=orphan_chroma_count,
        orphan_chroma_ids=orphan_chroma_ids,
        orphan_text_vector_count=orphan_text_vector_count,
        orphan_text_vector_ids=orphan_text_vector_ids,
        stale_deleted_in_chroma_count=stale_deleted_in_chroma_count,
        stale_deleted_in_chroma_ids=stale_deleted_in_chroma_ids,
        bm25_doc_count=bm25_doc_count,
        bm25_missing_ids=bm25_missing_ids,
        bm25_orphan_ids=bm25_orphan_ids,
        bm25_stale=bm25_stale,
        text_vector_count=text_vector_count,
        stores_in_sync=stores_in_sync,
        is_healthy=is_healthy,
        elapsed_sec=elapsed,
    )


def reconcile_index_safe(*, dry_run: bool = False) -> dict:
    """Cheap cross-store fixes: purge orphan/stale vectors and rebuild BM25 when drifted."""
    t0 = time.perf_counter()
    report = assess_index_health(include_weak=True)

    purge_chroma = list(
        dict.fromkeys(report.orphan_chroma_ids + report.stale_deleted_in_chroma_ids)
    )
    purge_text = list(
        dict.fromkeys(report.orphan_text_vector_ids + report.stale_deleted_in_chroma_ids)
    )

    stats: Dict[str, Any] = {
        "dry_run": dry_run,
        "orphan_chroma_purged": 0,
        "orphan_text_purged": 0,
        "bm25_rebuilt": False,
        "stores_in_sync_before": report.stores_in_sync,
        "is_healthy_before": report.is_healthy,
    }

    if not dry_run:
        if purge_chroma:
            vector_store.delete(purge_chroma)
            stats["orphan_chroma_purged"] = len(purge_chroma)
        if purge_text:
            vector_store.delete_text(purge_text)
            stats["orphan_text_purged"] = len(purge_text)
        if report.bm25_stale or not bm25_index.is_loaded():
            bm25_index.rebuild_from_records(get_all_records(include_deleted=False))
            stats["bm25_rebuilt"] = True
    else:
        stats["orphan_chroma_purged"] = len(purge_chroma)
        stats["orphan_text_purged"] = len(purge_text)
        stats["bm25_rebuilt"] = report.bm25_stale or not bm25_index.is_loaded()

    final = assess_index_health(include_weak=True) if not dry_run else report
    stats["stores_in_sync_after"] = final.stores_in_sync
    stats["is_healthy_after"] = final.is_healthy
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    logger.info(
        "Index reconcile (dry_run=%s): chroma_purged=%s text_purged=%s bm25_rebuilt=%s "
        "in_sync %s -> %s (%.2fs)",
        dry_run,
        stats["orphan_chroma_purged"],
        stats["orphan_text_purged"],
        stats["bm25_rebuilt"],
        report.stores_in_sync,
        final.stores_in_sync,
        stats["elapsed_sec"],
    )
    return stats


def _load_cached_image(record: ImageRecord) -> Optional[Image.Image]:
    try:
        return open_record_image(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %s: %s", record.image_path, exc)
        return None


def _caption_from_record(record: ImageRecord, img: Image.Image) -> CaptionJSON:
    vocab = set(load_tag_vocab())
    ctx = caption_context_from_record(record)
    captioner = get_captioner()
    cap = captioner.caption_image(
        img,
        context=ctx,
        source_file=record.source_file,
    )
    if cap.short_caption != CAPTION_FAILED:
        cap.search.tags = normalize_tags(list(cap.search.tags), vocab)
        cap = enrich_caption_search_terms(cap)
        cap.caption_quality = assess_caption(cap)
    else:
        cap.caption_quality = "failed"
    return cap


def _apply_caption_to_record(record: ImageRecord, caption: CaptionJSON) -> None:
    if caption.image_name:
        record.image_name = caption.image_name
    record.caption_short = caption.short_caption
    record.caption_detailed = caption.detailed_description
    record.use_case = caption.use_case
    record.theme = caption.theme
    record.scene = caption.scene
    record.text_overlay_summary = caption.text_overlay_summary
    record.text_read_uncertain = 1 if caption.text_read_uncertain else 0
    record.objects_json = metadata_db.serialize_list(caption.objects)
    record.tags_json = metadata_db.serialize_list(caption.tags)
    record.recommended_cases_json = metadata_db.serialize_list(caption.recommended_cases)
    record.search_aliases_json = metadata_db.serialize_list(caption.aliases)
    record.caption_quality = caption.caption_quality or "ok"
    record.asset_type = caption.asset_type or None


def _persist_record(record: ImageRecord) -> None:
    with session_scope() as s:
        s.merge(record)


def _upsert_record_embedding(record: ImageRecord, embedding: np.ndarray) -> None:
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    vector_store.upsert(
        image_ids=[record.image_id],
        embeddings=embedding,
        metadatas=[_chroma_metadata(record)],
    )


def refresh_text_vector(record: ImageRecord) -> bool:
    """Re-embed the record's caption document into the text lane (fail-soft)."""
    emb = embed_caption_document(record)
    if emb is None:
        return False
    vector_store.upsert_text(
        image_ids=[record.image_id],
        embeddings=emb.reshape(1, -1),
    )
    return True


def _apply_regenerated_caption(
    record: ImageRecord,
    caption: CaptionJSON,
    *,
    embedding: Optional[np.ndarray] = None,
) -> None:
    """Persist caption fields and refresh search vectors."""
    _apply_caption_to_record(record, caption)
    _persist_record(record)
    if embedding is not None:
        _upsert_record_embedding(record, embedding)
    refresh_text_vector(record)


def _caption_with_retry(record: ImageRecord, img: Image.Image) -> CaptionJSON:
    caption = _caption_from_record(record, img)
    if caption.caption_quality in ("weak", "failed"):
        retry = _caption_from_record(record, img)
        if retry.caption_quality == "ok" or (
            retry.caption_quality == "weak" and caption.caption_quality == "failed"
        ):
            caption = retry
        elif retry.caption_quality == "weak" and caption.caption_quality == "weak":
            caption = retry
    return caption


def rescan_caption_quality(*, include_deleted: bool = False) -> dict:
    """Re-assess stored captions and update caption_quality flags (no VLM)."""
    t0 = time.perf_counter()
    records = get_all_records(include_deleted=include_deleted)
    stats = {
        "scanned": 0,
        "updated": 0,
        "ok": 0,
        "weak": 0,
        "failed": 0,
    }
    for record in records:
        stats["scanned"] += 1
        cap = caption_json_from_record(record)
        quality = assess_caption(cap)
        stats[quality] += 1
        prev = (record.caption_quality or "ok").lower()
        if quality != prev:
            record.caption_quality = quality
            _persist_record(record)
            stats["updated"] += 1
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def regenerate_caption(image_id: str, *, rebuild_bm25: bool = True) -> dict:
    """Re-caption one image, re-embed, and refresh BM25."""
    from imagecb.storage.metadata_db import get_record

    record = get_record(image_id)
    if record is None:
        raise ValueError("image not found")
    img = _load_cached_image(record)
    if img is None:
        raise ValueError("cached image missing")

    caption = _caption_with_retry(record, img)
    _apply_caption_to_record(record, caption)
    embedder = get_embedder()
    ctx = embed_context_from_record(record)
    emb = embedder.embed_image_with_context(img, ctx or None)
    _persist_record(record)
    _upsert_record_embedding(record, emb)
    refresh_text_vector(record)
    if rebuild_bm25:
        bm25_index.rebuild_from_records(get_all_records())

    _, reasons = assess_caption_with_reasons(caption)
    quality = caption.caption_quality or "ok"
    return {
        "image_id": image_id,
        "caption_quality": quality,
        "needs_regeneration": needs_regeneration(quality),
        "quality_reasons": reasons,
        "caption_short": caption.short_caption,
        "caption_detailed": caption.detailed_description,
        "image_name": caption.image_name,
        "tags": list(caption.tags),
    }


def reindex_image(image_id: str, *, rebuild_bm25: bool = True) -> dict:
    """Re-embed one image from its stored caption and refresh search indexes."""
    from imagecb.storage.metadata_db import get_record

    record = get_record(image_id)
    if record is None:
        raise ValueError("image not found")
    img = _load_cached_image(record)
    if img is None:
        raise ValueError("cached image missing")

    embedder = get_embedder()
    ctx = embed_context_from_record(record)
    emb = embedder.embed_image_with_context(img, ctx or None)
    _upsert_record_embedding(record, emb)
    refresh_text_vector(record)
    if rebuild_bm25:
        bm25_index.rebuild_from_records(get_all_records())

    quality = (record.caption_quality or "ok").lower()
    return {
        "image_id": image_id,
        "reindexed": True,
        "caption_short": record.caption_short,
        "caption_quality": quality,
    }


def _repair_one_caption(record: ImageRecord) -> Tuple[str, bool, Optional[str]]:
    img = _load_cached_image(record)
    if img is None:
        return record.image_id, False, "cached image missing"
    try:
        caption = _caption_with_retry(record, img)
        _apply_regenerated_caption(record, caption)
        return record.image_id, True, None
    except Exception as exc:  # noqa: BLE001
        return record.image_id, False, str(exc)


def repair_failed_captions(
    *,
    workers: Optional[int] = None,
    include_weak: bool = False,
    scope: Optional[Literal["failed", "weak", "failed_and_weak"]] = None,
    rebuild_bm25: bool = True,
) -> dict:
    """Re-caption rows where VLM failed or quality is weak."""
    workers = max(1, workers if workers is not None else SETTINGS.ingest_workers)
    if scope == "weak":
        records = records_with_weak_captions()
    elif scope == "failed":
        records = records_with_failed_captions()
    elif scope == "failed_and_weak":
        records = records_needing_regen(include_weak=True)
    else:
        records = records_needing_regen(include_weak=include_weak)
    stats = {
        "attempted": len(records),
        "repaired": 0,
        "errors": 0,
        "include_weak": include_weak,
        "scope": scope,
        "workers": workers,
    }
    if not records:
        return stats

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_repair_one_caption, r): r for r in records}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Repair captions",
            unit="img",
            disable=not sys.stderr.isatty(),
        )
        for future in pbar:
            _id, ok, err = future.result()
            if ok:
                stats["repaired"] += 1
            else:
                stats["errors"] += 1
                if err:
                    logger.warning("Caption repair failed for %s: %s", _id, err)

    if rebuild_bm25:
        bm25_index.rebuild_from_records(get_all_records())
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def regenerate_missing_thumbs() -> dict:
    """Create display thumbnails for active images that are missing one.

    Deterministic key per image_id — existing thumbs are skipped so this never
    creates duplicate thumbnail files for the same image.
    """
    records = get_all_records(include_deleted=False)
    stats: dict = {
        "scanned": len(records),
        "created": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    t0 = time.perf_counter()
    for record in records:
        image_id = record.image_id
        if blob_store.thumb_exists(image_id):
            stats["skipped"] += 1
            continue
        img = _load_cached_image(record)
        if img is None:
            stats["failed"] += 1
            if len(stats["errors"]) < 20:
                stats["errors"].append({"image_id": image_id, "error": "image blob not found"})
            continue
        try:
            blob_store.persist_image_thumb(image_id, make_thumbnail(img))
            stats["created"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            logger.warning("Thumb regenerate failed for %s: %s", image_id, exc)
            if len(stats["errors"]) < 20:
                stats["errors"].append({"image_id": image_id, "error": str(exc)})
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def backfill_asset_types(
    *,
    workers: Optional[int] = None,
    dry_run: bool = False,
    rebuild_bm25: bool = True,
    all_rows: bool = False,
) -> dict:
    """Re-caption rows to populate asset_type from cached PNGs (no re-upload)."""
    workers = max(1, workers if workers is not None else SETTINGS.ingest_workers)
    records = get_all_records() if all_rows else records_missing_asset_type()
    stats = {
        "scanned": len(records),
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "dry_run": dry_run,
        "all_rows": all_rows,
        "workers": workers,
    }
    if not records:
        return stats

    if dry_run:
        stats["skipped"] = len(records)
        return stats

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_repair_one_caption, r): r for r in records}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Backfill asset types",
            unit="img",
            disable=not sys.stderr.isatty(),
        )
        for future in pbar:
            _id, ok, err = future.result()
            if ok:
                stats["updated"] += 1
            else:
                stats["errors"] += 1
                if err:
                    logger.warning("Asset type backfill failed for %s: %s", _id, err)

    if rebuild_bm25 and stats["updated"]:
        bm25_index.rebuild_from_records(get_all_records())
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def _record_to_caption_json(record: ImageRecord) -> CaptionJSON:
    return caption_json_from_record(record)


def repair_search_terms(*, rebuild_bm25: bool = True) -> dict:
    """Re-clean aliases and recommended_cases from stored captions (no VLM)."""
    t0 = time.perf_counter()
    records = get_all_records()
    updated = 0

    for record in records:
        if _caption_failed(record):
            continue
        cap = _record_to_caption_json(record)
        if not cap.search.tags:
            continue
        enriched = enrich_caption_search_terms(cap)
        record.recommended_cases_json = metadata_db.serialize_list(enriched.recommended_cases)
        record.search_aliases_json = metadata_db.serialize_list(enriched.aliases)
        with session_scope() as s:
            s.merge(record)
        updated += 1

    if rebuild_bm25 and updated:
        bm25_index.rebuild_from_records(get_all_records())

    return {
        "records": len(records),
        "updated": updated,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }


def _reindex_one(record: ImageRecord, embedder) -> Tuple[str, bool, Optional[np.ndarray], Optional[str]]:
    img = _load_cached_image(record)
    if img is None:
        return record.image_id, False, None, "cached image missing"
    try:
        ctx = embed_context_from_record(record)
        emb = embedder.embed_image_with_context(img, ctx or None)
        return record.image_id, True, emb, None
    except Exception as exc:  # noqa: BLE001
        return record.image_id, False, None, str(exc)


def reindex_embeddings(
    *,
    workers: Optional[int] = None,
    batch_upsert: Optional[int] = None,
    image_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Re-embed indexed images from cached PNGs (picks up context-aware vectors)."""
    workers = max(1, workers if workers is not None else SETTINGS.ingest_workers)
    batch_upsert = max(1, batch_upsert if batch_upsert is not None else SETTINGS.ingest_batch_upsert)
    if image_ids is not None:
        id_set = set(image_ids)
        records = [r for r in get_records(image_ids) if r.image_id in id_set]
    else:
        records = get_all_records()
    stats = {
        "records": len(records),
        "reindexed": 0,
        "errors": 0,
        "workers": workers,
        "elapsed_sec": 0.0,
    }
    if not records:
        return stats

    embedder = get_embedder()
    chroma_batch: List[Tuple[str, np.ndarray, dict]] = []
    chroma_lock = threading.Lock()
    t0 = time.perf_counter()

    def _submit(record: ImageRecord):
        return _reindex_one(record, embedder)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_submit, r): r for r in records}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Reindex embeddings",
            unit="img",
            disable=not sys.stderr.isatty(),
        )
        for future in pbar:
            record = futures[future]
            image_id, ok, emb, err = future.result()
            if not ok or emb is None:
                stats["errors"] += 1
                if err:
                    logger.warning("Reindex failed for %s: %s", image_id, err)
                continue
            stats["reindexed"] += 1
            pending: Optional[List[Tuple[str, np.ndarray, dict]]] = None
            with chroma_lock:
                chroma_batch.append((image_id, emb, _chroma_metadata(record)))
                if len(chroma_batch) >= batch_upsert:
                    pending = list(chroma_batch)
                    chroma_batch.clear()
            if pending:
                _flush_chroma_batch(pending)

    with chroma_lock:
        if chroma_batch:
            _flush_chroma_batch(list(chroma_batch))

    if SETTINGS.caption_text_lane_enabled:
        text_stats = backfill_text_vectors(
            image_ids=[r.image_id for r in records], workers=workers
        )
        stats["text_vectors"] = text_stats

    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def backfill_text_vectors(
    *,
    image_ids: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> dict:
    """Embed caption documents into the text lane (no image loading)."""
    workers = max(1, workers if workers is not None else SETTINGS.ingest_workers)
    if image_ids is not None:
        records = get_records(image_ids)
    else:
        records = get_all_records()
    records = [r for r in records if caption_document_text(r).strip()]

    stats = {
        "records": len(records),
        "embedded": 0,
        "errors": 0,
        "workers": workers,
        "elapsed_sec": 0.0,
    }
    if not records or not SETTINGS.caption_text_lane_enabled:
        return stats

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(refresh_text_vector, r): r for r in records}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Text vectors",
            unit="img",
            disable=not sys.stderr.isatty(),
        )
        for future in pbar:
            try:
                ok = future.result()
            except Exception as exc:  # noqa: BLE001
                ok = False
                logger.warning(
                    "Text vector backfill failed for %s: %s",
                    futures[future].image_id,
                    exc,
                )
            if ok:
                stats["embedded"] += 1
            else:
                stats["errors"] += 1

    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def repair_missing_cache(
    records: Optional[List[ImageRecord]] = None,
    *,
    workers: Optional[int] = None,
    hold_ingest_lock: bool = True,
) -> dict:
    """Re-extract from source files to rebuild missing cached PNGs.

    ``hold_ingest_lock=False`` is required when called from inside an ingest
    (the non-reentrant ingest lock is already held there; re-acquiring would
    raise IngestInProgressError after data was written).
    """
    t0 = time.perf_counter()
    if records is None:
        report = assess_index_health()
        records = report.missing_cache_records

    repair_records: dict[str, ImageRecord] = {}
    local_repair_queue: set[Path] = set()
    unrecoverable = 0
    for record in records:
        local_source = resolve_source_file(record)
        if local_source is not None:
            local_repair_queue.add(local_source)
        elif source_exists(record):
            repair_records.setdefault(record.source_file, record)
        else:
            unrecoverable += 1
            logger.warning(
                "Unrecoverable missing cache for %s (source missing: %s)",
                record.image_id,
                record.source_file or "(none)",
            )

    stats = {
        "source_files_attempted": len(repair_records) + len(local_repair_queue),
        "source_files_repaired": 0,
        "images_updated": 0,
        "errors": 0,
        "unrecoverable": unrecoverable,
    }
    if not repair_records and not local_repair_queue:
        stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        return stats

    from imagecb.ingest import ingest_paths

    with ExitStack() as stack:
        repair_queue: list[Path] = list(local_repair_queue)
        for record in repair_records.values():
            source = stack.enter_context(materialize_source(record))
            if source is not None:
                repair_queue.append(source)
        ingest_stats = ingest_paths(
            sorted(repair_queue, key=str),
            force=True,
            auto_repair=False,
            rebuild_bm25=False,
            refresh_vocab=False,
            workers=workers,
            _hold_ingest_lock=hold_ingest_lock,
        )
    stats["source_files_repaired"] = ingest_stats.get("files", 0)
    stats["images_updated"] = ingest_stats.get("images_updated", 0) + ingest_stats.get("images_added", 0)
    stats["errors"] = ingest_stats.get("errors", 0)
    stats["ingest"] = ingest_stats
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return stats


def repair_index_issues(
    *,
    workers: Optional[int] = None,
    include_weak_captions: Optional[bool] = None,
    repair_missing_vectors: Optional[bool] = None,
    skip_caption_phases: bool = False,
    hold_ingest_lock: bool = True,
) -> dict:
    """Phased repair: missing cache, failed captions, optional weak, missing Chroma vectors."""
    if include_weak_captions is None:
        include_weak_captions = SETTINGS.post_ingest_repair_include_weak
    if repair_missing_vectors is None:
        repair_missing_vectors = SETTINGS.post_ingest_repair_reindex_vectors

    t0 = time.perf_counter()
    reconcile_stats = reconcile_index_safe()
    rescan_stats = rescan_caption_quality()
    report = assess_index_health(include_weak=include_weak_captions)
    if report.is_healthy and report.missing_asset_type_count == 0:
        logger.info("Post-ingest repair skipped: index is healthy")
        return {
            "skipped": True,
            "is_healthy": True,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "assess": {"elapsed_sec": report.elapsed_sec},
            "rescan": rescan_stats,
            "reconcile": reconcile_stats,
        }

    logger.info(
        "Post-ingest repair starting: missing_cache=%s failed_captions=%s "
        "missing_chroma=%s missing_asset_types=%s",
        report.missing_cache_count,
        report.failed_caption_count,
        report.missing_chroma_count,
        report.missing_asset_type_count,
    )

    phases: dict = {}
    any_phase_ran = False

    if report.missing_cache_count > 0:
        any_phase_ran = True
        phases["cache"] = repair_missing_cache(
            workers=workers, hold_ingest_lock=hold_ingest_lock
        )

    if not skip_caption_phases:
        report = assess_index_health(include_weak=include_weak_captions)
        if report.failed_caption_count > 0:
            any_phase_ran = True
            phases["captions"] = repair_failed_captions(
                workers=workers,
                include_weak=False,
                rebuild_bm25=False,
            )

        if include_weak_captions:
            report = assess_index_health(include_weak=True)
            if report.weak_caption_count > 0:
                any_phase_ran = True
                phases["weak_captions"] = repair_failed_captions(
                    workers=workers,
                    include_weak=True,
                    rebuild_bm25=False,
                )

        report = assess_index_health(include_weak=include_weak_captions)
        if report.missing_asset_type_count > 0:
            any_phase_ran = True
            phases["asset_types"] = backfill_asset_types(
                workers=workers,
                rebuild_bm25=False,
            )

    if repair_missing_vectors:
        report = assess_index_health(include_weak=include_weak_captions)
        if report.missing_chroma_count > 0:
            records_by_id = {r.image_id: r for r in get_all_records()}
            reindex_ids = [
                iid
                for iid in report.missing_chroma_ids
                if iid in records_by_id and not _cache_missing(records_by_id[iid])
            ]
            if reindex_ids:
                any_phase_ran = True
                phases["vectors"] = reindex_embeddings(workers=workers, image_ids=reindex_ids)

    if SETTINGS.caption_text_lane_enabled:
        report = assess_index_health(include_weak=include_weak_captions)
        if report.missing_text_vector_count > 0:
            any_phase_ran = True
            phases["text_vectors"] = backfill_text_vectors(
                image_ids=report.missing_text_vector_ids,
                workers=workers,
            )

    if any_phase_ran:
        bm25_index.rebuild_from_records(get_all_records())

    final_reconcile = reconcile_index_safe()
    final = assess_index_health(include_weak=include_weak_captions)
    elapsed = round(time.perf_counter() - t0, 1)

    summary = {
        "skipped": False,
        "is_healthy": final.is_healthy,
        "phases": phases,
        "reconcile": reconcile_stats,
        "final_reconcile": final_reconcile,
        "elapsed_sec": elapsed,
        "cache_recached": phases.get("cache", {}).get("images_updated", 0),
        "captions_repaired": phases.get("captions", {}).get("repaired", 0)
        + phases.get("weak_captions", {}).get("repaired", 0),
        "asset_types_backfilled": phases.get("asset_types", {}).get("updated", 0),
        "vectors_reindexed": phases.get("vectors", {}).get("reindexed", 0),
        "text_vectors_backfilled": phases.get("text_vectors", {}).get("embedded", 0),
        "source_files_attempted": phases.get("cache", {}).get("source_files_attempted", 0),
        "unrecoverable": phases.get("cache", {}).get("unrecoverable", 0),
        "remaining_missing_cache": final.missing_cache_count,
        "remaining_failed_captions": final.failed_caption_count,
        "remaining_weak_captions": final.weak_caption_count,
        "remaining_missing_chroma": final.missing_chroma_count,
        "remaining_missing_asset_types": final.missing_asset_type_count,
        "rescan": rescan_stats,
    }
    logger.info(
        "Post-ingest repair done in %.1fs: recached=%s captions=%s asset_types=%s "
        "vectors=%s healthy=%s",
        elapsed,
        summary["cache_recached"],
        summary["captions_repaired"],
        summary["asset_types_backfilled"],
        summary["vectors_reindexed"],
        final.is_healthy,
    )
    return summary


def format_post_repair_summary(repair_stats: dict) -> str:
    """Human-readable one-liner for ingest API/UI; empty when repair was skipped."""
    if not repair_stats or repair_stats.get("skipped"):
        return ""

    parts: List[str] = []
    cache = repair_stats.get("cache_recached", 0)
    sources = repair_stats.get("source_files_attempted", 0)
    if cache or sources:
        parts.append(f"re-cached {cache} images from {sources} source file(s)")
    captions = repair_stats.get("captions_repaired", 0)
    if captions:
        parts.append(f"re-captioned {captions}")
    asset_types = repair_stats.get("asset_types_backfilled", 0)
    if asset_types:
        parts.append(f"backfilled asset types for {asset_types} images")
    vectors = repair_stats.get("vectors_reindexed", 0)
    if vectors:
        parts.append(f"re-indexed {vectors} vectors")
    text_vectors = repair_stats.get("text_vectors_backfilled", 0)
    if text_vectors:
        parts.append(f"backfilled {text_vectors} caption-text vectors")
    unrecoverable = repair_stats.get("unrecoverable", 0)
    if unrecoverable:
        parts.append(f"{unrecoverable} unrecoverable (source file missing)")

    if not parts:
        return ""
    return "Post-ingest repair: " + "; ".join(parts) + "."
