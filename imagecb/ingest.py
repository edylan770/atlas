"""End-to-end ingest pipeline.

For each source file under a root path:
  1. Dispatch to the right extractor.
  2. For each extracted image (optionally in parallel):
     a. Compute a content hash and skip if already ingested.
     b. Cache the image as a PNG under the image cache dir.
     c. Run OCR (optional).
     d. Call the VLM for a structured caption.
     e. Embed with Bedrock.
     f. Upsert SQLite row + Chroma vector (batched).
  3. After everything is in SQLite, rebuild the BM25 index.
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from imagecb.config import SETTINGS
from imagecb.extractors.dispatch import extract_path, iter_corpus
from imagecb.extractors.types import ExtractedImage
from imagecb.caption.context import slide_body_from_provenance
from imagecb.caption.document import caption_document_text
from imagecb.caption.pipeline import generate_caption, refresh_vocab_cache
from imagecb.ingest_context import embed_context_from_caption_and_provenance
from imagecb.models.embedder import BedrockEmbedder, get_embedder, get_text_embedder
from imagecb.models.ocr import extract_text as ocr_extract
from imagecb.models.vlm import CaptionJSON, VLMCaptioner, get_captioner
from imagecb.storage import bm25_index, vector_store
from imagecb.storage.metadata_db import (
    ImageRecord,
    existing_hashes,
    get_all_records,
    get_record_by_hash,
    new_image_id,
    serialize_list,
    session_scope,
)

logger = logging.getLogger(__name__)

_ingest_lock = threading.Lock()


class IngestInProgressError(Exception):
    """Raised when a second ingest starts while one is already running."""


def ingest_in_progress() -> bool:
    return _ingest_lock.locked()

_STAT_KEYS = (
    "files",
    "images_seen",
    "images_added",
    "images_updated",
    "skipped_duplicates",
    "errors",
    "captions_weak",
    "captions_failed",
    "workers",
    "elapsed_sec",
    "batches",
)


@dataclass
class _IngestWorkItem:
    file_path: Path
    extracted: ExtractedImage


@dataclass
class _IngestOutcome:
    skipped_duplicate: bool = False
    added: bool = False
    updated: bool = False
    record: Optional[ImageRecord] = None
    embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    error: Optional[str] = None


def embed_caption_document(record: ImageRecord) -> Optional[np.ndarray]:
    """Embed the record's caption document for the text dense lane (fail-soft)."""
    if not SETTINGS.caption_text_lane_enabled:
        return None
    doc = caption_document_text(record).strip()
    if not doc:
        return None
    try:
        return get_text_embedder().embed_document(doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Caption-text embedding failed for %s: %s", record.image_id, exc)
        return None


def _empty_stats(*, workers: int = 1) -> dict:
    return {
        "files": 0,
        "images_seen": 0,
        "images_added": 0,
        "images_updated": 0,
        "skipped_duplicates": 0,
        "errors": 0,
        "captions_weak": 0,
        "captions_failed": 0,
        "workers": workers,
        "elapsed_sec": 0.0,
        "batches": 0,
    }


def _merge_stats(total: dict, batch: dict) -> None:
    for key in _STAT_KEYS:
        if key in ("files", "workers", "elapsed_sec", "batches"):
            continue
        total[key] = total.get(key, 0) + batch.get(key, 0)


def _hash_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _cache_image(img: Image.Image, image_id: str) -> Path:
    out_path = SETTINGS.image_cache_dir / f"{image_id}.png"
    img.convert("RGB").save(out_path, format="PNG")
    return out_path


def _default_image_name(extracted: ExtractedImage) -> str:
    p = extracted.provenance
    base = Path(p.source_file or "").stem or "image"
    if p.source_type == "pptx" and p.slide_index is not None:
        return f"{base} — slide {p.slide_index}"
    if p.source_type == "pdf" and p.page_index is not None:
        return f"{base} — page {p.page_index}"
    return base


def _chroma_metadata(record: ImageRecord) -> dict:
    """Compact, JSON-safe metadata for Chroma filtering & display."""

    def _iso(v: Optional[datetime]) -> Optional[str]:
        return v.isoformat() if isinstance(v, datetime) else None

    return {
        "image_id": record.image_id,
        "source_type": record.source_type or "",
        "source_file": record.source_file or "",
        "author": record.author or "",
        "slide_index": int(record.slide_index) if record.slide_index else 0,
        "page_index": int(record.page_index) if record.page_index else 0,
        "source_modified_at": _iso(record.source_modified_at) or "",
    }


def _record_for(
    *,
    image_id: str,
    extracted: ExtractedImage,
    image_path: Path,
    content_hash: str,
    ocr_text: str,
    caption: CaptionJSON,
) -> ImageRecord:
    p = extracted.provenance
    return ImageRecord(
        image_id=image_id,
        content_hash=content_hash,
        image_path=str(image_path),
        source_file=p.source_file,
        source_type=p.source_type,
        source_modified_at=p.source_modified_at,
        source_created_at=p.source_created_at,
        author=p.author,
        slide_index=p.slide_index,
        page_index=p.page_index,
        slide_title=p.slide_title,
        slide_notes=p.slide_notes,
        ocr_text=ocr_text,
        image_name=(caption.image_name or "").strip() or _default_image_name(extracted),
        caption_short=caption.short_caption,
        caption_detailed=caption.detailed_description,
        use_case=caption.use_case,
        scene=caption.scene,
        text_overlay_summary=caption.text_overlay_summary,
        objects_json=serialize_list(caption.objects),
        tags_json=serialize_list(caption.tags),
        recommended_cases_json=serialize_list(caption.recommended_cases),
        theme=caption.theme,
        search_aliases_json=serialize_list(caption.aliases),
        slide_body_text=slide_body_from_provenance(p) or None,
        caption_quality=caption.caption_quality or "ok",
        text_read_uncertain=1 if caption.text_read_uncertain else 0,
        asset_type=caption.asset_type or None,
    )


def _caption_and_embed(
    extracted: ExtractedImage,
    *,
    captioner: Optional[VLMCaptioner],
    embedder: BedrockEmbedder,
    max_image_side: int,
) -> Tuple[CaptionJSON, np.ndarray]:
    """Caption first (with context), then embed with interpretive context."""

    if captioner is None:
        caption = CaptionJSON.empty()
        ctx = embed_context_from_caption_and_provenance(caption, extracted.provenance)
        emb = embedder.embed_image_with_context(extracted.image, ctx or None)
        return caption, emb

    caption = generate_caption(extracted, captioner, max_side=max_image_side)
    ctx = embed_context_from_caption_and_provenance(caption, extracted.provenance)
    emb = embedder.embed_image_with_context(extracted.image, ctx or None)
    return caption, emb


def _ingest_one_image(
    item: _IngestWorkItem,
    *,
    known: Set[str],
    known_lock: threading.Lock,
    force: bool,
    skip_caption: bool,
    skip_ocr: bool,
    captioner: Optional[VLMCaptioner],
    embedder: BedrockEmbedder,
    max_image_side: int,
) -> _IngestOutcome:
    extracted = item.extracted
    try:
        content_hash = _hash_image(extracted.image)
        with known_lock:
            existing = get_record_by_hash(content_hash) if content_hash in known else None
            if existing is not None and not force:
                return _IngestOutcome(skipped_duplicate=True)
            if existing is not None:
                image_id = existing.image_id
                outcome = _IngestOutcome(updated=True)
            else:
                image_id = new_image_id()
                outcome = _IngestOutcome(added=True)

        cached_path = _cache_image(extracted.image, image_id)
        ocr_text = "" if skip_ocr else ocr_extract(extracted.image)
        caption, emb = _caption_and_embed(
            extracted,
            captioner=captioner,
            embedder=embedder,
            max_image_side=max_image_side,
        )
        record = _record_for(
            image_id=image_id,
            extracted=extracted,
            image_path=cached_path,
            content_hash=content_hash,
            ocr_text=ocr_text,
            caption=caption,
        )
        with session_scope() as s:
            s.merge(record)
        with known_lock:
            known.add(content_hash)
        outcome.record = record
        outcome.embedding = emb
        outcome.text_embedding = embed_caption_document(record)
        return outcome
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to ingest an image from %s: %s", item.file_path, exc)
        return _IngestOutcome(error=str(exc))


def _flush_chroma_batch(batch: List[Tuple[str, np.ndarray, dict]]) -> None:
    if not batch:
        return
    ids = [b[0] for b in batch]
    embeddings = np.stack([b[1] for b in batch])
    metadatas = [b[2] for b in batch]
    vector_store.upsert(image_ids=ids, embeddings=embeddings, metadatas=metadatas)


def _flush_text_batch(batch: List[Tuple[str, np.ndarray]]) -> None:
    if not batch:
        return
    ids = [b[0] for b in batch]
    embeddings = np.stack([b[1] for b in batch])
    vector_store.upsert_text(image_ids=ids, embeddings=embeddings)


def _collect_work_items(paths: Iterable[Path]) -> Tuple[List[_IngestWorkItem], int]:
    items: List[_IngestWorkItem] = []
    errors = 0
    for file_path in paths:
        try:
            for extracted in extract_path(file_path):
                items.append(_IngestWorkItem(file_path=file_path, extracted=extracted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extractor failed for %s: %s", file_path, exc)
            errors += 1
    return items, errors


def _iter_work_items(paths: Iterable[Path]) -> Iterator[Tuple[Optional[_IngestWorkItem], int]]:
    """Stream work items file-by-file. Yields (item, extract_errors_so_far)."""
    errors = 0
    for file_path in paths:
        try:
            for extracted in extract_path(file_path):
                yield _IngestWorkItem(file_path=file_path, extracted=extracted), errors
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extractor failed for %s: %s", file_path, exc)
            errors += 1
            yield None, errors


def _apply_outcome(
    outcome: _IngestOutcome,
    *,
    stats: dict,
    chroma_batch: List[Tuple[str, np.ndarray, dict]],
    text_batch: List[Tuple[str, np.ndarray]],
    chroma_lock: threading.Lock,
    batch_upsert: int,
) -> None:
    if outcome.skipped_duplicate:
        stats["skipped_duplicates"] += 1
        return
    if outcome.error:
        stats["errors"] += 1
        return
    if outcome.added:
        stats["images_added"] += 1
    if outcome.updated:
        stats["images_updated"] += 1
    if outcome.record is not None:
        q = (outcome.record.caption_quality or "ok").lower()
        if q == "weak":
            stats["captions_weak"] += 1
        elif q == "failed":
            stats["captions_failed"] += 1
    if outcome.record is not None and outcome.embedding is not None:
        pending: Optional[List[Tuple[str, np.ndarray, dict]]] = None
        pending_text: Optional[List[Tuple[str, np.ndarray]]] = None
        with chroma_lock:
            chroma_batch.append(
                (
                    outcome.record.image_id,
                    outcome.embedding,
                    _chroma_metadata(outcome.record),
                )
            )
            if outcome.text_embedding is not None:
                text_batch.append((outcome.record.image_id, outcome.text_embedding))
            if len(chroma_batch) >= batch_upsert:
                pending = list(chroma_batch)
                chroma_batch.clear()
                pending_text = list(text_batch)
                text_batch.clear()
        if pending:
            _flush_chroma_batch(pending)
        if pending_text:
            _flush_text_batch(pending_text)


def _finalize_ingest(*, rebuild_bm25: bool, refresh_vocab: bool) -> None:
    if refresh_vocab:
        refresh_vocab_cache()
    from imagecb.repair import reconcile_index_safe, rescan_caption_quality

    rescan_caption_quality()
    if rebuild_bm25:
        records = get_all_records()
        bm25_index.rebuild_from_records(records)
    if SETTINGS.index_reconcile_after_ingest:
        reconcile_index_safe()


def _drain_future(
    future: Future[_IngestOutcome],
    item: _IngestWorkItem,
    *,
    stats: dict,
    chroma_batch: List[Tuple[str, np.ndarray, dict]],
    text_batch: List[Tuple[str, np.ndarray]],
    chroma_lock: threading.Lock,
    batch_upsert: int,
    image_timeout_sec: int,
) -> None:
    try:
        outcome = future.result(timeout=image_timeout_sec)
    except FuturesTimeoutError:
        logger.warning(
            "Timed out ingesting image from %s after %ss",
            item.file_path,
            image_timeout_sec,
        )
        stats["errors"] += 1
        future.cancel()
        return
    _apply_outcome(
        outcome,
        stats=stats,
        chroma_batch=chroma_batch,
        text_batch=text_batch,
        chroma_lock=chroma_lock,
        batch_upsert=batch_upsert,
    )


def _run_ingest_pool(
    work_items: Iterable[_IngestWorkItem],
    *,
    known: Set[str],
    known_lock: threading.Lock,
    force: bool,
    skip_caption: bool,
    skip_ocr: bool,
    captioner: Optional[VLMCaptioner],
    embedder: BedrockEmbedder,
    max_image_side: int,
    workers: int,
    batch_upsert: int,
    image_timeout_sec: int,
    stats: dict,
    total_images: Optional[int] = None,
) -> None:
    chroma_batch: List[Tuple[str, np.ndarray, dict]] = []
    text_batch: List[Tuple[str, np.ndarray]] = []
    chroma_lock = threading.Lock()

    def _submit(item: _IngestWorkItem) -> _IngestOutcome:
        return _ingest_one_image(
            item,
            known=known,
            known_lock=known_lock,
            force=force,
            skip_caption=skip_caption,
            skip_ocr=skip_ocr,
            captioner=captioner,
            embedder=embedder,
            max_image_side=max_image_side,
        )

    max_in_flight = max(workers * 2, workers)
    pending: Deque[Tuple[Future[_IngestOutcome], _IngestWorkItem]] = deque()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pbar = tqdm(
            total=total_images,
            desc="Images",
            unit="img",
            disable=not sys.stderr.isatty(),
        )
        try:
            for item in work_items:
                pending.append((pool.submit(_submit, item), item))
                if len(pending) >= max_in_flight:
                    future, queued_item = pending.popleft()
                    pbar.set_postfix_str(queued_item.file_path.name)
                    _drain_future(
                        future,
                        queued_item,
                        stats=stats,
                        chroma_batch=chroma_batch,
                        text_batch=text_batch,
                        chroma_lock=chroma_lock,
                        batch_upsert=batch_upsert,
                        image_timeout_sec=image_timeout_sec,
                    )
                    pbar.update(1)

            while pending:
                future, queued_item = pending.popleft()
                pbar.set_postfix_str(queued_item.file_path.name)
                _drain_future(
                    future,
                    queued_item,
                    stats=stats,
                    chroma_batch=chroma_batch,
                    text_batch=text_batch,
                    chroma_lock=chroma_lock,
                    batch_upsert=batch_upsert,
                    image_timeout_sec=image_timeout_sec,
                )
                pbar.update(1)
        finally:
            pbar.close()

    with chroma_lock:
        if chroma_batch:
            _flush_chroma_batch(list(chroma_batch))
            chroma_batch.clear()
        if text_batch:
            _flush_text_batch(list(text_batch))
            text_batch.clear()


def ingest_paths(
    paths: Iterable[Path],
    *,
    skip_caption: bool = False,
    skip_ocr: bool = False,
    force: bool = False,
    workers: Optional[int] = None,
    max_image_side: Optional[int] = None,
    batch_upsert: Optional[int] = None,
    rebuild_bm25: bool = True,
    refresh_vocab: bool = True,
    image_timeout_sec: Optional[int] = None,
    auto_repair: bool = True,
    _hold_ingest_lock: bool = True,
) -> dict:
    """Ingest a list of source files. Returns a stats dict."""
    SETTINGS.ensure_dirs()
    acquired = False
    if _hold_ingest_lock:
        if not _ingest_lock.acquire(blocking=False):
            raise IngestInProgressError("Another ingest is already in progress")
        acquired = True
    try:
        return _ingest_paths_locked(
            paths,
            skip_caption=skip_caption,
            skip_ocr=skip_ocr,
            force=force,
            workers=workers,
            max_image_side=max_image_side,
            batch_upsert=batch_upsert,
            rebuild_bm25=rebuild_bm25,
            refresh_vocab=refresh_vocab,
            image_timeout_sec=image_timeout_sec,
            auto_repair=auto_repair,
        )
    finally:
        if acquired:
            _ingest_lock.release()


def _ingest_paths_locked(
    paths: Iterable[Path],
    *,
    skip_caption: bool = False,
    skip_ocr: bool = False,
    force: bool = False,
    workers: Optional[int] = None,
    max_image_side: Optional[int] = None,
    batch_upsert: Optional[int] = None,
    rebuild_bm25: bool = True,
    refresh_vocab: bool = True,
    image_timeout_sec: Optional[int] = None,
    auto_repair: bool = True,
) -> dict:
    paths = list(paths)
    workers = workers if workers is not None else SETTINGS.ingest_workers
    workers = max(1, workers)
    max_image_side = max_image_side if max_image_side is not None else SETTINGS.ingest_max_image_side
    batch_upsert = batch_upsert if batch_upsert is not None else SETTINGS.ingest_batch_upsert
    batch_upsert = max(1, batch_upsert)
    image_timeout_sec = (
        image_timeout_sec
        if image_timeout_sec is not None
        else SETTINGS.ingest_image_timeout_sec
    )
    image_timeout_sec = max(30, image_timeout_sec)

    stats = _empty_stats(workers=workers)
    stats["files"] = len(paths)
    if not paths:
        return stats

    t0 = time.perf_counter()
    extract_errors = 0
    images_seen = 0

    def _stream_items() -> Iterator[_IngestWorkItem]:
        nonlocal extract_errors, images_seen
        for item, err_count in _iter_work_items(paths):
            extract_errors = err_count
            if item is not None:
                images_seen += 1
                yield item

    known = existing_hashes()
    known_lock = threading.Lock()
    embedder = get_embedder()
    captioner = None if skip_caption else get_captioner()

    _run_ingest_pool(
        _stream_items(),
        known=known,
        known_lock=known_lock,
        force=force,
        skip_caption=skip_caption,
        skip_ocr=skip_ocr,
        captioner=captioner,
        embedder=embedder,
        max_image_side=max_image_side,
        workers=workers,
        batch_upsert=batch_upsert,
        image_timeout_sec=image_timeout_sec,
        stats=stats,
        total_images=None,
    )

    stats["errors"] += extract_errors
    stats["images_seen"] = images_seen

    if images_seen > 0:
        _finalize_ingest(rebuild_bm25=rebuild_bm25, refresh_vocab=refresh_vocab)

    stats["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    if stats["captions_weak"] or stats["captions_failed"]:
        logger.info(
            "Caption quality: weak=%s failed=%s (run repair-captions --include-weak to retry)",
            stats["captions_weak"],
            stats["captions_failed"],
        )

    if auto_repair and SETTINGS.post_ingest_repair_enabled:
        from imagecb.repair import repair_index_issues

        repair_stats = repair_index_issues(
            workers=workers,
            skip_caption_phases=skip_caption,
        )
        stats["post_repair"] = repair_stats

    return stats


def ingest_paths_batched(
    paths: Iterable[Path],
    *,
    batch_size: int,
    skip_caption: bool = False,
    skip_ocr: bool = False,
    force: bool = False,
    workers: Optional[int] = None,
    max_image_side: Optional[int] = None,
    batch_upsert: Optional[int] = None,
    defer_bm25: bool = True,
    image_timeout_sec: Optional[int] = None,
    auto_repair: bool = True,
) -> dict:
    """Ingest source files in file batches; rebuild BM25 once at the end."""
    if not _ingest_lock.acquire(blocking=False):
        raise IngestInProgressError("Another ingest is already in progress")
    try:
        paths = list(paths)
        batch_size = max(1, batch_size)
        workers = workers if workers is not None else SETTINGS.ingest_workers
        workers = max(1, workers)

        total = _empty_stats(workers=workers)
        total["files"] = len(paths)
        if not paths:
            return total

        batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
        total["batches"] = len(batches)
        t0 = time.perf_counter()

        for idx, chunk in enumerate(batches, start=1):
            logger.info("Ingest batch %s/%s (%s files)", idx, len(batches), len(chunk))
            batch_stats = ingest_paths(
                chunk,
                skip_caption=skip_caption,
                skip_ocr=skip_ocr,
                force=force,
                workers=workers,
                max_image_side=max_image_side,
                batch_upsert=batch_upsert,
                rebuild_bm25=not defer_bm25,
                refresh_vocab=False,
                image_timeout_sec=image_timeout_sec,
                auto_repair=False,
                _hold_ingest_lock=False,
            )
            _merge_stats(total, batch_stats)
            logger.info(
                "Batch %s/%s done: added=%s updated=%s duplicates=%s errors=%s",
                idx,
                len(batches),
                batch_stats.get("images_added", 0),
                batch_stats.get("images_updated", 0),
                batch_stats.get("skipped_duplicates", 0),
                batch_stats.get("errors", 0),
            )

        if defer_bm25:
            _finalize_ingest(rebuild_bm25=True, refresh_vocab=True)

        if auto_repair and SETTINGS.post_ingest_repair_enabled:
            from imagecb.repair import repair_index_issues

            repair_stats = repair_index_issues(
                workers=workers,
                skip_caption_phases=skip_caption,
            )
            total["post_repair"] = repair_stats

        total["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        return total
    finally:
        _ingest_lock.release()


def ingest_root(
    root: Path,
    *,
    skip_caption: bool = False,
    skip_ocr: bool = False,
    force: bool = False,
    workers: Optional[int] = None,
    max_image_side: Optional[int] = None,
    batch_upsert: Optional[int] = None,
    batch_size: Optional[int] = None,
    defer_bm25: bool = True,
    image_timeout_sec: Optional[int] = None,
    auto_repair: bool = True,
) -> dict:
    paths = list(iter_corpus(root))
    batch_size = batch_size if batch_size is not None else SETTINGS.ingest_batch_size
    if batch_size and batch_size > 0:
        return ingest_paths_batched(
            paths,
            batch_size=batch_size,
            skip_caption=skip_caption,
            skip_ocr=skip_ocr,
            force=force,
            workers=workers,
            max_image_side=max_image_side,
            batch_upsert=batch_upsert,
            defer_bm25=defer_bm25,
            image_timeout_sec=image_timeout_sec,
            auto_repair=auto_repair,
        )
    return ingest_paths(
        paths,
        skip_caption=skip_caption,
        skip_ocr=skip_ocr,
        force=force,
        workers=workers,
        max_image_side=max_image_side,
        batch_upsert=batch_upsert,
        image_timeout_sec=image_timeout_sec,
        auto_repair=auto_repair,
    )
