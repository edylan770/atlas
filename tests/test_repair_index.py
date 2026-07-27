"""Tests for index health assessment and post-ingest repair orchestrator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imagecb.caption.quality import CAPTION_FAILED
from imagecb.repair import (
    IndexHealthReport,
    assess_index_health,
    format_post_repair_summary,
    reconcile_index_safe,
    repair_failed_captions,
    repair_index_issues,
    repair_missing_cache,
)
from imagecb.storage.metadata_db import ImageRecord


def _record(
    image_id: str,
    *,
    image_path: str = "/nonexistent/cache.png",
    source_file: str = "/docs/deck.pptx",
    caption_short: str = "A slide about security",
    caption_quality: str = "ok",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=image_path,
        source_file=source_file,
        source_type="pptx",
        source_modified_at=datetime(2024, 9, 15),
        source_created_at=None,
        author=None,
        slide_index=1,
        page_index=None,
        slide_title=None,
        slide_notes=None,
        ocr_text=None,
        caption_short=caption_short,
        caption_detailed=None,
        objects_json=None,
        tags_json=None,
        scene=None,
        text_overlay_summary=None,
        created_at=datetime.utcnow(),
        caption_quality=caption_quality,
    )


def test_assess_index_health_healthy():
    records = [_record("a"), _record("b", image_path="/exists/a.png")]
    with patch("imagecb.repair.get_all_records", return_value=records), patch(
        "imagecb.repair._cache_missing", side_effect=[False, False]
    ), patch("imagecb.repair.vector_store.count", return_value=2), patch(
        "imagecb.repair.vector_store.list_ids", return_value={"a", "b"}
    ), patch("imagecb.repair.vector_store.list_text_ids", return_value={"a", "b"}), patch(
        "imagecb.repair.metadata_db.get_deleted_records", return_value=[]
    ), patch("imagecb.repair.bm25_index.count", return_value=2), patch(
        "imagecb.repair.bm25_index.list_ids", return_value={"a", "b"}
    ):
        report = assess_index_health()
    assert report.is_healthy is True
    assert report.stores_in_sync is True
    assert report.missing_cache_count == 0
    assert report.failed_caption_count == 0
    assert report.missing_chroma_count == 0


def test_assess_index_health_detects_issues(tmp_path):
    cache_path = tmp_path / "cached.png"
    cache_path.write_bytes(b"png")
    missing = _record("missing", image_path=str(tmp_path / "gone.png"))
    failed = _record("failed", image_path=str(cache_path), caption_short=CAPTION_FAILED)
    no_chroma = _record("no-vec", image_path=str(cache_path))
    records = [missing, failed, no_chroma]

    def _cache_missing(record: ImageRecord) -> bool:
        return record.image_id == "missing"

    with patch("imagecb.repair.get_all_records", return_value=records), patch(
        "imagecb.repair._cache_missing", side_effect=_cache_missing
    ), patch(
        "imagecb.repair.resolve_source_file",
        side_effect=lambda r: Path(r.source_file) if r.image_id == "missing" else None,
    ), patch("imagecb.repair.vector_store.count", return_value=2), patch(
        "imagecb.repair.vector_store.list_ids", return_value={"failed", "no-vec"}
    ), patch("imagecb.repair.vector_store.list_text_ids", return_value=set()), patch(
        "imagecb.repair.metadata_db.get_deleted_records", return_value=[]
    ), patch("imagecb.repair.bm25_index.count", return_value=0), patch(
        "imagecb.repair.bm25_index.list_ids", return_value=set()
    ):
        report = assess_index_health(include_weak=True)

    assert report.is_healthy is False
    assert report.missing_cache_count == 1
    assert report.failed_caption_count == 1
    assert report.missing_chroma_count == 1
    assert report.missing_chroma_ids == ["missing"]
    assert report.unrecoverable_source_missing_count == 0
    assert str(Path("/docs/deck.pptx")) in report.recoverable_source_files or "/docs/deck.pptx" in report.recoverable_source_files


def test_assess_index_health_unrecoverable_when_source_missing():
    records = [_record("x", source_file="/gone/deck.pptx")]
    with patch("imagecb.repair.get_all_records", return_value=records), patch(
        "imagecb.repair._cache_missing", return_value=True
    ), patch("imagecb.repair.resolve_source_file", return_value=None), patch(
        "imagecb.repair.vector_store.count", return_value=1
    ), patch("imagecb.repair.vector_store.list_ids", return_value={"x"}), patch(
        "imagecb.repair.vector_store.list_text_ids", return_value=set()
    ), patch("imagecb.repair.metadata_db.get_deleted_records", return_value=[]), patch(
        "imagecb.repair.bm25_index.count", return_value=1
    ), patch("imagecb.repair.bm25_index.list_ids", return_value={"x"}):
        report = assess_index_health()
    assert report.unrecoverable_source_missing_count == 1
    assert report.recoverable_source_files == []


def test_repair_missing_cache_calls_ingest_paths_with_force(tmp_path):
    src = tmp_path / "deck.pptx"
    src.touch()
    records = [_record("a", source_file=str(src))]

    ingest_calls: list[dict] = []

    def fake_ingest(paths, **kwargs):
        ingest_calls.append({"paths": list(paths), **kwargs})
        return {"files": len(paths), "images_updated": 1, "images_added": 0, "errors": 0}

    with patch("imagecb.repair.resolve_source_file", return_value=src), patch(
        "imagecb.ingest.ingest_paths", side_effect=fake_ingest
    ):
        stats = repair_missing_cache(records)

    assert stats["source_files_attempted"] == 1
    assert len(ingest_calls) == 1
    call = ingest_calls[0]
    assert call["paths"] == [src]
    assert call["force"] is True
    assert call["auto_repair"] is False
    assert call["rebuild_bm25"] is False


def test_repair_failed_captions_scope_weak_only():
    weak = _record("weak-1", caption_quality="weak")
    with patch("imagecb.repair.records_with_weak_captions", return_value=[weak]), patch(
        "imagecb.repair._repair_one_caption", return_value=("weak-1", True, None)
    ), patch("imagecb.repair.bm25_index.rebuild_from_records"):
        stats = repair_failed_captions(scope="weak")
    assert stats["attempted"] == 1
    assert stats["repaired"] == 1
    assert stats["scope"] == "weak"


def test_repair_index_issues_noop_when_healthy():
    healthy = IndexHealthReport(
        total_records=1,
        chroma_vectors=1,
        missing_cache_count=0,
        stores_in_sync=True,
        is_healthy=True,
    )
    with patch("imagecb.repair.reconcile_index_safe", return_value={}), patch(
        "imagecb.repair.assess_index_health", return_value=healthy
    ), patch("imagecb.repair.repair_missing_cache") as cache_mock, patch(
        "imagecb.repair.repair_failed_captions"
    ) as cap_mock:
        stats = repair_index_issues()
    assert stats["skipped"] is True
    cache_mock.assert_not_called()
    cap_mock.assert_not_called()


def test_repair_index_issues_skips_caption_phases():
    unhealthy = IndexHealthReport(
        total_records=2,
        chroma_vectors=1,
        missing_cache_count=0,
        failed_caption_count=1,
        missing_chroma_count=1,
        missing_chroma_ids=["b"],
        is_healthy=False,
    )
    final = IndexHealthReport(
        total_records=2,
        chroma_vectors=2,
        missing_cache_count=0,
        failed_caption_count=1,
        missing_chroma_count=0,
        is_healthy=False,
    )
    records_by_id = {
        "b": _record("b", image_path="/cache/b.png"),
    }

    with patch("imagecb.repair.reconcile_index_safe", return_value={}), patch(
        "imagecb.repair.assess_index_health", side_effect=[unhealthy, unhealthy, unhealthy, final, final]
    ), patch("imagecb.repair.repair_failed_captions") as cap_mock, patch(
        "imagecb.repair.reindex_embeddings", return_value={"reindexed": 1}
    ) as vec_mock, patch("imagecb.repair.get_all_records", return_value=list(records_by_id.values())), patch(
        "imagecb.repair._cache_missing", return_value=False
    ), patch("imagecb.repair.bm25_index.rebuild_from_records"):
        stats = repair_index_issues(skip_caption_phases=True)

    cap_mock.assert_not_called()
    vec_mock.assert_called_once()
    assert stats["skipped"] is False


def test_format_post_repair_summary_skipped():
    assert format_post_repair_summary({"skipped": True}) == ""
    assert format_post_repair_summary({}) == ""


def test_format_post_repair_summary_with_work():
    text = format_post_repair_summary(
        {
            "skipped": False,
            "cache_recached": 14,
            "source_files_attempted": 3,
            "captions_repaired": 1,
            "vectors_reindexed": 13,
        }
    )
    assert "Post-ingest repair:" in text
    assert "14" in text
    assert "re-captioned 1" in text
    assert "13" in text


def test_ingest_paths_invokes_repair_when_enabled(tmp_path):
    from imagecb.ingest import ingest_paths

    path = tmp_path / "file.png"
    path.touch()

    with patch("imagecb.ingest._run_ingest_pool"), patch(
        "imagecb.ingest._iter_work_items", return_value=iter([])
    ), patch("imagecb.ingest.existing_hashes", return_value=set()), patch(
        "imagecb.ingest.get_embedder"
    ), patch("imagecb.repair.repair_index_issues", return_value={"skipped": True}) as repair_mock:
        stats = ingest_paths([path], auto_repair=True)

    repair_mock.assert_called_once()
    assert stats["post_repair"]["skipped"] is True


def test_ingest_paths_no_repair_when_auto_repair_false():
    from imagecb.ingest import ingest_paths

    with patch("imagecb.ingest._run_ingest_pool"), patch(
        "imagecb.ingest._iter_work_items", return_value=iter([])
    ), patch("imagecb.ingest.existing_hashes", return_value=set()), patch(
        "imagecb.ingest.get_embedder"
    ), patch("imagecb.repair.repair_index_issues") as repair_mock:
        ingest_paths([], auto_repair=False)

    repair_mock.assert_not_called()


def test_ingest_paths_batched_runs_repair_once(tmp_path):
    from imagecb.ingest import ingest_paths_batched

    paths = [tmp_path / f"file{i}.png" for i in range(4)]
    for p in paths:
        p.touch()

    batch_stats = {
        "files": 2,
        "images_seen": 1,
        "images_added": 1,
        "images_updated": 0,
        "skipped_duplicates": 0,
        "errors": 0,
        "captions_weak": 0,
        "captions_failed": 0,
        "workers": 2,
        "elapsed_sec": 1.0,
    }
    ingest_calls: list[dict] = []
    repair_calls: list[dict] = []

    def fake_ingest(chunk, **kwargs):
        ingest_calls.append(kwargs)
        return dict(batch_stats, files=len(chunk))

    with patch("imagecb.ingest.ingest_paths", side_effect=fake_ingest), patch(
        "imagecb.ingest._finalize_ingest"
    ), patch(
        "imagecb.repair.repair_index_issues",
        side_effect=lambda **kw: repair_calls.append(kw) or {"skipped": True},
    ):
        stats = ingest_paths_batched(paths, batch_size=2, defer_bm25=True, auto_repair=True)

    assert all(call.get("auto_repair") is False for call in ingest_calls)
    assert len(repair_calls) == 1
    assert stats["post_repair"]["skipped"] is True


def test_ingest_paths_skip_caption_passes_skip_caption_phases(tmp_path):
    from imagecb.ingest import ingest_paths

    path = tmp_path / "file.png"
    path.touch()
    repair_calls: list[dict] = []

    with patch("imagecb.ingest.SETTINGS") as mock_settings, patch(
        "imagecb.ingest._run_ingest_pool"
    ), patch("imagecb.ingest._iter_work_items", return_value=iter([])), patch(
        "imagecb.ingest.existing_hashes", return_value=set()
    ), patch(
        "imagecb.ingest.get_embedder"
    ), patch(
        "imagecb.repair.repair_index_issues",
        side_effect=lambda **kw: repair_calls.append(kw) or {"skipped": True},
    ):
        mock_settings.post_ingest_repair_enabled = True
        mock_settings.ingest_workers = 2
        mock_settings.ingest_max_image_side = 1024
        mock_settings.ingest_batch_upsert = 16
        mock_settings.ingest_image_timeout_sec = 300
        ingest_paths([path], skip_caption=True, auto_repair=True)

    assert repair_calls[0]["skip_caption_phases"] is True


def test_assess_index_health_detects_orphan_chroma():
    records = [_record("a")]
    with patch("imagecb.repair.get_all_records", return_value=records), patch(
        "imagecb.repair._cache_missing", return_value=False
    ), patch("imagecb.repair.vector_store.count", return_value=2), patch(
        "imagecb.repair.vector_store.list_ids", return_value={"a", "orphan-1"}
    ), patch("imagecb.repair.vector_store.list_text_ids", return_value=set()), patch(
        "imagecb.repair.metadata_db.get_deleted_records", return_value=[]
    ), patch("imagecb.repair.bm25_index.count", return_value=1), patch(
        "imagecb.repair.bm25_index.list_ids", return_value={"a"}
    ):
        report = assess_index_health()
    assert report.orphan_chroma_count == 1
    assert report.orphan_chroma_ids == ["orphan-1"]
    assert report.stores_in_sync is False


def test_assess_index_health_unhealthy_when_text_vectors_missing():
    records = [_record("a", caption_short="dashboard screenshot")]
    with patch("imagecb.repair.get_all_records", return_value=records), patch(
        "imagecb.repair._cache_missing", return_value=False
    ), patch("imagecb.repair.caption_document_text", return_value="dashboard screenshot"), patch(
        "imagecb.repair.vector_store.count", return_value=1
    ), patch("imagecb.repair.vector_store.list_ids", return_value={"a"}), patch(
        "imagecb.repair.vector_store.list_text_ids", return_value=set()
    ), patch("imagecb.repair.metadata_db.get_deleted_records", return_value=[]), patch(
        "imagecb.repair.bm25_index.count", return_value=1
    ), patch("imagecb.repair.bm25_index.list_ids", return_value={"a"}):
        report = assess_index_health()
    assert report.missing_text_vector_count == 1
    assert report.is_healthy is False


def test_reconcile_index_safe_purges_orphans():
    before = IndexHealthReport(
        total_records=1,
        chroma_vectors=2,
        missing_cache_count=0,
        orphan_chroma_count=1,
        orphan_chroma_ids=["orphan-1"],
        orphan_text_vector_count=1,
        orphan_text_vector_ids=["orphan-txt"],
        stale_deleted_in_chroma_ids=[],
        bm25_stale=True,
        stores_in_sync=False,
        is_healthy=False,
    )
    after = IndexHealthReport(
        total_records=1,
        chroma_vectors=1,
        missing_cache_count=0,
        stores_in_sync=True,
        is_healthy=True,
    )
    with patch("imagecb.repair.assess_index_health", side_effect=[before, after]), patch(
        "imagecb.repair.vector_store.delete"
    ) as chroma_del, patch("imagecb.repair.vector_store.delete_text") as text_del, patch(
        "imagecb.repair.bm25_index.is_loaded", return_value=False
    ), patch("imagecb.repair.bm25_index.rebuild_from_records") as bm25_rebuild, patch(
        "imagecb.repair.get_all_records", return_value=[]
    ):
        stats = reconcile_index_safe()
    chroma_del.assert_called_once_with(["orphan-1"])
    text_del.assert_called_once_with(["orphan-txt"])
    bm25_rebuild.assert_called_once()
    assert stats["stores_in_sync_after"] is True


def test_repair_missing_cache_reingest_while_ingest_lock_held(tmp_path):
    """Auto-repair runs inside ingest with the (non-reentrant) lock held.

    Regression: repair used to re-acquire the ingest lock and raise
    IngestInProgressError after the ingest had already written data, which
    the job runner then requeued forever.
    """
    from imagecb import ingest as ingest_mod

    src = tmp_path / "deck.pptx"
    src.touch()
    records = [_record("a", source_file=str(src))]

    seen_kwargs: list[dict] = []
    real_ingest_paths = ingest_mod.ingest_paths

    def spying_ingest(paths, **kwargs):
        seen_kwargs.append(dict(kwargs))
        # Exercise the real lock logic but stop before any pipeline work.
        with patch("imagecb.ingest._ingest_paths_locked", return_value={"files": 0}):
            return real_ingest_paths(paths, **kwargs)

    assert ingest_mod._ingest_lock.acquire(blocking=False)
    try:
        with patch("imagecb.repair.resolve_source_file", return_value=src), patch(
            "imagecb.ingest.ingest_paths", side_effect=spying_ingest
        ):
            stats = repair_missing_cache(records, hold_ingest_lock=False)
    finally:
        ingest_mod._ingest_lock.release()

    assert stats["errors"] == 0
    assert seen_kwargs and seen_kwargs[0]["_hold_ingest_lock"] is False


def test_repair_missing_cache_takes_lock_when_standalone(tmp_path):
    """Admin/CLI-triggered repair (no outer ingest) still serializes on the lock."""
    src = tmp_path / "deck.pptx"
    src.touch()
    records = [_record("a", source_file=str(src))]

    seen_kwargs: list[dict] = []

    def fake_ingest(paths, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return {"files": 1, "images_updated": 1, "images_added": 0, "errors": 0}

    with patch("imagecb.repair.resolve_source_file", return_value=src), patch(
        "imagecb.ingest.ingest_paths", side_effect=fake_ingest
    ):
        repair_missing_cache(records)

    assert seen_kwargs and seen_kwargs[0]["_hold_ingest_lock"] is True
