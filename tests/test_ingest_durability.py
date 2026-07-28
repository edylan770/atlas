"""Phase 5 B3: ingest durability regressions."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

from PIL import Image

from imagecb.images import make_thumbnail, resize_for_model


def test_extreme_aspect_ratio_thumbnail_never_zero_dim():
    strip = Image.new("RGB", (2000, 3), "blue")
    data = make_thumbnail(strip, max_side=256, quality=80)
    out = Image.open(__import__("io").BytesIO(data))
    assert out.size[0] >= 1 and out.size[1] >= 1


def test_transparency_composites_to_white_not_black():
    logo = Image.new("RGBA", (10, 10), (0, 0, 0, 0))  # fully transparent
    rgb = resize_for_model(logo, 10)
    assert rgb.getpixel((5, 5)) == (255, 255, 255)


def test_bm25_save_is_atomic(tmp_path, monkeypatch):
    from imagecb.storage import bm25_index

    idx = bm25_index.BM25Index()
    idx._state = bm25_index._BM25State(image_ids=["a"], docs=[["hello"]])
    target = tmp_path / "bm25.pkl"
    idx.save(target)
    assert target.exists()
    with open(target, "rb") as f:
        state = pickle.load(f)
    assert state.image_ids == ["a"]
    assert not (tmp_path / "bm25.pkl.tmp").exists()


def test_image_exists_requires_cache_for_derived_records():
    from imagecb import paths

    record = MagicMock()
    record.image_path = "/gone/cached.png"
    record.source_file = "/present/deck.pptx"
    record.source_type = "pptx"
    record.image_id = "x"

    def fake_exists(ref, fallbacks=()):
        return ref == "/present/deck.pptx"

    with patch.object(paths.blob_store, "exists", side_effect=fake_exists):
        # a live pptx is NOT a displayable image cache
        assert paths.image_exists(record) is False

    record.source_type = "image"
    record.source_file = "/present/photo.jpg"

    def fake_exists2(ref, fallbacks=()):
        return ref == "/present/photo.jpg"

    with patch.object(paths.blob_store, "exists", side_effect=fake_exists2):
        assert paths.image_exists(record) is True


def test_finish_job_failure_preserves_cumulative_stats(tmp_path, monkeypatch):
    from imagecb import ingest_jobs

    ingest_jobs.ensure_job_schema()
    job_id = ingest_jobs.new_job_id()
    from imagecb.ingest_jobs import IngestJob, _finish_job, session_scope
    import json

    with session_scope() as session:
        session.add(
            IngestJob(
                job_id=job_id,
                status="running",
                files_json="[]",
                options_json="{}",
                stats_json=json.dumps({"images_added": 900, "images_seen": 900}),
                files_total=10,
                files_done=9,
            )
        )

    _finish_job(job_id, status="failed", stats=None, error="boom")

    with session_scope() as session:
        row = session.get(IngestJob, job_id)
        stats = json.loads(row.stats_json)
        assert stats.get("images_added") == 900, "failure must not wipe progress"
        assert row.images_processed == 900
