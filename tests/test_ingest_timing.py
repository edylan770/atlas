from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from imagecb.config import SETTINGS
from imagecb.ingest_timing import ImageTimingDetail, IngestTimingSession
from imagecb.storage import blob_store


def test_format_report_includes_aggregates_and_per_image_detail():
    session = IngestTimingSession(
        mode="ingest",
        enabled=True,
        meta={"workers": 2, "skip_caption": False, "skip_ocr": False, "force": True},
    )
    session.record("persist_source", 0.05)
    session.record("extract", 0.10)
    session.add_image_detail(
        ImageTimingDetail(
            image_id="img-1",
            source_file="deck.pptx",
            outcome="added",
            steps={
                "hash_image": 0.2,
                "cache_image": 0.3,
                "ocr": 0.5,
                "caption_vlm": 0.4,
                "embed_image": 0.35,
                "sqlite_write": 0.01,
                "embed_text": 0.02,
            },
            total_sec=1.78,
        )
    )
    report = session.format_report(
        {
            "files": 1,
            "images_seen": 1,
            "images_added": 1,
            "images_updated": 0,
            "skipped_duplicates": 0,
            "errors": 0,
            "workers": 2,
            "elapsed_sec": 2.0,
        }
    )

    assert "ImageCB ingest timing report" in report
    assert "AGGREGATE BY STEP" in report
    assert "PER-IMAGE DETAIL" in report
    assert "ocr" in report
    assert "image_id=img-1" in report
    assert "outcome=added" in report
    assert "mode:           ingest" in report


def test_persist_report_uploads_plain_text_under_ingest_logs(tmp_path):
    fake_objects: dict[tuple[str, str], tuple[bytes, str]] = {}

    class FakeS3:
        def put_object(self, *, Bucket, Key, Body, ContentType=None):
            fake_objects[(Bucket, Key)] = (bytes(Body), ContentType or "")

    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="atlas",
        s3_region="us-east-1",
        ingest_timing_log=True,
        data_dir=tmp_path,
    )
    session = IngestTimingSession(mode="ingest", enabled=True, meta={"workers": 1})
    session.record("finalize", 0.01)
    session.add_image_detail(
        ImageTimingDetail(
            image_id="img-2",
            source_file="a.png",
            outcome="added",
            steps={"hash_image": 0.01},
            total_sec=0.01,
        )
    )

    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.ingest_timing.SETTINGS", settings
    ), patch("imagecb.storage.blob_store.get_s3_client", return_value=FakeS3()):
        ref = session.persist_report({"files": 1, "images_seen": 1, "workers": 1, "elapsed_sec": 0.1})

    assert ref is not None
    assert ref.startswith("s3://private-corpus/atlas/ingest-logs/")
    assert ref.endswith(".txt")
    assert len(fake_objects) == 1
    ((_bucket, key), (body, content_type)) = next(iter(fake_objects.items()))
    assert "/ingest-logs/" in key
    assert content_type.startswith("text/plain")
    assert b"ImageCB ingest timing report" in body


def test_persist_report_disabled_returns_none():
    session = IngestTimingSession(enabled=False)
    assert session.persist_report({"files": 0}) is None


def test_ingest_log_key_shape():
    settings = replace(SETTINGS, s3_prefix="imagecb")
    when = datetime(2026, 7, 20, 17, 15, 30, tzinfo=timezone.utc)
    with patch("imagecb.storage.blob_store.SETTINGS", settings):
        key = blob_store.ingest_log_key("a1b2c3d4", when=when)
    assert key == "imagecb/ingest-logs/20260720_171530_a1b2c3d4.txt"
