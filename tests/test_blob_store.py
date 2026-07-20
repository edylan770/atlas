from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

from imagecb.config import SETTINGS
from imagecb.storage import blob_store


class _Body(io.BytesIO):
    pass


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = (bytes(Body), ContentType or "application/octet-stream")

    def upload_fileobj(self, handle, bucket, key, ExtraArgs=None):
        content_type = (ExtraArgs or {}).get("ContentType", "application/octet-stream")
        self.objects[(bucket, key)] = (handle.read(), content_type)

    def head_object(self, *, Bucket, Key):
        data, content_type = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "ContentType": content_type}

    def get_object(self, *, Bucket, Key):
        data, content_type = self.objects[(Bucket, Key)]
        return {"Body": _Body(data), "ContentType": content_type}


def test_s3_uri_round_trip_and_filename_safety():
    assert blob_store.parse_s3_uri("s3://bucket/a/b.png") == ("bucket", "a/b.png")
    assert blob_store.safe_filename("../../bad:name?.png") == "bad_name_.png"


def test_ingest_log_key_under_prefix():
    settings = replace(SETTINGS, s3_prefix="atlas")
    when = datetime(2026, 1, 2, 3, 4, 5)
    with patch("imagecb.storage.blob_store.SETTINGS", settings):
        key = blob_store.ingest_log_key("run99", when=when)
    assert key == "atlas/ingest-logs/20260102_030405_run99.txt"


def test_local_image_persistence(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        image_cache_dir=tmp_path / "images",
    )
    with patch("imagecb.storage.blob_store.SETTINGS", settings):
        ref = blob_store.persist_image_png("abc", b"png-data")

    assert (tmp_path / "images" / "abc.png").read_bytes() == b"png-data"
    assert ref.endswith("abc.png")


def test_private_s3_put_describe_read_and_stream(tmp_path):
    fake = FakeS3()
    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="atlas",
        s3_region="us-east-1",
    )
    source = tmp_path / "deck.pptx"
    source.write_bytes(b"deck-bytes")

    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.storage.blob_store.get_s3_client",
        return_value=fake,
    ):
        ref = blob_store.persist_source(source)
        assert ref.startswith("s3://private-corpus/atlas/uploads/")
        assert blob_store.exists(ref)
        assert blob_store.read_bytes(ref) == b"deck-bytes"
        assert b"".join(blob_store.iter_bytes(ref, chunk_size=3)) == b"deck-bytes"
        info = blob_store.describe(ref)

    assert info.filename == "deck.pptx"
    assert info.content_length == len(b"deck-bytes")
