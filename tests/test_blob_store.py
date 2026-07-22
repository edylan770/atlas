from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pytest

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

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


class FakeS3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


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


def test_s3_exists_returns_false_for_true_not_found():
    fake = FakeS3()
    with patch("imagecb.storage.blob_store.get_s3_client", return_value=fake), patch.object(
        fake,
        "head_object",
        side_effect=FakeS3Error("404"),
    ):
        assert blob_store.exists("s3://private-corpus/missing.png") is False


@pytest.mark.parametrize("code", ["AccessDenied", "ExpiredToken", "SlowDown"])
def test_s3_exists_surfaces_operational_failures(code):
    fake = FakeS3()
    error = FakeS3Error(code)
    with patch("imagecb.storage.blob_store.get_s3_client", return_value=fake), patch.object(
        fake,
        "head_object",
        side_effect=error,
    ):
        with pytest.raises(FakeS3Error) as exc_info:
            blob_store.exists("s3://private-corpus/image.png")

    assert exc_info.value is error


def test_local_delete_removes_file_and_fallbacks(tmp_path):
    primary = tmp_path / "primary.png"
    fallback = tmp_path / "fallback.png"
    primary.write_bytes(b"a")
    fallback.write_bytes(b"b")

    assert blob_store.delete(primary, fallbacks=(fallback,)) is True
    assert not primary.exists()
    assert not fallback.exists()
    assert blob_store.delete(primary, fallbacks=(fallback,)) is False


def test_s3_delete_issues_delete_object():
    fake = FakeS3()
    fake.objects[("private-corpus", "atlas/images/x.png")] = (b"png", "image/png")

    with patch("imagecb.storage.blob_store.get_s3_client", return_value=fake):
        assert blob_store.delete("s3://private-corpus/atlas/images/x.png") is True

    assert ("private-corpus", "atlas/images/x.png") not in fake.objects


def test_s3_delete_treats_missing_as_success():
    fake = FakeS3()

    def _missing(*, Bucket, Key):
        raise FakeS3Error("404")

    with patch("imagecb.storage.blob_store.get_s3_client", return_value=fake), patch.object(
        fake,
        "delete_object",
        side_effect=_missing,
    ):
        assert blob_store.delete("s3://private-corpus/missing.png") is False


@pytest.mark.parametrize("code", ["AccessDenied", "ExpiredToken", "SlowDown"])
def test_s3_delete_surfaces_operational_failures(code):
    fake = FakeS3()
    error = FakeS3Error(code)
    with patch("imagecb.storage.blob_store.get_s3_client", return_value=fake), patch.object(
        fake,
        "delete_object",
        side_effect=error,
    ):
        with pytest.raises(FakeS3Error) as exc_info:
            blob_store.delete("s3://private-corpus/image.png")

    assert exc_info.value is error


def test_presign_upload_uses_browser_endpoint():
    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_region="us-east-1",
        s3_presign_endpoint_url="http://localhost:9000",
    )

    class PresignClient:
        def generate_presigned_url(self, ClientMethod, Params, ExpiresIn, HttpMethod):
            assert ClientMethod == "put_object"
            assert HttpMethod == "PUT"
            assert Params["Bucket"] == "private-corpus"
            return (
                f"http://localhost:9000/{Params['Bucket']}/{Params['Key']}"
                f"?content-type={Params['ContentType']}&Expires={ExpiresIn}"
            )

    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.storage.blob_store.get_presign_s3_client",
        return_value=PresignClient(),
    ):
        url, headers = blob_store.presign_upload(
            "atlas/staging/job/file.png",
            content_type="image/png",
        )

    assert url.startswith("http://localhost:9000/private-corpus/atlas/staging/")
    assert headers == {"Content-Type": "image/png"}
