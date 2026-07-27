from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from imagecb.api.routes import router
from imagecb.storage.blob_store import BlobInfo
from imagecb.storage.metadata_db import ImageRecord


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_image_endpoint_streams_private_s3_blob():
    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.png",
        source_type="image",
    )
    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.describe",
        return_value=BlobInfo("image-1.png", "image/png", 7),
    ), patch(
        "imagecb.api.routes.blob_store.iter_bytes",
        return_value=iter([b"png", b"data"]),
    ):
        response = _client().get("/api/images/image-1")

    assert response.status_code == 200
    assert response.content == b"pngdata"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-length"] == "7"


def test_thumb_endpoint_streams_jpeg_when_present():
    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.png",
        source_type="image",
    )
    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.thumb_ref",
        return_value="s3://private/imagecb/thumbs/image-1.jpg",
    ), patch(
        "imagecb.api.routes.blob_store.describe",
        return_value=BlobInfo("image-1.jpg", "image/jpeg", 5),
    ), patch(
        "imagecb.api.routes.blob_store.iter_bytes",
        return_value=iter([b"thumb"]),
    ):
        response = _client().get("/api/images/image-1/thumb")

    assert response.status_code == 200
    assert response.content == b"thumb"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"


def test_thumb_endpoint_generates_jpeg_when_missing():
    from PIL import Image

    from imagecb.images import make_thumbnail

    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.png",
        source_type="image",
    )
    source = Image.new("RGB", (120, 80), (10, 20, 30))
    expected = make_thumbnail(source)
    persisted: list[tuple[str, bytes]] = []

    def _describe(_ref, **_kwargs):
        err = Exception("missing")
        err.response = {"Error": {"Code": "404"}}  # type: ignore[attr-defined]
        raise err

    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.thumb_ref",
        return_value="s3://private/imagecb/thumbs/image-1.jpg",
    ), patch(
        "imagecb.api.routes.blob_store.describe",
        side_effect=_describe,
    ), patch(
        "imagecb.api.routes.open_record_image",
        return_value=source,
    ), patch(
        "imagecb.api.routes.blob_store.persist_image_thumb",
        side_effect=lambda image_id, data: persisted.append((image_id, data)) or f"s3://x/{image_id}",
    ):
        response = _client().get("/api/images/image-1/thumb")

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert persisted == [("image-1", expected)]


def test_thumb_endpoint_serves_memory_jpeg_when_persist_fails():
    from PIL import Image

    from imagecb.images import make_thumbnail

    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.png",
        source_type="image",
    )
    source = Image.new("RGB", (64, 64), (1, 2, 3))
    expected = make_thumbnail(source)

    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.describe",
        side_effect=FileNotFoundError("no thumb"),
    ), patch(
        "imagecb.api.routes.open_record_image",
        return_value=source,
    ), patch(
        "imagecb.api.routes.blob_store.persist_image_thumb",
        side_effect=RuntimeError("s3 down"),
    ):
        response = _client().get("/api/images/image-1/thumb")

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "image/jpeg"


def test_source_endpoint_keeps_download_filename():
    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.pptx",
        source_type="pptx",
    )
    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.describe",
        return_value=BlobInfo(
            "source.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            4,
        ),
    ), patch(
        "imagecb.api.routes.blob_store.iter_bytes",
        return_value=iter([b"deck"]),
    ):
        response = _client().get("/api/sources/image-1")

    assert response.status_code == 200
    assert response.content == b"deck"
    assert "source.pptx" in response.headers["content-disposition"]
