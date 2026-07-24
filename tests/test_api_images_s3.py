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
        "imagecb.api.routes.blob_store.thumb_exists",
        return_value=True,
    ), patch(
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


def test_thumb_endpoint_falls_back_to_full_image_when_missing():
    record = ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path="s3://private/imagecb/images/image-1.png",
        source_file="s3://private/imagecb/uploads/aa/hash/source.png",
        source_type="image",
    )
    with patch("imagecb.api.routes.metadata_db.get_record", return_value=record), patch(
        "imagecb.api.routes.blob_store.thumb_exists",
        return_value=False,
    ), patch(
        "imagecb.api.routes.blob_store.describe",
        return_value=BlobInfo("image-1.png", "image/png", 7),
    ), patch(
        "imagecb.api.routes.blob_store.iter_bytes",
        return_value=iter([b"png", b"data"]),
    ):
        response = _client().get("/api/images/image-1/thumb")

    assert response.status_code == 200
    assert response.content == b"pngdata"
    assert response.headers["content-type"] == "image/png"


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
