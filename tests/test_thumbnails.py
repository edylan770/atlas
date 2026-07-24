"""Display thumbnail generation, backfill, and dedupe-by-key behavior."""

from __future__ import annotations

import io
import uuid
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.images import make_thumbnail
from imagecb.repair import regenerate_missing_thumbs
from imagecb.storage import blob_store
from imagecb.storage.metadata_db import ImageRecord, get_engine
from imagecb.telemetry.schema import ensure_telemetry_schema


@pytest.fixture(autouse=True)
def _db():
    get_engine()
    ensure_telemetry_schema()
    yield


def _rgb(size=(400, 300), color=(20, 80, 160)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_make_thumbnail_is_small_jpeg():
    data = make_thumbnail(_rgb(), max_side=256, quality=80)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI
    thumb = Image.open(io.BytesIO(data))
    assert max(thumb.size) <= 256


def test_cache_thumb_writes_single_deterministic_file(tmp_path):
    from imagecb.ingest import _cache_thumb

    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        image_cache_dir=tmp_path / "images",
    )
    image_id = "thumb-one"
    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.images.SETTINGS", settings
    ):
        ref1 = _cache_thumb(_rgb(), image_id)
        ref2 = _cache_thumb(_rgb(color=(1, 2, 3)), image_id)
        thumbs = list((tmp_path / "images" / "thumbs").glob("*"))
        exists = blob_store.thumb_exists(image_id)

    assert ref1 == ref2
    assert len(thumbs) == 1
    assert thumbs[0].name == f"{image_id}.jpg"
    assert exists is True


def test_regenerate_missing_thumbs_creates_and_skips(tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        image_cache_dir=tmp_path / "images",
    )
    missing_id = f"miss-{uuid.uuid4().hex[:8]}"
    present_id = f"have-{uuid.uuid4().hex[:8]}"
    png_missing = tmp_path / "images" / f"{missing_id}.png"
    png_present = tmp_path / "images" / f"{present_id}.png"
    png_missing.parent.mkdir(parents=True, exist_ok=True)
    _rgb().save(png_missing, format="PNG")
    _rgb().save(png_present, format="PNG")

    missing = ImageRecord(
        image_id=missing_id,
        content_hash=f"h-{missing_id}",
        image_path=str(png_missing),
        source_file=str(tmp_path / "a.png"),
        source_type="image",
        created_at=datetime.utcnow(),
    )
    present = ImageRecord(
        image_id=present_id,
        content_hash=f"h-{present_id}",
        image_path=str(png_present),
        source_file=str(tmp_path / "b.png"),
        source_type="image",
        created_at=datetime.utcnow(),
    )

    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.images.SETTINGS", settings
    ), patch("imagecb.repair.SETTINGS", settings), patch(
        "imagecb.paths.SETTINGS", settings
    ), patch(
        "imagecb.repair.get_all_records",
        return_value=[missing, present],
    ):
        blob_store.persist_image_thumb(present_id, make_thumbnail(_rgb()))
        first = regenerate_missing_thumbs()
        second = regenerate_missing_thumbs()

        assert first == {
            "scanned": 2,
            "created": 1,
            "skipped": 1,
            "failed": 0,
            "errors": [],
            "elapsed_sec": first["elapsed_sec"],
        }
        assert second["created"] == 0
        assert second["skipped"] == 2
        assert second["failed"] == 0
        assert blob_store.thumb_exists(missing_id)
        assert blob_store.thumb_exists(present_id)
        assert len(list((tmp_path / "images" / "thumbs").glob(f"{missing_id}*"))) == 1
        assert len(list((tmp_path / "images" / "thumbs").glob(f"{present_id}*"))) == 1


def test_admin_regenerate_missing_thumbs_endpoint():
    patched = replace(SETTINGS, admin_api_key="test-admin-secret")
    with patch("imagecb.api.auth.SETTINGS", patched), patch(
        "imagecb.repair.regenerate_missing_thumbs",
        return_value={
            "scanned": 2,
            "created": 1,
            "skipped": 1,
            "failed": 0,
            "errors": [],
            "elapsed_sec": 0.1,
        },
    ) as mock_regen:
        client = TestClient(create_app())
        res = client.post(
            "/api/admin/corpus/regenerate-missing-thumbs",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["created"] == 1
    assert body["skipped"] == 1
    mock_regen.assert_called_once()
