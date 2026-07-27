"""Tests for idle-only S3 orphan blob GC."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from imagecb.admin.orphan_blobs import OrphanBlobError, assess_orphan_blobs, purge_orphan_blobs
from imagecb.api.server import create_app
from imagecb.config import SETTINGS
from imagecb.storage import blob_store
from imagecb.storage.blob_store import ListedObject
from imagecb.storage.metadata_db import ImageRecord, get_engine
from imagecb.telemetry.schema import ensure_telemetry_schema


@pytest.fixture(autouse=True)
def _db():
    get_engine()
    ensure_telemetry_schema()
    yield


def _record(
    image_id: str,
    *,
    source_file: str = "",
    deleted_at: datetime | None = None,
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"s3://bucket/atlas/images/{image_id}.png",
        source_file=source_file,
        source_type="image",
        created_at=datetime.utcnow(),
        deleted_at=deleted_at,
    )


def _old(hours: float = 5) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _fresh(minutes: float = 10) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def _s3_settings():
    return replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="atlas",
        s3_region="us-east-1",
        admin_api_key="test-admin-secret",
    )


def test_assess_keeps_referenced_and_soft_deleted():
    keep = _record("keep", source_file="s3://private-corpus/atlas/uploads/aa/aabb/keep.pptx")
    soft = _record(
        "soft",
        source_file="s3://private-corpus/atlas/uploads/bb/bbcc/soft.pptx",
        deleted_at=datetime.utcnow(),
    )
    objects = {
        "images": [
            ListedObject("atlas/images/keep.png", _old()),
            ListedObject("atlas/images/soft.png", _old()),
            ListedObject("atlas/images/orphan.png", _old()),
            ListedObject("atlas/images/fresh.png", _fresh()),
        ],
        "thumbs": [
            ListedObject("atlas/thumbs/keep.jpg", _old()),
            ListedObject("atlas/thumbs/soft.jpg", _old()),
            ListedObject("atlas/thumbs/orphan.jpg", _old()),
        ],
        "uploads": [
            ListedObject("atlas/uploads/aa/aabb/keep.pptx", _old()),
            ListedObject("atlas/uploads/bb/bbcc/soft.pptx", _old()),
            ListedObject("atlas/uploads/cc/ccdd/orphan.pptx", _old()),
        ],
        "staging": [
            ListedObject("atlas/staging/job1/f1/deck.pptx", _old()),
            ListedObject("atlas/staging/busy/f2/deck.pptx", _old()),
        ],
    }

    def list_objects(prefix: str, *, max_keys=None):
        if prefix.startswith("atlas/images"):
            return objects["images"]
        if prefix.startswith("atlas/thumbs"):
            return objects["thumbs"]
        if prefix.startswith("atlas/uploads"):
            return objects["uploads"]
        if prefix.startswith("atlas/staging"):
            return objects["staging"]
        return []

    settings = _s3_settings()
    with patch("imagecb.admin.orphan_blobs.SETTINGS", settings), patch(
        "imagecb.storage.blob_store.SETTINGS", settings
    ), patch("imagecb.admin.orphan_blobs.list_busy_jobs", return_value=[]), patch(
        "imagecb.admin.orphan_blobs.busy_staging_uris",
        return_value={"s3://private-corpus/atlas/staging/busy/f2/deck.pptx"},
    ), patch(
        "imagecb.admin.orphan_blobs.get_all_records",
        return_value=[keep, soft],
    ), patch("imagecb.admin.orphan_blobs.blob_store.list_objects", side_effect=list_objects):
        report = assess_orphan_blobs(min_age_hours=1)

    assert {c.image_id for c in report.images} == {"orphan"}
    assert {c.image_id for c in report.thumbs} == {"orphan"}
    assert [c.uri for c in report.uploads] == [
        "s3://private-corpus/atlas/uploads/cc/ccdd/orphan.pptx"
    ]
    assert [c.uri for c in report.staging] == [
        "s3://private-corpus/atlas/staging/job1/f1/deck.pptx"
    ]
    assert any(c.image_id == "fresh" for c in report.skipped_too_new)


def test_purge_dry_run_does_not_delete():
    settings = _s3_settings()
    deleted: list[str] = []

    def list_objects(prefix: str, *, max_keys=None):
        if "images" in prefix:
            return [ListedObject("atlas/images/x.png", _old())]
        return []

    with patch("imagecb.admin.orphan_blobs.SETTINGS", settings), patch(
        "imagecb.storage.blob_store.SETTINGS", settings
    ), patch("imagecb.admin.orphan_blobs.list_busy_jobs", return_value=[]), patch(
        "imagecb.admin.orphan_blobs.busy_staging_uris", return_value=set()
    ), patch("imagecb.admin.orphan_blobs.get_all_records", return_value=[]), patch(
        "imagecb.admin.orphan_blobs.blob_store.list_objects", side_effect=list_objects
    ), patch(
        "imagecb.admin.orphan_blobs.blob_store.delete",
        side_effect=lambda uri, **kwargs: deleted.append(uri) or True,
    ):
        result = purge_orphan_blobs(dry_run=True, min_age_hours=1)
        assert result["dry_run"] is True
        assert result["orphan_image_count"] == 1
        assert result["deleted_count"] == 0
        assert deleted == []

        result = purge_orphan_blobs(dry_run=False, min_age_hours=1)
        assert result["dry_run"] is False
        assert result["deleted_count"] == 1
        assert deleted == ["s3://private-corpus/atlas/images/x.png"]


def test_refuse_when_ingest_busy():
    settings = _s3_settings()
    with patch("imagecb.admin.orphan_blobs.SETTINGS", settings), patch(
        "imagecb.admin.orphan_blobs.list_busy_jobs",
        return_value=[{"job_id": "j1", "status": "running"}],
    ):
        with pytest.raises(OrphanBlobError, match="Refusing"):
            assess_orphan_blobs()


def test_admin_orphan_blob_endpoints():
    settings = _s3_settings()
    fake_report = {
        "dry_run": True,
        "min_age_hours": 1.0,
        "orphan_image_count": 2,
        "orphan_thumb_count": 2,
        "orphan_upload_count": 1,
        "orphan_staging_count": 0,
        "skipped_too_new_count": 0,
        "purgeable_count": 5,
        "deleted_count": 0,
        "failed_count": 0,
        "elapsed_sec": 0.1,
        "samples": {},
        "deleted_sample": [],
        "failed": [],
    }
    with patch("imagecb.api.auth.SETTINGS", settings), patch(
        "imagecb.admin.orphan_blobs.assess_orphan_blobs"
    ) as mock_assess, patch(
        "imagecb.admin.orphan_blobs.purge_orphan_blobs", return_value={**fake_report, "dry_run": False, "deleted_count": 5}
    ) as mock_purge:
        from imagecb.admin.orphan_blobs import OrphanBlobReport

        mock_assess.return_value = OrphanBlobReport(
            dry_run=True,
            min_age_hours=1.0,
            images=[],
            thumbs=[],
        )
        # Force to_dict shape via patching return of assess through module used by route
        mock_assess.return_value.to_dict = lambda: fake_report  # type: ignore[method-assign]

        client = TestClient(create_app())
        headers = {"X-Admin-Api-Key": "test-admin-secret"}
        get_res = client.get("/api/admin/corpus/orphan-blobs", headers=headers)
        assert get_res.status_code == 200
        assert get_res.json()["orphan_image_count"] == 2

        post_res = client.post(
            "/api/admin/corpus/purge-orphan-blobs",
            headers=headers,
            json={"dry_run": False, "min_age_hours": 1},
        )
        assert post_res.status_code == 200
        assert post_res.json()["deleted_count"] == 5
        mock_purge.assert_called_once()


def test_admin_orphan_blobs_conflict_when_busy():
    settings = _s3_settings()
    with patch("imagecb.api.auth.SETTINGS", settings), patch(
        "imagecb.admin.orphan_blobs.assess_orphan_blobs",
        side_effect=OrphanBlobError("Refusing orphan blob GC while ingest jobs are active"),
    ):
        client = TestClient(create_app())
        res = client.get(
            "/api/admin/corpus/orphan-blobs",
            headers={"X-Admin-Api-Key": "test-admin-secret"},
        )
        assert res.status_code == 409
        assert "Refusing" in res.json()["detail"]


def test_list_image_ids_s3():
    fake_objects = {
        ("private-corpus", "atlas/images/one.png"): (b"a", "image/png"),
        ("private-corpus", "atlas/images/two.png"): (b"b", "image/png"),
        ("private-corpus", "atlas/thumbs/one.jpg"): (b"c", "image/jpeg"),
    }

    class Fake:
        def list_objects_v2(self, **kwargs):
            prefix = kwargs["Prefix"]
            contents = [
                {"Key": key, "LastModified": _old()}
                for (bucket, key), _ in fake_objects.items()
                if bucket == kwargs["Bucket"] and key.startswith(prefix)
            ]
            return {"Contents": contents, "IsTruncated": False}

    settings = _s3_settings()
    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.storage.blob_store.get_s3_client", return_value=Fake()
    ):
        assert blob_store.list_image_ids() == {"one", "two"}
        objs = blob_store.list_objects("atlas/images/")
        assert len(objs) == 2
        assert objs[0].last_modified is not None
