from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from imagecb.config import SETTINGS
from imagecb.storage.blob_migration import migrate_local_blobs_to_s3
from imagecb.storage.metadata_db import ImageRecord


@contextmanager
def _session_for(record):
    class Session:
        def get(self, _model, image_id):
            return record if image_id == record.image_id else None

    yield Session()


def _record(tmp_path):
    image = tmp_path / "cached.png"
    source = tmp_path / "source.pptx"
    image.write_bytes(b"png")
    source.write_bytes(b"deck")
    return ImageRecord(
        image_id="image-1",
        content_hash="hash-1",
        image_path=str(image),
        source_file=str(source),
        source_type="pptx",
    )


def test_blob_migration_dry_run_does_not_write(tmp_path):
    record = _record(tmp_path)
    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private",
        s3_prefix="atlas",
    )
    with patch("imagecb.storage.blob_migration.SETTINGS", settings), patch(
        "imagecb.storage.blob_migration.get_all_records",
        return_value=[record],
    ), patch("imagecb.storage.blob_migration.blob_store.put_file") as put:
        stats = migrate_local_blobs_to_s3(dry_run=True)

    assert stats["image_candidates"] == 1
    assert stats["source_candidates"] == 1
    put.assert_not_called()
    assert not record.image_path.startswith("s3://")


def test_blob_migration_rewrites_only_after_upload(tmp_path):
    record = _record(tmp_path)
    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private",
        s3_prefix="atlas",
    )

    def fake_put(path, key, **_kwargs):
        return f"s3://private/{key}"

    with patch("imagecb.storage.blob_migration.SETTINGS", settings), patch(
        "imagecb.storage.blob_migration.get_all_records",
        return_value=[record],
    ), patch(
        "imagecb.storage.blob_migration.session_scope",
        side_effect=lambda: _session_for(record),
    ), patch(
        "imagecb.storage.blob_migration.blob_store.put_file",
        side_effect=fake_put,
    ), patch(
        "imagecb.storage.blob_migration.blob_store.SETTINGS",
        settings,
    ):
        stats = migrate_local_blobs_to_s3(dry_run=False)

    assert stats["images_migrated"] == 1
    assert stats["sources_migrated"] == 1
    assert record.image_path == "s3://private/atlas/images/image-1.png"
    assert record.source_file.startswith("s3://private/atlas/uploads/")
