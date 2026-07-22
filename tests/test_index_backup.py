from __future__ import annotations

import io
import json
import pickle
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from imagecb.config import SETTINGS
from imagecb.storage import blob_store, index_backup, metadata_db


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

    def list_objects_v2(self, *, Bucket, Prefix="", MaxKeys=1000, ContinuationToken=None):
        keys = sorted(
            key for (bucket, key) in self.objects if bucket == Bucket and key.startswith(Prefix)
        )
        start = 0
        if ContinuationToken:
            start = int(ContinuationToken)
        page = keys[start : start + MaxKeys]
        truncated = start + MaxKeys < len(keys)
        response = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": truncated,
        }
        if truncated:
            response["NextContinuationToken"] = str(start + MaxKeys)
        return response


def _seed_index(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sqlite_path = data_dir / "imagecb.db"
    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "marker.txt").write_text("chroma-v1", encoding="utf-8")
    bm25_path = data_dir / "bm25.pkl"
    hubness_path = data_dir / "hubness.pkl"
    bm25_path.write_bytes(pickle.dumps({"image_ids": ["a"], "docs": [["tag"]]}))
    hubness_path.write_bytes(pickle.dumps({"count": 1}))

    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="imagecb",
        s3_region="us-east-1",
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        chroma_dir=chroma_dir,
        bm25_path=bm25_path,
        hubness_path=hubness_path,
    )
    with patch("imagecb.storage.metadata_db.SETTINGS", settings):
        metadata_db.dispose_engine()
        engine = metadata_db.get_engine()
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO images (image_id, content_hash, image_path, source_file, source_type) "
                "VALUES ('img-1', 'hash-1', 's3://private-corpus/imagecb/images/img-1.png', "
                "'s3://private-corpus/imagecb/uploads/x/y/file.jpg', 'image')"
            )
        metadata_db.dispose_engine()
    return {
        "settings": settings,
        "sqlite_path": sqlite_path,
        "chroma_dir": chroma_dir,
        "bm25_path": bm25_path,
        "hubness_path": hubness_path,
    }


@pytest.fixture
def seeded(tmp_path):
    return _seed_index(tmp_path)


def test_index_backup_key_helpers():
    settings = replace(SETTINGS, s3_prefix="atlas")
    with patch("imagecb.storage.blob_store.SETTINGS", settings):
        assert blob_store.index_backup_prefix() == "atlas/index-backups"
        assert blob_store.index_backup_key("snap1", "manifest.json") == (
            "atlas/index-backups/snap1/manifest.json"
        )


def test_list_keys_paginates(monkeypatch):
    fake = FakeS3()
    for i in range(5):
        fake.objects[("private-corpus", f"imagecb/index-backups/a/{i}.txt")] = (b"x", "text/plain")
    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="imagecb",
    )
    monkeypatch.setattr(blob_store, "SETTINGS", settings)
    monkeypatch.setattr(blob_store, "get_s3_client", lambda: fake)
    keys = blob_store.list_keys("imagecb/index-backups/", max_keys=3)
    assert len(keys) == 3


def test_incomplete_backup_not_listed(seeded, monkeypatch):
    fake = FakeS3()
    settings = seeded["settings"]
    monkeypatch.setattr(blob_store, "SETTINGS", settings)
    monkeypatch.setattr(index_backup, "SETTINGS", settings)
    monkeypatch.setattr(blob_store, "get_s3_client", lambda: fake)
    fake.objects[("private-corpus", "imagecb/index-backups/partial/archive.tar.gz")] = (
        b"not-a-real-archive",
        "application/gzip",
    )
    assert index_backup.list_backups() == []


def test_backup_restore_round_trip(seeded, monkeypatch):
    fake = FakeS3()
    settings = seeded["settings"]

    monkeypatch.setattr(blob_store, "SETTINGS", settings)
    monkeypatch.setattr(index_backup, "SETTINGS", settings)
    monkeypatch.setattr(metadata_db, "SETTINGS", settings)
    monkeypatch.setattr("imagecb.storage.vector_store.SETTINGS", settings)
    monkeypatch.setattr("imagecb.storage.bm25_index.SETTINGS", settings)
    monkeypatch.setattr("imagecb.retrieval.hubness.SETTINGS", settings)
    monkeypatch.setattr(blob_store, "get_s3_client", lambda: fake)

    monkeypatch.setattr(index_backup, "_cancel_active_jobs", lambda: [])
    monkeypatch.setattr(index_backup, "_wait_for_idle", lambda **_: None)
    monkeypatch.setattr(
        "imagecb.ingest_jobs.stop_job_runner",
        lambda: None,
    )
    monkeypatch.setattr(
        "imagecb.ingest_jobs.start_job_runner",
        lambda: None,
    )
    monkeypatch.setattr(index_backup, "_reopen_live_stores", lambda: metadata_db.reopen_engine())
    monkeypatch.setattr(index_backup, "_dispose_live_stores", lambda: metadata_db.dispose_engine())

    result = index_backup.create_backup(label="unit-test")
    backup_id = result["backup_id"]
    assert result["ok"] is True
    assert result["label"] == "unit-test"

    backups = index_backup.list_backups()
    assert len(backups) == 1
    assert backups[0]["id"] == backup_id
    manifest_key = f"imagecb/index-backups/{backup_id}/manifest.json"
    archive_key = f"imagecb/index-backups/{backup_id}/archive.tar.gz"
    assert ("private-corpus", manifest_key) in fake.objects
    assert ("private-corpus", archive_key) in fake.objects
    manifest = json.loads(fake.objects[("private-corpus", manifest_key)][0])
    assert manifest["archive_sha256"] == result["archive_sha256"]

    # Corrupt live index, then restore.
    metadata_db.dispose_engine()
    seeded["sqlite_path"].unlink()
    seeded["chroma_dir"].joinpath("marker.txt").write_text("wiped", encoding="utf-8")
    seeded["bm25_path"].unlink()
    seeded["hubness_path"].unlink()

    restored = index_backup.restore_backup(backup_id, confirm=True)
    assert restored["ok"] is True
    assert seeded["sqlite_path"].is_file()
    assert seeded["chroma_dir"].joinpath("marker.txt").read_text(encoding="utf-8") == "chroma-v1"
    assert seeded["bm25_path"].is_file()
    assert seeded["hubness_path"].is_file()

    metadata_db.reopen_engine()
    with metadata_db.session_scope() as session:
        row = session.get(metadata_db.ImageRecord, "img-1")
        assert row is not None
        assert row.content_hash == "hash-1"


def test_restore_requires_confirm(seeded, monkeypatch):
    settings = seeded["settings"]
    monkeypatch.setattr(index_backup, "SETTINGS", settings)
    with pytest.raises(index_backup.IndexBackupError, match="confirm=true"):
        index_backup.restore_backup("missing", confirm=False)


def test_backup_requires_s3(monkeypatch, tmp_path):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        s3_bucket=None,
        data_dir=tmp_path,
    )
    monkeypatch.setattr(index_backup, "SETTINGS", settings)
    with pytest.raises(index_backup.IndexBackupError, match="BLOB_STORAGE_BACKEND=s3"):
        index_backup.create_backup()
