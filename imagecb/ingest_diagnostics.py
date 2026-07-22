"""Authenticated runtime and dependency diagnostics for ingestion."""

from __future__ import annotations

import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from sqlalchemy import text

from imagecb.config import SETTINGS
from imagecb.ingest_jobs import runner_health
from imagecb.storage import blob_store
from imagecb.storage.metadata_db import get_engine


def _check(name: str, operation: Callable[[], Any]) -> dict:
    started = time.perf_counter()
    try:
        detail = operation()
        return {
            "name": name,
            "ok": True,
            "detail": str(detail or "ok"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }


def runtime_diagnostics() -> dict:
    engine = get_engine()
    sqlite_path = SETTINGS.sqlite_path.resolve()
    stat = sqlite_path.stat() if sqlite_path.exists() else None
    sqlite_identity = (
        f"{sqlite_path}:{getattr(stat, 'st_dev', 0)}:{getattr(stat, 'st_ino', 0)}"
        if stat
        else f"{sqlite_path}:missing"
    )
    from imagecb.storage.index_backup import last_checkpoint_info, startup_restore_info

    return {
        "build_id": os.environ.get("APP_BUILD_ID", "development"),
        "runtime_id": f"{socket.gethostname()}:{os.getpid()}",
        "pid": os.getpid(),
        "storage_backend": SETTINGS.blob_storage_backend,
        "aws_region": SETTINGS.aws_region,
        "s3_region": SETTINGS.s3_region,
        "s3_bucket": SETTINGS.s3_bucket,
        "s3_prefix": SETTINGS.s3_prefix,
        "embedding_model": SETTINGS.embedding_model,
        "text_embedding_model": SETTINGS.text_embedding_model,
        "vlm_model": SETTINGS.vlm_model,
        "sqlite_path": str(sqlite_path),
        "sqlite_identity": sqlite_identity,
        "database_url": engine.url.render_as_string(hide_password=True),
        "runner": runner_health(),
        "index_checkpoint_enabled": SETTINGS.index_checkpoint_enabled,
        "index_checkpoint_every_n": SETTINGS.index_checkpoint_every_n,
        "index_auto_restore_on_startup": SETTINGS.index_auto_restore_on_startup,
        "last_checkpoint": last_checkpoint_info(),
        "startup_restore": startup_restore_info(),
    }


def _writable_data_dir() -> str:
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=SETTINGS.data_dir,
        prefix=".ingest-preflight-",
        delete=False,
    ) as handle:
        path = Path(handle.name)
        handle.write(b"atlas-ingest-preflight")
    path.unlink()
    return str(SETTINGS.data_dir.resolve())


def _sqlite_round_trip() -> str:
    with get_engine().begin() as connection:
        value = connection.execute(text("SELECT 1")).scalar_one()
        database = connection.exec_driver_sql("PRAGMA database_list").fetchone()
    return f"value={value}, database={database[2] if database else 'unknown'}"


def _s3_round_trip() -> str:
    if SETTINGS.blob_storage_backend != "s3":
        return "not configured; local blob backend"
    key = "/".join(
        part
        for part in (
            SETTINGS.s3_prefix.strip("/"),
            "preflight",
            f"{uuid.uuid4().hex}.txt",
        )
        if part
    )
    client = blob_store.get_s3_client()
    try:
        client.put_object(
            Bucket=SETTINGS.s3_bucket,
            Key=key,
            Body=b"atlas-ingest-preflight",
            ContentType="text/plain",
        )
        response = client.get_object(Bucket=SETTINGS.s3_bucket, Key=key)
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            body.close()
        if payload != b"atlas-ingest-preflight":
            raise RuntimeError("S3 preflight payload did not round-trip")
        return f"s3://{SETTINGS.s3_bucket}/{key}"
    finally:
        try:
            client.delete_object(Bucket=SETTINGS.s3_bucket, Key=key)
        except Exception:
            pass


def _image_embedding_probe() -> str:
    from imagecb.models.embedder import get_embedder

    vector = get_embedder().embed_image(Image.new("RGB", (2, 2), "white"))
    return f"{SETTINGS.embedding_model}: dimensions={len(vector)}"


def _text_embedding_probe() -> str:
    if not SETTINGS.caption_text_lane_enabled:
        return "disabled"
    from imagecb.models.embedder import get_text_embedder

    vector = get_text_embedder().embed_document("ATLAS ingestion preflight")
    return f"{SETTINGS.text_embedding_model}: dimensions={len(vector)}"


def _caption_probe() -> str:
    from imagecb.models.vlm import get_captioner

    caption = get_captioner().caption_image(
        Image.new("RGB", (8, 8), "white"),
        context="ATLAS ingestion preflight test image.",
    )
    if caption.caption_quality == "failed":
        raise RuntimeError(caption.detailed_description or "caption model returned failure")
    return f"{SETTINGS.vlm_model}: available"


def run_ingest_preflight() -> dict:
    checks = [
        _check("data_directory", _writable_data_dir),
        _check("sqlite", _sqlite_round_trip),
        _check("blob_storage", _s3_round_trip),
        _check("image_embedding", _image_embedding_probe),
        _check("text_embedding", _text_embedding_probe),
        _check("caption_model", _caption_probe),
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "runtime": runtime_diagnostics(),
        "checks": checks,
    }
