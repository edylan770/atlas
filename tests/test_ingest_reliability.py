"""Tests for ingest batching, Bedrock gating, and stats aggregation."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imagecb.ingest import _merge_stats, ingest_paths_batched


def test_merge_stats_aggregates_batches():
    total = {
        "files": 5,
        "images_seen": 0,
        "images_added": 0,
        "images_updated": 0,
        "skipped_duplicates": 0,
        "errors": 0,
        "captions_weak": 0,
        "captions_failed": 0,
        "workers": 2,
        "elapsed_sec": 0.0,
        "batches": 0,
    }
    _merge_stats(
        total,
        {
            "files": 3,
            "images_seen": 10,
            "images_added": 8,
            "images_updated": 1,
            "skipped_duplicates": 1,
            "errors": 0,
            "captions_weak": 2,
            "captions_failed": 0,
        },
    )
    _merge_stats(
        total,
        {
            "files": 2,
            "images_seen": 5,
            "images_added": 4,
            "images_updated": 0,
            "skipped_duplicates": 1,
            "errors": 1,
            "captions_weak": 0,
            "captions_failed": 1,
        },
    )
    assert total["files"] == 5  # file count is set once by ingest_paths_batched, not merged
    assert total["images_seen"] == 15
    assert total["images_added"] == 12
    assert total["images_updated"] == 1
    assert total["skipped_duplicates"] == 2
    assert total["errors"] == 1
    assert total["captions_weak"] == 2
    assert total["captions_failed"] == 1


def test_ingest_paths_batched_defers_bm25_once(tmp_path):
    paths = [tmp_path / f"file{i}.png" for i in range(5)]
    for p in paths:
        p.touch()

    batch_stats = {
        "files": 2,
        "images_seen": 3,
        "images_added": 2,
        "images_updated": 0,
        "skipped_duplicates": 1,
        "errors": 0,
        "captions_weak": 0,
        "captions_failed": 0,
        "workers": 2,
        "elapsed_sec": 1.0,
    }
    ingest_calls: list[dict] = []
    finalize_calls: list[dict] = []

    def fake_ingest_paths(chunk, **kwargs):
        ingest_calls.append({"files": len(chunk), **kwargs})
        return dict(batch_stats, files=len(chunk))

    def fake_finalize(**kwargs):
        finalize_calls.append(kwargs)

    with patch("imagecb.ingest.ingest_paths", side_effect=fake_ingest_paths), patch(
        "imagecb.ingest._finalize_ingest", side_effect=fake_finalize
    ):
        stats = ingest_paths_batched(paths, batch_size=2, defer_bm25=True, workers=2)

    assert stats["batches"] == 3
    assert stats["files"] == 5
    assert stats["images_added"] == 6
    assert len(ingest_calls) == 3
    assert all(call["rebuild_bm25"] is False for call in ingest_calls)
    assert all(call["refresh_vocab"] is False for call in ingest_calls)
    assert len(finalize_calls) == 1
    assert finalize_calls[0] == {"rebuild_bm25": True, "refresh_vocab": True}


def test_bedrock_gate_limits_concurrency(monkeypatch):
    import imagecb.models.bedrock_client as bc

    monkeypatch.setattr(bc, "_client", None)
    monkeypatch.setattr(bc, "_semaphore", None)

    fake_settings = MagicMock()
    fake_settings.aws_region = "us-east-1"
    fake_settings.bedrock_connect_timeout = 10
    fake_settings.bedrock_read_timeout = 120
    fake_settings.bedrock_max_retries = 3
    fake_settings.bedrock_max_concurrent = 2
    monkeypatch.setattr(bc, "SETTINGS", fake_settings)

    active = 0
    peak = 0
    lock = threading.Lock()

    mock_client = MagicMock()

    def slow_converse(**_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with lock:
            active -= 1
        return {"output": {"message": {"content": []}}}

    mock_client.converse.side_effect = slow_converse
    monkeypatch.setattr(bc, "get_bedrock_runtime", lambda: mock_client)

    errors: list[Exception] = []

    def worker():
        try:
            bc.bedrock_converse(modelId="test")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert peak <= 2
    assert mock_client.converse.call_count == 4
