"""Search and interaction telemetry (blob/S3 store)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from imagecb.config import SETTINGS
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import ImageRecord
from imagecb.telemetry import s3_store
from imagecb.telemetry.recorder import record_interaction, record_search_from_results


def _sample_record(image_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="/tmp/doc.pptx",
        source_type="pptx",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def telemetry_dir(tmp_path, monkeypatch):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path,
        s3_prefix="imagecb",
        telemetry_retention_days=90,
        telemetry_default_window_days=90,
    )
    monkeypatch.setattr(s3_store, "SETTINGS", settings)
    monkeypatch.setattr("imagecb.storage.blob_store.SETTINGS", settings)
    monkeypatch.setattr("imagecb.telemetry.recorder.SETTINGS", settings)
    s3_store.invalidate_quality_cache()
    yield tmp_path


def test_record_search_and_interaction_linkage(telemetry_dir):
    results = [
        RankedResult(
            image_id="img-a",
            score=0.5,
            record=_sample_record("img-a"),
            provenance_line="slide 1",
            score_kind="rerank",
        )
    ]
    spec = QuerySpec(semantic_query="charts", raw_text="charts")
    event_id = record_search_from_results(
        query_text="charts",
        user_id="user-1",
        session_id="sess-1",
        search_kind="chat",
        results=results,
        spec=spec,
    )

    iid = record_interaction(
        search_event_id=event_id,
        image_id="img-a",
        interaction_type="view",
        user_id="user-1",
        rank=1,
    )
    assert iid

    row = s3_store.get_search_event(event_id)
    assert row is not None
    assert row["served_image_ids"] == ["img-a"]
    assert row["top_score"] == 0.5
    assert row["result_count"] == 1
    assert row["has_interaction"] is True

    rollup = s3_store.load_daily_rollup(s3_store._dt_str(datetime.utcnow()))
    assert rollup["total_searches"] == 1
    assert rollup["interaction_count"] == 1
    assert rollup["no_interaction_count"] == 0


def test_interaction_rejects_unknown_image(telemetry_dir):
    event_id = record_search_from_results(
        query_text="q",
        user_id="u",
        session_id=None,
        search_kind="chat",
        results=[
            RankedResult(
                image_id="only-one",
                score=0.3,
                record=_sample_record("only-one"),
                provenance_line="",
            )
        ],
    )
    with pytest.raises(ValueError, match="not in the originating"):
        record_interaction(
            search_event_id=event_id,
            image_id="other",
            interaction_type="view",
        )


def test_bump_daily_rollup_does_not_wipe_on_transient_get(telemetry_dir, monkeypatch):
    """Transient S3 GET failure must not overwrite an existing rollup with zeros+delta."""
    dt = s3_store._dt_str(datetime.utcnow())
    s3_store.bump_daily_rollup(dt, {"total_searches": 10, "interaction_count": 3})
    before = s3_store.load_daily_rollup(dt)
    assert before["total_searches"] == 10

    key = s3_store.daily_rollup_key(dt)
    real_get_raw = s3_store._get_raw

    def flaky_get(k: str, *, strict: bool = False):
        if k == key:
            raise RuntimeError("simulated S3 timeout")
        return real_get_raw(k, strict=strict)

    monkeypatch.setattr(s3_store, "_get_raw", flaky_get)
    with pytest.raises(RuntimeError, match="simulated S3 timeout"):
        s3_store.bump_daily_rollup(dt, {"total_searches": 1})

    monkeypatch.setattr(s3_store, "_get_raw", real_get_raw)
    after = s3_store.load_daily_rollup(dt)
    assert after["total_searches"] == 10
    assert after["interaction_count"] == 3
