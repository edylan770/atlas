"""Admin analytics classification (blob/S3 telemetry)."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime

import pytest

from imagecb.admin import analytics
from imagecb.config import SETTINGS
from imagecb.telemetry import s3_store


@pytest.fixture
def telemetry_dir(tmp_path, monkeypatch):
    settings = replace(
        SETTINGS,
        blob_storage_backend="local",
        data_dir=tmp_path,
        s3_prefix="imagecb",
        weak_result_score_threshold=0.25,
        telemetry_retention_days=90,
        telemetry_default_window_days=90,
    )
    monkeypatch.setattr(s3_store, "SETTINGS", settings)
    monkeypatch.setattr("imagecb.storage.blob_store.SETTINGS", settings)
    monkeypatch.setattr("imagecb.admin.analytics.SETTINGS", settings)
    monkeypatch.setattr("imagecb.telemetry.recorder.SETTINGS", settings)
    s3_store.invalidate_quality_cache()
    yield tmp_path


def _add_search(
    *,
    result_count: int,
    top_score: float | None,
    served: list[str],
    query_text: str = "test",
    parsed_semantic_query: str | None = None,
    search_kind: str = "chat",
    total_ms: float | None = None,
    ask_ms: float | None = None,
    reply_ms: float | None = None,
    timings: dict | None = None,
    timing_log: str | None = None,
) -> str:
    eid = str(uuid.uuid4())
    created = datetime.utcnow()
    is_weak = (
        result_count > 0
        and top_score is not None
        and top_score < 0.25
    )
    event = {
        "id": eid,
        "created_at": created.isoformat(),
        "query_text": query_text,
        "user_id": "u",
        "session_id": None,
        "search_kind": search_kind,
        "served_image_ids": served,
        "result_count": result_count,
        "top_score": top_score,
        "top_score_kind": "rerank" if top_score is not None else None,
        "parsed_semantic_query": parsed_semantic_query,
        "total_ms": total_ms,
        "ask_ms": ask_ms,
        "reply_ms": reply_ms,
        "timings": timings,
        "timing_log": timing_log,
        "has_interaction": False,
    }
    s3_store.put_search_event(event)
    s3_store.bump_daily_rollup(
        s3_store._dt_str(created),
        {
            "total_searches": 1,
            "zero_result_count": 1 if result_count == 0 else 0,
            "weak_result_count": 1 if is_weak else 0,
            "searches_with_results": 1 if result_count > 0 else 0,
            "no_interaction_count": 1 if result_count > 0 else 0,
            "interaction_count": 0,
        },
    )
    return eid


def _add_interaction(search_event_id: str, image_id: str) -> None:
    search = s3_store.get_search_event(search_event_id)
    assert search is not None
    created = datetime.utcnow()
    s3_store.put_interaction_event(
        {
            "id": str(uuid.uuid4()),
            "search_event_id": search_event_id,
            "image_id": image_id,
            "interaction_type": "view",
            "created_at": created.isoformat(),
            "user_id": "u",
            "rank": 1,
        }
    )
    if not search.get("has_interaction"):
        search["has_interaction"] = True
        s3_store.put_search_event(search)
        search_dt = s3_store._dt_str(
            s3_store._parse_created_at(search["created_at"]) or created
        )
        s3_store.bump_daily_rollup(search_dt, {"no_interaction_count": -1})
    s3_store.bump_daily_rollup(s3_store._dt_str(created), {"interaction_count": 1})
    s3_store.invalidate_quality_cache()


def test_search_quality_categories(telemetry_dir):
    zero_id = _add_search(result_count=0, top_score=None, served=[])
    weak_id = _add_search(result_count=2, top_score=0.1, served=["a", "b"])
    served_id = _add_search(result_count=1, top_score=0.9, served=["c"])
    _add_interaction(served_id, "c")
    no_ix_id = _add_search(result_count=3, top_score=0.8, served=["d", "e", "f"])

    data = analytics.search_quality_lists(limit=100, weak_score_threshold=0.25)
    zero_ids = {r["search_event_id"] for r in data["zero_result"]}
    weak_ids = {r["search_event_id"] for r in data["weak_result"]}
    no_ix_ids = {r["search_event_id"] for r in data["no_interaction"]}

    assert zero_id in zero_ids
    assert weak_id in weak_ids
    assert no_ix_id in no_ix_ids
    assert served_id not in no_ix_ids


def test_display_query_prefers_semantic_for_chat(telemetry_dir):
    eid = _add_search(
        result_count=1,
        top_score=0.5,
        served=["x"],
        query_text="find charts",
        parsed_semantic_query="quarterly revenue charts in presentations",
    )
    data = analytics.search_quality_lists(limit=10)
    row = next(
        r
        for r in data["weak_result"] + data["no_interaction"] + data["zero_result"]
        if r["search_event_id"] == eid
    )
    assert row["display_query"] == "quarterly revenue charts in presentations"
    assert row["user_message"] == "find charts"


def test_display_query_similar_uses_query_text(telemetry_dir):
    eid = _add_search(
        result_count=2,
        top_score=0.4,
        served=["a", "b"],
        query_text="[similar image search]",
        parsed_semantic_query="visually similar images",
        search_kind="similar",
    )
    data = analytics.search_quality_lists(limit=10)
    all_rows = data["weak_result"] + data["no_interaction"] + data["zero_result"]
    row = next(r for r in all_rows if r["search_event_id"] == eid)
    assert row["display_query"] == "[similar image search]"


def test_event_dict_includes_timing_fields(telemetry_dir):
    eid = _add_search(
        result_count=0,
        top_score=None,
        served=[],
        query_text="slow query",
        total_ms=12345.0,
        ask_ms=8000.0,
        reply_ms=4000.0,
        timings={"parse_query": 5000.0, "embed_visual": 2000.0},
        timing_log="s3://bucket/prefix/query-logs/x.txt",
    )
    data = analytics.search_quality_lists(limit=100)
    row = next(r for r in data["zero_result"] if r["search_event_id"] == eid)
    assert row["total_ms"] == 12345.0
    assert row["ask_ms"] == 8000.0
    assert row["reply_ms"] == 4000.0
    assert row["timings"]["parse_query"] == 5000.0
    assert row["timing_log"].endswith("query-logs/x.txt")


def test_analytics_summary_from_rollups(telemetry_dir):
    _add_search(result_count=0, top_score=None, served=[])
    weak = _add_search(result_count=1, top_score=0.1, served=["a"])
    good = _add_search(result_count=1, top_score=0.9, served=["b"])
    _add_interaction(good, "b")

    summary = analytics.analytics_summary(days=90)
    assert summary["total_searches"] == 3
    assert summary["zero_result_count"] == 1
    assert summary["weak_result_count"] == 1
    assert summary["searches_with_results"] == 2
    assert summary["no_interaction_count"] == 1  # weak still no interaction
    assert summary["interaction_count"] == 1
    assert weak  # silence unused if needed
