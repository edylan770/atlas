"""Admin analytics classification."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pytest

from imagecb.admin import analytics
from imagecb.storage.metadata_db import get_engine, session_scope
from imagecb.telemetry.models import InteractionEvent, SearchEvent
from imagecb.telemetry.schema import ensure_telemetry_schema


@pytest.fixture(autouse=True)
def _db():
    get_engine()
    ensure_telemetry_schema()
    yield


def _add_search(
    *,
    result_count: int,
    top_score: float | None,
    served: list[str],
    query_text: str = "test",
    parsed_semantic_query: str | None = None,
    search_kind: str = "chat",
) -> str:
    eid = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            SearchEvent(
                id=eid,
                query_text=query_text,
                user_id="u",
                session_id=None,
                search_kind=search_kind,
                served_image_ids_json=json.dumps(served),
                result_count=result_count,
                top_score=top_score,
                top_score_kind="rerank" if top_score is not None else None,
                parsed_semantic_query=parsed_semantic_query,
            )
        )
    return eid


def test_search_quality_categories():
    zero_id = _add_search(result_count=0, top_score=None, served=[])
    weak_id = _add_search(result_count=2, top_score=0.1, served=["a", "b"])
    served_id = _add_search(result_count=1, top_score=0.9, served=["c"])

    with session_scope() as s:
        s.add(
            InteractionEvent(
                id=str(uuid.uuid4()),
                search_event_id=served_id,
                image_id="c",
                interaction_type="view",
                user_id="u",
            )
        )

    no_ix_id = _add_search(result_count=3, top_score=0.8, served=["d", "e", "f"])

    data = analytics.search_quality_lists(limit=100, weak_score_threshold=0.25)
    zero_ids = {r["search_event_id"] for r in data["zero_result"]}
    weak_ids = {r["search_event_id"] for r in data["weak_result"]}
    no_ix_ids = {r["search_event_id"] for r in data["no_interaction"]}

    assert zero_id in zero_ids
    assert weak_id in weak_ids
    assert no_ix_id in no_ix_ids
    assert served_id not in no_ix_ids


def test_display_query_prefers_semantic_for_chat():
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


def test_display_query_similar_uses_query_text():
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


def test_event_dict_includes_timing_fields():
    eid = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            SearchEvent(
                id=eid,
                query_text="slow query",
                user_id="u",
                session_id=None,
                search_kind="chat",
                served_image_ids_json="[]",
                result_count=0,
                top_score=None,
                top_score_kind=None,
                total_ms=12345.0,
                ask_ms=8000.0,
                reply_ms=4000.0,
                timings_json=json.dumps({"parse_query": 5000.0, "embed_visual": 2000.0}),
                timing_log="s3://bucket/prefix/query-logs/x.txt",
            )
        )
    data = analytics.search_quality_lists(limit=100)
    row = next(r for r in data["zero_result"] if r["search_event_id"] == eid)
    assert row["total_ms"] == 12345.0
    assert row["ask_ms"] == 8000.0
    assert row["reply_ms"] == 4000.0
    assert row["timings"]["parse_query"] == 5000.0
    assert row["timing_log"].endswith("query-logs/x.txt")


def test_search_events_timing_columns_exist():
    from sqlalchemy import inspect

    engine = get_engine()
    ensure_telemetry_schema()
    cols = {c["name"] for c in inspect(engine).get_columns("search_events")}
    for name in ("total_ms", "ask_ms", "reply_ms", "timings_json", "timing_log"):
        assert name in cols
