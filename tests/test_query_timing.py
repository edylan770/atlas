from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from imagecb.config import SETTINGS
from imagecb.query_timing import QueryTimingSession, finalize_query_timing
from imagecb.storage import blob_store


def test_format_report_includes_steps():
    session = QueryTimingSession(enabled=True, meta={"query_text": "gpus"})
    session.record("parse_query", 1.2)
    session.record("embed_visual", 0.4)
    session.record("chroma_visual", 0.05)
    session.record("ask_total", 2.0)
    report = session.format_report(
        {"search_event_id": "evt-1", "result_count": 3, "search_kind": "chat"}
    )

    assert "ImageCB query timing report" in report
    assert "STEPS (ms)" in report
    assert "parse_query" in report
    assert "ask_total" in report
    assert "search_event_id: evt-1" in report


def test_timings_ms_and_helpers():
    session = QueryTimingSession(enabled=True)
    session.record("ask_total", 1.5)
    session.record("conversational_reply", 2.0)
    session.record("request_total", 4.0)
    assert session.ask_ms() == 1500.0
    assert session.reply_ms() == 2000.0
    assert session.total_ms() == 4000.0
    assert session.timings_ms()["ask_total"] == 1500.0


def test_rrf_rank_accumulates():
    session = QueryTimingSession(enabled=True)
    session.record("rrf_rank", 0.01)
    session.record("rrf_rank", 0.02)
    assert abs(session.timings_sec()["rrf_rank"] - 0.03) < 1e-9


def test_persist_report_uploads_under_query_logs(tmp_path):
    fake_objects: dict[tuple[str, str], tuple[bytes, str]] = {}

    class FakeS3:
        def put_object(self, *, Bucket, Key, Body, ContentType=None):
            fake_objects[(Bucket, Key)] = (bytes(Body), ContentType or "")

    settings = replace(
        SETTINGS,
        blob_storage_backend="s3",
        s3_bucket="private-corpus",
        s3_prefix="atlas",
        s3_region="us-east-1",
        query_timing_log=True,
        query_timing_persist=True,
        data_dir=tmp_path,
    )
    session = QueryTimingSession(enabled=True, persist=True)
    session.record("parse_query", 0.1)

    with patch("imagecb.storage.blob_store.SETTINGS", settings), patch(
        "imagecb.query_timing.SETTINGS", settings
    ), patch("imagecb.storage.blob_store.get_s3_client", return_value=FakeS3()):
        ref = session.persist_report({"search_event_id": "abcdef12-3456-7890"})

    assert ref is not None
    assert ref.startswith("s3://private-corpus/atlas/query-logs/")
    assert ref.endswith(".txt")
    assert len(fake_objects) == 1
    ((_bucket, key), (body, content_type)) = next(iter(fake_objects.items()))
    assert "/query-logs/" in key
    assert content_type.startswith("text/plain")
    assert b"ImageCB query timing report" in body


def test_persist_report_disabled_returns_none():
    session = QueryTimingSession(enabled=False)
    assert session.persist_report({"search_event_id": "x"}) is None

    session = QueryTimingSession(enabled=True, persist=False)
    assert session.persist_report({"search_event_id": "x"}) is None


def test_query_log_key_shape():
    settings = replace(SETTINGS, s3_prefix="imagecb")
    when = datetime(2026, 7, 23, 18, 30, 0, tzinfo=timezone.utc)
    with patch("imagecb.storage.blob_store.SETTINGS", settings):
        key = blob_store.query_log_key("a1b2c3d4-eeee", when=when)
    assert key == "imagecb/query-logs/20260723_183000_a1b2c3d4-eeee.txt"


def test_finalize_records_request_total_and_logs():
    session = QueryTimingSession(enabled=True, persist=False)
    session.record("ask_total", 0.5)
    with patch.object(session, "persist_report", return_value=None) as persist:
        ref = finalize_query_timing(session, search_event_id="evt-99")
    assert ref is None
    assert "request_total" in session.timings_sec()
    persist.assert_called_once()


def test_timed_context_records_elapsed():
    session = QueryTimingSession(enabled=True)
    with session.timed("parse_query"):
        pass
    assert session.timings_sec()["parse_query"] >= 0.0


def test_session_ask_records_expected_steps():
    from imagecb.retrieval.session import ChatSession
    from imagecb.retrieval.hybrid import SearchOutcome
    from imagecb.retrieval.query_parser import QuerySpec

    timing = QueryTimingSession(enabled=True)
    spec = QuerySpec(raw_text="cyber", semantic_query="cyber", top_k=5)

    with patch("imagecb.retrieval.session.parse_query", return_value=spec), patch(
        "imagecb.retrieval.session.search", return_value=SearchOutcome(candidates=[])
    ) as mock_search, patch(
        "imagecb.retrieval.session._rank_by_fused_score", return_value=[]
    ), patch(
        "imagecb.retrieval.session._apply_min_match", return_value=([], False)
    ), patch(
        "imagecb.retrieval.sort.resolve_sort", return_value="relevance"
    ), patch(
        "imagecb.retrieval.sort.sort_ranked_results", return_value=[]
    ):
        ChatSession().ask("cyber", timing=timing)

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("timing") is timing
    steps = timing.timings_sec()
    assert "ask_total" in steps
    assert "parse_query" in steps
    assert "rrf_rank" in steps
