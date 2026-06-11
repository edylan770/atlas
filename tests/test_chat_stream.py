"""Tests for POST /api/chat/stream SSE endpoint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.session import AskResult


def _parse_sse_events(raw: str) -> list[dict]:
    events = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def client():
    return TestClient(create_app())


def _ask_result() -> AskResult:
    return AskResult(
        spec=QuerySpec(semantic_query="test", raw_text="test"),
        results=[],
    )


@patch("imagecb.api.routes.generate_follow_up_suggestions", return_value=["only charts", "finance dashboards"])
@patch("imagecb.api.routes.record_search_from_results", return_value="evt-1")
@patch("imagecb.api.routes.get_or_create_session")
@patch("imagecb.api.routes.iter_conversational_reply_text")
def test_chat_stream_emits_metadata_tokens_done(
    mock_iter,
    mock_session_factory,
    _mock_record,
    _mock_follow_up,
    client,
):
    mock_session = MagicMock()
    mock_session.ask.return_value = _ask_result()
    mock_session_factory.return_value = ("sess-1", mock_session)
    mock_iter.return_value = iter(["Hello", " world"])

    with patch("imagecb.api.routes.vector_store") as mock_vs:
        mock_vs.count.return_value = 0
        res = client.post(
            "/api/chat/stream",
            json={"message": "find charts", "top_k": 5, "min_match_percent": 0},
        )

    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    events = _parse_sse_events(res.text)
    types = [e["type"] for e in events]
    assert types[0] == "metadata"
    assert "token" in types
    assert types[-1] == "done"
    assert events[-1]["assistant_message"] == "Hello world"
    assert events[-1]["follow_up_suggestions"] == ["only charts", "finance dashboards"]
    mock_session.record_turn.assert_called_once_with("find charts", "Hello world")
    assert events[0]["session_id"] == "sess-1"
    assert events[0]["search_event_id"] == "evt-1"


@patch("imagecb.api.routes.get_or_create_session")
def test_chat_stream_requires_message(mock_session_factory, client):
    res = client.post("/api/chat/stream", json={"message": "  "})
    assert res.status_code == 400
