"""Tests for POST /api/chat/stream SSE endpoint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.session import AskResult
from imagecb.suggestions.corpus_summary import CorpusContext


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
@pytest.mark.xfail(
    reason="expects SSE status/stage events - backend half of the progress "
    "indicator feature is not implemented yet",
    strict=False,
)
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

    with patch(
        "imagecb.api.routes.build_corpus_context",
        return_value=CorpusContext(indexed_count=17, fingerprint="sqlite"),
    ):
        res = client.post(
            "/api/chat/stream",
            json={"message": "find charts", "top_k": 5, "min_match_percent": 0},
        )

    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    events = _parse_sse_events(res.text)
    types = [e["type"] for e in events]
    # status (stage) events now precede metadata: an immediate ack plus live
    # pipeline stages replace the old silent wait.
    assert types[0] == "status"
    first_meta = types.index("metadata")
    assert all(t == "status" for t in types[:first_meta])
    assert "token" in types
    assert types[-1] == "done"
    assert events[-1]["assistant_message"] == "Hello world"
    assert events[-1]["follow_up_suggestions"] == ["only charts", "finance dashboards"]
    mock_session.record_turn.assert_called_once_with("find charts", "Hello world")
    assert events[0]["session_id"] == "sess-1"
    metadata = events[types.index("metadata")]
    assert metadata["search_event_id"] == "evt-1"
    assert mock_iter.call_args.kwargs["indexed_count"] == 17


@patch("imagecb.api.routes.get_or_create_session")
def test_chat_stream_requires_message(mock_session_factory, client):
    res = client.post("/api/chat/stream", json={"message": "  "})
    assert res.status_code == 400
