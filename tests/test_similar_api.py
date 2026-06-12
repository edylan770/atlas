"""API tests for similar search endpoint."""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from imagecb.api.server import create_app
from imagecb.models.vlm import ImageQueryJSON
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.similar import SimilarSearchOutcome


@pytest.fixture
def client():
    return TestClient(create_app())


def _fake_outcome() -> SimilarSearchOutcome:
    return SimilarSearchOutcome(
        results=[],
        facets=ImageQueryJSON(search_query="hero banner"),
        spec=QuerySpec(raw_text="[Find similar] test", top_k=10),
    )


def test_similar_json_image_id(client):
    with mock.patch("imagecb.api.routes.search_similar", return_value=_fake_outcome()) as mock_search:
        res = client.post(
            "/api/similar",
            json={
                "image_id": "abc-123",
                "top_k": 10,
                "min_match_percent": 0,
                "similarity_axis": "balanced",
            },
        )
    assert res.status_code == 200
    mock_search.assert_called_once()
    kwargs = mock_search.call_args.kwargs
    assert kwargs["image_id"] == "abc-123"
    assert kwargs["exclude_image_id"] == "abc-123"


def test_similar_multipart_image_id(client):
    with mock.patch("imagecb.api.routes.search_similar", return_value=_fake_outcome()) as mock_search:
        res = client.post(
            "/api/similar",
            data={
                "image_id": "abc-123",
                "top_k": "10",
                "min_match_percent": "0",
                "similarity_axis": "balanced",
            },
        )
    assert res.status_code == 200
    mock_search.assert_called_once()
    kwargs = mock_search.call_args.kwargs
    assert kwargs["image_id"] == "abc-123"
    assert kwargs["exclude_image_id"] == "abc-123"


def test_similar_rejects_missing_input(client):
    res = client.post("/api/similar", json={})
    assert res.status_code == 400
    assert "image_id or image file is required" in res.json()["detail"]


def test_similar_does_not_restrict_to_session_pool(client):
    from imagecb.retrieval.session import ChatSession

    session = ChatSession()
    session.last_candidate_ids = ["img-1", "img-2"]
    session.last_spec = QuerySpec(raw_text="prior", is_refinement=True, top_k=10)

    with mock.patch("imagecb.api.routes.get_session", return_value=session):
        with mock.patch(
            "imagecb.api.routes.search_similar",
            return_value=_fake_outcome(),
        ) as mock_search:
            res = client.post(
                "/api/similar",
                json={
                    "image_id": "abc-123",
                    "session_id": "sess-1",
                    "top_k": 10,
                    "min_match_percent": 0,
                    "similarity_axis": "balanced",
                },
            )

    assert res.status_code == 200
    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("restrict_to") is None
