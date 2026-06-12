"""Tests for strict structured caption parsing (no lenient salvage)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from imagecb.caption.quality import CAPTION_FAILED
from imagecb.models.vlm import (
    CaptionJSON,
    ImageQueryJSON,
    VLMCaptioner,
    _accept_caption_payload,
)


def _valid_caption_dict() -> dict:
    return {
        "image_name": "Sales Chart",
        "grounded": {
            "objects": ["bar chart"],
            "scene": "presentation slide",
            "readable_text": "Q3 2024",
            "text_read_uncertain": False,
            "asset_type": "chart",
        },
        "interpretive": {
            "theme": "revenue growth",
            "use_case": "quarterly business review",
            "short_caption": "Bar chart of quarterly revenue by region",
            "detailed_description": "Colorful bars show revenue increasing each quarter.",
        },
        "search": {
            "tags": ["chart", "revenue", "quarterly"],
            "recommended_cases": [
                "quarterly revenue chart",
                "sales by region",
                "bar chart revenue",
            ],
            "aliases": ["sales", "Q3 results"],
        },
    }


def test_accept_caption_payload_accepts_valid_dict():
    data = _valid_caption_dict()
    assert _accept_caption_payload(data) == data


def test_accept_caption_payload_accepts_valid_json_string():
    data = _valid_caption_dict()
    raw = json.dumps(data)
    assert _accept_caption_payload(raw) == data


def test_accept_caption_payload_rejects_markdown_wrapped_json():
    raw = '```json\n' + json.dumps(_valid_caption_dict()) + "\n```"
    assert _accept_caption_payload(raw) is None


def test_accept_caption_payload_rejects_incomplete_dict():
    data = _valid_caption_dict()
    del data["grounded"]["objects"]
    assert _accept_caption_payload(data) is None


def test_accept_caption_payload_rejects_too_few_tags():
    data = _valid_caption_dict()
    data["search"]["tags"] = ["chart"]
    assert _accept_caption_payload(data) is None


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(128, 128, 128))


@patch("imagecb.models.bedrock_client.bedrock_converse")
def test_bedrock_tool_use_valid_caption(mock_converse):
    payload = _valid_caption_dict()
    mock_converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {"name": "emit_caption", "input": payload}},
                ],
            },
        },
    }
    cap = VLMCaptioner(provider="bedrock", model="test-model").caption_image(_tiny_image())
    assert cap.short_caption == payload["interpretive"]["short_caption"]
    assert cap.caption_quality != "failed"


@patch("imagecb.models.bedrock_client.bedrock_converse")
def test_bedrock_text_only_response_fails(mock_converse):
    mock_converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": json.dumps(_valid_caption_dict())},
                ],
            },
        },
    }
    cap = VLMCaptioner(provider="bedrock", model="test-model").caption_image(_tiny_image())
    assert cap.short_caption == CAPTION_FAILED


@patch("openai.OpenAI")
def test_openai_strict_json_string_ok(mock_openai_cls):
    payload = _valid_caption_dict()
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
    )
    cap = VLMCaptioner(provider="openai", model="gpt-test").caption_image(_tiny_image())
    assert cap.image_name == "Sales Chart"
    assert cap.short_caption != CAPTION_FAILED


@patch("anthropic.Anthropic")
def test_anthropic_tool_use_valid(mock_anthropic_cls):
    payload = _valid_caption_dict()
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="emit_caption", input=payload),
        ],
    )
    cap = VLMCaptioner(provider="anthropic", model="claude-test").caption_image(_tiny_image())
    assert cap.tags == ["chart", "revenue", "quarterly"]


@patch("anthropic.Anthropic")
def test_anthropic_text_only_response_fails(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text=json.dumps(_valid_caption_dict())),
        ],
    )
    cap = VLMCaptioner(provider="anthropic", model="claude-test").caption_image(_tiny_image())
    assert cap.short_caption == CAPTION_FAILED


def test_image_query_json_failed_not_usable():
    facets = ImageQueryJSON.failed("timeout")
    assert facets.query_quality == "failed"
    assert facets.error_message == "timeout"
    assert facets.search_query == ""
    assert not facets.is_usable()


def test_image_query_json_from_dict_empty_not_usable():
    facets = ImageQueryJSON.from_dict({})
    assert facets.query_quality == "ok"
    assert not facets.is_usable()


def test_image_query_json_usable_with_search_query():
    facets = ImageQueryJSON(search_query="hero banner")
    assert facets.is_usable()


@patch.object(VLMCaptioner, "_query_bedrock", side_effect=RuntimeError("timeout"))
def test_query_image_exception_returns_failed(_mock_query):
    facets = VLMCaptioner(provider="bedrock", model="test-model").query_image(_tiny_image())
    assert facets.query_quality == "failed"
    assert "timeout" in facets.error_message
    assert not facets.is_usable()


@patch.object(VLMCaptioner, "_query_bedrock", return_value="{}")
def test_query_image_empty_response_returns_failed(_mock_query):
    facets = VLMCaptioner(provider="bedrock", model="test-model").query_image(_tiny_image())
    assert facets.query_quality == "failed"
    assert not facets.is_usable()
