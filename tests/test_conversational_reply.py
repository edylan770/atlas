"""Tests for conversational assistant replies."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from imagecb.formatting.conversational_reply import (
    _build_llm_payload,
    _results_block,
    build_conversational_reply,
    iter_conversational_reply_text,
)
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import AskResult
from imagecb.storage.metadata_db import ImageRecord


def _record(
    image_id: str = "id-1",
    *,
    source_file: str = "/docs/Q3.pptx",
    source_type: str = "pptx",
    slide_index: int | None = 1,
    caption_short: str = "Chart",
    image_name: str | None = "Quarterly Revenue Chart",
    use_case: str | None = "Board decks and earnings reviews",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"data/images/{image_id}.png",
        source_file=source_file,
        source_type=source_type,
        source_modified_at=datetime(2024, 9, 15),
        source_created_at=None,
        author=None,
        slide_index=slide_index,
        page_index=None,
        slide_title=None,
        slide_notes=None,
        ocr_text=None,
        caption_short=caption_short,
        caption_detailed=None,
        objects_json=None,
        tags_json=None,
        scene=None,
        text_overlay_summary=None,
        created_at=datetime.utcnow(),
        image_name=image_name,
        use_case=use_case,
    )


def _ranked(record: ImageRecord | None = None) -> RankedResult:
    from imagecb.retrieval.rerank import _format_provenance

    rec = record or _record()
    return RankedResult(
        image_id=rec.image_id,
        score=0.87,
        record=rec,
        provenance_line=_format_provenance(rec),
    )


def _ask_result(results: list[RankedResult] | None = None) -> AskResult:
    return AskResult(
        spec=QuerySpec(semantic_query="charts", raw_text="charts"),
        results=results or [_ranked()],
    )


@patch("imagecb.formatting.conversational_reply.SETTINGS")
def test_fallback_when_llm_disabled(mock_settings):
    mock_settings.enable_conversational_llm = False
    reply = build_conversational_reply("find charts", _ask_result(), [])
    assert "found" in reply.message.lower() or "image" in reply.message.lower()
    assert len(reply.results) == 1
    assert reply.results[0].match_percent == 94


@patch("imagecb.formatting.conversational_reply.SETTINGS")
@patch("imagecb.formatting.conversational_reply.get_conversation_llm")
def test_llm_reply_used_when_enabled(mock_get_llm, mock_settings):
    mock_settings.enable_conversational_llm = True
    mock_llm = MagicMock()
    mock_llm.reply.return_value = "Here are your charts. Try refining with **only Q3**."
    mock_get_llm.return_value = mock_llm

    reply = build_conversational_reply("find charts", _ask_result(), ["note"])
    assert "Here are your charts" in reply.message
    mock_llm.reply.assert_called_once()


@patch("imagecb.formatting.conversational_reply.SETTINGS")
@patch("imagecb.formatting.conversational_reply.get_conversation_llm")
def test_llm_failure_falls_back_to_template(mock_get_llm, mock_settings):
    mock_settings.enable_conversational_llm = True
    mock_llm = MagicMock()
    mock_llm.reply.side_effect = RuntimeError("api down")
    mock_get_llm.return_value = mock_llm

    reply = build_conversational_reply("find charts", _ask_result(), [])
    assert reply.results[0].match_percent == 94
    assert len(reply.message) > 0


@patch("imagecb.formatting.conversational_reply.SETTINGS")
def test_iter_yields_template_when_llm_disabled(mock_settings):
    mock_settings.enable_conversational_llm = False
    chunks = list(iter_conversational_reply_text("find charts", _ask_result(), []))
    assert len(chunks) == 1
    assert len(chunks[0]) > 0


@patch("imagecb.formatting.conversational_reply.SETTINGS")
@patch("imagecb.formatting.conversational_reply.get_conversation_llm")
def test_iter_streams_llm_chunks(mock_get_llm, mock_settings):
    mock_settings.enable_conversational_llm = True
    mock_llm = MagicMock()
    mock_llm.reply_stream.return_value = iter(["Hello", " there"])
    mock_get_llm.return_value = mock_llm

    chunks = list(iter_conversational_reply_text("find charts", _ask_result(), []))
    assert "".join(chunks) == "Hello there"


@patch("imagecb.formatting.conversational_reply.SETTINGS")
@patch("imagecb.formatting.conversational_reply.get_conversation_llm")
def test_iter_falls_back_on_llm_failure(mock_get_llm, mock_settings):
    mock_settings.enable_conversational_llm = True
    mock_llm = MagicMock()
    mock_llm.reply_stream.side_effect = RuntimeError("api down")
    mock_get_llm.return_value = mock_llm

    chunks = list(iter_conversational_reply_text("find charts", _ask_result(), []))
    assert len(chunks) == 1
    assert len(chunks[0]) > 0


def test_results_block_uses_image_name_and_use_case():
    block = _results_block(
        [
            _ranked(
                _record(
                    source_file="/docs/AdobeStock_1025891723.jpeg",
                    source_type="image",
                    slide_index=None,
                    image_name="Healthcare Digital Innovation Concept",
                    use_case="Digital health marketing and product explainers",
                    caption_short="medical symbols with technology",
                )
            )
        ]
    )
    assert "Healthcare Digital Innovation Concept" in block
    assert "Use case: Digital health marketing and product explainers" in block
    assert "AdobeStock_1025891723.jpeg" not in block


def test_llm_payload_includes_titles_not_filenames():
    payload = _build_llm_payload(
        "digital healthcare concept",
        QuerySpec(semantic_query="digital healthcare", raw_text="digital healthcare concept"),
        [
            _ranked(
                _record(
                    source_file="/docs/AdobeStock_1025891723.jpeg",
                    source_type="image",
                    slide_index=None,
                    image_name="Healthcare Digital Innovation Concept",
                    use_case="Campaign visuals for telehealth launches",
                )
            )
        ],
        interpretation_notes=[],
        indexed_count=10,
    )
    assert "Healthcare Digital Innovation Concept" in payload
    assert "Campaign visuals for telehealth launches" in payload
    assert "AdobeStock_1025891723.jpeg" not in payload
