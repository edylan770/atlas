"""Tests for ingest embedding context text."""

from __future__ import annotations

from imagecb.extractors.types import Provenance
from imagecb.ingest_context import (
    embed_context_from_caption_and_provenance,
    embed_context_from_provenance,
    embed_context_from_record,
    embed_context_text,
)
from imagecb.models.vlm import CaptionJSON, GroundedCaption, InterpretiveCaption
from imagecb.storage.metadata_db import ImageRecord


def test_embed_context_prioritizes_interpretive_fields():
    text = embed_context_text(
        slide_title="Q3 Revenue",
        slide_notes="Year over year growth",
        theme="sales performance",
        short_caption="Quarterly revenue bar chart",
    )
    assert "sales performance" in text
    assert "Quarterly revenue" in text


def test_embed_context_joins_slide_fields_when_no_interpretive():
    text = embed_context_text(
        slide_title="Q3 Revenue",
        slide_notes="Year over year growth",
    )
    assert "Q3 Revenue" in text
    assert "Year over year" in text


def test_embed_context_truncates_long_text():
    long_notes = "x" * 600
    text = embed_context_text(slide_notes=long_notes)
    assert len(text) <= 480


def test_embed_context_includes_use_case_and_tags():
    text = embed_context_text(
        short_caption="Team meeting",
        use_case="internal communications",
        tags=["presentation", "office", "collaboration"],
    )
    assert "internal communications" in text
    assert "presentation" in text


def test_embed_context_empty():
    assert embed_context_text() == ""


def test_embed_context_from_provenance():
    prov = Provenance(
        source_file="/docs/deck.pptx",
        source_type="pptx",
        slide_title="Architecture",
        slide_index=3,
    )
    assert "Architecture" in embed_context_from_provenance(prov)


def test_embed_context_from_record():
    record = ImageRecord(
        image_id="id-1",
        content_hash="h1",
        image_path="data/images/id-1.png",
        source_file="/docs/report.pdf",
        source_type="pdf",
        page_index=2,
        slide_title="Summary",
        slide_index=None,
        slide_notes=None,
        source_modified_at=None,
        source_created_at=None,
        author=None,
        ocr_text=None,
        caption_short=None,
        caption_detailed=None,
        objects_json=None,
        tags_json=None,
        scene=None,
        text_overlay_summary=None,
    )
    assert "Summary" in embed_context_from_record(record)


def test_embed_context_from_caption_and_provenance():
    caption = CaptionJSON(
        interpretive=InterpretiveCaption(
            theme="cybersecurity",
            short_caption="Network security operations center",
        ),
        grounded=GroundedCaption(),
    )
    prov = Provenance(
        source_file="/docs/deck.pptx",
        source_type="pptx",
        slide_title="Security",
        slide_notes="Long speaker notes " * 20,
    )
    text = embed_context_from_caption_and_provenance(caption, prov)
    assert "cybersecurity" in text
    assert "Network security" in text


def test_embed_context_from_record_with_theme():
    record = ImageRecord(
        image_id="id-2",
        content_hash="h2",
        image_path="data/images/id-2.png",
        source_file="/docs/deck.pptx",
        source_type="pptx",
        theme="growth strategy",
        caption_short="Roadmap timeline infographic",
    )
    text = embed_context_from_record(record)
    assert "growth strategy" in text
    assert "Roadmap timeline" in text
