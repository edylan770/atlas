"""Tests for caption context builder."""

from __future__ import annotations

from imagecb.caption.context import (
    caption_context_from_provenance,
    caption_context_from_record,
    slide_body_from_provenance,
)
from imagecb.extractors.types import Provenance
from imagecb.storage.metadata_db import ImageRecord


def test_slide_body_from_pptx_provenance():
    prov = Provenance(
        source_file="/docs/deck.pptx",
        source_type="pptx",
        extra={"slide_body_text": "Bullet one\nBullet two"},
    )
    assert slide_body_from_provenance(prov) == "Bullet one\nBullet two"


def test_slide_body_from_pdf_provenance():
    prov = Provenance(
        source_file="/docs/report.pdf",
        source_type="pdf",
        extra={"nearby_text": "Page body text"},
    )
    assert slide_body_from_provenance(prov) == "Page body text"


def test_caption_context_from_provenance_includes_fields():
    prov = Provenance(
        source_file="/docs/deck.pptx",
        source_type="pptx",
        author="Jane Doe",
        slide_index=3,
        slide_title="Revenue Overview",
        slide_notes="Speaker notes here",
        extra={"slide_body_text": "Chart shows Q3 sales"},
    )
    ctx = caption_context_from_provenance(prov)
    assert "disambiguate" in ctx
    assert "Revenue Overview" in ctx
    assert "Speaker notes" in ctx
    assert "Chart shows Q3 sales" in ctx
    assert "deck.pptx" in ctx
    assert "Jane Doe" in ctx


def test_caption_context_from_record():
    record = ImageRecord(
        image_id="id-1",
        content_hash="h1",
        image_path="data/images/id-1.png",
        source_file="/docs/deck.pptx",
        source_type="pptx",
        slide_index=2,
        slide_title="KPIs",
        slide_body_text="Metrics dashboard",
    )
    ctx = caption_context_from_record(record)
    assert "KPIs" in ctx
    assert "Metrics dashboard" in ctx
