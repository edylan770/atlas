"""Tests for the shared caption document builder."""

from __future__ import annotations

from datetime import datetime

from imagecb.caption.document import caption_document_text
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _record(**overrides) -> ImageRecord:
    fields = dict(
        image_id="img-1",
        content_hash="hash-1",
        image_path="data/images/img-1.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        source_modified_at=datetime(2024, 9, 15),
        slide_index=1,
        slide_title="Q3 Review",
        ocr_text="Revenue up 12%",
        image_name="Quarterly Sales Chart",
        caption_short="Bar chart of quarterly sales",
        caption_detailed="Colorful bars compare sales across regions for each quarter.",
        scene="presentation slide",
        theme="sales performance",
        use_case="quarterly business review",
        asset_type="chart",
        objects_json=serialize_list(["bar chart", "legend"]),
        tags_json=serialize_list(["chart", "sales"]),
        recommended_cases_json=serialize_list(["quarterly sales chart"]),
        search_aliases_json=serialize_list(["revenue"]),
        created_at=datetime.utcnow(),
    )
    fields.update(overrides)
    return ImageRecord(**fields)


def test_document_includes_grounded_and_interpretive_fields():
    doc = caption_document_text(_record())
    assert "asset_type: Chart" in doc
    assert "presentation slide" in doc
    assert "Revenue up 12%" in doc
    assert "Q3 Review" in doc
    assert "bar chart" in doc
    assert "Quarterly Sales Chart" in doc
    assert "Colorful bars compare sales" in doc
    assert "quarterly business review" in doc
    assert "quarterly sales chart" in doc
    assert "revenue" in doc


def test_document_empty_for_blank_record():
    rec = _record(
        slide_title=None,
        ocr_text=None,
        image_name=None,
        caption_short=None,
        caption_detailed=None,
        scene=None,
        theme=None,
        use_case=None,
        asset_type=None,
        objects_json=None,
        tags_json=None,
        recommended_cases_json=None,
        search_aliases_json=None,
    )
    assert caption_document_text(rec).strip() == ""
