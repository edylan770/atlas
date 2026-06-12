"""Tests for caption quality assessment."""

from __future__ import annotations

from imagecb.caption.quality import (
    CAPTION_FAILED,
    assess_caption,
    assess_caption_with_reasons,
    caption_json_from_record,
    needs_regeneration,
)
from imagecb.models.vlm import (
    CaptionJSON,
    GroundedCaption,
    InterpretiveCaption,
    SearchTerms,
)
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _caption(**kwargs) -> CaptionJSON:
    grounded = kwargs.pop("grounded", GroundedCaption())
    interpretive = kwargs.pop("interpretive", InterpretiveCaption())
    search = kwargs.pop("search", SearchTerms())
    return CaptionJSON(
        image_name=kwargs.get("image_name", "Q3 Sales Chart"),
        grounded=grounded,
        interpretive=interpretive,
        search=search,
    )


def test_assess_ok_caption():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="Bar chart comparing quarterly revenue across regions",
            detailed_description="A colorful bar chart with four quarters labeled.",
        ),
        grounded=GroundedCaption(
            objects=["bar chart"],
            scene="business presentation slide",
        ),
        search=SearchTerms(
            tags=["revenue", "chart", "sales"],
            recommended_cases=[
                "quarterly revenue chart",
                "regional sales comparison",
                "bar chart revenue by region",
            ],
            aliases=["sales", "revenue growth", "q3 revenue"],
        ),
    )
    assert assess_caption(cap) == "ok"


def test_assess_failed_caption():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="[caption failed]",
            detailed_description="VLM error: timeout",
        ),
    )
    assert assess_caption(cap) == "failed"


def test_assess_weak_short_caption():
    cap = _caption(
        interpretive=InterpretiveCaption(short_caption="A photo"),
        grounded=GroundedCaption(),
    )
    assert assess_caption(cap) == "weak"


def test_assess_weak_generic_pattern():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="An image showing a business scene with people",
            detailed_description="People in an office setting during a meeting.",
        ),
        grounded=GroundedCaption(objects=["people"], scene="office"),
    )
    assert assess_caption(cap) == "weak"


def test_assess_weak_thin_search_fields():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="Bar chart comparing quarterly revenue across regions",
            detailed_description="A colorful bar chart with four quarters labeled.",
        ),
        grounded=GroundedCaption(objects=["bar chart"], scene="business slide"),
        search=SearchTerms(tags=["revenue"], recommended_cases=["revenue chart"], aliases=[]),
    )
    assert assess_caption(cap) == "weak"


def test_assess_weak_missing_detailed_description():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="Bar chart comparing quarterly revenue across regions",
            detailed_description="",
        ),
        grounded=GroundedCaption(objects=["bar chart"], scene="slide"),
        search=SearchTerms(
            tags=["revenue", "chart", "sales"],
            recommended_cases=["a", "b", "c"],
            aliases=["x", "y"],
        ),
    )
    quality, reasons = assess_caption_with_reasons(cap)
    assert quality == "weak"
    assert "missing_detailed_description" in reasons


def test_assess_weak_insufficient_tags():
    cap = _caption(
        interpretive=InterpretiveCaption(
            short_caption="Bar chart comparing quarterly revenue across regions",
            detailed_description="Detailed chart description here.",
        ),
        grounded=GroundedCaption(objects=["bar chart"], scene="slide"),
        search=SearchTerms(
            tags=["revenue"],
            recommended_cases=["a", "b", "c"],
            aliases=["x", "y"],
        ),
    )
    quality, reasons = assess_caption_with_reasons(cap)
    assert quality == "weak"
    assert "insufficient_tags" in reasons


def test_caption_json_from_record_round_trip():
    record = ImageRecord(
        image_id="id1",
        content_hash="hash",
        image_path="/tmp/x.png",
        source_file="deck.pptx",
        source_type="pptx",
        image_name="Sales Chart",
        caption_short="Bar chart of quarterly sales",
        caption_detailed="Colorful bars show sales by region.",
        use_case="QBR",
        theme="sales",
        scene="slide",
        objects_json=serialize_list(["bar chart"]),
        tags_json=serialize_list(["sales", "chart", "quarterly"]),
        recommended_cases_json=serialize_list(["sales chart", "quarterly chart", "revenue slide"]),
        search_aliases_json=serialize_list(["revenue", "Q3"]),
        caption_quality="ok",
    )
    cap = caption_json_from_record(record)
    assert cap.image_name == "Sales Chart"
    assert cap.tags == ["sales", "chart", "quarterly"]
    assert assess_caption(cap) == "ok"


def test_needs_regeneration():
    assert needs_regeneration("weak") is True
    assert needs_regeneration("failed") is True
    assert needs_regeneration("ok") is False
    assert needs_regeneration(None) is False


def test_failed_marker_constant():
    assert CAPTION_FAILED == "[caption failed]"
