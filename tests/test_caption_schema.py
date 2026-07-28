"""Tests for caption JSON schema and CaptionJSON parsing."""

from __future__ import annotations

from imagecb.caption.schema import validate_caption_dict
from imagecb.models.vlm import CaptionJSON


def _valid_nested() -> dict:
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


def test_validate_caption_dict_accepts_nested():
    assert validate_caption_dict(_valid_nested()) is True


def test_validate_caption_dict_rejects_incomplete():
    data = _valid_nested()
    del data["grounded"]["objects"]
    assert validate_caption_dict(data) is False


def test_validate_caption_dict_rejects_invalid_asset_type():
    data = _valid_nested()
    data["grounded"]["asset_type"] = "meme"
    assert validate_caption_dict(data) is False


def test_validate_caption_dict_rejects_missing_asset_type():
    data = _valid_nested()
    del data["grounded"]["asset_type"]
    assert validate_caption_dict(data) is False


def test_validate_caption_dict_rejects_too_few_tags():
    data = _valid_nested()
    data["search"]["tags"] = ["chart"]
    assert validate_caption_dict(data) is False


def test_caption_json_from_nested_dict():
    cap = CaptionJSON.from_dict(_valid_nested())
    assert cap.image_name == "Sales Chart"
    assert cap.objects == ["bar chart"]
    assert cap.theme == "revenue growth"
    assert cap.tags == ["chart", "revenue", "quarterly"]
    assert len(cap.recommended_cases) == 3
    assert cap.aliases == ["sales", "Q3 results"]
    assert cap.text_read_uncertain is False
    assert cap.asset_type == "chart"


def test_caption_json_from_legacy_flat_dict():
    cap = CaptionJSON.from_dict(
        {
            "image_name": "Logo",
            "short_caption": "Company logo on white background",
            "detailed_description": "A blue wordmark centered on white.",
            "use_case": "brand identity slide",
            "objects": ["logo"],
            "scene": "plain background",
            "text_overlay_summary": "ACME",
            "tags": ["logo", "brand"],
            "recommended_cases": ["company logo", "brand mark"],
        }
    )
    assert cap.short_caption == "Company logo on white background"
    assert cap.readable_text == "ACME"
    assert cap.tags == ["logo", "brand"]


def test_caption_prompt_demands_verbatim_transcription():
    """Batch A drift guard: the verbatim rules must survive future prompt edits."""
    from imagecb.caption.examples import format_few_shot_for_prompt
    from imagecb.caption.schema import CAPTION_JSON_SCHEMA
    from imagecb.models.vlm import CAPTION_USER_PROMPT_TEMPLATE

    prompt = CAPTION_USER_PROMPT_TEMPLATE
    assert "verbatim transcription of ALL clearly legible text" in prompt
    assert "natural reading order" in prompt
    assert "too much text to transcribe completely" in prompt

    schema_desc = CAPTION_JSON_SCHEMA["properties"]["grounded"]["properties"][
        "readable_text"
    ]["description"]
    assert "Verbatim transcription" in schema_desc
    assert "Never invent or guess text" in schema_desc

    examples = format_few_shot_for_prompt()
    # the text-dense example teaches exhaustive transcription
    assert "Support Plans" in examples
    assert "Enterprise | $1,999" in examples
    # the chart example transcribes its own declared axis labels/legend
    assert "North America" in examples
