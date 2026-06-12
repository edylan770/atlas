"""Tests for few-shot caption examples and prompt wiring."""

from __future__ import annotations

from imagecb.caption.examples import CAPTION_FEW_SHOT_EXAMPLES, format_few_shot_for_prompt
from imagecb.caption.normalize import normalize_tag
from imagecb.caption.schema import validate_caption_dict
from imagecb.models.vlm import _build_caption_user_prompt


def test_each_few_shot_example_validates():
    for _label, data in CAPTION_FEW_SHOT_EXAMPLES:
        assert validate_caption_dict(data) is True


def test_format_few_shot_for_prompt_contains_all_labels():
    block = format_few_shot_for_prompt()
    assert block
    assert "Example 1" in block
    assert "Example 2" in block
    assert "Example 3" in block
    assert "Standalone photo" in block
    assert "Standalone illustration" in block
    assert "Presentation slide" in block


def test_build_caption_user_prompt_includes_examples():
    prompt = _build_caption_user_prompt(context=None, source_file=None)
    assert "Match the granularity of these examples" in prompt
    assert "Example 1" in prompt
    assert "quarterly sales chart" in prompt.lower()
    assert "Asset type taxonomy" in prompt
    assert '"asset_type":"photo"' in prompt.replace(" ", "")


def test_few_shot_tags_are_lowercase_singular():
    for _label, data in CAPTION_FEW_SHOT_EXAMPLES:
        for tag in data["search"]["tags"]:
            assert tag == tag.lower()
            assert tag == normalize_tag(tag)
            assert " " not in tag
