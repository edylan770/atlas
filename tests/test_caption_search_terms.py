"""Tests for caption search-term cleanup."""

from __future__ import annotations

from imagecb.caption.pipeline import enrich_caption_search_terms
from imagecb.models.vlm import CaptionJSON, GroundedCaption, SearchTerms


def _caption(*, cases: list[str], aliases: list[str], asset_type: str = "diagram") -> CaptionJSON:
    return CaptionJSON(
        grounded=GroundedCaption(asset_type=asset_type),
        search=SearchTerms(
            tags=["cloud", "diagram"],
            recommended_cases=list(cases),
            aliases=list(aliases),
        ),
    )


def test_strips_bare_asset_type_from_recommended_cases():
    cap = _caption(
        cases=["diagram", "cloud systems diagram"],
        aliases=["schematic"],
    )
    out = enrich_caption_search_terms(cap)
    assert "diagram" not in out.search.recommended_cases
    assert "cloud systems diagram" in out.search.recommended_cases


def test_dedupes_and_lowercases_cases_and_aliases():
    cap = _caption(
        cases=["Cloud Systems Diagram", "cloud systems diagram", "network topology"],
        aliases=["Schematic", "schematic", "flowchart"],
    )
    out = enrich_caption_search_terms(cap)
    assert out.search.recommended_cases == ["cloud systems diagram", "network topology"]
    assert out.search.aliases == ["schematic", "flowchart"]


def test_caps_alias_and_case_counts():
    cap = _caption(
        cases=[f"specific case {i}" for i in range(10)],
        aliases=[f"alias {i}" for i in range(12)],
    )
    out = enrich_caption_search_terms(cap)
    assert len(out.search.recommended_cases) <= 5
    assert len(out.search.aliases) <= 8
