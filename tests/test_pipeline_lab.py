"""Tests for Pipeline Lab comparison (imagecb/experiments/variants.py)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from imagecb.experiments.variants import run_comparison
from imagecb.retrieval.hybrid import Candidate, SearchOutcome
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import AskResult
from imagecb.storage.metadata_db import ImageRecord


def _record(image_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"data/images/{image_id}.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        source_modified_at=datetime(2024, 9, 15),
        source_created_at=None,
        author=None,
        slide_index=1,
        page_index=None,
        slide_title=None,
        slide_notes=None,
        ocr_text=None,
        caption_short="Test caption",
        caption_detailed=None,
        objects_json=None,
        tags_json=None,
        scene=None,
        text_overlay_summary=None,
        created_at=datetime.utcnow(),
    )


def _ranked(image_id: str, record: ImageRecord) -> RankedResult:
    return RankedResult(
        image_id=image_id,
        score=0.9,
        record=record,
        provenance_line="Slide 1",
        score_kind="fusion",
    )


@patch("imagecb.experiments.variants.ChatSession")
@patch("imagecb.experiments.variants.metadata_db.get_records")
@patch("imagecb.experiments.variants.search")
@patch("imagecb.experiments.variants.parse_query")
def test_run_comparison_without_keyword_filters(
    mock_parse,
    mock_search,
    mock_get_records,
    mock_session_cls,
):
    spec = QuerySpec(
        semantic_query="dashboard screenshots",
        raw_text="dashboard screenshots",
    )
    mock_parse.return_value = spec

    candidates = [
        Candidate(image_id="a", dense_score=0.9, text_score=0.7, fused_score=0.8),
        Candidate(image_id="b", dense_score=0.5, text_score=0.85, fused_score=0.6),
    ]
    mock_search.return_value = SearchOutcome(candidates=candidates)

    records = [_record("a"), _record("b")]
    mock_get_records.return_value = records

    mock_session = MagicMock()
    mock_session.ask.return_value = AskResult(
        spec=spec,
        results=[_ranked("a", records[0]), _ranked("b", records[1])],
    )
    mock_session_cls.return_value = mock_session

    out = run_comparison("dashboard screenshots", top_k=5)

    assert out["parsed_query"]["must_have_keywords"] == []
    assert out["parsed_query"]["must_avoid_keywords"] == []
    assert out["candidate_count"] == 2

    by_key = {v["key"]: v for v in out["variants"]}
    visual_text = by_key["visual_text"]
    assert visual_text["count"] == 2
    assert {r["image_id"] for r in visual_text["results"]} == {"a", "b"}

    rrf_fusion = by_key["rrf_fusion"]
    assert rrf_fusion["count"] == 2
    assert {r["image_id"] for r in rrf_fusion["results"]} == {"a", "b"}


@patch("imagecb.experiments.variants.ChatSession")
@patch("imagecb.experiments.variants.metadata_db.get_records")
@patch("imagecb.experiments.variants.search")
@patch("imagecb.experiments.variants.parse_query")
def test_no_keyword_filters_variant_researches_without_keywords(
    mock_parse,
    mock_search,
    mock_get_records,
    mock_session_cls,
):
    spec = QuerySpec(
        semantic_query="dashboard screenshots",
        raw_text="dashboard screenshots",
        must_have_keywords=["revenue"],
        must_avoid_keywords=["logo"],
    )
    mock_parse.return_value = spec

    records = [_record("a"), _record("b")]
    mock_get_records.return_value = records

    def search_side_effect(cleared_spec):
        if cleared_spec.must_have_keywords or cleared_spec.must_avoid_keywords:
            return SearchOutcome(
                candidates=[
                    Candidate(
                        image_id="a",
                        dense_score=0.9,
                        text_score=0.7,
                        fused_score=0.8,
                    ),
                ]
            )
        return SearchOutcome(
            candidates=[
                Candidate(
                    image_id="a",
                    dense_score=0.9,
                    text_score=0.7,
                    fused_score=0.8,
                ),
                Candidate(
                    image_id="b",
                    dense_score=0.5,
                    text_score=0.85,
                    fused_score=0.6,
                ),
            ]
        )

    mock_search.side_effect = search_side_effect

    mock_session = MagicMock()
    mock_session.ask.return_value = AskResult(
        spec=spec,
        results=[_ranked("a", records[0])],
    )
    mock_session_cls.return_value = mock_session

    out = run_comparison("dashboard screenshots with revenue, no logo", top_k=5)

    assert mock_search.call_count == 2
    cleared_spec = mock_search.call_args_list[1].args[0]
    assert cleared_spec.must_have_keywords == []
    assert cleared_spec.must_avoid_keywords == []

    by_key = {v["key"]: v for v in out["variants"]}
    no_kw = by_key["no_keyword_filters"]
    assert no_kw["count"] == 2
    assert {r["image_id"] for r in no_kw["results"]} == {"a", "b"}
