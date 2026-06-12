"""Tests for short-query asset type inference and rerank boost."""

from __future__ import annotations

from unittest.mock import patch

from imagecb.retrieval.asset_type_boost import (
    apply_asset_type_boost,
    asset_type_rerank_multiplier,
    infer_asset_types_from_short_query,
)
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import ImageRecord


def _ranked(image_id: str, score: float, asset_type: str) -> RankedResult:
    record = ImageRecord(
        image_id=image_id,
        content_hash=f"h-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="/tmp/test.pptx",
        source_type="pptx",
        asset_type=asset_type,
    )
    return RankedResult(
        image_id=image_id,
        score=score,
        record=record,
        provenance_line="",
    )


@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_infer_asset_types_from_diagram(_mock_rate):
    spec = QuerySpec(semantic_query="diagram", raw_text="diagram")
    assert infer_asset_types_from_short_query(spec) == {"diagram"}


@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_infer_asset_types_skips_presentation(_mock_rate):
    spec = QuerySpec(semantic_query="presentation", raw_text="presentation")
    assert infer_asset_types_from_short_query(spec) == set()


@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_asset_type_multiplier_for_matching_record(_mock_rate):
    spec = QuerySpec(semantic_query="logo", raw_text="logo")
    assert asset_type_rerank_multiplier(spec, "logo") > 1.0
    assert asset_type_rerank_multiplier(spec, "photo") == 1.0


@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_apply_asset_type_boost_reorders(_mock_rate):
    spec = QuerySpec(semantic_query="screenshot", raw_text="screenshot")
    ranked = [
        _ranked("photo", 0.92, "photo"),
        _ranked("shot", 0.88, "screenshot"),
    ]
    boosted = apply_asset_type_boost(spec, ranked)
    assert boosted[0].image_id == "shot"
