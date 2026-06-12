"""Tests for rerank candidate text and asset-type boost."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from imagecb.retrieval.hybrid import Candidate
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import _candidate_text, rerank
from imagecb.storage.metadata_db import ImageRecord, serialize_list


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
        caption_detailed="Long detailed caption describing the cloud diagram in depth",
        objects_json=serialize_list(["server", "database"]),
        tags_json=serialize_list(["diagram", "architecture", "cloud", "network", "api", "data", "flow", "system"]),
        scene="cloud architecture",
        text_overlay_summary=None,
        theme="system design",
        use_case="technical documentation",
        asset_type="diagram",
        recommended_cases_json=serialize_list(
            [
                "cloud architecture diagram",
                "system design flow",
                "network topology",
                "bare diagram",
            ]
        ),
        search_aliases_json=serialize_list(["schematic", "flowchart"]),
        created_at=datetime.utcnow(),
    )


def test_candidate_text_includes_asset_type_and_caps_search_fields():
    text = _candidate_text(_record("x"))
    assert "asset_type: Diagram" in text
    assert "cloud architecture" in text
    assert "Long detailed caption" in text
    assert "technical documentation" in text
    assert "aliases: schematic, flowchart" in text
    assert text.count("recommended:") == 1
    assert "bare diagram" not in text
    assert "network topology" in text


def test_candidate_text_caps_ocr_length():
    record = _record("x")
    record.ocr_text = "word " * 500
    text = _candidate_text(record)
    start = text.index("word")
    ocr_block = text[start:].splitlines()[0]
    assert len(ocr_block) <= 600


@patch("imagecb.retrieval.rerank.get_reranker")
@patch("imagecb.retrieval.rerank.metadata_db.get_records")
@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_rerank_applies_asset_type_boost_for_diagram(mock_rate, mock_get_records, mock_get_reranker):
    diagram = _record("diagram-img")
    photo = _record("photo-img")
    photo.asset_type = "photo"
    photo.caption_short = "Office team photo"
    mock_get_records.return_value = [diagram, photo]
    mock_get_reranker.return_value.score.return_value = [0.80, 0.85]

    spec = QuerySpec(semantic_query="diagram", raw_text="diagram")
    candidates = [
        Candidate(image_id="diagram-img", fused_score=0.9),
        Candidate(image_id="photo-img", fused_score=0.8),
    ]

    results = rerank("diagram", candidates, top_k=2, min_match_percent=0, spec=spec)

    assert results[0].image_id == "diagram-img"


@patch("imagecb.retrieval.rerank.get_reranker")
@patch("imagecb.retrieval.rerank.metadata_db.get_records")
@patch("imagecb.retrieval.asset_type_boost._corpus_asset_type_unclassified_rate", return_value=0.0)
def test_rerank_skips_asset_type_boost_for_presentation(mock_rate, mock_get_records, mock_get_reranker):
    a = _record("a")
    b = _record("b")
    b.image_id = "b"
    mock_get_records.return_value = [a, b]
    mock_get_reranker.return_value.score.return_value = [0.70, 0.90]

    spec = QuerySpec(semantic_query="presentation", raw_text="presentation")
    candidates = [
        Candidate(image_id="a", fused_score=0.9),
        Candidate(image_id="b", fused_score=0.8),
    ]

    results = rerank("presentation", candidates, top_k=2, min_match_percent=0, spec=spec)

    assert results[0].image_id == "b"
