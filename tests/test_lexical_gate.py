"""Tests for the deterministic high-confidence lexical match check."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from imagecb.retrieval.hybrid import Candidate
from imagecb.retrieval.lexical import has_high_confidence_lexical_hit
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _record(image_id: str, *, caption: str = "", tags=None, name: str = "") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"data/images/{image_id}.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        slide_index=1,
        caption_short=caption,
        image_name=name,
        tags_json=serialize_list(tags or []),
        created_at=datetime.utcnow(),
    )


def _spec(query: str) -> QuerySpec:
    return QuerySpec(semantic_query=query, raw_text=query)


def test_literal_token_match_is_high_confidence():
    records = [_record("a", caption="A deep hole in the ground")]
    candidates = [Candidate(image_id="a", sparse_score=5.0)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("hole"), candidates) is True


def test_no_literal_match_is_not_confident():
    records = [_record("a", caption="Developer coding with augmented reality")]
    candidates = [Candidate(image_id="a", sparse_score=5.0)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("uninterested"), candidates) is False


def test_requires_bm25_candidate():
    # Even if the text would match, a candidate with no sparse signal is ignored.
    records = [_record("a", caption="A deep hole")]
    candidates = [Candidate(image_id="a", sparse_score=0.0, dense_score=0.9)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("hole"), candidates) is False


def test_multi_token_partial_coverage_below_threshold():
    # Only one of two content tokens present -> 50% coverage < 90% threshold.
    records = [_record("a", caption="A team at work", tags=["team"])]
    candidates = [Candidate(image_id="a", sparse_score=2.0)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("team meeting"), candidates) is False


def test_multi_token_full_coverage():
    records = [_record("a", caption="Team meeting in the office", tags=["team", "meeting"])]
    candidates = [Candidate(image_id="a", sparse_score=4.0)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("team meeting"), candidates) is True


def test_stopword_only_query_no_hit():
    records = [_record("a", caption="The of and")]
    candidates = [Candidate(image_id="a", sparse_score=1.0)]
    with patch("imagecb.retrieval.lexical.metadata_db.get_records", return_value=records):
        assert has_high_confidence_lexical_hit(_spec("the of"), candidates) is False
