"""Tests for ranked-result deduplication."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from imagecb.retrieval.dedupe import cosine_similarity, dedupe_results
from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import ImageRecord


def _record(image_id: str, *, content_hash: str | None = None) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=content_hash if content_hash is not None else f"hash-{image_id}",
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


def _ranked(image_id: str, score: float, *, content_hash: str | None = None) -> RankedResult:
    rec = _record(image_id, content_hash=content_hash)
    return RankedResult(
        image_id=image_id,
        score=score,
        record=rec,
        provenance_line=image_id,
    )


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


@patch("imagecb.retrieval.dedupe.vector_store.get_embeddings", return_value={})
def test_dedupe_same_content_hash_keeps_highest_score(_mock_embed):
    results = [
        _ranked("a", 0.95, content_hash="same-hash"),
        _ranked("b", 0.80, content_hash="same-hash"),
    ]

    kept = dedupe_results(results, top_k=5)

    assert [r.image_id for r in kept] == ["a"]


@patch("imagecb.retrieval.dedupe.vector_store.get_embeddings")
def test_dedupe_near_duplicate_embeddings(mock_get_embed):
    base = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    near = np.array([0.99, 0.14, 0.0], dtype=np.float64)
    near = near / np.linalg.norm(near)
    assert cosine_similarity(base, near) >= 0.98

    results = [_ranked("a", 0.95), _ranked("b", 0.90)]
    mock_get_embed.return_value = {"a": base, "b": near}

    kept = dedupe_results(results, top_k=5)

    assert [r.image_id for r in kept] == ["a"]


@patch("imagecb.retrieval.dedupe.vector_store.get_embeddings")
def test_dedupe_different_embeddings_both_kept(mock_get_embed):
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == 0.0

    results = [_ranked("a", 0.95), _ranked("b", 0.90)]
    mock_get_embed.return_value = {"a": a, "b": b}

    kept = dedupe_results(results, top_k=5)

    assert [r.image_id for r in kept] == ["a", "b"]


@patch("imagecb.retrieval.dedupe.vector_store.get_embeddings")
def test_dedupe_missing_embedding_keeps_both(mock_get_embed):
    results = [_ranked("a", 0.95), _ranked("b", 0.90)]
    mock_get_embed.return_value = {"a": np.array([1.0, 0.0, 0.0])}

    kept = dedupe_results(results, top_k=5)

    assert [r.image_id for r in kept] == ["a", "b"]


@patch("imagecb.retrieval.dedupe.vector_store.get_embeddings", return_value={})
def test_dedupe_backfills_from_pool(_mock_embed):
    results = [
        _ranked("a", 0.95, content_hash="hash-a"),
        _ranked("b", 0.90, content_hash="hash-a"),
        _ranked("c", 0.85, content_hash="hash-c"),
    ]
    pool = [_ranked("d", 0.80, content_hash="hash-d")]

    kept = dedupe_results(results, top_k=3, pool=pool)

    assert [r.image_id for r in kept] == ["a", "c", "d"]


@patch("imagecb.retrieval.dedupe.dedupe_results")
@patch("imagecb.retrieval.rerank.get_reranker")
@patch("imagecb.retrieval.rerank.metadata_db.get_records")
def test_rerank_calls_dedupe_results(mock_get_records, mock_get_reranker, mock_dedupe):
    from imagecb.retrieval.hybrid import Candidate
    from imagecb.retrieval.rerank import rerank

    records = [_record("a"), _record("b")]
    mock_get_records.return_value = records
    mock_get_reranker.return_value.score.return_value = [0.9, 0.8]
    ranked = [
        RankedResult(
            image_id="a",
            score=0.9,
            record=records[0],
            provenance_line="a",
        )
    ]
    mock_dedupe.return_value = ranked

    candidates = [
        Candidate(image_id="a", fused_score=0.9),
        Candidate(image_id="b", fused_score=0.8),
    ]
    out = rerank("query", candidates, top_k=1, min_match_percent=0)

    mock_dedupe.assert_called_once()
    assert mock_dedupe.call_args.kwargs["top_k"] == 1
    assert out == ranked
