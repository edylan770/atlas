"""Tests for deck pipeline manifest cache invalidation."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from imagecb.deck.pipeline import SlideSuggestion, process_deck_upload


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    cache = tmp_path / "deck_cache"
    fake_settings = Mock()
    fake_settings.deck_cache_enabled = True
    fake_settings.deck_cache_dir = cache
    fake_settings.deck_max_upload_bytes = 50 * 1024 * 1024
    fake_settings.deck_llm_batch_size = 8
    monkeypatch.setattr("imagecb.deck.pipeline.SETTINGS", fake_settings)
    monkeypatch.setattr("imagecb.deck.cache.SETTINGS", fake_settings)

    def slide_path(content_hash: str):
        return cache / "slides" / f"{content_hash}.json"

    def deck_path(deck_hash: str):
        return cache / "decks" / f"{deck_hash}.json"

    monkeypatch.setattr("imagecb.deck.cache._slide_cache_path", slide_path)
    monkeypatch.setattr("imagecb.deck.cache._deck_cache_path", deck_path)
    monkeypatch.setattr("imagecb.deck.pipeline.vector_store.count", lambda: 10)
    monkeypatch.setattr(
        "imagecb.deck.cache.corpus_fingerprint",
        lambda: "corp_v1",
    )
    return cache


def test_manifest_hit_when_request_fingerprint_matches(pipeline_env):
    slide = {
        "slide_index": 1,
        "title": "T",
        "body_preview": "body",
        "notes_preview": "",
        "body": "body",
        "notes": "",
        "content_hash": "abc",
        "status": "image_needed",
        "description": "office photo",
        "reason": "",
        "results": [{"image_id": "img1"}],
        "search_cached": True,
    }
    with patch(
        "imagecb.deck.pipeline.extract_slides_from_bytes",
        return_value=[],
    ), patch(
        "imagecb.deck.pipeline.deck_hash",
        return_value="deck1",
    ), patch(
        "imagecb.deck.pipeline._slide_hashes",
        return_value=["abc"],
    ), patch(
        "imagecb.deck.cache.request_fingerprint",
        return_value="reqfp",
    ):
        from imagecb.deck import cache as deck_cache

        deck_cache.put_deck_manifest(
            "deck1",
            "demo.pptx",
            ["abc"],
            [slide],
            request_fingerprint="reqfp",
        )
        result = process_deck_upload(b"pptx", "demo.pptx", top_k=10, min_match_percent=0)

    assert result.deck_cached is True
    assert len(result.slides) == 1
    assert result.slides[0].results == [{"image_id": "img1"}]


def test_manifest_refresh_when_top_k_changes(pipeline_env):
    stale_results = [{"image_id": "stale"}]
    fresh_results = [{"image_id": "fresh"}]
    slide_entry = {
        "slide_index": 1,
        "title": "T",
        "body_preview": "body",
        "notes_preview": "",
        "body": "body",
        "notes": "",
        "content_hash": "abc",
        "status": "image_needed",
        "description": "office photo",
        "reason": "",
        "results": stale_results,
        "search_cached": True,
    }

    call_count = {"n": 0}

    def fake_request_fp(*, top_k: int, min_match_percent: int, sort: str = "relevance") -> str:
        return f"k{top_k}"

    def fake_run_search(slide, llm_out, *, top_k, min_match_percent, sort="relevance"):
        call_count["n"] += 1
        return fresh_results, False

    with patch(
        "imagecb.deck.pipeline.extract_slides_from_bytes",
        return_value=[],
    ), patch(
        "imagecb.deck.pipeline.deck_hash",
        return_value="deck1",
    ), patch(
        "imagecb.deck.pipeline._slide_hashes",
        return_value=["abc"],
    ), patch(
        "imagecb.deck.cache.request_fingerprint",
        side_effect=fake_request_fp,
    ), patch(
        "imagecb.deck.pipeline._run_search_for_slide",
        side_effect=fake_run_search,
    ):
        from imagecb.deck import cache as deck_cache

        deck_cache.put_deck_manifest(
            "deck1",
            "demo.pptx",
            ["abc"],
            [slide_entry],
            request_fingerprint="k10",
        )
        result = process_deck_upload(b"pptx", "demo.pptx", top_k=5, min_match_percent=0)

    assert result.deck_cached is False
    assert call_count["n"] == 1
    assert result.slides[0].results == fresh_results

    manifest = deck_cache.get_deck_manifest("deck1")
    assert manifest is not None
    assert manifest.request_fingerprint == "k5"
    assert manifest.slides[0]["results"] == fresh_results
