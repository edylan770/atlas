"""Tests for calibrated match % display mapping."""

from __future__ import annotations

from imagecb.formatting.match_display import (
    ScoreKind,
    display_match_percent,
    meets_min_match_percent,
)


def test_rerank_anchor_points():
    assert display_match_percent(0.12, ScoreKind.RERANK) == 10
    assert display_match_percent(0.25, ScoreKind.RERANK) == 28
    assert display_match_percent(0.95, ScoreKind.RERANK) == 100


def test_rerank_interpolated():
    assert display_match_percent(0.27, ScoreKind.RERANK) == 31
    assert display_match_percent(0.45, ScoreKind.RERANK) == 54
    assert display_match_percent(0.90, ScoreKind.RERANK) == 97


def test_dense_anchor_and_excellent():
    assert display_match_percent(0.35, ScoreKind.DENSE) == 40
    assert display_match_percent(0.92, ScoreKind.DENSE) == 100


def test_clamping():
    assert display_match_percent(-0.1, ScoreKind.RERANK) == 0
    assert display_match_percent(1.5, ScoreKind.RERANK) == 100


def test_rerank_monotonic():
    raws = [0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 0.88, 0.99]
    percents = [display_match_percent(r, ScoreKind.RERANK) for r in raws]
    assert percents == sorted(percents)


def test_score_kind_string():
    assert display_match_percent(0.25, "rerank") == 28


def test_rerank_087_calibrated_for_cards():
    assert display_match_percent(0.87, ScoreKind.RERANK) == 94


def test_fusion_linear_display():
    assert display_match_percent(0.5, ScoreKind.FUSION) == 50
    assert display_match_percent(1.0, ScoreKind.FUSION) == 100
    assert display_match_percent(0.0, ScoreKind.FUSION) == 0


def test_meets_min_match_percent_rerank():
    assert meets_min_match_percent(0.70, ScoreKind.RERANK, 80) is True
    assert meets_min_match_percent(0.69, ScoreKind.RERANK, 80) is False
    assert meets_min_match_percent(0.80, ScoreKind.RERANK, 80) is True
    assert meets_min_match_percent(0.72, ScoreKind.RERANK, 0) is True


def test_meets_min_match_percent_fusion():
    assert meets_min_match_percent(0.5, ScoreKind.FUSION, 50) is True
    assert meets_min_match_percent(0.49, ScoreKind.FUSION, 50) is False
