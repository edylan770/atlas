"""Tests for per-image hubness statistics and the CSLS-style adjuster."""

from __future__ import annotations

import numpy as np

from imagecb.retrieval.hubness import HubnessStats, _compute_stats


def _unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def test_compute_stats_ranks_hub_highest():
    # Three vectors clustered together (hubs) plus one far-away outlier.
    pairs = [
        ("h1", _unit([1.0, 0.05, 0.0])),
        ("h2", _unit([1.0, 0.0, 0.05])),
        ("h3", _unit([0.98, 0.02, 0.02])),
        ("out", _unit([0.0, 0.0, 1.0])),
    ]
    stats = _compute_stats(pairs, knn=2)

    assert stats.count == 4
    # Clustered images have higher local density than the outlier.
    assert stats.r_img["h1"] > stats.r_img["out"]
    assert stats.r_img["h2"] > stats.r_img["out"]
    # mean is the average of per-image scores
    assert abs(stats.mean_r_img - np.mean(list(stats.r_img.values()))) < 1e-9


def test_compute_stats_empty():
    stats = _compute_stats([], knn=10)
    assert stats == HubnessStats(r_img={}, mean_r_img=0.0, count=0)


def test_compute_stats_single_vector_no_neighbours():
    pairs = [("only", _unit([1.0, 0.0, 0.0]))]
    stats = _compute_stats(pairs, knn=10)
    assert stats.count == 1
    assert stats.r_img["only"] == 0.0


def test_knn_clamped_to_available_neighbours():
    pairs = [
        ("a", _unit([1.0, 0.0])),
        ("b", _unit([0.0, 1.0])),
    ]
    # knn larger than n-1 must not raise.
    stats = _compute_stats(pairs, knn=50)
    assert set(stats.r_img) == {"a", "b"}
