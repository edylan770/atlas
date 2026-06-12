"""Unit tests for search evaluation metrics."""

from __future__ import annotations

from imagecb.eval.metrics import aggregate_cases, score_case


def test_score_case_hit_and_mrr():
    metrics = score_case(
        case_id="q1",
        ranked_ids=["a", "b", "c", "d"],
        relevant_ids=["b", "z"],
        k_values=[1, 3],
    )

    assert metrics.hit_at[1] is False
    assert metrics.hit_at[3] is True
    assert metrics.recall_at[3] == 0.5
    assert metrics.mrr == 0.5
    assert metrics.relevant_ranks[0].image_id == "b"
    assert metrics.relevant_ranks[0].rank == 2
    assert metrics.relevant_ranks[1].image_id == "z"
    assert metrics.relevant_ranks[1].rank is None


def test_score_case_no_relevant_ids():
    metrics = score_case(
        case_id="empty",
        ranked_ids=["a", "b"],
        relevant_ids=[],
        k_values=[1, 2],
    )

    assert metrics.hit_at[1] is False
    assert metrics.recall_at[1] == 0.0
    assert metrics.mrr == 0.0


def test_score_case_first_relevant_at_rank_one():
    metrics = score_case(
        case_id="top",
        ranked_ids=["x", "y"],
        relevant_ids=["x"],
        k_values=[1],
    )

    assert metrics.hit_at[1] is True
    assert metrics.mrr == 1.0


def test_aggregate_cases_averages():
    cases = [
        score_case(case_id="a", ranked_ids=["x"], relevant_ids=["x"], k_values=[1, 3]),
        score_case(case_id="b", ranked_ids=["y"], relevant_ids=["z"], k_values=[1, 3]),
    ]
    agg = aggregate_cases("chat", cases, [1, 3])

    assert agg.case_count == 2
    assert agg.hit_at[1] == 0.5
    assert agg.mrr == 0.5
