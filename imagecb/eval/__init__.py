"""Offline search evaluation harness."""

from imagecb.eval.dataset import GoldenSetValidationError, load_golden_set
from imagecb.eval.metrics import AggregateMetrics, CaseMetrics, aggregate_cases, score_case
from imagecb.eval.runner import EvalRunResult, run_eval
from imagecb.eval.schema import GoldenSet, SimilarCase, TextCase

__all__ = [
    "AggregateMetrics",
    "CaseMetrics",
    "EvalRunResult",
    "GoldenSet",
    "GoldenSetValidationError",
    "SimilarCase",
    "TextCase",
    "aggregate_cases",
    "load_golden_set",
    "run_eval",
    "score_case",
]
