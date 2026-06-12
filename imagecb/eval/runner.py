"""Execute golden-set cases through live search pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

from imagecb.deck.search import search_for_description
from imagecb.eval.dataset import filter_cases, load_golden_set
from imagecb.eval.metrics import AggregateMetrics, CaseMetrics, aggregate_cases, score_case
from imagecb.eval.schema import EvalMode, GoldenSet, SimilarCase, TextCase
from imagecb.retrieval.rerank import RankedResult
from imagecb.retrieval.session import ChatSession
from imagecb.retrieval.similar import search_similar

logger = logging.getLogger(__name__)

RunMode = Literal["all", "chat", "retrieval", "similar"]


@dataclass
class CaseResult:
    case_id: str
    mode: EvalMode
    metrics: CaseMetrics
    scores: List[float] = field(default_factory=list)


@dataclass
class EvalRunResult:
    chat: AggregateMetrics | None = None
    retrieval: AggregateMetrics | None = None
    similar: AggregateMetrics | None = None
    case_results: List[CaseResult] = field(default_factory=list)


def _ranked_ids(results: Sequence[RankedResult]) -> List[str]:
    return [r.image_id for r in results]


def _ranked_scores(results: Sequence[RankedResult]) -> List[float]:
    return [r.score for r in results]


def run_text_chat(case: TextCase) -> List[RankedResult]:
    session = ChatSession()
    outcome = session.ask(case.query, top_k=case.top_k, min_match_percent=0)
    return list(outcome.results)


def run_text_retrieval(case: TextCase) -> List[RankedResult]:
    _cards, ranked = search_for_description(
        case.query,
        top_k=case.top_k,
        min_match_percent=0,
    )
    return ranked


def run_similar(case: SimilarCase) -> List[RankedResult]:
    outcome = search_similar(
        image_id=case.image_id,
        top_k=case.top_k,
        min_match_percent=0,
        similarity_axis=case.similarity_axis,
        exclude_image_id=case.image_id,
    )
    return list(outcome.results)


def run_eval(
    golden: GoldenSet,
    *,
    mode: RunMode = "all",
    k_values: Sequence[int] = (1, 3, 5, 10),
    case_id: str | None = None,
) -> EvalRunResult:
    text_cases, similar_cases = filter_cases(golden, case_id=case_id)
    report_ks = sorted(set(int(k) for k in k_values))

    result = EvalRunResult()
    case_results: List[CaseResult] = []

    if mode in ("all", "chat") and text_cases:
        chat_metrics: List[CaseMetrics] = []
        for case in text_cases:
            ranked = run_text_chat(case)
            case_k = max(report_ks + [case.top_k])
            metrics = score_case(
                case_id=case.id,
                ranked_ids=_ranked_ids(ranked),
                relevant_ids=case.relevant_ids,
                k_values=report_ks,
            )
            chat_metrics.append(metrics)
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    mode="chat",
                    metrics=metrics,
                    scores=_ranked_scores(ranked[:case_k]),
                )
            )
            logger.info("eval chat %s first_hit=%s", case.id, metrics.first_relevant_rank)
        result.chat = aggregate_cases("chat", chat_metrics, report_ks)

    if mode in ("all", "retrieval") and text_cases:
        retrieval_metrics: List[CaseMetrics] = []
        for case in text_cases:
            ranked = run_text_retrieval(case)
            case_k = max(report_ks + [case.top_k])
            metrics = score_case(
                case_id=case.id,
                ranked_ids=_ranked_ids(ranked),
                relevant_ids=case.relevant_ids,
                k_values=report_ks,
            )
            retrieval_metrics.append(metrics)
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    mode="retrieval",
                    metrics=metrics,
                    scores=_ranked_scores(ranked[:case_k]),
                )
            )
            logger.info("eval retrieval %s first_hit=%s", case.id, metrics.first_relevant_rank)
        result.retrieval = aggregate_cases("retrieval", retrieval_metrics, report_ks)

    if mode in ("all", "similar") and similar_cases:
        similar_metrics: List[CaseMetrics] = []
        for case in similar_cases:
            ranked = run_similar(case)
            case_k = max(report_ks + [case.top_k])
            metrics = score_case(
                case_id=case.id,
                ranked_ids=_ranked_ids(ranked),
                relevant_ids=case.relevant_ids,
                k_values=report_ks,
            )
            similar_metrics.append(metrics)
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    mode="similar",
                    metrics=metrics,
                    scores=_ranked_scores(ranked[:case_k]),
                )
            )
            logger.info("eval similar %s first_hit=%s", case.id, metrics.first_relevant_rank)
        result.similar = aggregate_cases("similar", similar_metrics, report_ks)

    result.case_results = case_results
    return result


def run_eval_from_path(
    golden_path,
    *,
    mode: RunMode = "all",
    k_values: Sequence[int] = (1, 3, 5, 10),
    case_id: str | None = None,
    validate_ids: bool = True,
) -> EvalRunResult:
    from pathlib import Path

    golden = load_golden_set(Path(golden_path), validate_ids=validate_ids)
    return run_eval(golden, mode=mode, k_values=k_values, case_id=case_id)
