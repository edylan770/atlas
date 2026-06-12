"""Information-retrieval metrics for search evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


@dataclass
class RelevantRank:
    image_id: str
    rank: Optional[int]  # 1-based; None if absent from ranked list


@dataclass
class CaseMetrics:
    case_id: str
    relevant_ranks: List[RelevantRank] = field(default_factory=list)
    ranked_ids: List[str] = field(default_factory=list)
    hit_at: Dict[int, bool] = field(default_factory=dict)
    recall_at: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    @property
    def first_relevant_rank(self) -> Optional[int]:
        for item in self.relevant_ranks:
            if item.rank is not None:
                return item.rank
        return None

    @property
    def failed_at_k(self) -> Dict[int, bool]:
        """True when no relevant hit appears in top-k."""
        return {k: not hit for k, hit in self.hit_at.items()}


def _rank_map(ranked_ids: Sequence[str]) -> Dict[str, int]:
    return {image_id: index + 1 for index, image_id in enumerate(ranked_ids)}


def score_case(
    *,
    case_id: str,
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int],
) -> CaseMetrics:
    """Score one case against an ordered list of retrieved image IDs."""
    relevant: Set[str] = set(relevant_ids)
    ranks = _rank_map(ranked_ids)

    relevant_ranks = [
        RelevantRank(image_id=image_id, rank=ranks.get(image_id))
        for image_id in relevant_ids
    ]

    first_rank = next(
        (item.rank for item in relevant_ranks if item.rank is not None),
        None,
    )
    mrr = 1.0 / first_rank if first_rank else 0.0

    hit_at: Dict[int, bool] = {}
    recall_at: Dict[int, float] = {}
    denom = len(relevant) if relevant else 0

    for k in k_values:
        top = set(ranked_ids[:k])
        found = len(relevant & top)
        hit_at[k] = found > 0
        recall_at[k] = (found / denom) if denom else 0.0

    return CaseMetrics(
        case_id=case_id,
        relevant_ranks=relevant_ranks,
        ranked_ids=list(ranked_ids),
        hit_at=hit_at,
        recall_at=recall_at,
        mrr=mrr,
    )


@dataclass
class AggregateMetrics:
    mode: str
    case_count: int
    hit_at: Dict[int, float] = field(default_factory=dict)
    recall_at: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    cases: List[CaseMetrics] = field(default_factory=list)


def aggregate_cases(mode: str, cases: Sequence[CaseMetrics], k_values: Sequence[int]) -> AggregateMetrics:
    """Average per-case metrics into summary percentages."""
    n = len(cases)
    if n == 0:
        return AggregateMetrics(mode=mode, case_count=0)

    hit_at = {k: sum(1 for c in cases if c.hit_at.get(k)) / n for k in k_values}
    recall_at = {k: sum(c.recall_at.get(k, 0.0) for c in cases) / n for k in k_values}
    mrr = sum(c.mrr for c in cases) / n

    return AggregateMetrics(
        mode=mode,
        case_count=n,
        hit_at=hit_at,
        recall_at=recall_at,
        mrr=mrr,
        cases=list(cases),
    )
