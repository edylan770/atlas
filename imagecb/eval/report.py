"""Format evaluation results for CLI output and JSON export."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from imagecb.eval.metrics import AggregateMetrics
from imagecb.eval.runner import EvalRunResult


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _format_aggregate_line(label: str, agg: AggregateMetrics, k_values: Sequence[int]) -> str:
    parts = [f"{label:<16} n={agg.case_count}"]
    for k in k_values:
        parts.append(f"Hit@{k}={_pct(agg.hit_at.get(k, 0.0))}")
    parts.append(f"MRR={agg.mrr:.2f}")
    max_k = max(k_values) if k_values else 10
    parts.append(f"Recall@{max_k}={_pct(agg.recall_at.get(max_k, 0.0))}")
    return "  ".join(parts)


def format_summary(result: EvalRunResult, k_values: Sequence[int]) -> str:
    lines: List[str] = []
    if result.chat is not None:
        lines.append(_format_aggregate_line("Text (chat)", result.chat, k_values))
    if result.retrieval is not None:
        lines.append(_format_aggregate_line("Text (retrieval)", result.retrieval, k_values))
    if result.similar is not None:
        lines.append(_format_aggregate_line("Similar", result.similar, k_values))
    if not lines:
        lines.append("No active cases matched the requested mode.")
    return "\n".join(lines)


def _format_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "—"


def format_failures(
    result: EvalRunResult,
    k_values: Sequence[int],
    *,
    failures_only: bool = False,
) -> str:
    max_k = max(k_values) if k_values else 10
    lines: List[str] = []

    for case_result in result.case_results:
        metrics = case_result.metrics
        failed = metrics.failed_at_k.get(max_k, True)
        if failures_only and not failed:
            continue

        header = f"FAIL  {metrics.case_id} [{case_result.mode}]"
        if not failed:
            header = f"OK    {metrics.case_id} [{case_result.mode}]"
        lines.append(header)

        for item in metrics.relevant_ranks:
            lines.append(f"  expected {item.image_id}  rank={_format_rank(item.rank)}")

        preview = metrics.ranked_ids[:max_k]
        if preview:
            lines.append(f"  got: {', '.join(preview)}")
        lines.append("")

    if failures_only:
        lines = [block for block in "\n".join(lines).split("\n\n") if block.startswith("FAIL")]
        return "\n\n".join(lines).rstrip()

    return "\n".join(lines).rstrip()


def result_to_dict(result: EvalRunResult, k_values: Sequence[int]) -> Dict[str, Any]:
    def agg_dict(agg: AggregateMetrics | None) -> Dict[str, Any] | None:
        if agg is None:
            return None
        return {
            "mode": agg.mode,
            "case_count": agg.case_count,
            "hit_at": {str(k): agg.hit_at.get(k, 0.0) for k in k_values},
            "recall_at": {str(k): agg.recall_at.get(k, 0.0) for k in k_values},
            "mrr": agg.mrr,
            "cases": [
                {
                    "case_id": c.case_id,
                    "relevant_ranks": [
                        {"image_id": r.image_id, "rank": r.rank}
                        for r in c.relevant_ranks
                    ],
                    "ranked_ids": c.ranked_ids,
                    "hit_at": {str(k): c.hit_at.get(k) for k in k_values},
                    "recall_at": {str(k): c.recall_at.get(k, 0.0) for k in k_values},
                    "mrr": c.mrr,
                }
                for c in agg.cases
            ],
        }

    return {
        "k_values": list(k_values),
        "chat": agg_dict(result.chat),
        "retrieval": agg_dict(result.retrieval),
        "similar": agg_dict(result.similar),
        "case_results": [
            {
                "case_id": cr.case_id,
                "mode": cr.mode,
                "scores": cr.scores,
                "metrics": asdict(cr.metrics),
            }
            for cr in result.case_results
        ],
    }


def write_json_report(path, result: EvalRunResult, k_values: Sequence[int]) -> None:
    from pathlib import Path

    payload = result_to_dict(result, k_values)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
