"""Read-only audit of stored asset_type labels against corpus signals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from imagecb.caption.asset_type import (
    ASSET_TYPE_TAXONOMY_VERSION,
    ASSET_TYPES,
    CONFUSION_PAIR_MIN,
    DEFAULT_ASSET_TYPE,
    OTHER_WARN_PCT,
    taxonomy_snapshot_path,
)
from imagecb.storage.metadata_db import ImageRecord, deserialize_list, get_all_records


@dataclass
class ConfusionSignal:
    name: str
    description: str
    count: int
    sample_image_ids: List[str] = field(default_factory=list)


@dataclass
class AssetTypeAuditReport:
    total_records: int
    unclassified_count: int
    distribution: Dict[str, int]
    distribution_pct: Dict[str, float]
    other_pct: float
    unclassified_pct: float
    confusion_signals: List[ConfusionSignal]
    warnings: List[str]
    taxonomy_version: int = ASSET_TYPE_TAXONOMY_VERSION
    values: List[str] = field(default_factory=lambda: list(ASSET_TYPES))

    def to_dict(self) -> dict:
        return {
            "version": self.taxonomy_version,
            "frozen_at": None,
            "values": list(self.values),
            "total_records": self.total_records,
            "unclassified_count": self.unclassified_count,
            "unclassified_pct": round(self.unclassified_pct, 2),
            "distribution": dict(self.distribution),
            "distribution_pct": {k: round(v, 2) for k, v in self.distribution_pct.items()},
            "other_pct": round(self.other_pct, 2),
            "confusion_signals": [
                {
                    "name": s.name,
                    "description": s.description,
                    "count": s.count,
                    "sample_image_ids": list(s.sample_image_ids),
                }
                for s in self.confusion_signals
            ],
            "warnings": list(self.warnings),
        }

    def passes_freeze_checks(self) -> bool:
        return not self.warnings


# (signal_name, description, tag/caption needles, expected asset_type if matched)
_CONFUSION_RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "chart_tag_vs_diagram",
        "tags mention chart but asset_type is diagram",
        ("chart", "graph", "bar"),
        "diagram",
    ),
    (
        "screenshot_tag_vs_photo",
        "tags mention screenshot but asset_type is photo",
        ("screenshot", "ui", "browser"),
        "photo",
    ),
    (
        "logo_tag_vs_illustration",
        "tags mention logo but asset_type is illustration",
        ("logo", "wordmark", "brand"),
        "illustration",
    ),
    (
        "flowchart_caption_vs_chart",
        "caption mentions flowchart/infographic but asset_type is chart",
        ("flowchart", "infographic", "process diagram"),
        "chart",
    ),
)


def _record_text_blob(record: ImageRecord) -> str:
    tags = deserialize_list(record.tags_json)
    parts = [
        record.caption_short or "",
        record.caption_detailed or "",
        record.theme or "",
        record.use_case or "",
        " ".join(tags),
    ]
    return " ".join(p for p in parts if p).lower()


def _tags_contain_needle(tags: Sequence[str], needles: tuple[str, ...]) -> bool:
    joined = " ".join(tags).lower()
    return any(n in joined for n in needles)


def _caption_contains_needle(blob: str, needles: tuple[str, ...]) -> bool:
    return any(n in blob for n in needles)


def _collect_confusion_signals(records: Sequence[ImageRecord]) -> List[ConfusionSignal]:
    signals: List[ConfusionSignal] = []
    for name, description, needles, wrong_type in _CONFUSION_RULES:
        hits: List[str] = []
        for record in records:
            asset = (record.asset_type or "").strip().lower()
            if asset != wrong_type:
                continue
            tags = deserialize_list(record.tags_json)
            blob = _record_text_blob(record)
            if name == "flowchart_caption_vs_chart":
                matched = _caption_contains_needle(blob, needles)
            else:
                matched = _tags_contain_needle(tags, needles)
            if matched:
                hits.append(record.image_id)
        signals.append(
            ConfusionSignal(
                name=name,
                description=description,
                count=len(hits),
                sample_image_ids=hits[:5],
            )
        )
    return signals


def audit_asset_types(*, records: Optional[Sequence[ImageRecord]] = None) -> AssetTypeAuditReport:
    """Scan active records and report distribution plus heuristic confusion signals."""
    rows = list(records if records is not None else get_all_records())
    total = len(rows)

    distribution: Dict[str, int] = {t: 0 for t in ASSET_TYPES}
    unclassified = 0
    for record in rows:
        raw = (record.asset_type or "").strip().lower()
        if not raw:
            unclassified += 1
            continue
        if raw in distribution:
            distribution[raw] += 1
        else:
            distribution[DEFAULT_ASSET_TYPE] += 1

    distribution_pct = {
        k: (100.0 * v / total if total else 0.0) for k, v in distribution.items()
    }
    other_pct = distribution_pct.get(DEFAULT_ASSET_TYPE, 0.0)
    unclassified_pct = 100.0 * unclassified / total if total else 0.0

    classified = [r for r in rows if (r.asset_type or "").strip()]
    confusion_signals = _collect_confusion_signals(classified)

    warnings: List[str] = []
    if unclassified > 0:
        warnings.append(
            f"WARN: {unclassified} row(s) ({unclassified_pct:.1f}%) lack asset_type"
        )
    if other_pct >= OTHER_WARN_PCT:
        warnings.append(f"WARN: other is {other_pct:.1f}% (threshold {OTHER_WARN_PCT:.0f}%)")
    for signal in confusion_signals:
        if signal.count >= CONFUSION_PAIR_MIN:
            warnings.append(
                f"WARN: {signal.name} — {signal.count} cases "
                f"({signal.description})"
            )

    return AssetTypeAuditReport(
        total_records=total,
        unclassified_count=unclassified,
        distribution=distribution,
        distribution_pct=distribution_pct,
        other_pct=other_pct,
        unclassified_pct=unclassified_pct,
        confusion_signals=confusion_signals,
        warnings=warnings,
    )


def format_audit_report(report: AssetTypeAuditReport, *, verbose: bool = False) -> str:
    lines = [
        f"Asset type audit (taxonomy v{report.taxonomy_version})",
        f"Total records: {report.total_records}",
        f"Unclassified: {report.unclassified_count} ({report.unclassified_pct:.1f}%)",
        "",
        "Distribution:",
    ]
    for name in ASSET_TYPES:
        count = report.distribution.get(name, 0)
        pct = report.distribution_pct.get(name, 0.0)
        lines.append(f"  {name:14s} {count:5d}  ({pct:5.1f}%)")

    if report.confusion_signals:
        lines.append("")
        lines.append("Confusion signals (heuristic):")
        for signal in report.confusion_signals:
            lines.append(f"  {signal.name}: {signal.count}")
            if verbose and signal.sample_image_ids:
                lines.append(f"    samples: {', '.join(signal.sample_image_ids)}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  {w}")

    return "\n".join(lines)


def freeze_asset_types(
    *,
    force: bool = False,
    output_path: Optional[Path] = None,
) -> dict:
    """Validate audit and write a frozen taxonomy snapshot JSON."""
    report = audit_asset_types()
    if not force and not report.passes_freeze_checks():
        raise ValueError(
            "Audit has warnings; fix corpus or pass --force. "
            + "; ".join(report.warnings)
        )

    path = output_path or Path(taxonomy_snapshot_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "frozen": True,
        "path": str(path),
        "version": report.taxonomy_version,
        "total_records": report.total_records,
        "other_pct": report.other_pct,
        "warnings": list(report.warnings),
        "forced": force,
    }


__all__ = [
    "AssetTypeAuditReport",
    "ConfusionSignal",
    "audit_asset_types",
    "format_audit_report",
    "freeze_asset_types",
]
