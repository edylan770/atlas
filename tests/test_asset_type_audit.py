"""Tests for asset-type corpus audit and freeze workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagecb.caption.asset_type import ASSET_TYPE_TAXONOMY_VERSION
from imagecb.caption.asset_type_audit import (
    audit_asset_types,
    format_audit_report,
    freeze_asset_types,
)
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def _record(
    image_id: str,
    *,
    asset_type: str | None = "photo",
    tags: list[str] | None = None,
    caption_short: str = "",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="deck.pptx",
        source_type="pptx",
        asset_type=asset_type,
        tags_json=serialize_list(tags or []),
        caption_short=caption_short,
    )


def test_audit_distribution_counts():
    records = [
        _record("a", asset_type="photo"),
        _record("b", asset_type="chart"),
        _record("c", asset_type=None),
        _record("d", asset_type="other"),
    ]
    report = audit_asset_types(records=records)
    assert report.total_records == 4
    assert report.distribution["photo"] == 1
    assert report.distribution["chart"] == 1
    assert report.distribution["other"] == 1
    assert report.unclassified_count == 1
    assert report.unclassified_pct == 25.0
    assert any("lack asset_type" in w for w in report.warnings)


def test_audit_confusion_chart_tag_vs_diagram():
    records = [
        _record("x1", asset_type="diagram", tags=["chart", "sales"]),
        _record("x2", asset_type="diagram", tags=["bar", "revenue"]),
    ]
    report = audit_asset_types(records=records)
    signal = next(s for s in report.confusion_signals if s.name == "chart_tag_vs_diagram")
    assert signal.count == 2
    assert "x1" in signal.sample_image_ids


def test_audit_confusion_flowchart_caption_vs_chart():
    records = [
        _record(
            "f1",
            asset_type="chart",
            tags=["sales"],
            caption_short="Process flowchart with approval steps",
        ),
    ]
    report = audit_asset_types(records=records)
    signal = next(
        s for s in report.confusion_signals if s.name == "flowchart_caption_vs_chart"
    )
    assert signal.count == 1


def test_format_audit_report_includes_distribution():
    report = audit_asset_types(records=[_record("a", asset_type="photo")])
    text = format_audit_report(report)
    assert "photo" in text
    assert f"taxonomy v{ASSET_TYPE_TAXONOMY_VERSION}" in text


def test_freeze_asset_types_writes_snapshot(tmp_path: Path, monkeypatch):
    records = [_record(f"id-{i}", asset_type="photo") for i in range(5)]
    monkeypatch.setattr(
        "imagecb.caption.asset_type_audit.get_all_records",
        lambda: records,
    )
    report = audit_asset_types(records=records)
    assert report.passes_freeze_checks()

    out = tmp_path / "frozen.json"
    result = freeze_asset_types(force=False, output_path=out)
    assert result["frozen"] is True
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == ASSET_TYPE_TAXONOMY_VERSION
    assert payload["total_records"] == 5
    assert payload["frozen_at"]
    assert payload["distribution"]["photo"] == 5


def test_freeze_fails_on_unclassified(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "imagecb.caption.asset_type_audit.get_all_records",
        lambda: [_record("a", asset_type=None)],
    )
    with pytest.raises(ValueError, match="warnings"):
        freeze_asset_types(force=False, output_path=tmp_path / "x.json")


def test_audit_warns_on_high_other_share():
    records = [_record(f"id-{i}", asset_type="other") for i in range(20)]
    records += [_record("p1", asset_type="photo")]
    report = audit_asset_types(records=records)
    assert report.other_pct > 15.0
    assert any("other is" in w for w in report.warnings)
