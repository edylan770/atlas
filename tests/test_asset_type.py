"""Tests for asset-type taxonomy normalization and metadata filtering."""

from __future__ import annotations

from imagecb.caption.asset_type import (
    ASSET_TYPES,
    format_asset_type_label,
    normalize_asset_type,
    normalize_asset_types,
)
from imagecb.retrieval.query_parser import _build_spec
from imagecb.storage.metadata_db import ImageRecord, filter_image_ids, get_engine


def _make_record(image_id: str, asset_type: str | None) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        content_hash=f"hash-{image_id}",
        image_path=f"/tmp/{image_id}.png",
        source_file="deck.pptx",
        source_type="pptx",
        asset_type=asset_type,
    )


def test_normalize_asset_type_synonyms():
    assert normalize_asset_type("Photo") == "photo"
    assert normalize_asset_type("flowchart") == "diagram"
    assert normalize_asset_type("bar chart") == "chart"
    assert normalize_asset_type("screenshots") == "screenshot"
    assert normalize_asset_type("") == "other"
    assert normalize_asset_type("unknown-thing") == "other"


def test_normalize_asset_types_dedupes():
    assert normalize_asset_types(["photo", "photos", "picture"]) == ["photo"]
    assert normalize_asset_types(["chart", "diagram"]) == ["chart", "diagram"]


def test_format_asset_type_label():
    assert format_asset_type_label("photo") == "Photo"
    assert format_asset_type_label("other") == "Other"
    assert format_asset_type_label("") == ""


def test_build_spec_parses_asset_types():
    spec = _build_spec(
        {
            "semantic_query": "sales visuals",
            "source_filters": {"asset_types": ["photos", "graphs"]},
        },
        "photos and graphs",
    )
    assert spec.source_filters.asset_types == ["photo", "chart"]


def test_resolve_query_asset_type_maps_flowchart():
    from imagecb.caption.asset_type import resolve_query_asset_type

    assert resolve_query_asset_type("flowchart") == "diagram"
    assert resolve_query_asset_type("presentation") is None
    assert resolve_query_asset_type("diagram") == "diagram"


def test_filter_image_ids_excludes_null_when_asset_types_set(tmp_path, monkeypatch):
    import imagecb.storage.metadata_db as metadata_db

    db_path = tmp_path / "asset_type_filter.db"
    monkeypatch.setattr(metadata_db, "_engine", None)
    monkeypatch.setattr(metadata_db, "_SessionLocal", None)
    monkeypatch.setattr(metadata_db, "_engine_url", lambda: f"sqlite:///{db_path}")

    get_engine()

    from imagecb.storage.metadata_db import session_scope

    with session_scope() as s:
        s.add(_make_record("photo-1", "photo"))
        s.add(_make_record("chart-1", "chart"))
        s.add(_make_record("missing-1", None))

    photo_ids = filter_image_ids(asset_types=["photo"])
    assert photo_ids == ["photo-1"]

    both = filter_image_ids(asset_types=["photo", "chart"])
    assert set(both) == {"photo-1", "chart-1"}

    assert "missing-1" not in filter_image_ids(asset_types=["photo", "chart", "other"])


def test_asset_types_constant_count():
    assert len(ASSET_TYPES) == 10
    assert "other" in ASSET_TYPES
