"""Unit tests for golden-set loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from imagecb.eval.dataset import (
    GoldenSetValidationError,
    active_similar_cases,
    active_text_cases,
    filter_cases,
    load_golden_set,
)
from imagecb.eval.schema import GoldenSet


def _write_golden(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_golden_set_parses_cases(tmp_path: Path):
    path = _write_golden(
        tmp_path,
        {
            "version": 1,
            "text_cases": [
                {
                    "id": "q1",
                    "query": "dashboards",
                    "relevant_ids": ["id-a"],
                    "top_k": 5,
                }
            ],
            "similar_cases": [],
        },
    )

    with patch("imagecb.eval.dataset.metadata_db.get_record", return_value=object()):
        golden = load_golden_set(path)

    assert len(golden.text_cases) == 1
    assert golden.text_cases[0].query == "dashboards"
    assert active_text_cases(golden)[0].id == "q1"


def test_load_golden_set_missing_ids_raise(tmp_path: Path):
    path = _write_golden(
        tmp_path,
        {
            "version": 1,
            "text_cases": [
                {"id": "q1", "query": "dashboards", "relevant_ids": ["missing-id"]}
            ],
            "similar_cases": [],
        },
    )

    with patch("imagecb.eval.dataset.metadata_db.get_record", return_value=None):
        with pytest.raises(GoldenSetValidationError, match="missing-id"):
            load_golden_set(path)


def test_template_cases_skip_validation(tmp_path: Path):
    path = _write_golden(
        tmp_path,
        {
            "version": 1,
            "text_cases": [
                {
                    "id": "template-q",
                    "query": "example",
                    "relevant_ids": ["missing-id"],
                    "template": True,
                }
            ],
            "similar_cases": [],
        },
    )

    with patch("imagecb.eval.dataset.metadata_db.get_record", return_value=None):
        golden = load_golden_set(path)

    assert active_text_cases(golden) == []


def test_filter_cases_by_id(tmp_path: Path):
    golden = GoldenSet.model_validate(
        {
            "text_cases": [
                {"id": "a", "query": "one", "relevant_ids": []},
                {"id": "b", "query": "two", "relevant_ids": []},
            ],
            "similar_cases": [
                {
                    "id": "s1",
                    "image_id": "ref",
                    "relevant_ids": [],
                    "template": True,
                }
            ],
        }
    )

    text_cases, similar_cases = filter_cases(golden, case_id="a")
    assert [c.id for c in text_cases] == ["a"]
    assert similar_cases == []

    with pytest.raises(ValueError, match="No active case"):
        filter_cases(golden, case_id="missing")


def test_active_similar_cases_excludes_templates():
    golden = GoldenSet.model_validate(
        {
            "similar_cases": [
                {
                    "id": "tpl",
                    "image_id": "ref",
                    "relevant_ids": [],
                    "template": True,
                },
                {
                    "id": "live",
                    "image_id": "ref2",
                    "relevant_ids": ["n1"],
                },
            ]
        }
    )
    assert [c.id for c in active_similar_cases(golden)] == ["live"]
