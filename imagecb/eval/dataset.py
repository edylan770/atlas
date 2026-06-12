"""Load and validate the evaluation golden set."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Set, Tuple

from imagecb.eval.schema import GoldenSet, SimilarCase, TextCase
from imagecb.storage import metadata_db

logger = logging.getLogger(__name__)


class GoldenSetValidationError(ValueError):
    """Raised when the golden set references missing image IDs."""


def load_golden_set(path: Path, *, validate_ids: bool = True) -> GoldenSet:
    """Parse golden.json and optionally verify image IDs exist in SQLite."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    golden = GoldenSet.model_validate(raw)
    _check_duplicate_ids(golden)
    if validate_ids:
        missing = find_missing_ids(golden)
        if missing:
            raise GoldenSetValidationError(
                "Golden set references image IDs not found in the index: "
                + ", ".join(sorted(missing))
            )
    return golden


def active_text_cases(golden: GoldenSet) -> List[TextCase]:
    return [case for case in golden.text_cases if not case.template]


def short_query_cases(golden: GoldenSet) -> List[TextCase]:
    return [case for case in active_text_cases(golden) if case.short_query]


def active_similar_cases(golden: GoldenSet) -> List[SimilarCase]:
    return [case for case in golden.similar_cases if not case.template]


def find_missing_ids(golden: GoldenSet) -> Set[str]:
    """Return image IDs referenced by active cases that are absent from metadata."""
    referenced: Set[str] = set()
    for case in active_text_cases(golden):
        referenced.update(case.relevant_ids)
    for case in active_similar_cases(golden):
        referenced.add(case.image_id)
        referenced.update(case.relevant_ids)

    missing: Set[str] = set()
    for image_id in referenced:
        if metadata_db.get_record(image_id) is None:
            missing.add(image_id)
    return missing


def _check_duplicate_ids(golden: GoldenSet) -> None:
    seen: Set[str] = set()
    duplicates: List[str] = []
    for case in [*golden.text_cases, *golden.similar_cases]:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)
    if duplicates:
        logger.warning("Duplicate case IDs in golden set: %s", ", ".join(sorted(set(duplicates))))


def filter_cases(
    golden: GoldenSet,
    *,
    case_id: str | None = None,
) -> Tuple[List[TextCase], List[SimilarCase]]:
    """Return active cases, optionally narrowed to a single case id."""
    text_cases = active_text_cases(golden)
    similar_cases = active_similar_cases(golden)
    if case_id is None:
        return text_cases, similar_cases
    text_cases = [c for c in text_cases if c.id == case_id]
    similar_cases = [c for c in similar_cases if c.id == case_id]
    if not text_cases and not similar_cases:
        raise ValueError(f"No active case with id {case_id!r}")
    return text_cases, similar_cases
