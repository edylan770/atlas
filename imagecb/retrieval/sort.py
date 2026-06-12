"""Shared result ordering for search and catalog listing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from sqlalchemy import func

from imagecb.retrieval.rerank import RankedResult
from imagecb.storage.metadata_db import ImageRecord

ResultSort = Literal["relevance", "newest", "oldest", "name", "source"]
VALID_SORTS = frozenset({"relevance", "newest", "oldest", "name", "source"})

_MIN_DT = datetime.min


class InvalidSortError(ValueError):
    pass


def parse_sort(sort: str | None) -> Optional[ResultSort]:
    if sort is None or sort == "":
        return None
    normalized = sort.strip().lower()
    if normalized not in VALID_SORTS:
        raise InvalidSortError(
            f"sort must be one of: {', '.join(sorted(VALID_SORTS))}"
        )
    return normalized  # type: ignore[return-value]


def resolve_sort(sort: str | None, *, is_search: bool) -> ResultSort:
    parsed = parse_sort(sort)
    if parsed is not None:
        return parsed
    return "relevance" if is_search else "newest"


def catalog_sort_for_sql(sort: ResultSort) -> List:
    """Map catalog sort to SQLAlchemy order clauses. relevance -> newest."""
    effective: ResultSort = "newest" if sort == "relevance" else sort
    if effective == "newest":
        return [ImageRecord.created_at.desc(), ImageRecord.image_id.asc()]
    if effective == "oldest":
        return [ImageRecord.created_at.asc(), ImageRecord.image_id.asc()]
    if effective == "name":
        return [
            func.lower(func.coalesce(ImageRecord.image_name, ImageRecord.source_file)).asc(),
            ImageRecord.image_id.asc(),
        ]
    if effective == "source":
        return [ImageRecord.source_file.asc(), ImageRecord.image_id.asc()]
    raise InvalidSortError(f"unknown sort: {sort}")


def _record_name(rec: ImageRecord) -> str:
    name = (rec.image_name or "").strip()
    if name:
        return name.lower()
    return Path(rec.source_file or "").name.lower()


def _record_created(rec: ImageRecord) -> datetime:
    return rec.created_at if rec.created_at is not None else _MIN_DT


def sort_ranked_results(
    results: Sequence[RankedResult],
    sort: ResultSort,
) -> List[RankedResult]:
    if sort == "relevance" or not results:
        return list(results)

    items = list(results)
    if sort == "newest":
        items.sort(key=lambda r: (_record_created(r.record), r.image_id), reverse=True)
    elif sort == "oldest":
        items.sort(key=lambda r: (_record_created(r.record), r.image_id))
    elif sort == "name":
        items.sort(key=lambda r: (_record_name(r.record), r.image_id))
    elif sort == "source":
        items.sort(key=lambda r: ((r.record.source_file or "").lower(), r.image_id))
    return items


def sort_image_records(
    records: Sequence[ImageRecord],
    sort: ResultSort,
) -> List[ImageRecord]:
    if sort == "relevance" or not records:
        return list(records)

    effective: ResultSort = "newest" if sort == "relevance" else sort
    items = list(records)
    if effective == "newest":
        items.sort(key=lambda r: (_record_created(r), r.image_id), reverse=True)
    elif effective == "oldest":
        items.sort(key=lambda r: (_record_created(r), r.image_id))
    elif effective == "name":
        items.sort(key=lambda r: (_record_name(r), r.image_id))
    elif effective == "source":
        items.sort(key=lambda r: ((r.source_file or "").lower(), r.image_id))
    return items
