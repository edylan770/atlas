"""Convert a chat turn into a validated `QuerySpec` via the LLM."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, List, Optional

from imagecb.caption.asset_type import normalize_asset_types
from imagecb.config import SETTINGS
from imagecb.models.llm import get_query_llm

if TYPE_CHECKING:
    from imagecb.retrieval.rerank import RankedResult

logger = logging.getLogger(__name__)


@dataclass
class SourceFilters:
    file_types: List[str] = field(default_factory=list)
    asset_types: List[str] = field(default_factory=list)
    filename_contains: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)


@dataclass
class TimeFilter:
    before: Optional[datetime] = None
    after: Optional[datetime] = None


@dataclass
class QuerySpec:
    semantic_query: str = ""
    must_have_keywords: List[str] = field(default_factory=list)
    must_avoid_keywords: List[str] = field(default_factory=list)
    source_filters: SourceFilters = field(default_factory=SourceFilters)
    time_filter: TimeFilter = field(default_factory=TimeFilter)
    top_k: int = 10
    is_refinement: bool = False
    raw_text: str = ""
    sanitization_notes: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Extra context for query parsing on follow-up turns."""

    previous_spec: Optional[QuerySpec] = None
    previous_results_summary: str = ""


def _serialize_previous_spec(spec: QuerySpec) -> str:
    sf = spec.source_filters
    tf = spec.time_filter
    payload = {
        "source_filters": {
            "file_types": list(sf.file_types),
            "asset_types": list(sf.asset_types),
            "filename_contains": list(sf.filename_contains),
            "authors": list(sf.authors),
        },
        "time_filter": {
            "before": tf.before.date().isoformat() if tf.before else None,
            "after": tf.after.date().isoformat() if tf.after else None,
        },
        "must_have_keywords": list(spec.must_have_keywords),
        "must_avoid_keywords": list(spec.must_avoid_keywords),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_session_context(
    previous_spec: Optional[QuerySpec],
    previous_results: "List[RankedResult]",
    *,
    max_results: int = 5,
) -> Optional[SessionContext]:
    if previous_spec is None and not previous_results:
        return None
    lines: List[str] = []
    for r in previous_results[:max_results]:
        cap = (r.record.caption_short or "")[:80]
        lines.append(f"- {r.provenance_line}: {cap}".strip())
    return SessionContext(
        previous_spec=previous_spec,
        previous_results_summary="\n".join(lines),
    )


def _parse_iso(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Accept date-only or full ISO.
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        logger.debug("Could not parse date: %s", value)
        return None


def _slist(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


_ALLOWED_FILE_TYPES = {"pptx", "pdf", "image"}


def _normalize_file_types(values: List[str]) -> List[str]:
    out = []
    for v in values:
        v2 = v.lower().strip().lstrip(".")
        if v2 in {"jpg", "jpeg", "png", "webp", "gif", "tiff", "tif", "bmp", "img", "images"}:
            v2 = "image"
        if v2 in _ALLOWED_FILE_TYPES and v2 not in out:
            out.append(v2)
    return out


# Explicit source-format filter language (not bare content words like "diagram").
_EXPLICIT_ASSET_FILTER_RE = re.compile(
    r"\b(?:only|just|exclude|excluding|without|no|not)\s+(?:the\s+)?"
    r"(?:photos?|pictures?|diagrams?|flowcharts?|charts?|graphs?|screenshots?|"
    r"logos?|illustrations?|icons?|tables?|maps?)\b"
    r"|\b(?:photos?|pictures?|diagrams?|flowcharts?|charts?|graphs?|screenshots?|"
    r"logos?|illustrations?|icons?|tables?|maps?)\s+only\b",
    re.IGNORECASE,
)

_EXPLICIT_FILE_TYPE_RE = re.compile(
    r"\b(?:pptx|powerpoint|power[\s-]?point|\.pptx?|pdf|\.pdf|image\s+files?|"
    r"(?:pptx|pdf|image)\s+(?:files?|only)|only\s+(?:pptx|pdf|image)(?:\s+files?)?)\b",
    re.IGNORECASE,
)

# Corpus is "unclassified" when most rows lack asset_type metadata.
_ASSET_TYPE_UNCLASSIFIED_THRESHOLD = 0.5


def _user_explicitly_requested_asset_types(text: str) -> bool:
    return bool(_EXPLICIT_ASSET_FILTER_RE.search(text or ""))


def _user_explicitly_requested_file_types(text: str) -> bool:
    return bool(_EXPLICIT_FILE_TYPE_RE.search(text or ""))


def _corpus_asset_type_unclassified_rate() -> float:
    from sqlalchemy import func, select

    from imagecb.storage.metadata_db import ImageRecord, get_engine, session_scope

    get_engine()
    with session_scope() as s:
        total = s.execute(
            select(func.count())
            .select_from(ImageRecord)
            .where(ImageRecord.deleted_at.is_(None))
        ).scalar() or 0
        if total == 0:
            return 1.0
        classified = s.execute(
            select(func.count())
            .select_from(ImageRecord)
            .where(
                ImageRecord.deleted_at.is_(None),
                ImageRecord.asset_type.isnot(None),
                ImageRecord.asset_type != "",
            )
        ).scalar() or 0
    return 1.0 - (classified / total)


def _file_type_filter_has_index_coverage(file_types: List[str]) -> bool:
    if not file_types:
        return True
    from imagecb.storage import metadata_db, vector_store

    ids = metadata_db.filter_image_ids(file_types=file_types)
    if not ids:
        return False
    active = set(metadata_db.get_active_image_ids())
    ids = [i for i in ids if i in active]
    if not ids:
        return False
    embeddings = vector_store.get_embeddings(ids)
    return any(embeddings.get(i) is not None for i in ids)


def sanitize_query_spec(spec: QuerySpec) -> QuerySpec:
    """Strip LLM-inferred metadata filters that would search empty corpus subsets."""
    raw = (spec.raw_text or "").strip()
    notes: List[str] = list(spec.sanitization_notes)
    sf = spec.source_filters

    if sf.asset_types:
        strip = False
        reason = ""
        if not _user_explicitly_requested_asset_types(raw):
            strip = True
            reason = "asset type filter was inferred from content words, not an explicit request"
        elif _corpus_asset_type_unclassified_rate() >= _ASSET_TYPE_UNCLASSIFIED_THRESHOLD:
            strip = True
            reason = "corpus asset types are not populated enough for format filtering"
        if strip:
            logger.warning(
                "Stripped asset_types %s from query %r (%s).",
                sf.asset_types,
                raw,
                reason,
            )
            sf.asset_types = []
            notes.append(f"Removed asset type filter ({reason}).")

    if sf.file_types:
        strip = False
        reason = ""
        if not _user_explicitly_requested_file_types(raw):
            strip = True
            reason = "file type filter was inferred, not explicitly requested"
        elif not _file_type_filter_has_index_coverage(sf.file_types):
            strip = True
            reason = "filtered file types have no indexed images"
        if strip:
            logger.warning(
                "Stripped file_types %s from query %r (%s).",
                sf.file_types,
                raw,
                reason,
            )
            sf.file_types = []
            notes.append(f"Removed file type filter ({reason}).")

    if not (spec.semantic_query or "").strip() and raw:
        spec.semantic_query = raw

    spec.sanitization_notes = notes
    return spec


def parse_query(
    text: str,
    history_summary: str = "",
    *,
    session_context: Optional[SessionContext] = None,
) -> QuerySpec:
    """Use the LLM to convert `text` into a validated `QuerySpec`."""
    text = (text or "").strip()
    if not text:
        return sanitize_query_spec(QuerySpec(raw_text=text))

    today_iso = date.today().isoformat()
    prev_spec_json = ""
    prev_results = ""
    if session_context is not None:
        if session_context.previous_spec is not None:
            prev_spec_json = _serialize_previous_spec(session_context.previous_spec)
        prev_results = session_context.previous_results_summary or ""

    try:
        raw = get_query_llm().parse(
            text,
            history_summary,
            today_iso,
            previous_spec_json=prev_spec_json,
            previous_results_summary=prev_results,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query LLM failed (%s); falling back to literal query.", exc)
        return sanitize_query_spec(QuerySpec(semantic_query=text, raw_text=text))

    return sanitize_query_spec(_build_spec(raw, text))


def _build_spec(raw: dict, original_text: str) -> QuerySpec:
    sf_raw = raw.get("source_filters") or {}
    tf_raw = raw.get("time_filter") or {}
    spec = QuerySpec(
        semantic_query=str(raw.get("semantic_query") or original_text).strip() or original_text,
        must_have_keywords=_slist(raw.get("must_have_keywords")),
        must_avoid_keywords=_slist(raw.get("must_avoid_keywords")),
        source_filters=SourceFilters(
            file_types=_normalize_file_types(_slist(sf_raw.get("file_types"))),
            asset_types=normalize_asset_types(_slist(sf_raw.get("asset_types"))),
            filename_contains=_slist(sf_raw.get("filename_contains")),
            authors=_slist(sf_raw.get("authors")),
        ),
        time_filter=TimeFilter(
            before=_parse_iso(tf_raw.get("before")),
            after=_parse_iso(tf_raw.get("after")),
        ),
        top_k=int(raw.get("top_k") or SETTINGS.default_top_k),
        is_refinement=bool(raw.get("is_refinement", False)),
        raw_text=original_text,
    )
    # Clamp top_k to sane bounds.
    spec.top_k = max(1, min(spec.top_k, 50))
    return spec


def summarize_history(history: List[tuple[str, str]], max_turns: int = 6) -> str:
    """Compact (user, assistant) history into a few bullet lines for the LLM."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for user, assistant in recent:
        if user:
            lines.append(f"- user: {user.strip()[:200]}")
        if assistant:
            lines.append(f"  assistant: {assistant.strip()[:200]}")
    return "\n".join(lines)
