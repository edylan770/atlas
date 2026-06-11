"""Cross-encoder reranking of fused candidates.

We build a single text representation per candidate by concatenating
caption + OCR + slide context, and ask the cross-encoder to score each
against the user's query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, TYPE_CHECKING

from imagecb.caption.asset_type import format_asset_type_label
from imagecb.config import SETTINGS
from imagecb.formatting.match_display import meets_min_match_percent
from imagecb.models.reranker import get_reranker
from imagecb.retrieval.hybrid import Candidate
from imagecb.storage import metadata_db
from imagecb.storage.metadata_db import ImageRecord, deserialize_list

if TYPE_CHECKING:
    from imagecb.retrieval.query_parser import QuerySpec

_RERANK_MAX_TAGS = 8
_RERANK_MAX_RECOMMENDED_CASES = 3
_RERANK_MAX_ALIASES = 8
_RERANK_MAX_OCR_CHARS = 600


@dataclass
class RankedResult:
    image_id: str
    score: float
    record: ImageRecord
    provenance_line: str
    score_kind: Literal["rerank", "dense", "fusion"] = "rerank"

    @property
    def image_path(self) -> str:
        return self.record.image_path

    def explanation(self) -> str:
        bits = []
        if self.record.caption_short:
            bits.append(self.record.caption_short)
        tags = deserialize_list(self.record.tags_json)
        if tags:
            bits.append("tags: " + ", ".join(tags[:6]))
        if self.record.ocr_text:
            ocr = self.record.ocr_text.strip()
            if ocr:
                bits.append('OCR: "' + (ocr[:120] + ("..." if len(ocr) > 120 else "")) + '"')
        return " | ".join(bits)


def _candidate_text(r: ImageRecord) -> str:
    parts: List[str] = []

    asset_label = format_asset_type_label(r.asset_type)
    if asset_label:
        parts.append(f"asset_type: {asset_label}")

    for v in (
        r.scene,
        r.text_overlay_summary,
    ):
        if v:
            parts.append(v)

    objects = deserialize_list(r.objects_json)
    if objects:
        parts.append("objects: " + ", ".join(objects))

    for v in (
        r.slide_title,
        r.slide_notes,
        r.slide_body_text,
    ):
        if v:
            parts.append(v)

    ocr = (r.ocr_text or "").strip()
    if ocr:
        parts.append(ocr[:_RERANK_MAX_OCR_CHARS])

    for v in (
        r.image_name,
        r.caption_short,
        r.caption_detailed,
        r.use_case,
        r.theme,
    ):
        if v:
            parts.append(v)

    tags = deserialize_list(r.tags_json)
    if tags:
        parts.append("tags: " + ", ".join(tags[:_RERANK_MAX_TAGS]))

    aliases = deserialize_list(r.search_aliases_json)
    if aliases:
        parts.append("aliases: " + ", ".join(aliases[:_RERANK_MAX_ALIASES]))

    cases = deserialize_list(r.recommended_cases_json)
    if cases:
        parts.append("recommended: " + "; ".join(cases[:_RERANK_MAX_RECOMMENDED_CASES]))

    return "\n".join(parts)


def _format_provenance(r: ImageRecord) -> str:
    src_name = Path(r.source_file or "").name or "(unknown)"
    loc: Optional[str] = None
    if r.source_type == "pptx" and r.slide_index:
        loc = f"Slide {r.slide_index} of {src_name}"
    elif r.source_type == "pdf" and r.page_index:
        loc = f"Page {r.page_index} of {src_name}"
    else:
        loc = src_name

    modified = ""
    if isinstance(r.source_modified_at, datetime):
        modified = f", modified {r.source_modified_at.date().isoformat()}"
    author = f", by {r.author}" if r.author else ""
    return f"{loc}{modified}{author}"


def _fusion_tail_results(
    tail: Sequence[Candidate],
    *,
    exclude_ids: set[str],
    records: dict[str, ImageRecord],
) -> List[RankedResult]:
    built: List[RankedResult] = []
    for c in tail:
        if c.image_id in exclude_ids:
            continue
        record = records.get(c.image_id)
        if record is None:
            continue
        built.append(
            RankedResult(
                image_id=c.image_id,
                score=float(c.fused_score),
                record=record,
                provenance_line=_format_provenance(record),
                score_kind="fusion",
            )
        )
    built.sort(key=lambda r: r.score, reverse=True)
    return built


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    top_k: int,
    top_n: Optional[int] = None,
    min_match_percent: int = 0,
    spec: Optional["QuerySpec"] = None,
) -> List[RankedResult]:
    if not candidates:
        return []
    top_n = top_n or SETTINGS.rerank_top_n
    head = list(candidates[:top_n])
    ids = [c.image_id for c in head]
    records = {r.image_id: r for r in metadata_db.get_records(ids)}
    head = [c for c in head if c.image_id in records]
    if not head:
        return []
    docs = [_candidate_text(records[c.image_id]) for c in head]
    scores = get_reranker().score(query, docs)

    ranked = sorted(
        (
            RankedResult(
                image_id=c.image_id,
                score=float(s),
                record=records[c.image_id],
                provenance_line=_format_provenance(records[c.image_id]),
            )
            for c, s in zip(head, scores)
        ),
        key=lambda r: r.score,
        reverse=True,
    )

    if spec is not None:
        from imagecb.retrieval.asset_type_boost import apply_asset_type_boost

        ranked = apply_asset_type_boost(spec, ranked)

    from imagecb.retrieval.dedupe import dedupe_results

    if min_match_percent > 0:
        above = [
            r
            for r in ranked
            if meets_min_match_percent(r.score, r.score_kind, min_match_percent)
        ]
        below = [
            r
            for r in ranked
            if not meets_min_match_percent(r.score, r.score_kind, min_match_percent)
        ]
        results = dedupe_results(above, top_k=top_k, pool=below)
    else:
        results = dedupe_results(ranked, top_k=top_k)

    if len(results) < top_k and len(candidates) > top_n:
        kept_ids = {r.image_id for r in results}
        tail_candidates = list(candidates[top_n:])
        tail_ids = [c.image_id for c in tail_candidates if c.image_id not in kept_ids]
        tail_records = {
            r.image_id: r for r in metadata_db.get_records(tail_ids)
        }
        tail_ranked = _fusion_tail_results(
            tail_candidates,
            exclude_ids=kept_ids,
            records=tail_records,
        )
        if tail_ranked:
            results = dedupe_results(results, top_k=top_k, pool=tail_ranked)

    return results
