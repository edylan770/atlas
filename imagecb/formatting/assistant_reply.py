"""Template-based assistant chat replies and structured result cards."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from imagecb.caption.asset_type import format_asset_type_label
from imagecb.formatting.match_display import display_match_percent
from imagecb.paths import resolve_image_file, resolve_source_file
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.rerank import RankedResult
from imagecb.caption.quality import needs_regeneration
from imagecb.storage.metadata_db import ImageRecord, deserialize_list

_CAPTION_FAILED = "[caption failed]"
_CAPTION_TRUNC = 120
_HIGHLIGHT_CAPTION_TRUNC = 100
_MATCH_HINT_TRUNC = 160


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_type: str  # pptx | pdf | image
    slide_index: Optional[int] = None
    page_index: Optional[int] = None
    modified: Optional[str] = None  # ISO date
    author: Optional[str] = None
    asset_type: str = ""

    def location_label(self) -> str:
        if self.source_type == "pptx" and self.slide_index is not None:
            return f"Slide {self.slide_index}"
        if self.source_type == "pdf" and self.page_index is not None:
            return f"Page {self.page_index}"
        return self.source_name

    def chips(self) -> List[str]:
        out: List[str] = []
        asset_label = format_asset_type_label(self.asset_type)
        if asset_label:
            out.append(asset_label)
        if self.source_type == "pptx" and self.slide_index is not None:
            out.append(f"Slide {self.slide_index}")
        elif self.source_type == "pdf" and self.page_index is not None:
            out.append(f"Page {self.page_index}")
        out.append(self.source_name)
        if self.modified:
            out.append(self.modified)
        if self.author:
            out.append(self.author)
        return out


@dataclass
class ResultCard:
    rank: int
    image_id: str
    image_url: str
    provenance: Provenance
    caption: str
    match_hint: Optional[str]
    match_percent: int
    has_image_file: bool
    image_name: str = ""
    use_case: str = ""
    tags: List[str] = field(default_factory=list)
    recommended_cases: List[str] = field(default_factory=list)
    theme: str = ""
    aliases: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    source_location: str = ""
    source_path: Optional[str] = None
    caption_quality: str = "ok"
    needs_regeneration: bool = False
    created_at: Optional[str] = None
    asset_type: str = ""


@dataclass
class AssistantReply:
    message: str
    results: List[ResultCard]


def catalog_fields_from_record(
    record: ImageRecord,
) -> tuple[str, str, List[str], List[str], str, List[str]]:
    name = (record.image_name or "").strip() or Path(record.source_file or "").name or "(unknown)"
    use_case = (record.use_case or "").strip()
    tags = deserialize_list(record.tags_json)
    recommended = deserialize_list(record.recommended_cases_json)
    theme = (record.theme or "").strip()
    aliases = deserialize_list(record.search_aliases_json)
    return name, use_case, tags, recommended, theme, aliases


def provenance_from_record(record: ImageRecord) -> Provenance:
    src_name = Path(record.source_file or "").name or "(unknown)"
    modified: Optional[str] = None
    if isinstance(record.source_modified_at, datetime):
        modified = record.source_modified_at.date().isoformat()
    author = (record.author or "").strip() or None
    return Provenance(
        source_name=src_name,
        source_type=record.source_type or "image",
        slide_index=record.slide_index,
        page_index=record.page_index,
        modified=modified,
        author=author,
        asset_type=(record.asset_type or "").strip(),
    )


def _display_caption(record: ImageRecord) -> str:
    cap = (record.caption_short or record.caption_detailed or "").strip()
    if cap == _CAPTION_FAILED:
        detail = (record.caption_detailed or "").strip()
        if detail.startswith("VLM error:"):
            return detail
        return "Caption unavailable (VLM failed during ingest)."
    return cap


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _short_match_hint(result: RankedResult) -> Optional[str]:
    hint = result.explanation().strip()
    if not hint:
        return None
    return _truncate(hint, _MATCH_HINT_TRUNC)


def source_location_label(record: ImageRecord) -> str:
    if record.source_type == "pptx" and record.slide_index is not None:
        return f"Slide {record.slide_index}"
    if record.source_type == "pdf" and record.page_index is not None:
        return f"Page {record.page_index}"
    return Path(record.source_file or "").name or ""


def _caption_quality_fields(record: ImageRecord) -> tuple[str, bool]:
    quality = (record.caption_quality or "ok").lower()
    return quality, needs_regeneration(quality)


def build_result_cards(
    results: Sequence[RankedResult],
    *,
    image_url_prefix: str = "/api/images",
    source_url_prefix: str = "/api/sources",
) -> List[ResultCard]:
    cards: List[ResultCard] = []
    for rank, r in enumerate(results, start=1):
        prov = provenance_from_record(r.record)
        cap = _display_caption(r.record)
        src_path = resolve_source_file(r.record)
        image_name, use_case, tags, recommended, theme, aliases = catalog_fields_from_record(
            r.record
        )
        caption_quality, regen = _caption_quality_fields(r.record)
        created_at: Optional[str] = None
        if isinstance(r.record.created_at, datetime):
            created_at = r.record.created_at.isoformat()
        cards.append(
            ResultCard(
                rank=rank,
                image_id=r.image_id,
                image_url=f"{image_url_prefix}/{r.image_id}",
                provenance=prov,
                caption=cap,
                match_hint=_short_match_hint(r),
                match_percent=display_match_percent(r.score, r.score_kind),
                has_image_file=resolve_image_file(r.record) is not None,
                image_name=image_name,
                use_case=use_case,
                tags=tags,
                recommended_cases=recommended,
                theme=theme,
                aliases=aliases,
                source_url=f"{source_url_prefix}/{r.image_id}" if src_path else None,
                source_location=source_location_label(r.record),
                source_path=str(src_path) if src_path else (r.record.source_file or None),
                caption_quality=caption_quality,
                needs_regeneration=regen,
                created_at=created_at,
                asset_type=(r.record.asset_type or "").strip(),
            )
        )
    return cards


def _group_by_source(results: Sequence[RankedResult]) -> dict[str, List[RankedResult]]:
    groups: dict[str, List[RankedResult]] = defaultdict(list)
    for r in results:
        name = Path(r.record.source_file or "").name or "(unknown)"
        groups[name].append(r)
    return dict(groups)


def _source_summary(groups: dict[str, List[RankedResult]]) -> str:
    parts: List[str] = []
    for name, items in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        n = len(items)
        if n == 1 and items[0].record.source_type in ("pptx", "pdf"):
            prov = provenance_from_record(items[0].record)
            loc = prov.location_label()
            parts.append(f"1 from **{name}** ({loc.lower()})")
        elif n == 1:
            parts.append(f"1 standalone image (**{name}**)")
        else:
            slides = sorted(
                {
                    items[i].record.slide_index
                    for i in range(len(items))
                    if items[i].record.slide_index is not None
                }
            )
            pages = sorted(
                {
                    items[i].record.page_index
                    for i in range(len(items))
                    if items[i].record.page_index is not None
                }
            )
            if slides:
                if len(slides) <= 4:
                    loc = ", ".join(str(s) for s in slides)
                else:
                    loc = f"{slides[0]}, {slides[1]}, … and {len(slides) - 2} more"
                parts.append(f"{n} from **{name}** (slides {loc})")
            elif pages:
                if len(pages) <= 4:
                    loc = ", ".join(str(p) for p in pages)
                else:
                    loc = f"{pages[0]}, {pages[1]}, … and {len(pages) - 2} more"
                parts.append(f"{n} from **{name}** (pages {loc})")
            else:
                parts.append(f"{n} from **{name}**")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _highlight_line(r: RankedResult, *, multi_source: bool) -> str:
    prov = provenance_from_record(r.record)
    cap = _display_caption(r.record)
    cap_part = _truncate(cap, _HIGHLIGHT_CAPTION_TRUNC) if cap else "No caption"
    if multi_source:
        return f"• **{prov.location_label()}** ({prov.source_name}) — {cap_part}"
    if prov.source_type in ("pptx", "pdf"):
        return f"• **{prov.location_label()}** — {cap_part}"
    return f"• {cap_part}"


def _missing_files_footer(results: Sequence[RankedResult]) -> str:
    missing = sum(1 for r in results if resolve_image_file(r.record) is None)
    if not missing:
        return ""
    noun = "result" if missing == 1 else "results"
    return (
        f"\n\n({missing} {noun} missing image files on disk — "
        "re-ingest your source folder with `--force`.)"
    )


def _refinement_prefix(spec: Optional[QuerySpec]) -> str:
    if spec and spec.is_refinement:
        return "Narrowed from your previous search — "
    return ""


def _closing_hint(spec: Optional[QuerySpec], *, count: int) -> str:
    lines: List[str] = []
    if count > 3:
        lines.append(f"See all {count} in the results panel.")
    if spec and spec.is_refinement:
        lines.append('You can keep refining (e.g. "only the charts" or "from last month").')
    elif count > 0:
        lines.append('You can refine with something like "only the charts" or "from last month."')
    return " ".join(lines)


def format_assistant_message(
    results: Sequence[RankedResult],
    spec: Optional[QuerySpec] = None,
) -> str:
    prefix = _refinement_prefix(spec)
    if not results:
        return (
            prefix
            + "I couldn't find any images that match. Try loosening your filters or "
            "rephrasing your question."
        ).strip()

    n = len(results)
    missing_footer = _missing_files_footer(results)

    if n <= 3:
        groups = _group_by_source(results)
        summary = _source_summary(groups)
        caps = [_truncate(_display_caption(r.record), _CAPTION_TRUNC) for r in results]
        caps = [c for c in caps if c]
        if caps:
            if len(caps) == 1:
                body = f"I found {n} image{'s' if n != 1 else ''} ({summary}): {caps[0]}"
            else:
                joined = "; ".join(caps)
                body = f"I found {n} images ({summary}): {joined}"
        else:
            body = f"I found {n} image{'s' if n != 1 else ''} ({summary})."
        close = _closing_hint(spec, count=n)
        if close:
            body = f"{body} {close}"
        return (prefix + body + missing_footer).strip()

    groups = _group_by_source(results)
    summary = _source_summary(groups)
    multi_source = len(groups) > 1
    lines = [
        prefix + f"I found **{n}** images that match. {summary.capitalize() if summary[0].islower() else summary}."
    ]
    highlights = results[:3]
    if highlights:
        lines.append("")
        lines.append("**Highlights:**")
        for r in highlights:
            lines.append(_highlight_line(r, multi_source=multi_source))
    close = _closing_hint(spec, count=n)
    if close:
        lines.append("")
        lines.append(close)
    return ("\n".join(lines) + missing_footer).strip()


def build_assistant_reply(
    results: Sequence[RankedResult],
    spec: Optional[QuerySpec] = None,
    *,
    image_url_prefix: str = "/api/images",
) -> AssistantReply:
    return AssistantReply(
        message=format_assistant_message(results, spec),
        results=build_result_cards(results, image_url_prefix=image_url_prefix),
    )
