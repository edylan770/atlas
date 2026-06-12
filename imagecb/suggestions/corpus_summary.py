"""Aggregate indexed corpus metadata for suggestion prompts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from imagecb.storage import metadata_db
from imagecb.storage.metadata_db import deserialize_list

_CAPTION_FAILED = "[caption failed]"
_MAX_SAMPLE_CAPTIONS = 12
_MAX_RECOMMENDED_CASES = 8
_MAX_TOP_TAGS = 12


@dataclass(frozen=True)
class SourceFileStat:
    name: str
    source_type: str
    count: int


@dataclass(frozen=True)
class CorpusContext:
    indexed_count: int
    source_files: Tuple[SourceFileStat, ...] = ()
    authors: Tuple[str, ...] = ()
    file_type_counts: Tuple[Tuple[str, int], ...] = ()
    modified_after: Optional[str] = None
    modified_before: Optional[str] = None
    sample_captions: Tuple[str, ...] = ()
    top_tags: Tuple[str, ...] = ()
    sample_recommended_cases: Tuple[str, ...] = ()
    fingerprint: str = ""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.date().isoformat()


def _summarize_records(records: Sequence) -> CorpusContext:
    if not records:
        return CorpusContext(indexed_count=0, fingerprint=_fingerprint_from_parts({}))

    source_counter: Counter[str] = Counter()
    source_type_by_name: dict[str, str] = {}
    type_counter: Counter[str] = Counter()
    author_counter: Counter[str] = Counter()
    modified_dates: List[datetime] = []
    captions: List[str] = []
    tag_counter: Counter[str] = Counter()
    recommended_cases: List[str] = []

    for r in records:
        src_path = r.source_file or ""
        name = Path(src_path).name or src_path
        st = (r.source_type or "").lower() or "image"
        source_counter[name] += 1
        source_type_by_name[name] = st
        type_counter[st] += 1
        author = (r.author or "").strip()
        if author:
            author_counter[author] += 1
        if r.source_modified_at is not None:
            modified_dates.append(r.source_modified_at)
        cap = (r.caption_short or "").strip()
        if cap and cap != _CAPTION_FAILED and len(cap) <= 200:
            captions.append(cap)
        for tag in deserialize_list(r.tags_json):
            t = tag.strip().lower()
            if t:
                tag_counter[t] += 1
        for case in deserialize_list(r.recommended_cases_json):
            c = case.strip()
            if c:
                recommended_cases.append(c)

    top_sources = [
        SourceFileStat(name=n, source_type=source_type_by_name.get(n, ""), count=c)
        for n, c in source_counter.most_common(8)
    ]
    top_authors = tuple(a for a, _ in author_counter.most_common(5))
    type_counts = tuple(type_counter.most_common())
    mod_after = _iso(min(modified_dates)) if modified_dates else None
    mod_before = _iso(max(modified_dates)) if modified_dates else None

    seen_cap: set[str] = set()
    sample_caps: List[str] = []
    for cap in captions:
        key = cap.lower()
        if key in seen_cap:
            continue
        seen_cap.add(key)
        sample_caps.append(cap)
        if len(sample_caps) >= _MAX_SAMPLE_CAPTIONS:
            break

    top_tags = tuple(t for t, _ in tag_counter.most_common(_MAX_TOP_TAGS))

    seen_case: set[str] = set()
    sample_cases: List[str] = []
    for case in recommended_cases:
        key = case.lower()
        if key in seen_case:
            continue
        seen_case.add(key)
        sample_cases.append(case)
        if len(sample_cases) >= _MAX_RECOMMENDED_CASES:
            break

    parts = {
        "count": len(records),
        "sources": [(s.name, s.source_type, s.count) for s in top_sources],
        "authors": list(top_authors),
        "types": list(type_counts),
        "modified": [mod_after, mod_before],
        "captions": sample_caps,
        "tags": list(top_tags),
        "recommended_cases": sample_cases,
    }
    fp = _fingerprint_from_parts(parts)

    return CorpusContext(
        indexed_count=len(records),
        source_files=tuple(top_sources),
        authors=top_authors,
        file_type_counts=type_counts,
        modified_after=mod_after,
        modified_before=mod_before,
        sample_captions=tuple(sample_caps),
        top_tags=top_tags,
        sample_recommended_cases=tuple(sample_cases),
        fingerprint=fp,
    )


def _fingerprint_from_parts(parts: dict) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_corpus_context() -> CorpusContext:
    records = metadata_db.get_all_records()
    return _summarize_records(records)


def context_to_prompt_text(ctx: CorpusContext) -> str:
    lines = [f"Indexed images: {ctx.indexed_count}"]
    if ctx.file_type_counts:
        types = ", ".join(f"{t}={n}" for t, n in ctx.file_type_counts)
        lines.append(f"File types: {types}")
    if ctx.top_tags:
        lines.append(f"Common tags: {', '.join(ctx.top_tags)}")
    if ctx.sample_captions:
        lines.append("Sample captions:")
        for cap in ctx.sample_captions:
            lines.append(f"  - {cap}")
    if ctx.sample_recommended_cases:
        lines.append("Sample recommended search phrases:")
        for case in ctx.sample_recommended_cases:
            lines.append(f"  - {case}")
    if ctx.source_files:
        lines.append("Top source files (grounding only — do not suggest filename-filter searches):")
        for s in ctx.source_files:
            lines.append(f"  - {s.name} ({s.source_type}): {s.count}")
    if ctx.authors:
        lines.append(f"Authors: {', '.join(ctx.authors)}")
    if ctx.modified_after or ctx.modified_before:
        lines.append(
            f"Source modified date range: {ctx.modified_after or '?'} to {ctx.modified_before or '?'}"
        )
    return "\n".join(lines)
