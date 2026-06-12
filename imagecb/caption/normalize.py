"""Tag and token normalization: case, whitespace, plurality."""

from __future__ import annotations

import re
from typing import Dict, List, Set

_PLURAL_EXCEPTIONS = frozenset(
    {
        "news",
        "series",
        "status",
        "business",
        "graphics",
        "analytics",
        "sales",
        "process",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_tag(term: str) -> str:
    """Lowercase, collapse whitespace, and singularize a tag-like term."""
    t = (term or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    if t.endswith("ies") and len(t) > 4:
        candidate = t[:-3] + "y"
        if candidate not in _PLURAL_EXCEPTIONS:
            t = candidate
    elif t.endswith("s") and len(t) > 3 and not t.endswith("ss") and t not in _PLURAL_EXCEPTIONS:
        t = t[:-1]
    return t


def tokenize_text(text: str) -> List[str]:
    """Lowercase alphanumeric tokens from free text."""
    return _TOKEN_RE.findall((text or "").lower())


def normalize_tags(
    raw_tags: List[str],
    vocab: Set[str],
    *,
    min_length: int = 2,
) -> List[str]:
    """Map tags to canonical forms, preferring existing vocab casing."""
    vocab_normalized: Dict[str, str] = {normalize_tag(v): v for v in vocab}

    seen: set[str] = set()
    out: List[str] = []

    for raw in raw_tags:
        canonical = normalize_tag(raw)
        if not canonical or len(canonical) < min_length:
            continue
        canonical = normalize_tag(vocab_normalized.get(canonical, canonical))
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)

    return out


__all__ = ["normalize_tag", "normalize_tags", "tokenize_text"]
