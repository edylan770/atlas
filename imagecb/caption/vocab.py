"""Corpus-derived tag vocabulary for controlled caption tagging."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Set

from imagecb.caption.normalize import normalize_tag
from imagecb.storage.metadata_db import deserialize_list, get_all_records

_MAX_VOCAB_PROMPT = 200

_vocab_cache: Optional[List[str]] = None


def build_tag_vocab_from_records(records) -> Set[str]:
    vocab: Set[str] = set()
    for r in records:
        for tag in deserialize_list(r.tags_json):
            norm = normalize_tag(tag)
            if norm:
                vocab.add(norm)
    return vocab


def load_tag_vocab(*, refresh: bool = False) -> List[str]:
    """Load sorted tag vocabulary from active corpus records (cached per process)."""
    global _vocab_cache
    if _vocab_cache is not None and not refresh:
        return list(_vocab_cache)
    vocab = build_tag_vocab_from_records(get_all_records())
    _vocab_cache = sorted(vocab)
    return list(_vocab_cache)


def vocab_for_prompt(
    *,
    source_file: Optional[str] = None,
    refresh: bool = False,
) -> List[str]:
    """Return up to _MAX_VOCAB_PROMPT terms, prioritizing same-source-file tags."""
    all_vocab = load_tag_vocab(refresh=refresh)
    if len(all_vocab) <= _MAX_VOCAB_PROMPT:
        return all_vocab

    if not source_file:
        return all_vocab[:_MAX_VOCAB_PROMPT]

    source_lower = source_file.lower()
    source_tags: List[str] = []
    freq = Counter()
    for r in get_all_records():
        if (r.source_file or "").lower() != source_lower:
            continue
        for tag in deserialize_list(r.tags_json):
            norm = normalize_tag(tag)
            if norm:
                freq[norm] += 1
                if norm not in source_tags:
                    source_tags.append(norm)

    remaining = [t for t in all_vocab if t not in set(source_tags)]
    by_freq = sorted(freq.keys(), key=lambda t: (-freq[t], t))
    combined = by_freq + [t for t in source_tags if t not in by_freq]
    combined += remaining
    seen: set[str] = set()
    out: List[str] = []
    for t in combined:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= _MAX_VOCAB_PROMPT:
            break
    return out


def format_vocab_for_prompt(vocab: Sequence[str]) -> str:
    if not vocab:
        return "(no existing tags in corpus yet — you may invent concise new tags when nothing fits)"
    return ", ".join(vocab)
