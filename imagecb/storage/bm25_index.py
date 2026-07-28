"""BM25 sparse index over caption + OCR + slide context text.

Rebuilt at the end of every ingest run and persisted via pickle. The
corpus is small (prototype scale), so we keep the whole thing in memory.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
class _BM25State:
    image_ids: List[str]
    docs: List[List[str]]


class BM25Index:
    def __init__(self) -> None:
        self._state: Optional[_BM25State] = None
        self._bm25 = None

    def build(self, image_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(image_ids) != len(texts):
            raise ValueError("image_ids and texts must have the same length")
        docs = [tokenize(t) for t in texts]
        self._state = _BM25State(image_ids=list(image_ids), docs=docs)
        self._fit()

    def _fit(self) -> None:
        if self._state is None or not self._state.docs:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._state.docs)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or SETTINGS.bm25_path
        if self._state is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: a crash mid-dump must not leave a truncated pickle
        # that silently loads as an empty index.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._state, f)
        os.replace(tmp, path)

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or SETTINGS.bm25_path
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                self._state = pickle.load(f)
            self._fit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load BM25 index from %s: %s", path, exc)
            self._state = None
            self._bm25 = None
            return False

    def is_loaded(self) -> bool:
        return self._state is not None and bool(self._state.image_ids)

    def count(self) -> int:
        if self._state is None:
            return 0
        return len(self._state.image_ids)

    def list_ids(self) -> set[str]:
        if self._state is None:
            return set()
        return set(self._state.image_ids)

    def query(
        self,
        text: str,
        *,
        top_k: int,
        allowed_ids: Optional[Iterable[str]] = None,
    ) -> List[tuple[str, float]]:
        if self._bm25 is None or self._state is None:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        allowed = set(allowed_ids) if allowed_ids is not None else None
        candidates = []
        for image_id, score in zip(self._state.image_ids, scores):
            if allowed is not None and image_id not in allowed:
                continue
            if score <= 0:
                continue
            candidates.append((image_id, float(score)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


_index: Optional[BM25Index] = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
        _index.load()
    return _index


def reset_cache() -> None:
    """Clear the in-process BM25 singleton (used after index restore)."""
    global _index
    _index = None


def reload_index() -> BM25Index:
    """Drop the cached index and reload from BM25_PATH."""
    reset_cache()
    return get_index()


def is_loaded() -> bool:
    return get_index().is_loaded()


def count() -> int:
    return get_index().count()


def list_ids() -> set[str]:
    return get_index().list_ids()


def rebuild_from_records(records) -> None:
    """Rebuild and persist the index from current SQLite records."""
    from imagecb.caption.document import caption_document_text

    global _index
    ids: List[str] = []
    texts: List[str] = []
    for r in records:
        ids.append(r.image_id)
        texts.append(caption_document_text(r))

    # Build off to the side so concurrent searches keep using the complete
    # previous index until the replacement is ready.
    replacement = BM25Index()
    replacement.build(ids, texts)
    replacement.save()
    _index = replacement
