"""Per-image hubness statistics for CSLS-style visual score correction.

Cross-modal embeddings exhibit *hubness*: a few image vectors sit near the
centre of the space and are close to almost any query, so they dominate
short/ambiguous queries regardless of relevance. To counteract this we
precompute, for each image, ``r_img`` = the mean cosine similarity to its K
nearest image neighbours (a local-density estimate). Hubs have a high
``r_img``; the visual lane subtracts a mean-centred version of it so universally
close images are demoted while query-specific matches are preserved.

The statistics are cached to disk (mirrors ``bm25.pkl``) and lazily rebuilt
when the cache is missing or the image count changed, so the various index
rebuild sites do not all need an explicit hook.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from imagecb.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class HubnessStats:
    """Per-image local-density scores and their corpus mean."""

    r_img: Dict[str, float]
    mean_r_img: float
    count: int


def _compute_stats(
    pairs: List[Tuple[str, np.ndarray]], *, knn: int
) -> HubnessStats:
    """Compute mean top-K neighbour cosine similarity per image.

    Vectors are L2-normalized so the Gram matrix entries are cosine
    similarities. For each row we average the K largest off-diagonal values.
    """
    if not pairs:
        return HubnessStats(r_img={}, mean_r_img=0.0, count=0)

    ids = [p[0] for p in pairs]
    mat = np.vstack([p[1] for p in pairs]).astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)

    n = len(ids)
    k = max(1, min(knn, n - 1)) if n > 1 else 0

    r_img: Dict[str, float] = {}
    if k == 0:
        for image_id in ids:
            r_img[image_id] = 0.0
        return HubnessStats(r_img=r_img, mean_r_img=0.0, count=n)

    for i, image_id in enumerate(ids):
        row = sims[i]
        top_k = np.partition(row, n - k)[n - k :]
        r_img[image_id] = float(np.mean(top_k))

    mean_r_img = float(np.mean(list(r_img.values()))) if r_img else 0.0
    return HubnessStats(r_img=r_img, mean_r_img=mean_r_img, count=n)


class HubnessIndex:
    def __init__(self) -> None:
        self._stats: Optional[HubnessStats] = None

    def build_from_embeddings(
        self, pairs: List[Tuple[str, np.ndarray]], *, knn: Optional[int] = None
    ) -> HubnessStats:
        knn = knn if knn is not None else SETTINGS.hubness_knn
        self._stats = _compute_stats(pairs, knn=knn)
        return self._stats

    def save(self, path: Optional[Path] = None) -> None:
        path = path or SETTINGS.hubness_path
        if self._stats is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._stats, f)

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or SETTINGS.hubness_path
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                self._stats = pickle.load(f)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load hubness stats from %s: %s", path, exc)
            self._stats = None
            return False

    def stats(self) -> Optional[HubnessStats]:
        return self._stats


_index: Optional[HubnessIndex] = None


def _current_image_count() -> int:
    try:
        from imagecb.storage import vector_store

        return vector_store.count()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read image vector count: %s", exc)
        return -1


def rebuild_from_embeddings() -> HubnessStats:
    """Recompute and persist hubness stats from the current image vectors."""
    from imagecb.storage import vector_store

    idx = _get_index()
    pairs = vector_store.get_all_embeddings()
    stats = idx.build_from_embeddings(pairs)
    idx.save()
    return stats


def _get_index() -> HubnessIndex:
    global _index
    if _index is None:
        _index = HubnessIndex()
        _index.load()
    return _index


def get_hubness_stats() -> HubnessStats:
    """Return cached stats, rebuilding lazily if missing or stale by count."""
    idx = _get_index()
    stats = idx.stats()
    count = _current_image_count()
    if stats is None or (count >= 0 and stats.count != count):
        try:
            stats = rebuild_from_embeddings()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hubness rebuild failed: %s", exc)
            return stats or HubnessStats(r_img={}, mean_r_img=0.0, count=0)
    return stats


def reset_cache() -> None:
    """Clear the in-process cache (used by tests)."""
    global _index
    _index = None
