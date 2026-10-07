"""Cluster zero/weak chat searches into content-gap themes for admins."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from imagecb.admin.analytics import _display_query, _parse_since
from imagecb.config import SETTINGS
from imagecb.telemetry import s3_store

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_EXAMPLE_LIMIT = 5


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalize_gap_query(text: str) -> str:
    """Lowercase, strip light punctuation, collapse whitespace."""
    s = (text or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _is_gap_event(row: dict, *, threshold: float) -> Optional[str]:
    """Return 'zero_result' or 'weak_result', or None if not a content gap."""
    if (row.get("search_kind") or "chat") == "similar":
        return None
    result_count = int(row.get("result_count") or 0)
    if result_count == 0:
        return "zero_result"
    top_score = row.get("top_score")
    if top_score is not None and float(top_score) < threshold:
        return "weak_result"
    return None


def _embed_unique_labels(labels: Sequence[str]) -> Optional[np.ndarray]:
    """Embed unique labels; return (n, dim) array or None on failure."""
    if not labels:
        return None
    try:
        from imagecb.models.embedder import get_text_embedder

        embedder = get_text_embedder()
        vectors = [embedder.embed_query(label) for label in labels]
        return np.stack(vectors, axis=0)
    except Exception:
        logger.warning(
            "content gaps: embedding failed; falling back to exact-string themes",
            exc_info=True,
        )
        return None


def _cluster_indices(embs: np.ndarray, *, threshold: float) -> List[List[int]]:
    n = embs.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(embs[i], embs[j]) >= threshold:
                uf.union(i, j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def _theme_payload(
    *,
    label_counts: Counter[str],
    zero_count: int,
    weak_count: int,
    users: set[str],
    last_seen_at: Optional[str],
) -> dict:
    search_count = zero_count + weak_count
    theme_label = label_counts.most_common(1)[0][0] if label_counts else "—"
    example_queries = [q for q, _ in label_counts.most_common(_EXAMPLE_LIMIT)]
    parts = [f'{search_count} time{"s" if search_count != 1 else ""}']
    detail_bits = []
    if zero_count:
        detail_bits.append(f"{zero_count} zero-result")
    if weak_count:
        detail_bits.append(f"{weak_count} weak")
    if detail_bits:
        parts.append(f"({', '.join(detail_bits)})")
    summary = (
        f'People searched for "{theme_label}" {" ".join(parts)}—no strong matches.'
    )
    return {
        "theme_label": theme_label,
        "search_count": search_count,
        "zero_count": zero_count,
        "weak_count": weak_count,
        "unique_users": len(users),
        "example_queries": example_queries,
        "last_seen_at": last_seen_at,
        "summary": summary,
    }


def content_gaps_report(
    *,
    since: Optional[str] = None,
    days: Optional[int] = None,
    weak_score_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Group zero/weak chat searches into ranked content-gap themes."""
    if days is None:
        days = SETTINGS.telemetry_default_window_days
    days = max(1, min(int(days), SETTINGS.telemetry_retention_days))

    since_dt = _parse_since(since)
    if since_dt is None:
        since_dt = datetime.utcnow() - timedelta(days=days)

    cutoff = s3_store.retention_cutoff()
    if since_dt < cutoff:
        since_dt = cutoff

    threshold = (
        weak_score_threshold
        if weak_score_threshold is not None
        else SETTINGS.weak_result_score_threshold
    )
    sim_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else SETTINGS.content_gap_similarity_threshold
    )

    cache_key = (
        f"content_gaps|{since_dt.isoformat()}|{days}|{threshold}|{sim_threshold}"
    )
    cached = s3_store.get_quality_cache(cache_key)
    if cached is not None:
        return cached

    # Exact-normalized buckets of gap events.
    # key -> list of (display_query, category, user_id, created_at)
    buckets: Dict[str, list[tuple[str, str, str, Optional[str]]]] = defaultdict(list)

    for row in s3_store.iter_search_events(since=since_dt):
        category = _is_gap_event(row, threshold=threshold)
        if category is None:
            continue
        display = _display_query(row)
        if not display or display == "—":
            continue
        norm = normalize_gap_query(display)
        if not norm:
            continue
        created = row.get("created_at")
        if isinstance(created, datetime):
            created = created.isoformat()
        buckets[norm].append(
            (
                display,
                category,
                str(row.get("user_id") or "anonymous"),
                created,
            )
        )

    unique_norms = sorted(buckets.keys())
    # Representative label for embedding: most common display string in bucket.
    reps: List[str] = []
    for norm in unique_norms:
        counts = Counter(d for d, _, _, _ in buckets[norm])
        reps.append(counts.most_common(1)[0][0])

    embs = _embed_unique_labels(reps)
    if embs is not None and len(reps) == embs.shape[0]:
        index_groups = _cluster_indices(embs, threshold=sim_threshold)
        clustering_mode = "embedding"
    else:
        index_groups = [[i] for i in range(len(reps))]
        clustering_mode = "exact"

    themes: List[dict] = []
    one_offs: List[dict] = []

    for indices in index_groups:
        label_counts: Counter[str] = Counter()
        zero_count = 0
        weak_count = 0
        users: set[str] = set()
        last_seen: Optional[str] = None
        for idx in indices:
            norm = unique_norms[idx]
            for display, category, user_id, created in buckets[norm]:
                label_counts[display] += 1
                if category == "zero_result":
                    zero_count += 1
                else:
                    weak_count += 1
                users.add(user_id)
                if created and (last_seen is None or created > last_seen):
                    last_seen = created
        payload = _theme_payload(
            label_counts=label_counts,
            zero_count=zero_count,
            weak_count=weak_count,
            users=users,
            last_seen_at=last_seen,
        )
        if payload["search_count"] >= 2:
            themes.append(payload)
        else:
            one_offs.append(payload)

    themes.sort(key=lambda t: (-t["search_count"], t["theme_label"].lower()))
    one_offs.sort(key=lambda t: (t["theme_label"].lower(),))

    payload = {
        "since": since_dt.isoformat(),
        "window_days": days,
        "weak_score_threshold": threshold,
        "similarity_threshold": sim_threshold,
        "clustering_mode": clustering_mode,
        "gap_event_count": sum(t["search_count"] for t in themes)
        + sum(t["search_count"] for t in one_offs),
        "themes": themes,
        "one_offs": one_offs,
    }
    s3_store.set_quality_cache(cache_key, payload)
    return payload
