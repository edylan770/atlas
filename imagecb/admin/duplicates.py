"""Near-duplicate clustering from Chroma embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from imagecb.config import SETTINGS
from imagecb.storage import metadata_db, vector_store


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_duplicate_clusters(
    *,
    threshold: float | None = None,
) -> List[dict]:
    """Return clusters of images with pairwise cosine similarity >= threshold."""
    threshold = threshold if threshold is not None else SETTINGS.duplicate_similarity_threshold
    pairs = vector_store.get_all_embeddings()
    if len(pairs) < 2:
        return []

    dim = pairs[0][1].shape[0]
    filtered = [(i, e) for i, e in pairs if e.ndim == 1 and e.shape[0] == dim]
    if len(filtered) < 2:
        return []

    ids = [p[0] for p in filtered]
    embs = np.stack([p[1] for p in filtered], axis=0)
    n = len(ids)
    uf = _UnionFind(n)
    max_sim: Dict[tuple[str, str], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embs[i], embs[j])
            if sim >= threshold:
                uf.union(i, j)
                key = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                max_sim[key] = sim

    clusters_map: Dict[int, Set[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters_map.setdefault(root, set()).add(i)

    records = {r.image_id: r for r in metadata_db.get_all_records(include_deleted=False)}
    clusters_out: List[dict] = []

    for members in clusters_map.values():
        if len(members) < 2:
            continue
        member_ids = [ids[i] for i in sorted(members)]
        cluster_max = 0.0
        for a in range(len(member_ids)):
            for b in range(a + 1, len(member_ids)):
                k = (
                    (member_ids[a], member_ids[b])
                    if member_ids[a] < member_ids[b]
                    else (member_ids[b], member_ids[a])
                )
                cluster_max = max(cluster_max, max_sim.get(k, 0.0))

        items = []
        for iid in member_ids:
            rec = records.get(iid)
            items.append(
                {
                    "image_id": iid,
                    "caption_short": rec.caption_short if rec else None,
                    "source_file": Path(rec.source_file).name if rec and rec.source_file else None,
                    "image_url": f"/api/images/{iid}",
                    "thumb_url": f"/api/images/{iid}/thumb",
                }
            )
        clusters_out.append(
            {
                "cluster_id": member_ids[0],
                "size": len(member_ids),
                "max_similarity": round(cluster_max, 4),
                "image_ids": member_ids,
                "images": items,
            }
        )

    clusters_out.sort(key=lambda c: (-c["size"], -c["max_similarity"]))
    return clusters_out
