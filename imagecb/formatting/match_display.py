"""Map raw retrieval scores to intuitive 0–100% display values.

Chat search ranking uses Cohere rerank raw scores; min-match filtering and
result cards share the calibrated display scale defined here.
Similar-image search uses normalized RRF fusion scores in [0, 1].

Rerank anchors target Cohere Rerank 3.5 relevance (conservative absolute scale).
Dense anchors target Chroma cosine similarity (typically lower in practice).
Fusion anchors are linear on normalized RRF scores.
100% is reserved for near-excellent raw rerank/dense scores (>= 0.93 / >= 0.92)
or fusion scores at 1.0.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Tuple


class ScoreKind(str, Enum):
    RERANK = "rerank"
    DENSE = "dense"
    FUSION = "fusion"


# (raw_score, display_percent) — monotonic, sorted by raw
_RERANK_ANCHORS: List[Tuple[float, int]] = [
    (0.00, 0),
    (0.12, 10),
    (0.25, 28),
    (0.40, 48),
    (0.55, 65),
    (0.70, 80),
    (0.85, 92),
    (0.93, 100),
]

_DENSE_ANCHORS: List[Tuple[float, int]] = [
    (0.00, 0),
    (0.20, 15),
    (0.35, 40),
    (0.50, 62),
    (0.65, 78),
    (0.80, 90),
    (0.92, 100),
]

_FUSION_ANCHORS: List[Tuple[float, int]] = [
    (0.00, 0),
    (1.00, 100),
]

_ANCHORS = {
    ScoreKind.RERANK: _RERANK_ANCHORS,
    ScoreKind.DENSE: _DENSE_ANCHORS,
    ScoreKind.FUSION: _FUSION_ANCHORS,
}


def _interpolate(raw: float, anchors: Sequence[Tuple[float, int]]) -> int:
    if not anchors:
        return 0
    if raw <= anchors[0][0]:
        return anchors[0][1]
    if raw >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(1, len(anchors)):
        r0, d0 = anchors[i - 1]
        r1, d1 = anchors[i]
        if raw <= r1:
            if r1 == r0:
                return d1
            t = (raw - r0) / (r1 - r0)
            return round(d0 + t * (d1 - d0))
    return anchors[-1][1]


def display_match_percent(raw: float, kind: ScoreKind | str = ScoreKind.RERANK) -> int:
    """Convert a raw score in [0, 1] to a display percentage in [0, 100]."""
    if isinstance(kind, str):
        kind = ScoreKind(kind)
    raw_clamped = max(0.0, min(float(raw), 1.0))
    anchors = _ANCHORS[kind]
    return max(0, min(100, _interpolate(raw_clamped, anchors)))


def meets_min_match_percent(
    raw: float,
    kind: ScoreKind | str,
    min_match_percent: int,
) -> bool:
    """Return True when the calibrated display % meets the user threshold."""
    if min_match_percent <= 0:
        return True
    return display_match_percent(raw, kind) >= min_match_percent
