"""Deterministic high-confidence lexical match check.

Used to decide whether a (typically short) query has a trustworthy literal
keyword hit. If so, the chat path keeps the fused/rerank ranking instead of
dropping to pure visual similarity. The check is purely lexical -- it counts how
many of the query's content tokens literally appear in a candidate's searchable
text -- so it is immune to the vector-space hubness that makes the visual lane
return false-positive "high" matches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Set

from imagecb.config import SETTINGS
from imagecb.storage import metadata_db
from imagecb.storage.bm25_index import tokenize

if TYPE_CHECKING:
    from imagecb.retrieval.hybrid import Candidate
    from imagecb.retrieval.query_parser import QuerySpec

# Common words that should not, on their own, count as content tokens. Keeps
# stopword-only or near-stopword queries from triggering a false lexical hit.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "are", "be", "this", "that", "these", "those",
    "it", "its", "into", "over", "about", "show", "me", "find", "image",
    "images", "photo", "photos", "picture", "pictures", "some", "any",
}

# Only inspect the strongest sparse candidates; a confident lexical hit will be
# near the top of BM25.
_MAX_CANDIDATES = 50


def _query_content_tokens(spec: "QuerySpec") -> List[str]:
    parts: List[str] = []
    semantic = (spec.semantic_query or spec.raw_text or "").strip()
    if semantic:
        parts.append(semantic)
    parts.extend(k for k in spec.must_have_keywords if k)
    tokens = tokenize(" ".join(parts))
    seen: Set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t in _STOPWORDS or len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _candidate_token_set(record) -> Set[str]:
    from imagecb.caption.document import caption_document_text

    return set(tokenize(caption_document_text(record)))


def has_high_confidence_lexical_hit(
    spec: "QuerySpec", candidates: Sequence["Candidate"]
) -> bool:
    """True when some BM25 candidate literally covers the query's content tokens.

    Coverage = (query content tokens present in candidate text) / (query content
    tokens). A hit requires coverage >= ``lexical_high_confidence_coverage``.
    """
    query_tokens = _query_content_tokens(spec)
    if not query_tokens:
        return False

    sparse = [c for c in candidates if getattr(c, "sparse_score", 0.0) > 0]
    if not sparse:
        return False
    sparse.sort(key=lambda c: c.sparse_score, reverse=True)
    head = sparse[:_MAX_CANDIDATES]

    ids = [c.image_id for c in head]
    records = {r.image_id: r for r in metadata_db.get_records(ids)}

    threshold = SETTINGS.lexical_high_confidence_coverage
    total = len(query_tokens)
    query_set = set(query_tokens)
    for c in head:
        record = records.get(c.image_id)
        if record is None:
            continue
        cand_tokens = _candidate_token_set(record)
        covered = sum(1 for t in query_set if t in cand_tokens)
        if covered / total >= threshold:
            return True
    return False
