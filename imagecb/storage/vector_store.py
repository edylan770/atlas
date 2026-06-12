"""Chroma-backed dense vector stores.

Two collections, one record per image_id each:
- image embeddings (Titan multimodal) with compact provenance metadata
  so Chroma can apply `where` filters server-side;
- caption-document text embeddings for the text-to-text dense lane.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from imagecb.config import SETTINGS

_IMAGE_COLLECTION = "imagecb_images"
_CAPTION_TEXT_COLLECTION = "imagecb_caption_text"


_client = None
_collections: Dict[str, object] = {}


def _get_collection(name: str = _IMAGE_COLLECTION):
    global _client
    col = _collections.get(name)
    if col is not None:
        return col
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(SETTINGS.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    col = _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    _collections[name] = col
    return col


def _upsert(
    name: str,
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    metadatas: Sequence[dict],
) -> None:
    if not image_ids:
        return
    col = _get_collection(name)
    col.upsert(
        ids=list(image_ids),
        embeddings=[e.tolist() for e in embeddings],
        metadatas=list(metadatas),
    )


def _query(
    name: str,
    query_embedding: np.ndarray,
    *,
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[tuple[str, float]]:
    col = _get_collection(name)
    where = None
    if allowed_ids is not None:
        ids_list = list(allowed_ids)
        if not ids_list:
            return []
        # Chroma uses image_id stored in metadata to allow $in filtering.
        where = {"image_id": {"$in": ids_list}}
    res = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
    )
    ids = res.get("ids", [[]])[0]
    distances = res.get("distances", [[]])[0]
    # Convert cosine distance (1 - similarity) to similarity.
    return [(i, 1.0 - float(d)) for i, d in zip(ids, distances)]


def _list_ids(name: str, *, batch_size: int = 500) -> set[str]:
    col = _get_collection(name)
    n = col.count()
    if n == 0:
        return set()
    out: set[str] = set()

    def _consume(res: dict) -> None:
        for i in res.get("ids") or []:
            out.add(str(i))

    try:
        offset = 0
        while offset < n:
            limit = min(batch_size, n - offset)
            res = col.get(include=[], limit=limit, offset=offset)
            offset += limit
            _consume(res)
    except TypeError:
        res = col.get(include=[], limit=n)
        _consume(res)
    return out


# --- Image embedding collection ---


def upsert(
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    metadatas: Sequence[dict],
) -> None:
    _upsert(_IMAGE_COLLECTION, image_ids=image_ids, embeddings=embeddings, metadatas=metadatas)


def query(
    query_embedding: np.ndarray,
    *,
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[tuple[str, float]]:
    return _query(_IMAGE_COLLECTION, query_embedding, top_k=top_k, allowed_ids=allowed_ids)


def count() -> int:
    return _get_collection(_IMAGE_COLLECTION).count()


def delete(image_ids: Sequence[str]) -> None:
    if not image_ids:
        return
    _get_collection(_IMAGE_COLLECTION).delete(ids=list(image_ids))


def list_ids(*, batch_size: int = 500) -> set[str]:
    """Return all image_id values present in the image collection."""
    return _list_ids(_IMAGE_COLLECTION, batch_size=batch_size)


# --- Caption-text embedding collection ---


def upsert_text(*, image_ids: Sequence[str], embeddings: np.ndarray) -> None:
    _upsert(
        _CAPTION_TEXT_COLLECTION,
        image_ids=image_ids,
        embeddings=embeddings,
        metadatas=[{"image_id": i} for i in image_ids],
    )


def query_text(
    query_embedding: np.ndarray,
    *,
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[tuple[str, float]]:
    return _query(_CAPTION_TEXT_COLLECTION, query_embedding, top_k=top_k, allowed_ids=allowed_ids)


def delete_text(image_ids: Sequence[str]) -> None:
    if not image_ids:
        return
    _get_collection(_CAPTION_TEXT_COLLECTION).delete(ids=list(image_ids))


def list_text_ids(*, batch_size: int = 500) -> set[str]:
    """Return all image_id values present in the caption-text collection."""
    return _list_ids(_CAPTION_TEXT_COLLECTION, batch_size=batch_size)


def get_embeddings(image_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Return stored image embeddings for the given image IDs."""
    if not image_ids:
        return {}
    col = _get_collection(_IMAGE_COLLECTION)
    res = col.get(ids=list(image_ids), include=["embeddings"])
    out: dict[str, np.ndarray] = {}
    ids = res.get("ids") or []
    raw_embs = res.get("embeddings")
    if not ids or raw_embs is None:
        return out
    if isinstance(raw_embs, np.ndarray):
        if raw_embs.ndim == 2:
            emb_list = [raw_embs[i] for i in range(raw_embs.shape[0])]
        elif raw_embs.ndim == 1 and len(ids) == 1:
            emb_list = [raw_embs]
        else:
            emb_list = []
    else:
        emb_list = list(raw_embs)
    for i, emb in zip(ids, emb_list):
        if emb is None:
            continue
        try:
            arr = np.asarray(emb, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if arr.ndim != 1 or arr.size == 0:
            continue
        out[str(i)] = arr
    return out


def get_all_embeddings(*, batch_size: int = 500) -> List[tuple[str, np.ndarray]]:
    """Return (image_id, embedding) for all vectors in the image collection."""
    col = _get_collection(_IMAGE_COLLECTION)
    n = col.count()
    if n == 0:
        return []
    out: List[tuple[str, np.ndarray]] = []

    def _consume(res: dict) -> None:
        ids = res.get("ids") or []
        embs = res.get("embeddings") or []
        for i, emb in zip(ids, embs):
            if emb is None:
                continue
            try:
                arr = np.asarray(emb, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if arr.ndim != 1 or arr.size == 0:
                continue
            out.append((str(i), arr))

    try:
        offset = 0
        while offset < n:
            limit = min(batch_size, n - offset)
            res = col.get(
                include=["embeddings"],
                limit=limit,
                offset=offset,
            )
            offset += limit
            _consume(res)
    except TypeError:
        res = col.get(include=["embeddings"], limit=n)
        _consume(res)
    return out
