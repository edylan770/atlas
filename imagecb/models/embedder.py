"""Bedrock embedding wrappers.

``BedrockEmbedder`` wraps Titan Multimodal Embeddings (or Cohere Embed)
for image and cross-modal text inputs in a shared space.
``BedrockTextEmbedder`` wraps a text embedding model (Titan Text v2 or
Cohere Embed) for the caption-text dense lane. Vectors are L2-normalized
so cosine similarity equals inner product in Chroma.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from imagecb.config import SETTINGS
from imagecb.models.bedrock_client import bedrock_invoke_model

_MAX_IMAGE_SIDE = 2048
_MAX_TEXT_CHARS = 8000


def _prepare_image(image: Image.Image) -> str:
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMAGE_SIDE:
        scale = _MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def _is_cohere_embed(model_id: str) -> bool:
    return "cohere" in model_id and "embed" in model_id


def _invoke_cohere(
    model_id: str,
    dim: int,
    *,
    input_text: Optional[str] = None,
    input_image_b64: Optional[str] = None,
    input_type: str,
) -> np.ndarray:
    body: dict = {
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": dim,
    }
    if input_text is not None:
        body["texts"] = [input_text]
    if input_image_b64 is not None:
        body["images"] = [f"data:image/png;base64,{input_image_b64}"]

    response = bedrock_invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="*/*",
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    embedding = np.array(result["embeddings"]["float"][0], dtype=np.float32)
    return _normalize(embedding)


class BedrockEmbedder:
    """Multimodal embedder for images and cross-modal text queries."""

    def __init__(self, model_id: Optional[str] = None, dim: Optional[int] = None) -> None:
        self.model_id = model_id or SETTINGS.embedding_model
        self._dim = dim or SETTINGS.embedding_dim
        self._cohere = _is_cohere_embed(self.model_id)

    @property
    def dim(self) -> int:
        return self._dim

    def _invoke_titan(
        self, *, input_text: Optional[str] = None, input_image_b64: Optional[str] = None
    ) -> np.ndarray:
        body: dict = {"embeddingConfig": {"outputEmbeddingLength": self._dim}}
        if input_text is not None:
            body["inputText"] = input_text
        if input_image_b64 is not None:
            body["inputImage"] = input_image_b64

        response = bedrock_invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        embedding = np.array(result["embedding"], dtype=np.float32)
        return _normalize(embedding)

    def _invoke(self, *, input_text: Optional[str] = None, input_image_b64: Optional[str] = None) -> np.ndarray:
        if self._cohere:
            input_type = "search_query" if input_text is not None else "search_document"
            return _invoke_cohere(
                self.model_id,
                self._dim,
                input_text=input_text,
                input_image_b64=input_image_b64,
                input_type=input_type,
            )
        return self._invoke_titan(input_text=input_text, input_image_b64=input_image_b64)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_image_with_context(
        self, image: Image.Image, context: Optional[str] = None
    ) -> np.ndarray:
        """Embed image; when context is set, include slide/PDF text (Titan only)."""
        ctx = (context or "").strip()
        b64 = _prepare_image(image)
        if ctx and not self._cohere:
            return self._invoke(input_text=ctx, input_image_b64=b64)
        return self._invoke(input_image_b64=b64)

    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        return np.stack([self._invoke(input_image_b64=_prepare_image(img)) for img in images])

    def embed_text(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._invoke(input_text=t) for t in texts])


class BedrockTextEmbedder:
    """Text embedder for the caption-text dense lane (Titan Text v2 or Cohere)."""

    def __init__(self, model_id: Optional[str] = None, dim: Optional[int] = None) -> None:
        self.model_id = model_id or SETTINGS.text_embedding_model
        self._dim = dim or SETTINGS.text_embedding_dim
        self._cohere = _is_cohere_embed(self.model_id)

    @property
    def dim(self) -> int:
        return self._dim

    def _invoke_titan_text(self, text: str) -> np.ndarray:
        body = {"inputText": text, "dimensions": self._dim, "normalize": True}
        response = bedrock_invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        embedding = np.array(result["embedding"], dtype=np.float32)
        return _normalize(embedding)

    def _embed(self, text: str, *, input_type: str) -> np.ndarray:
        text = (text or "").strip()[:_MAX_TEXT_CHARS]
        if not text:
            raise ValueError("cannot embed empty text")
        if self._cohere:
            return _invoke_cohere(self.model_id, self._dim, input_text=text, input_type=input_type)
        return self._invoke_titan_text(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text, input_type="search_query")

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text, input_type="search_document")


_embedder: Optional[BedrockEmbedder] = None
_text_embedder: Optional[BedrockTextEmbedder] = None


def get_embedder() -> BedrockEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BedrockEmbedder()
    return _embedder


def get_text_embedder() -> BedrockTextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = BedrockTextEmbedder()
    return _text_embedder
