"""Bedrock reranker wrapper.

Uses Cohere Rerank via ``bedrock-runtime`` ``invoke_model`` so auth matches
the rest of the stack (including ``AWS_BEARER_TOKEN_BEDROCK``). The separate
``bedrock-agent-runtime`` ``rerank`` API only supports standard IAM credentials.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from imagecb.config import SETTINGS
from imagecb.models.bedrock_client import bedrock_invoke_model

_MAX_DOC_CHARS = 8000


class BedrockReranker:
    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or SETTINGS.reranker_model

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        if not docs:
            return []

        documents = [(d or "")[:_MAX_DOC_CHARS] for d in docs]
        body = json.dumps(
            {
                "query": query,
                "documents": documents,
                "top_n": len(docs),
                "api_version": 2,
            }
        )
        response = bedrock_invoke_model(
            modelId=self.model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())

        scores = [0.0] * len(docs)
        for item in result.get("results", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(docs):
                rel = item.get("relevance_score", item.get("relevanceScore", 0.0))
                scores[idx] = float(rel)
        return scores


_reranker: Optional[BedrockReranker] = None


def get_reranker() -> BedrockReranker:
    global _reranker
    if _reranker is None:
        _reranker = BedrockReranker()
    return _reranker
