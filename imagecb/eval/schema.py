"""Pydantic models for the search evaluation golden set."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


SimilarityAxis = Literal["balanced", "subject", "style", "layout"]
EvalMode = Literal["chat", "retrieval", "similar"]


class TextCase(BaseModel):
    id: str
    query: str
    relevant_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    notes: Optional[str] = None
    template: bool = False
    short_query: bool = False

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("case id must not be blank")
        return value


class SimilarCase(BaseModel):
    id: str
    image_id: str
    relevant_ids: List[str] = Field(default_factory=list)
    similarity_axis: SimilarityAxis = "balanced"
    top_k: int = Field(default=10, ge=1, le=50)
    notes: Optional[str] = None
    template: bool = False

    @field_validator("id", "image_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be blank")
        return value


class GoldenSet(BaseModel):
    version: int = 1
    text_cases: List[TextCase] = Field(default_factory=list)
    similar_cases: List[SimilarCase] = Field(default_factory=list)
