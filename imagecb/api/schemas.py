"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProvenanceOut(BaseModel):
    source_name: str
    source_type: str
    slide_index: Optional[int] = None
    page_index: Optional[int] = None
    modified: Optional[str] = None
    author: Optional[str] = None
    chips: List[str] = Field(default_factory=list)


class ResultCardOut(BaseModel):
    rank: int
    image_id: str
    image_url: str
    provenance: ProvenanceOut
    caption: str
    match_hint: Optional[str] = None
    match_percent: int = 0
    has_image_file: bool = True
    image_name: str = ""
    use_case: str = ""
    tags: List[str] = Field(default_factory=list)
    recommended_cases: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source_location: str = ""
    source_path: Optional[str] = None


class CatalogItemOut(BaseModel):
    image_id: str
    image_url: str
    image_name: str
    use_case: str = ""
    tags: List[str] = Field(default_factory=list)
    recommended_cases: List[str] = Field(default_factory=list)
    caption: str = ""
    source_name: str = ""


class CorpusCatalogResponse(BaseModel):
    items: List[CatalogItemOut] = Field(default_factory=list)
    indexed_count: int = 0


class SourceFiltersOut(BaseModel):
    file_types: List[str] = Field(default_factory=list)
    filename_contains: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)


class TimeFilterOut(BaseModel):
    after: Optional[str] = None
    before: Optional[str] = None


class ParsedQueryOut(BaseModel):
    semantic_query: str = ""
    must_have_keywords: List[str] = Field(default_factory=list)
    must_avoid_keywords: List[str] = Field(default_factory=list)
    source_filters: SourceFiltersOut = Field(default_factory=SourceFiltersOut)
    time_filter: TimeFilterOut = Field(default_factory=TimeFilterOut)
    is_refinement: bool = False
    top_k: int = 10
    interpretation_notes: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    min_match_percent: int = Field(default=0, ge=0, le=100)


class SimilarRequest(BaseModel):
    image_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    min_match_percent: int = Field(default=0, ge=0, le=100)
    similarity_axis: str = Field(default="balanced")


class SimilarResponse(BaseModel):
    session_id: Optional[str] = None
    assistant_message: str
    results: List[ResultCardOut]
    parsed_query: Optional[ParsedQueryOut] = None
    search_event_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    results: List[ResultCardOut]
    parsed_query: Optional[ParsedQueryOut] = None
    search_event_id: Optional[str] = None


class InteractionRequest(BaseModel):
    search_event_id: str
    image_id: str
    interaction_type: str  # view | download | similar
    rank: Optional[int] = None


class InteractionResponse(BaseModel):
    interaction_id: str
    ok: bool = True


class SessionResetRequest(BaseModel):
    session_id: str


class SessionResetResponse(BaseModel):
    session_id: str


class SuggestionsRequest(BaseModel):
    recent_titles: List[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=4, ge=2, le=8)


class SuggestionsResponse(BaseModel):
    suggestions: List[str]
    cached: bool = False


class StatusResponse(BaseModel):
    indexed_count: int


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestResponse(BaseModel):
    message: str
    indexed_count: int
    stats: dict = Field(default_factory=dict)


class SlideSuggestionOut(BaseModel):
    slide_index: int
    title: Optional[str] = None
    body_preview: str = ""
    notes_preview: str = ""
    content_hash: str = ""
    status: str  # image_needed | no_image_needed
    description: str = ""
    reason: str = ""
    results: List[ResultCardOut] = Field(default_factory=list)
    llm_cached: bool = False
    search_cached: bool = False


class DeckSuggestResponse(BaseModel):
    deck_hash: str
    filename: str
    slides: List[SlideSuggestionOut] = Field(default_factory=list)
    deck_cached: bool = False
    llm_batches: int = 0


class DeckForceRequest(BaseModel):
    deck_hash: str
    slide_index: int = Field(ge=1)
    top_k: int = Field(default=10, ge=1, le=30)
    min_match_percent: int = Field(default=0, ge=0, le=100)


class DeckForceResponse(BaseModel):
    slide: SlideSuggestionOut
