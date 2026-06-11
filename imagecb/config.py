"""Environment-driven configuration.

Loaded once at import time. All paths are resolved to absolute paths so
modules don't depend on the process working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _abspath(p: str | os.PathLike[str]) -> Path:
    return Path(p).expanduser().resolve()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


@dataclass(frozen=True)
class Settings:
    # Providers
    vlm_provider: str = field(default_factory=lambda: _env("VLM_PROVIDER", "bedrock") or "bedrock")
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "bedrock") or "bedrock")

    # API keys (only needed for cloud providers)
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))

    # AWS region for Bedrock. Bedrock auth (AWS_BEARER_TOKEN_BEDROCK or standard
    # AWS credentials) is resolved implicitly by boto3 from the environment.
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1") or "us-east-1")

    # Caption / query-parser models. Defaults assume Bedrock cross-region
    # inference profiles; override per provider via env.
    vlm_model: str = field(
        default_factory=lambda: _env("VLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    # Bedrock embedding / reranking models
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "amazon.titan-embed-image-v1")
        or "amazon.titan-embed-image-v1"
    )
    embedding_dim: int = field(
        default_factory=lambda: int(_env("EMBEDDING_DIM", "1024") or "1024")
    )
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cohere.rerank-v3-5:0")
        or "cohere.rerank-v3-5:0"
    )

    # Caption-text dense lane: text-to-text retrieval over caption documents.
    caption_text_lane_enabled: bool = field(
        default_factory=lambda: (_env("CAPTION_TEXT_LANE_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    text_embedding_model: str = field(
        default_factory=lambda: _env("TEXT_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        or "amazon.titan-embed-text-v2:0"
    )
    text_embedding_dim: int = field(
        default_factory=lambda: int(_env("TEXT_EMBEDDING_DIM", "1024") or "1024")
    )

    # Storage paths
    data_dir: Path = field(default_factory=lambda: _abspath(_env("DATA_DIR", "./data") or "./data"))
    chroma_dir: Path = field(default_factory=lambda: _abspath(_env("CHROMA_DIR", "./data/chroma") or "./data/chroma"))
    sqlite_path: Path = field(default_factory=lambda: _abspath(_env("SQLITE_PATH", "./data/imagecb.db") or "./data/imagecb.db"))
    image_cache_dir: Path = field(default_factory=lambda: _abspath(_env("IMAGE_CACHE_DIR", "./data/images") or "./data/images"))
    uploads_dir: Path = field(
        default_factory=lambda: _abspath(
            _env("UPLOADS_DIR")
            or str(_abspath(_env("DATA_DIR", "./data") or "./data") / "uploads")
        )
    )
    bm25_path: Path = field(default_factory=lambda: _abspath(_env("BM25_PATH", "./data/bm25.pkl") or "./data/bm25.pkl"))

    # OCR
    tesseract_cmd: Optional[str] = field(default_factory=lambda: _env("TESSERACT_CMD"))

    # Tunables
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 50
    short_query_max_tokens: int = field(
        default_factory=lambda: int(_env("SHORT_QUERY_MAX_TOKENS", "2") or "2")
    )
    short_query_rerank_top_n: int = field(
        default_factory=lambda: int(_env("SHORT_QUERY_RERANK_TOP_N", "100") or "100")
    )
    short_query_retrieval_top_k: int = field(
        default_factory=lambda: int(_env("SHORT_QUERY_RETRIEVAL_TOP_K", "100") or "100")
    )
    embed_context_max_chars: int = field(
        default_factory=lambda: int(_env("EMBED_CONTEXT_MAX_CHARS", "480") or "480")
    )
    default_top_k: int = 10
    asset_type_rerank_boost: float = field(
        default_factory=lambda: float(_env("ASSET_TYPE_RERANK_BOOST", "1.10") or "1.10")
    )
    enable_conversational_llm: bool = field(
        default_factory=lambda: (_env("ENABLE_CONVERSATIONAL_LLM", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    suggestions_cache_ttl_sec: int = field(
        default_factory=lambda: int(_env("SUGGESTIONS_CACHE_TTL_SEC", "300") or "300")
    )
    suggestions_limit: int = field(
        default_factory=lambda: int(_env("SUGGESTIONS_LIMIT", "4") or "4")
    )
    follow_up_suggestions_limit: int = field(
        default_factory=lambda: int(_env("FOLLOW_UP_SUGGESTIONS_LIMIT", "3") or "3")
    )
    enable_follow_up_suggestions: bool = field(
        default_factory=lambda: (_env("ENABLE_FOLLOW_UP_SUGGESTIONS", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    # Ingest performance
    ingest_workers: int = field(
        default_factory=lambda: int(_env("INGEST_WORKERS", "4") or "4")
    )
    ingest_max_image_side: int = field(
        default_factory=lambda: int(_env("INGEST_MAX_IMAGE_SIDE", "1024") or "1024")
    )
    ingest_batch_upsert: int = field(
        default_factory=lambda: int(_env("INGEST_BATCH_UPSERT", "16") or "16")
    )
    ingest_batch_size: int = field(
        default_factory=lambda: int(_env("INGEST_BATCH_SIZE", "0") or "0")
    )
    ingest_image_timeout_sec: int = field(
        default_factory=lambda: int(_env("INGEST_IMAGE_TIMEOUT_SEC", "300") or "300")
    )

    # Post-ingest index repair
    post_ingest_repair_enabled: bool = field(
        default_factory=lambda: (_env("POST_INGEST_REPAIR_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    post_ingest_repair_include_weak: bool = field(
        default_factory=lambda: (_env("POST_INGEST_REPAIR_INCLUDE_WEAK", "false") or "false").lower()
        in ("1", "true", "yes", "on")
    )
    post_ingest_repair_reindex_vectors: bool = field(
        default_factory=lambda: (_env("POST_INGEST_REPAIR_REINDEX_VECTORS", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )

    # Index consistency (safe cross-store reconcile)
    index_reconcile_on_startup: bool = field(
        default_factory=lambda: (_env("INDEX_RECONCILE_ON_STARTUP", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    index_reconcile_after_ingest: bool = field(
        default_factory=lambda: (_env("INDEX_RECONCILE_AFTER_INGEST", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )

    # Bedrock API resilience
    bedrock_max_concurrent: int = field(
        default_factory=lambda: int(_env("BEDROCK_MAX_CONCURRENT", "2") or "2")
    )
    bedrock_read_timeout: int = field(
        default_factory=lambda: int(_env("BEDROCK_READ_TIMEOUT", "120") or "120")
    )
    bedrock_connect_timeout: int = field(
        default_factory=lambda: int(_env("BEDROCK_CONNECT_TIMEOUT", "10") or "10")
    )
    bedrock_max_retries: int = field(
        default_factory=lambda: int(_env("BEDROCK_MAX_RETRIES", "6") or "6")
    )

    # Admin / telemetry
    admin_api_key: str = field(default_factory=lambda: _env("ADMIN_API_KEY", "") or "")
    weak_result_score_threshold: float = field(
        default_factory=lambda: float(_env("WEAK_RESULT_SCORE_THRESHOLD", "0.25") or "0.25")
    )
    duplicate_similarity_threshold: float = field(
        default_factory=lambda: float(_env("DUPLICATE_SIMILARITY_THRESHOLD", "0.95") or "0.95")
    )
    result_deduplicate_enabled: bool = field(
        default_factory=lambda: (_env("RESULT_DEDUPLICATE_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    result_deduplicate_similarity_threshold: float = field(
        default_factory=lambda: float(
            _env("RESULT_DEDUPLICATE_SIMILARITY_THRESHOLD", "0.98") or "0.98"
        )
    )

    # Deck slide-aware suggestion
    deck_cache_dir: Path = field(
        default_factory=lambda: _abspath(
            _env("DECK_CACHE_DIR")
            or str(_abspath(_env("DATA_DIR", "./data") or "./data") / "deck_cache")
        )
    )
    deck_llm_batch_size: int = field(
        default_factory=lambda: int(_env("DECK_LLM_BATCH_SIZE", "10") or "10")
    )
    deck_max_slides: int = field(
        default_factory=lambda: int(_env("DECK_MAX_SLIDES", "200") or "200")
    )
    deck_max_chars_per_slide: int = field(
        default_factory=lambda: int(_env("DECK_MAX_CHARS_PER_SLIDE", "6000") or "6000")
    )
    deck_cache_enabled: bool = field(
        default_factory=lambda: (_env("DECK_CACHE_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    deck_max_upload_bytes: int = field(
        default_factory=lambda: int(_env("DECK_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)) or str(50 * 1024 * 1024))
    )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.deck_cache_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
SETTINGS.ensure_dirs()
