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
    hubness_path: Path = field(default_factory=lambda: _abspath(_env("HUBNESS_PATH", "./data/hubness.pkl") or "./data/hubness.pkl"))
    blob_storage_backend: str = field(
        default_factory=lambda: (_env("BLOB_STORAGE_BACKEND", "local") or "local").lower()
    )
    s3_bucket: Optional[str] = field(default_factory=lambda: _env("S3_BUCKET"))
    s3_prefix: str = field(
        default_factory=lambda: (_env("S3_PREFIX", "imagecb") or "imagecb").strip("/")
    )
    s3_region: str = field(
        default_factory=lambda: _env("S3_REGION")
        or _env("AWS_REGION", "us-east-1")
        or "us-east-1"
    )
    s3_connect_timeout: int = field(
        default_factory=lambda: int(_env("S3_CONNECT_TIMEOUT", "10") or "10")
    )
    s3_read_timeout: int = field(
        default_factory=lambda: int(_env("S3_READ_TIMEOUT", "30") or "30")
    )
    s3_max_retries: int = field(
        default_factory=lambda: int(_env("S3_MAX_RETRIES", "3") or "3")
    )

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
    ingest_timing_log: bool = field(
        default_factory=lambda: (_env("INGEST_TIMING_LOG", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )

    # Optional bootstrap corpus: directory ingested in a background thread on
    # startup when the index is empty. Used to seed a small smoke-test set on a
    # fresh deploy (e.g. ECS). Empty disables the feature.
    bootstrap_corpus_dir: Optional[str] = field(
        default_factory=lambda: _env("BOOTSTRAP_CORPUS_DIR")
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

    # Visual fallback: when Cohere rerank's top result is weak, re-rank the
    # chat turn's candidates by pure Titan visual similarity (Candidate.dense_score).
    visual_fallback_enabled: bool = field(
        default_factory=lambda: (_env("VISUAL_FALLBACK_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    visual_fallback_max_display_percent: int = field(
        default_factory=lambda: int(_env("VISUAL_FALLBACK_MAX_DISPLAY_PERCENT", "20") or "20")
    )
    # Proactive short-query visual route: for 1-2 word visual queries, skip the
    # Cohere rerank entirely and rank by pure Titan visual similarity.
    visual_short_query_enabled: bool = field(
        default_factory=lambda: (_env("VISUAL_SHORT_QUERY_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )

    # CSLS-style hubness correction for the visual lane. Demotes images that are
    # universally close to any query (embedding hubs) so short-query ranking
    # responds to the actual query instead of vector-space centrality.
    hubness_correction_enabled: bool = field(
        default_factory=lambda: (_env("HUBNESS_CORRECTION_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    hubness_knn: int = field(
        default_factory=lambda: int(_env("HUBNESS_KNN", "10") or "10")
    )
    hubness_penalty_weight: float = field(
        default_factory=lambda: float(_env("HUBNESS_PENALTY_WEIGHT", "0.5") or "0.5")
    )

    # Soft confidence gating for visual-only results: when the top adjusted score
    # is below the floor or barely separated from the rest, the matches are flagged
    # as weak (results are still returned).
    visual_confidence_floor: float = field(
        default_factory=lambda: float(_env("VISUAL_CONFIDENCE_FLOOR", "0.5") or "0.5")
    )
    visual_confidence_margin: float = field(
        default_factory=lambda: float(_env("VISUAL_CONFIDENCE_MARGIN", "0.03") or "0.03")
    )

    # High-confidence lexical pre-check: fraction of query content tokens that must
    # literally appear in a candidate's text for it to count as a confident keyword
    # hit (routes to the fused/rerank path instead of pure visual).
    lexical_high_confidence_coverage: float = field(
        default_factory=lambda: float(
            _env("LEXICAL_HIGH_CONFIDENCE_COVERAGE", "0.9") or "0.9"
        )
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

    def validate_blob_storage(self) -> None:
        if self.blob_storage_backend not in {"local", "s3"}:
            raise ValueError("BLOB_STORAGE_BACKEND must be 'local' or 's3'")
        if self.blob_storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("S3_BUCKET is required when BLOB_STORAGE_BACKEND=s3")


SETTINGS = Settings()
SETTINGS.validate_blob_storage()
SETTINGS.ensure_dirs()
