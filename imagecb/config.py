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
    # Gemini / Nano Banana: prefer GEMINI_API_KEY for local; production loads from
    # Secrets Manager (see gemini_secret_name / gemini_secret_region).
    gemini_api_key: Optional[str] = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_secret_name: str = field(
        default_factory=lambda: _env("GEMINI_SECRET_NAME", "gemini") or "gemini"
    )
    gemini_secret_region: str = field(
        default_factory=lambda: _env("GEMINI_SECRET_REGION")
        or _env("AWS_REGION", "us-east-1")
        or "us-east-1"
    )
    nano_banana_model: str = field(
        default_factory=lambda: _env("NANO_BANANA_MODEL", "gemini-3.1-flash-image")
        or "gemini-3.1-flash-image"
    )

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

    # Visual dense lane: cross-modal retrieval over Titan image vectors.
    # Disable (with the caption-text lane on) for text-embedding-only ranking.
    visual_lane_enabled: bool = field(
        default_factory=lambda: (_env("VISUAL_LANE_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
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
    s3_presign_expiry_sec: int = field(
        default_factory=lambda: int(_env("S3_PRESIGN_EXPIRY_SEC", "3600") or "3600")
    )
    # Host the browser must reach for presigned PUTs. When set (e.g. MinIO
    # playground: http://localhost:9000), only URL signing uses this endpoint;
    # server-side S3 calls still follow AWS_ENDPOINT_URL_S3 / the default chain.
    s3_presign_endpoint_url: Optional[str] = field(
        default_factory=lambda: (_env("S3_PRESIGN_ENDPOINT_URL") or "").rstrip("/") or None
    )

    # CORS: comma-separated browser origins allowed to call the API.
    # Defaults cover local dev (Vite + same-host ports).
    cors_allow_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in (
                _env(
                    "CORS_ALLOW_ORIGINS",
                    "http://127.0.0.1:5173,http://localhost:5173,"
                    "http://127.0.0.1:8080,http://localhost:8080,"
                    "http://127.0.0.1:8081,http://localhost:8081",
                )
                or ""
            ).split(",")
            if o.strip()
        )
    )

    # OCR
    tesseract_cmd: Optional[str] = field(default_factory=lambda: _env("TESSERACT_CMD"))
    # Which source supplies verbatim visible-text for the embedded caption
    # document and BM25: "vlm" (VLM readable_text only; skips Tesseract -
    # measured faster and more accurate), "both" (adds Tesseract; consider for
    # corpora dominated by dense small text), or "tesseract" (Tesseract only).
    ocr_source: str = field(
        default_factory=lambda: (_env("OCR_SOURCE", "vlm") or "vlm").lower()
    )

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
    # Follow-up suggestion generation pool: must scale with expected
    # concurrent chats or responses queue up to 15s behind each other.
    follow_up_workers: int = field(
        default_factory=lambda: int(_env("FOLLOW_UP_WORKERS", "8") or "8")
    )
    # Concurrent per-slide searches during deck suggest (each is ~3 Bedrock
    # calls); bounded by BEDROCK_MAX_CONCURRENT globally.
    deck_search_workers: int = field(
        default_factory=lambda: int(_env("DECK_SEARCH_WORKERS", "4") or "4")
    )
    # TTL for the cached index-health report served by /api/status and
    # /api/ready (the uncached assessment is O(corpus) including S3 HEADs).
    status_cache_ttl_sec: int = field(
        default_factory=lambda: int(_env("STATUS_CACHE_TTL_SEC", "30") or "30")
    )
    # Start embedding the raw query while the parse LLM runs; reused when the
    # parsed dense query equals the raw text (the common case), cutting
    # time-to-results by roughly one embed round-trip.
    speculative_query_embed: bool = field(
        default_factory=lambda: (_env("SPECULATIVE_QUERY_EMBED", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    # Per-client-IP requests/minute across the LLM-backed public endpoints
    # (chat, similar, deck). 0 disables limiting.
    llm_rate_limit_per_minute: int = field(
        default_factory=lambda: int(_env("LLM_RATE_LIMIT_PER_MINUTE", "30") or "30")
    )
    # Chat session store: idle sessions are evicted after the TTL, and the
    # store is capped (LRU) so anonymous traffic can't grow memory unbounded.
    session_ttl_sec: int = field(
        default_factory=lambda: int(_env("SESSION_TTL_SEC", "7200") or "7200")
    )
    session_max_count: int = field(
        default_factory=lambda: int(_env("SESSION_MAX_COUNT", "1000") or "1000")
    )
    # Ingest performance
    ingest_workers: int = field(
        default_factory=lambda: int(_env("INGEST_WORKERS", "2") or "2")
    )
    ingest_max_image_side: int = field(
        default_factory=lambda: int(_env("INGEST_MAX_IMAGE_SIDE", "1024") or "1024")
    )
    thumb_max_side: int = field(
        default_factory=lambda: int(_env("THUMB_MAX_SIDE", "256") or "256")
    )
    thumb_jpeg_quality: int = field(
        default_factory=lambda: int(_env("THUMB_JPEG_QUALITY", "80") or "80")
    )
    ingest_batch_upsert: int = field(
        default_factory=lambda: int(_env("INGEST_BATCH_UPSERT", "16") or "16")
    )
    ingest_batch_size: int = field(
        default_factory=lambda: int(_env("INGEST_BATCH_SIZE", "0") or "0")
    )
    # Max source files processed per worker claim; remainder is requeued on the
    # same job_id so one large folder upload runs as short memory-friendly cycles.
    ingest_job_chunk_size: int = field(
        default_factory=lambda: int(_env("INGEST_JOB_CHUNK_SIZE", "25") or "25")
    )
    ingest_image_timeout_sec: int = field(
        default_factory=lambda: int(_env("INGEST_IMAGE_TIMEOUT_SEC", "300") or "300")
    )
    ingest_timing_log: bool = field(
        default_factory=lambda: (_env("INGEST_TIMING_LOG", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    query_timing_log: bool = field(
        default_factory=lambda: (_env("QUERY_TIMING_LOG", "true") or "true").lower()
        in ("1", "true", "yes", "on")
    )
    query_timing_persist: bool = field(
        default_factory=lambda: (_env("QUERY_TIMING_PERSIST", "true") or "true").lower()
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

    # S3 index durability: checkpoint local SQLite/Chroma to S3 and restore on boot
    # when the ephemeral host volume comes back empty.
    index_checkpoint_enabled: bool = field(
        default_factory=lambda: (
            _env("INDEX_CHECKPOINT_ENABLED")
            or (
                "true"
                if (_env("BLOB_STORAGE_BACKEND", "local") or "local").lower() == "s3"
                else "false"
            )
        ).lower()
        in ("1", "true", "yes", "on")
    )
    index_checkpoint_every_n: int = field(
        default_factory=lambda: int(_env("INDEX_CHECKPOINT_EVERY_N", "10") or "10")
    )
    index_auto_restore_on_startup: bool = field(
        default_factory=lambda: (_env("INDEX_AUTO_RESTORE_ON_STARTUP", "true") or "true").lower()
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
    # Proactive short-query visual route: for 1-2 word visual queries, skip the
    # Cohere rerank entirely and rank by pure Titan visual similarity.

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

    # High-confidence lexical pre-check: fraction of query content tokens that must
    # literally appear in a candidate's text for it to count as a confident keyword
    # hit (routes to the fused/rerank path instead of pure visual).
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
        (self.image_cache_dir / "thumbs").mkdir(parents=True, exist_ok=True)
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
