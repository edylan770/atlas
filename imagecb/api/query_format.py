"""Serialize QuerySpec and format API errors."""

from __future__ import annotations

from imagecb.api.schemas import ParsedQueryOut, SourceFiltersOut, TimeFilterOut
from imagecb.config import SETTINGS
from imagecb.retrieval.query_parser import QuerySpec

_RERANK_SUPPORTED_REGIONS = (
    "us-east-1",
    "us-west-2",
    "ca-central-1",
    "eu-central-1",
    "ap-northeast-1",
)


def spec_to_parsed_query(
    spec: QuerySpec | None,
    *,
    interpretation_notes: list[str] | None = None,
) -> ParsedQueryOut | None:
    if spec is None:
        return None
    tf = spec.time_filter
    return ParsedQueryOut(
        semantic_query=spec.semantic_query,
        must_have_keywords=list(spec.must_have_keywords),
        must_avoid_keywords=list(spec.must_avoid_keywords),
        source_filters=SourceFiltersOut(
            file_types=list(spec.source_filters.file_types),
            asset_types=list(spec.source_filters.asset_types),
            filename_contains=list(spec.source_filters.filename_contains),
            authors=list(spec.source_filters.authors),
        ),
        time_filter=TimeFilterOut(
            after=tf.after.date().isoformat() if tf.after else None,
            before=tf.before.date().isoformat() if tf.before else None,
        ),
        is_refinement=spec.is_refinement,
        top_k=spec.top_k,
        interpretation_notes=list(interpretation_notes or []),
    )


def format_query_error(exc: BaseException) -> str:
    msg = str(exc)
    name = type(exc).__name__
    lower = msg.lower()
    needs_rerank_hint = (
        "rerank" in lower
        or "model identifier is invalid" in lower
        or name in ("ValidationException", "NoCredentialsError", "AccessDeniedException")
    )
    if not needs_rerank_hint:
        return f"Error: {exc}"
    lines = [f"Error: {exc}", ""]
    lines.append(
        f"Reranker config: `AWS_REGION={SETTINGS.aws_region}`, "
        f"`RERANKER_MODEL={SETTINGS.reranker_model}`."
    )
    if SETTINGS.aws_region not in _RERANK_SUPPORTED_REGIONS:
        regions = ", ".join(_RERANK_SUPPORTED_REGIONS)
        lines.append(
            f"Cohere Rerank 3.5 is not available in `{SETTINGS.aws_region}`. "
            f"Set `AWS_REGION` in `.env` to one of: {regions}."
        )
    lines.append(
        "Enable `cohere.rerank-v3-5:0` in the Bedrock console for that region, "
        "or run `python -m imagecb.cli validate-reranker` to test access."
    )
    return "\n".join(lines)


_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "Throttling"}
_AUTH_CODES = {
    "ExpiredTokenException",
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "AccessDeniedException",
    "NoCredentialsError",
}


def _aws_error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return type(exc).__name__


def classify_model_error(exc: BaseException) -> tuple[int, str]:
    """Map a model-backend failure to (status_code, user-facing detail).

    Credentials expiring must surface loudly: with fail-soft retrieval lanes the
    old behavior was silent 0%-match results that looked like bad search.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        code = _aws_error_code(node)
        text = f"{code} {node}".lower()
        if code in _THROTTLE_CODES or "too many requests" in text:
            return 429, (
                "The model backend is rate limited right now; wait a moment and retry."
            )
        if code in _AUTH_CODES or "token has expired" in text or "expired" in text and "token" in text:
            return 503, (
                "The model backend rejected our credentials (expired or invalid "
                "AWS/Bedrock token). Searches cannot run until an administrator "
                "refreshes them."
            )
        if code in ("ServiceUnavailableException", "ModelNotReadyException"):
            return 503, "The model backend is temporarily unavailable; retry shortly."
        node = node.__cause__ or node.__context__
    return 500, format_query_error(exc)
