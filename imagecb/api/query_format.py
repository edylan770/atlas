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
