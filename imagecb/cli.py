"""Typer-based command-line entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from imagecb.config import SETTINGS

app = typer.Typer(add_completion=False, help="Conversational multimodal image retrieval.")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True, help="File or directory to ingest."),
    skip_caption: bool = typer.Option(
        False,
        "--skip-caption",
        help="Skip the VLM caption step (useful for an offline dry-run; quality of search will drop).",
    ),
    skip_ocr: bool = typer.Option(
        False,
        "--skip-ocr",
        help="Skip Tesseract OCR during ingest (faster; OCR fields will be empty).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-process images already in the index (re-cache PNGs, re-caption, refresh vectors).",
    ),
    workers: int = typer.Option(
        SETTINGS.ingest_workers,
        "--workers",
        min=1,
        help="Parallel ingest workers (Bedrock calls are I/O-bound; try 4-8).",
    ),
    max_image_side: int = typer.Option(
        SETTINGS.ingest_max_image_side,
        "--max-image-side",
        min=256,
        help="Longest edge sent to the VLM for captioning (smaller is faster).",
    ),
    batch_size: int = typer.Option(
        SETTINGS.ingest_batch_size,
        "--batch-size",
        min=0,
        help="Process files in batches of N (0 = single run). Recommended 25 for large corpora.",
    ),
    defer_bm25: bool = typer.Option(
        True,
        "--defer-bm25/--no-defer-bm25",
        help="When batching, rebuild BM25 once at the end instead of per batch.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Walk PATH and ingest every supported image into the index."""
    _configure_logging(verbose)
    from imagecb.ingest import ingest_root

    stats = ingest_root(
        path,
        skip_caption=skip_caption,
        skip_ocr=skip_ocr,
        force=force,
        workers=workers,
        max_image_side=max_image_side,
        batch_size=batch_size if batch_size > 0 else None,
        defer_bm25=defer_bm25,
    )
    elapsed = stats.get("elapsed_sec", 0)
    rate = ""
    processed = stats["images_added"] + stats["images_updated"]
    if elapsed > 0 and processed > 0:
        rate = f" ({processed / elapsed:.2f} images/s)"
    batches = stats.get("batches", 0)
    batch_note = f" batches={batches}" if batches else ""
    typer.echo(
        f"Done in {elapsed}s{rate}. workers={stats.get('workers', workers)}{batch_note} "
        f"files={stats['files']} images_seen={stats['images_seen']} "
        f"added={stats['images_added']} updated={stats['images_updated']} "
        f"duplicates={stats['skipped_duplicates']} errors={stats['errors']}"
    )


@app.command(name="serve-web")
def serve_web(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8080, help="Bind port."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Launch the FastAPI web UI (no npm required)."""
    _configure_logging(verbose)
    from imagecb.api.server import launch
    from imagecb.api.static_ui import format_serve_web_urls, warn_if_deck_route_missing

    for line in format_serve_web_urls(host=host, port=port):
        typer.echo(line)
    deck_warn = warn_if_deck_route_missing()
    if deck_warn:
        typer.echo(deck_warn, err=True)
    launch(host=host, port=port)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(7860, help="Bind port."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Launch the legacy Gradio chat UI."""
    _configure_logging(verbose)
    from imagecb.app import launch

    launch(host=host, port=port, share=share)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show affected source files and image IDs."),
) -> None:
    """Print a quick summary of the current index."""
    from imagecb.repair import assess_index_health

    SETTINGS.ensure_dirs()
    report = assess_index_health(include_weak=True)

    typer.echo(
        f"SQLite records: {report.total_records} | Chroma vectors: {report.chroma_vectors} | "
        f"Caption-text vectors: {report.text_vector_count} | BM25 docs: {report.bm25_doc_count} | "
        f"stores_in_sync: {report.stores_in_sync} | healthy: {report.is_healthy}"
    )
    typer.echo(
        f"Chroma dir: {SETTINGS.chroma_dir} | SQLite: {SETTINGS.sqlite_path}"
    )
    typer.echo(
        f"Missing cached PNGs: {report.missing_cache_count} | Captions failed: {report.failed_caption_count} | "
        f"Captions weak: {report.weak_caption_count} | Needs regeneration: {report.needs_regeneration_count} | "
        f"Missing Chroma vectors: {report.missing_chroma_count}"
    )
    typer.echo(
        f"Orphan Chroma: {report.orphan_chroma_count} | Orphan text vectors: {report.orphan_text_vector_count} | "
        f"Stale soft-deleted in index: {report.stale_deleted_in_chroma_count} | BM25 stale: {report.bm25_stale}"
    )
    typer.echo(
        f"Asset types missing: {report.missing_asset_type_count} / {report.total_records}"
    )
    if SETTINGS.caption_text_lane_enabled:
        typer.echo(
            f"Caption-text vectors missing: {report.missing_text_vector_count} / {report.total_records}"
        )

    has_issues = (
        not report.is_healthy
        or report.weak_caption_count
        or report.missing_asset_type_count
    )
    if not has_issues:
        if verbose:
            typer.echo("Index is healthy.")
        return

    if verbose or has_issues:
        if report.recoverable_source_files:
            typer.echo(
                f"\nRecoverable source files ({len(report.recoverable_source_files)}):"
            )
            for path in report.recoverable_source_files:
                typer.echo(f"  {path}")
        if report.unrecoverable_records:
            typer.echo(
                f"\nUnrecoverable rows ({report.unrecoverable_source_missing_count}) "
                "(missing cache and source file not on disk):"
            )
            for r in report.unrecoverable_records[:10]:
                typer.echo(f"  {r.image_id}  source={r.source_file or '(none)'}")
            if len(report.unrecoverable_records) > 10:
                typer.echo(f"  ... and {len(report.unrecoverable_records) - 10} more")

        def _sample_ids(label: str, records: list, limit: int = 10) -> None:
            if not records:
                return
            ids = [r.image_id for r in records[:limit]]
            suffix = f" ... +{len(records) - limit} more" if len(records) > limit else ""
            typer.echo(f"\n{label} sample image_ids: {', '.join(ids)}{suffix}")

        _sample_ids("Missing cache", report.missing_cache_records)
        _sample_ids("Failed captions", report.failed_caption_records)

        if report.missing_asset_type_ids:
            ids = report.missing_asset_type_ids[:10]
            suffix = (
                f" ... +{report.missing_asset_type_count - 10} more"
                if report.missing_asset_type_count > 10
                else ""
            )
            typer.echo(f"\nMissing asset type sample image_ids: {', '.join(ids)}{suffix}")

        if report.missing_chroma_ids:
            ids = report.missing_chroma_ids[:10]
            suffix = (
                f" ... +{len(report.missing_chroma_ids) - 10} more"
                if len(report.missing_chroma_ids) > 10
                else ""
            )
            typer.echo(f"\nMissing Chroma sample image_ids: {', '.join(ids)}{suffix}")

        if report.orphan_chroma_ids:
            ids = report.orphan_chroma_ids[:10]
            typer.echo(f"\nOrphan Chroma sample image_ids: {', '.join(ids)}")
        if report.bm25_orphan_ids:
            ids = report.bm25_orphan_ids[:10]
            typer.echo(f"\nOrphan BM25 sample image_ids: {', '.join(ids)}")

        typer.echo(
            "\nRun safe reconcile (no Bedrock calls):\n"
            "  python -m imagecb.cli reconcile-index\n"
            "Run targeted repair:\n"
            "  python -m imagecb.cli repair-index\n"
            "Or re-ingest only the affected source files listed above with --force."
        )


@app.command(name="reconcile-index")
def reconcile_index_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be fixed without mutating indexes.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Safe cross-store reconcile: purge orphan vectors and rebuild BM25 when drifted."""
    _configure_logging(verbose)
    from imagecb.repair import assess_index_health, reconcile_index_safe

    before = assess_index_health(include_weak=True)
    typer.echo(
        f"Before: stores_in_sync={before.stores_in_sync} healthy={before.is_healthy} "
        f"orphan_chroma={before.orphan_chroma_count} bm25_stale={before.bm25_stale}"
    )
    stats = reconcile_index_safe(dry_run=dry_run)
    typer.echo(
        f"{'Would purge' if dry_run else 'Purged'} chroma={stats['orphan_chroma_purged']} "
        f"text={stats['orphan_text_purged']} bm25_rebuilt={stats['bm25_rebuilt']}"
    )
    if not dry_run:
        typer.echo(
            f"After: stores_in_sync={stats['stores_in_sync_after']} "
            f"healthy={stats['is_healthy_after']} ({stats['elapsed_sec']}s)"
        )


@app.command(name="validate-reranker")
def validate_reranker_cmd(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Smoke-test Bedrock reranker access for the configured region and model."""
    _configure_logging(verbose)
    from imagecb.models.reranker import get_reranker

    region = SETTINGS.aws_region
    model = SETTINGS.reranker_model
    typer.echo(f"Region: {region}")
    typer.echo(f"Reranker model: {model}")
    try:
        scores = get_reranker().score(
            "cyber security",
            ["A network operations center with security dashboards.", "A beach at sunset."],
        )
        typer.echo(f"OK. Sample scores: {scores}")
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"FAILED: {exc}", fg=typer.colors.RED, err=True)
        msg = str(exc).lower()
        name = type(exc).__name__
        if name == "AccessDeniedException" or "authentication failed" in msg:
            typer.echo(
                "Bedrock rejected the API key for this region. Short-lived "
                "AWS_BEARER_TOKEN_BEDROCK keys are tied to the region where you "
                "created them. After changing AWS_REGION in .env, generate a NEW "
                "key in the Bedrock console for that same region (e.g. us-east-1), "
                "or use standard IAM credentials (aws configure).",
                err=True,
            )
            typer.echo(
                "In the Bedrock console (same region as AWS_REGION), open Model "
                "access and enable cohere.rerank-v3-5:0, your embedding model, "
                "and your Claude model.",
                err=True,
            )
        elif "model identifier is invalid" in msg:
            typer.echo(
                "Cohere Rerank 3.5 (cohere.rerank-v3-5:0) is supported in us-east-1, "
                "us-west-2, ca-central-1, eu-central-1, and ap-northeast-1 (not us-east-2). "
                "Set AWS_REGION to a supported region in .env.",
                err=True,
            )
        else:
            typer.echo(
                "Check AWS_REGION, RERANKER_MODEL, and Bedrock model access in the console.",
                err=True,
            )
        raise typer.Exit(1) from exc


@app.command(name="repair-index")
def repair_index_cmd(
    workers: int = typer.Option(
        SETTINGS.ingest_workers,
        "--workers",
        min=1,
        help="Parallel workers for repair phases.",
    ),
    include_weak: bool = typer.Option(
        False,
        "--include-weak",
        help="Also re-caption rows flagged as weak quality.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Assess and print issues only; do not mutate the index.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Repair missing cached PNGs, failed captions, and missing Chroma vectors."""
    _configure_logging(verbose)
    from imagecb.repair import assess_index_health, repair_index_issues

    if dry_run:
        report = assess_index_health(include_weak=include_weak)
        typer.echo(
            f"SQLite records: {report.total_records} | Chroma vectors: {report.chroma_vectors} | "
            f"BM25 docs: {report.bm25_doc_count} | stores_in_sync: {report.stores_in_sync} | "
            f"healthy: {report.is_healthy}"
        )
        typer.echo(
            f"Missing cache: {report.missing_cache_count} | Failed captions: {report.failed_caption_count} | "
            f"Weak captions: {report.weak_caption_count} | Missing Chroma: {report.missing_chroma_count} | "
            f"Orphan Chroma: {report.orphan_chroma_count} | BM25 stale: {report.bm25_stale} | "
            f"Unrecoverable: {report.unrecoverable_source_missing_count}"
        )
        if report.recoverable_source_files:
            typer.echo(f"\nSource files to re-ingest ({len(report.recoverable_source_files)}):")
            for path in report.recoverable_source_files:
                typer.echo(f"  {path}")
        if report.unrecoverable_records:
            typer.echo(f"\nUnrecoverable ({len(report.unrecoverable_records)}):")
            for r in report.unrecoverable_records[:10]:
                typer.echo(f"  {r.image_id}  source={r.source_file or '(none)'}")
        return

    stats = repair_index_issues(workers=workers, include_weak_captions=include_weak)
    if stats.get("skipped"):
        typer.echo("Index is healthy; no repair needed.")
        return
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"recached={stats.get('cache_recached', 0)} "
        f"captions={stats.get('captions_repaired', 0)} "
        f"vectors={stats.get('vectors_reindexed', 0)} "
        f"unrecoverable={stats.get('unrecoverable', 0)} "
        f"healthy={stats.get('is_healthy', False)}"
    )


@app.command(name="rescan-captions")
def rescan_captions_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-assess caption quality flags on all indexed images (no VLM calls)."""
    _configure_logging(verbose)
    from imagecb.repair import rescan_caption_quality

    stats = rescan_caption_quality()
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"scanned={stats['scanned']} updated={stats['updated']} "
        f"ok={stats['ok']} weak={stats['weak']} failed={stats['failed']}"
    )


@app.command(name="repair-captions")
def repair_captions_cmd(
    workers: int = typer.Option(
        SETTINGS.ingest_workers,
        "--workers",
        min=1,
        help="Parallel VLM workers.",
    ),
    include_weak: bool = typer.Option(
        False,
        "--include-weak",
        help="Also re-caption rows flagged as weak quality.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-run VLM captioning for images that failed or are weak."""
    _configure_logging(verbose)
    from imagecb.repair import repair_failed_captions

    stats = repair_failed_captions(workers=workers, include_weak=include_weak)
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"attempted={stats['attempted']} repaired={stats['repaired']} errors={stats['errors']} "
        f"include_weak={stats.get('include_weak', include_weak)}"
    )


@app.command(name="backfill-asset-types")
def backfill_asset_types_cmd(
    workers: int = typer.Option(
        SETTINGS.ingest_workers,
        "--workers",
        min=1,
        help="Parallel VLM workers.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report how many rows need backfill without calling the VLM.",
    ),
    all_rows: bool = typer.Option(
        False,
        "--all",
        help="Re-caption every active row (use after a taxonomy adjustment).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-caption images to populate asset_type from cached PNGs (no re-upload)."""
    _configure_logging(verbose)
    from imagecb.repair import backfill_asset_types

    stats = backfill_asset_types(workers=workers, dry_run=dry_run, all_rows=all_rows)
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"scanned={stats['scanned']} updated={stats['updated']} "
        f"skipped={stats['skipped']} errors={stats['errors']} "
        f"dry_run={stats.get('dry_run', dry_run)} all_rows={stats.get('all_rows', all_rows)}"
    )


@app.command(name="audit-asset-types")
def audit_asset_types_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_out: Optional[Path] = typer.Option(
        None,
        "--json-out",
        help="Write machine-readable audit report to this path.",
    ),
) -> None:
    """Report asset_type distribution and confusion heuristics for the corpus."""
    from imagecb.caption.asset_type_audit import audit_asset_types, format_audit_report

    report = audit_asset_types()
    typer.echo(format_audit_report(report, verbose=verbose))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        typer.echo(f"\nWrote JSON report to {json_out}")


@app.command(name="freeze-asset-types")
def freeze_asset_types_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        help="Write snapshot even when audit warnings are present.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Snapshot path (default: data/asset_type_taxonomy_vN.json).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Validate audit and freeze the taxonomy manifest for this corpus."""
    _configure_logging(verbose)
    from imagecb.caption.asset_type_audit import freeze_asset_types

    try:
        result = freeze_asset_types(force=force, output_path=output)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Taxonomy v{result['version']} frozen at {result['path']} "
        f"({result['total_records']} records, other={result['other_pct']:.1f}%)."
    )
    typer.echo(
        "Do not change ASSET_TYPES without bumping version and running "
        "backfill-asset-types --all."
    )
    if result.get("warnings") and verbose:
        typer.echo("Warnings recorded in snapshot:")
        for w in result["warnings"]:
            typer.echo(f"  {w}")
    if result.get("forced"):
        typer.echo("(frozen with --force)")


@app.command(name="reindex-embeddings")
def reindex_embeddings_cmd(
    workers: int = typer.Option(
        SETTINGS.ingest_workers,
        "--workers",
        min=1,
        help="Parallel embedding workers.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-embed all cached images (e.g. after enabling slide/PDF context vectors)."""
    _configure_logging(verbose)
    from imagecb.repair import reindex_embeddings

    stats = reindex_embeddings(workers=workers)
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"records={stats['records']} reindexed={stats['reindexed']} errors={stats['errors']}"
    )


@app.command(name="parse-query")
def parse_query_cmd(text: str = typer.Argument(..., help="A natural-language query to parse.")) -> None:
    """Debug helper: print the parsed QuerySpec for TEXT."""
    from imagecb.retrieval.query_parser import parse_query

    spec = parse_query(text)
    typer.echo(
        f"semantic_query : {spec.semantic_query}\n"
        f"must_have      : {spec.must_have_keywords}\n"
        f"must_avoid     : {spec.must_avoid_keywords}\n"
        f"file_types     : {spec.source_filters.file_types}\n"
        f"asset_types    : {spec.source_filters.asset_types}\n"
        f"filename_like  : {spec.source_filters.filename_contains}\n"
        f"authors        : {spec.source_filters.authors}\n"
        f"after          : {spec.time_filter.after}\n"
        f"before         : {spec.time_filter.before}\n"
        f"top_k          : {spec.top_k}\n"
        f"is_refinement  : {spec.is_refinement}"
    )


def _parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise typer.BadParameter("k values must be >= 1")
        values.append(k)
    if not values:
        raise typer.BadParameter("provide at least one k value, e.g. 1,3,5,10")
    return sorted(set(values))


def _format_short_query_line(mode: str, agg, k_values: list[int]) -> str:
    from imagecb.eval.report import _format_aggregate_line

    return _format_aggregate_line(f"Short ({mode})", agg, k_values)


@app.command(name="eval-search")
def eval_search_cmd(
    golden: Path = typer.Option(
        Path("eval/golden.json"),
        "--golden",
        help="Path to the golden-set JSON file.",
        exists=True,
        readable=True,
    ),
    mode: str = typer.Option(
        "all",
        "--mode",
        help="Which pipelines to run: all, chat, retrieval, or similar.",
    ),
    k: str = typer.Option("1,3,5,10", "--k", help="Comma-separated Hit@k values to report."),
    case_id: Optional[str] = typer.Option(None, "--case-id", help="Run a single case by id."),
    failures_only: bool = typer.Option(False, "--failures-only", help="Only print failing cases."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write machine-readable results here."),
    skip_id_validation: bool = typer.Option(
        False,
        "--skip-id-validation",
        help="Do not require golden-set image IDs to exist in the index (for drafting cases).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the golden-set search evaluation harness against the local index."""
    _configure_logging(verbose)
    from imagecb.eval.dataset import GoldenSetValidationError, load_golden_set
    from imagecb.eval.report import format_failures, format_summary, write_json_report
    from imagecb.eval.runner import run_eval

    allowed_modes = {"all", "chat", "retrieval", "similar"}
    if mode not in allowed_modes:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(sorted(allowed_modes))}")

    k_values = _parse_k_values(k)
    try:
        golden_set = load_golden_set(golden, validate_ids=not skip_id_validation)
    except GoldenSetValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    try:
        result = run_eval(golden_set, mode=mode, k_values=k_values, case_id=case_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_summary(result, k_values))

    from imagecb.eval.dataset import short_query_cases
    from imagecb.eval.metrics import aggregate_cases

    short_cases = short_query_cases(golden_set)
    if short_cases and case_id is None:
        for run_mode, agg in (
            ("chat", result.chat),
            ("retrieval", result.retrieval),
        ):
            if agg is None:
                continue
            short_ids = {c.id for c in short_cases}
            short_metrics = [c for c in agg.cases if c.case_id in short_ids]
            if not short_metrics:
                continue
            short_agg = aggregate_cases(run_mode, short_metrics, k_values)
            typer.echo(_format_short_query_line(run_mode, short_agg, k_values))

    detail = format_failures(result, k_values, failures_only=failures_only)
    if detail:
        typer.echo("")
        typer.echo(detail)
    if json_out is not None:
        write_json_report(json_out, result, k_values)
        typer.echo(f"\nWrote {json_out}")


@app.command(name="eval-suggest")
def eval_suggest_cmd(
    text: Optional[str] = typer.Argument(None, help="Text query to search (for labeling text cases)."),
    similar: Optional[str] = typer.Option(None, "--similar", help="Reference image_id for similar-search labeling."),
    top_k: int = typer.Option(15, "--top-k", min=1, max=50),
    suggest_mode: str = typer.Option(
        "retrieval",
        "--mode",
        help="Text search path for labeling: chat (production) or retrieval (stable).",
    ),
    axis: str = typer.Option("balanced", "--axis", help="Similarity axis when using --similar."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Print ranked image IDs to help build eval/golden.json."""
    _configure_logging(verbose)

    if similar:
        from imagecb.retrieval.similar import search_similar

        outcome = search_similar(
            image_id=similar,
            top_k=top_k,
            min_match_percent=0,
            similarity_axis=axis,
            exclude_image_id=similar,
        )
        typer.echo(f"similar  ref={similar}  axis={axis}  top_k={top_k}")
        for rank, row in enumerate(outcome.results, start=1):
            caption = (row.record.caption_short or "").strip()
            source = Path(row.record.source_file).name
            typer.echo(f"{rank:>2}  {row.image_id}  {source}  {caption}")
        return

    if not text:
        raise typer.BadParameter("provide TEXT or --similar IMAGE_ID")

    if suggest_mode == "chat":
        from imagecb.retrieval.session import ChatSession

        outcome = ChatSession().ask(text, top_k=top_k, min_match_percent=0)
        ranked = outcome.results
        typer.echo(f"chat  query={text!r}  top_k={top_k}")
    elif suggest_mode == "retrieval":
        from imagecb.deck.search import search_for_description

        _cards, ranked = search_for_description(text, top_k=top_k, min_match_percent=0)
        typer.echo(f"retrieval  query={text!r}  top_k={top_k}")
    else:
        raise typer.BadParameter("--mode must be chat or retrieval")

    for rank, row in enumerate(ranked, start=1):
        caption = (row.record.caption_short or "").strip()
        source = Path(row.record.source_file).name
        typer.echo(f"{rank:>2}  {row.image_id}  {source}  {caption}")


@app.command(name="repair-search-terms")
def repair_search_terms_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-enrich aliases and recommended_cases from stored tags (no VLM)."""
    _configure_logging(verbose)
    from imagecb.repair import repair_search_terms

    stats = repair_search_terms()
    typer.echo(
        f"Done in {stats.get('elapsed_sec', 0)}s. "
        f"records={stats['records']} updated={stats['updated']}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
