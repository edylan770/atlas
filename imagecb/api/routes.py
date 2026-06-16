"""FastAPI route handlers."""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from imagecb.api.interpretation import build_interpretation_notes
from imagecb.api.query_format import format_query_error, spec_to_parsed_query
from imagecb.api.auth import require_admin, resolve_user_id
from imagecb.api.schemas import (
    CatalogItemOut,
    ChatRequest,
    ChatResponse,
    CorpusCatalogResponse,
    DeckForceRequest,
    DeckForceResponse,
    DeckSuggestResponse,
    HealthResponse,
    ReadyResponse,
    IngestResponse,
    InteractionRequest,
    InteractionResponse,
    ProvenanceOut,
    ResultCardOut,
    SessionResetRequest,
    SessionResetResponse,
    SimilarRequest,
    SimilarResponse,
    SlideSuggestionOut,
    StatusResponse,
    SuggestionsRequest,
    SuggestionsResponse,
)
from imagecb.api.sessions import get_or_create_session, get_session, reset_session
from imagecb.config import SETTINGS
from imagecb.caption.quality import needs_regeneration
from imagecb.formatting.assistant_reply import (
    _display_caption,
    build_result_cards,
    catalog_fields_from_record,
    provenance_from_record,
)
from imagecb.telemetry.recorder import record_interaction, record_search_from_results
from imagecb.formatting.conversational_reply import (
    build_conversational_reply,
    iter_conversational_reply_text,
)
from imagecb.deck.pipeline import DeckSuggestResult, SlideSuggestion, force_slide_image, process_deck_upload
from imagecb.ingest import IngestInProgressError, ingest_paths
from imagecb.paths import resolve_image_file, resolve_source_file
from imagecb.retrieval.image_query import SimilarityAxis, axis_label
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.session import AskResult
from imagecb.retrieval.similar import search_similar
from imagecb.retrieval.sort import InvalidSortError, ResultSort, resolve_sort
from imagecb.storage import metadata_db, vector_store
from imagecb.suggestions import generate_follow_up_suggestions, generate_suggestions
from imagecb.suggestions.corpus_summary import build_corpus_context
from imagecb.uploads import save_uploads_from_files

logger = logging.getLogger(__name__)

_follow_up_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="follow-up")

router = APIRouter(prefix="/api")

_SOURCE_MEDIA = {
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _resolve_sort_param(sort: Optional[str], *, is_search: bool) -> ResultSort:
    try:
        return resolve_sort(sort, is_search=is_search)
    except InvalidSortError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _result_card_from_dict(d: dict) -> ResultCardOut:
    prov = d.get("provenance") or {}
    return ResultCardOut(
        rank=int(d.get("rank", 0)),
        image_id=str(d.get("image_id", "")),
        image_url=str(d.get("image_url", "")),
        provenance=ProvenanceOut(
            source_name=str(prov.get("source_name", "")),
            source_type=str(prov.get("source_type", "")),
            slide_index=prov.get("slide_index"),
            page_index=prov.get("page_index"),
            modified=prov.get("modified"),
            author=prov.get("author"),
            chips=list(prov.get("chips") or []),
        ),
        caption=str(d.get("caption", "")),
        match_hint=d.get("match_hint"),
        match_percent=int(d.get("match_percent", 0)),
        has_image_file=bool(d.get("has_image_file", True)),
        image_name=str(d.get("image_name", "")),
        use_case=str(d.get("use_case", "")),
        tags=list(d.get("tags") or []),
        recommended_cases=list(d.get("recommended_cases") or []),
        theme=str(d.get("theme", "")),
        aliases=list(d.get("aliases") or []),
        source_url=d.get("source_url"),
        source_location=str(d.get("source_location", "")),
        source_path=d.get("source_path"),
        caption_quality=str(d.get("caption_quality", "ok")),
        needs_regeneration=bool(d.get("needs_regeneration", False)),
        created_at=d.get("created_at"),
        asset_type=str(d.get("asset_type", "")),
    )


def _slide_suggestion_out(s: SlideSuggestion) -> SlideSuggestionOut:
    return SlideSuggestionOut(
        slide_index=s.slide_index,
        title=s.title,
        body_preview=s.body_preview,
        notes_preview=s.notes_preview,
        content_hash=s.content_hash,
        status=s.status,
        description=s.description,
        reason=s.reason,
        results=[_result_card_from_dict(r) for r in s.results],
        llm_cached=s.llm_cached,
        search_cached=s.search_cached,
    )


def _deck_suggest_response(result: DeckSuggestResult) -> DeckSuggestResponse:
    return DeckSuggestResponse(
        deck_hash=result.deck_hash,
        filename=result.filename,
        slides=[_slide_suggestion_out(s) for s in result.slides],
        deck_cached=result.deck_cached,
        llm_batches=result.llm_batches,
    )


def _result_card_out(card) -> ResultCardOut:
    prov = card.provenance
    return ResultCardOut(
        rank=card.rank,
        image_id=card.image_id,
        image_url=card.image_url,
        provenance=ProvenanceOut(
            source_name=prov.source_name,
            source_type=prov.source_type,
            slide_index=prov.slide_index,
            page_index=prov.page_index,
            modified=prov.modified,
            author=prov.author,
            chips=prov.chips(),
        ),
        caption=card.caption,
        match_hint=card.match_hint,
        match_percent=card.match_percent,
        has_image_file=card.has_image_file,
        image_name=card.image_name,
        use_case=card.use_case,
        tags=card.tags,
        recommended_cases=card.recommended_cases,
        theme=card.theme,
        aliases=card.aliases,
        source_url=card.source_url,
        source_location=card.source_location,
        source_path=card.source_path,
        caption_quality=card.caption_quality,
        needs_regeneration=card.needs_regeneration,
        created_at=card.created_at,
        asset_type=card.asset_type,
    )


def _parsed_with_notes(spec: QuerySpec, notes: List[str]):
    return spec_to_parsed_query(spec, interpretation_notes=notes)


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


def _index_status_payload() -> StatusResponse:
    from imagecb.repair import assess_index_health

    try:
        report = assess_index_health(include_weak=True)
    except Exception:  # noqa: BLE001
        logger.exception("Index health assessment failed")
        try:
            n = vector_store.count()
        except Exception:  # noqa: BLE001
            n = 0
        return StatusResponse(indexed_count=n)

    return StatusResponse(
        indexed_count=report.chroma_vectors,
        total_records=report.total_records,
        chroma_vectors=report.chroma_vectors,
        text_vector_count=report.text_vector_count,
        bm25_doc_count=report.bm25_doc_count,
        is_healthy=report.is_healthy,
        stores_in_sync=report.stores_in_sync,
    )


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return _index_status_payload()


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    from imagecb.repair import assess_index_health

    try:
        report = assess_index_health(include_weak=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Readiness check failed")
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "error": str(exc)},
        ) from exc

    issues: List[str] = []
    if not report.stores_in_sync:
        if report.missing_chroma_count:
            issues.append(f"{report.missing_chroma_count} active row(s) missing Chroma vectors")
        if report.missing_text_vector_count:
            issues.append(f"{report.missing_text_vector_count} missing caption-text vectors")
        if report.orphan_chroma_count:
            issues.append(f"{report.orphan_chroma_count} orphan Chroma vector(s)")
        if report.orphan_text_vector_count:
            issues.append(f"{report.orphan_text_vector_count} orphan caption-text vector(s)")
        if report.stale_deleted_in_chroma_count:
            issues.append(
                f"{report.stale_deleted_in_chroma_count} soft-deleted row(s) still indexed"
            )
        if report.bm25_stale:
            issues.append("BM25 index out of sync with SQLite")
    if report.missing_cache_count:
        issues.append(f"{report.missing_cache_count} missing cached PNG(s)")
    if report.failed_caption_count:
        issues.append(f"{report.failed_caption_count} failed caption(s)")

    body = ReadyResponse(
        ready=report.is_healthy,
        is_healthy=report.is_healthy,
        stores_in_sync=report.stores_in_sync,
        total_records=report.total_records,
        chroma_vectors=report.chroma_vectors,
        text_vector_count=report.text_vector_count,
        bm25_doc_count=report.bm25_doc_count,
        issues=issues,
    )
    if not report.is_healthy:
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


@router.post("/suggestions", response_model=SuggestionsResponse)
def suggestions(body: SuggestionsRequest) -> SuggestionsResponse:
    result = generate_suggestions(limit=body.limit)
    return SuggestionsResponse(
        suggestions=result.suggestions,
        cached=result.cached,
    )


@router.post("/telemetry/interaction", response_model=InteractionResponse)
def telemetry_interaction(
    body: InteractionRequest,
    user_id: str = Depends(resolve_user_id),
) -> InteractionResponse:
    if body.interaction_type not in ("view", "download", "similar"):
        raise HTTPException(status_code=400, detail="invalid interaction_type")
    try:
        iid = record_interaction(
            search_event_id=body.search_event_id,
            image_id=body.image_id,
            interaction_type=body.interaction_type,  # type: ignore[arg-type]
            user_id=user_id,
            rank=body.rank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InteractionResponse(interaction_id=iid)


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    user_id: str = Depends(resolve_user_id),
) -> ChatResponse:
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    session_id, session = get_or_create_session(body.session_id)

    try:
        ask_result = session.ask(
            message,
            top_k=body.top_k,
            min_match_percent=body.min_match_percent,
            sort=body.sort,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=format_query_error(exc)) from exc

    spec = ask_result.spec
    notes = build_interpretation_notes(
        spec,
        min_match_percent=ask_result.min_match_percent,
        relaxed_min_score=ask_result.relaxed_min_score,
        dense_failed=ask_result.dense_failed,
        sparse_failed=ask_result.sparse_failed,
        visual_fallback=ask_result.visual_fallback,
        low_confidence_visual=ask_result.low_confidence_visual,
    )
    try:
        indexed_count = vector_store.count()
    except Exception:  # noqa: BLE001
        indexed_count = ask_result.indexed_count

    corpus = build_corpus_context()
    follow_up_future = _follow_up_executor.submit(
        generate_follow_up_suggestions,
        message,
        ask_result,
        notes,
        corpus=corpus,
    )
    reply = build_conversational_reply(
        message,
        ask_result,
        notes,
        indexed_count=indexed_count,
    )
    try:
        follow_ups = follow_up_future.result(timeout=15)
    except Exception:  # noqa: BLE001
        logger.exception("Follow-up suggestions failed")
        follow_ups = []
    session.record_turn(message, reply.message)
    search_event_id = record_search_from_results(
        query_text=message,
        user_id=user_id,
        session_id=session_id,
        search_kind="chat",
        results=ask_result.results,
        spec=spec,
    )
    return ChatResponse(
        session_id=session_id,
        assistant_message=reply.message,
        results=[_result_card_out(c) for c in reply.results],
        parsed_query=_parsed_with_notes(spec, notes),
        search_event_id=search_event_id,
        follow_up_suggestions=follow_ups,
    )


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    user_id: str = Depends(resolve_user_id),
) -> StreamingResponse:
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    session_id, session = get_or_create_session(body.session_id)

    try:
        ask_result = session.ask(
            message,
            top_k=body.top_k,
            min_match_percent=body.min_match_percent,
            sort=body.sort,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=format_query_error(exc)) from exc

    spec = ask_result.spec
    notes = build_interpretation_notes(
        spec,
        min_match_percent=ask_result.min_match_percent,
        relaxed_min_score=ask_result.relaxed_min_score,
        dense_failed=ask_result.dense_failed,
        sparse_failed=ask_result.sparse_failed,
        visual_fallback=ask_result.visual_fallback,
        low_confidence_visual=ask_result.low_confidence_visual,
    )
    try:
        indexed_count = vector_store.count()
    except Exception:  # noqa: BLE001
        indexed_count = ask_result.indexed_count

    result_cards = build_result_cards(ask_result.results)
    results_out = [_result_card_out(c) for c in result_cards]
    parsed_out = _parsed_with_notes(spec, notes)
    search_event_id = record_search_from_results(
        query_text=message,
        user_id=user_id,
        session_id=session_id,
        search_kind="chat",
        results=ask_result.results,
        spec=spec,
    )

    corpus = build_corpus_context()
    follow_up_future = _follow_up_executor.submit(
        generate_follow_up_suggestions,
        message,
        ask_result,
        notes,
        corpus=corpus,
    )

    def event_stream() -> Iterator[str]:
        yield _sse_event(
            {
                "type": "metadata",
                "session_id": session_id,
                "search_event_id": search_event_id,
                "results": [r.model_dump() for r in results_out],
                "parsed_query": parsed_out.model_dump() if parsed_out else None,
            }
        )
        full_message: List[str] = []
        try:
            for chunk in iter_conversational_reply_text(
                message,
                ask_result,
                notes,
                indexed_count=indexed_count,
            ):
                if not chunk:
                    continue
                full_message.append(chunk)
                yield _sse_event({"type": "token", "text": chunk})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat stream failed")
            yield _sse_event({"type": "error", "detail": str(exc)})
            return

        assistant_message = "".join(full_message)
        try:
            follow_ups = follow_up_future.result(timeout=15)
        except Exception:  # noqa: BLE001
            logger.exception("Follow-up suggestions failed")
            follow_ups = []
        session.record_turn(message, assistant_message)
        yield _sse_event(
            {
                "type": "done",
                "assistant_message": assistant_message,
                "follow_up_suggestions": follow_ups,
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/similar", response_model=SimilarResponse)
async def similar(
    request: Request,
    user_id: str = Depends(resolve_user_id),
) -> SimilarResponse:
    """Find visually similar images (JSON body or multipart with optional file)."""
    ref_id: Optional[str] = None
    upload_image: Optional[Image.Image] = None
    upload_filename: Optional[str] = None
    sid: Optional[str] = None
    k = 10
    min_pct = 0
    axis = "balanced"
    sort_param: Optional[str] = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = SimilarRequest.model_validate(await request.json())
        ref_id = body.image_id
        sid = body.session_id
        k = body.top_k
        min_pct = body.min_match_percent
        axis = body.similarity_axis
        sort_param = body.sort
    else:
        form = await request.form()
        raw_id = form.get("image_id")
        if raw_id:
            ref_id = str(raw_id)
        raw_sid = form.get("session_id")
        if raw_sid:
            sid = str(raw_sid)
        raw_top_k = form.get("top_k")
        if raw_top_k:
            k = int(raw_top_k)
        raw_min = form.get("min_match_percent")
        if raw_min is not None and raw_min != "":
            min_pct = int(raw_min)
        raw_axis = form.get("similarity_axis")
        if raw_axis:
            axis = str(raw_axis)
        raw_sort = form.get("sort")
        if raw_sort:
            sort_param = str(raw_sort)

        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            filename = getattr(file, "filename", None)
            if filename:
                raw = await file.read()
                if not raw:
                    raise HTTPException(status_code=400, detail="empty image file")
                try:
                    upload_image = Image.open(io.BytesIO(raw))
                    upload_image.load()
                    upload_filename = filename
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    try:
        SimilarityAxis.parse(axis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not ref_id and upload_image is None:
        raise HTTPException(status_code=400, detail="image_id or image file is required")

    _resolve_sort_param(sort_param, is_search=True)

    try:
        outcome = search_similar(
            image_id=ref_id,
            image=upload_image,
            top_k=k,
            exclude_image_id=ref_id,
            min_match_percent=min_pct,
            similarity_axis=axis,
            sort=sort_param,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Similar search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = outcome.results
    spec = outcome.spec
    facets = outcome.facets
    parsed_axis = SimilarityAxis.parse(axis)

    if upload_image is not None and upload_filename:
        user_label = f"[Image search] {upload_filename}"
    elif ref_id:
        rec = metadata_db.get_record(ref_id)
        name = (rec.image_name if rec else None) or ref_id
        user_label = f"[Find similar] {name}"
    else:
        user_label = "[similar image search]"

    notes = [
        f"Blended visual similarity with text search ({axis_label(parsed_axis)}).",
        (
            f"Generated query: {facets.search_query}"
            if facets.is_usable() and facets.search_query
            else "Visual similarity only (image query unavailable)."
            if not facets.is_usable()
            else ""
        ),
        f"Showing visual matches at or above {min_pct}%." if min_pct > 0 else "",
    ]
    notes = [n for n in notes if n]

    try:
        indexed_count = vector_store.count()
    except Exception:  # noqa: BLE001
        indexed_count = 0

    ask_result = AskResult(
        spec=spec,
        results=results,
        min_match_percent=min_pct,
        indexed_count=indexed_count,
    )
    reply = build_conversational_reply(
        user_label,
        ask_result,
        notes,
        indexed_count=indexed_count,
    )
    msg = reply.message
    if not results:
        msg = "I couldn't find similar images in the index."

    out_session_id = sid
    if sid:
        session = get_session(sid)
        if session is not None:
            session.apply_similar_results(results, spec=spec)
        else:
            out_session_id, session = get_or_create_session(None)
            session.apply_similar_results(results, spec=spec)
            out_session_id = out_session_id

    search_event_id = record_search_from_results(
        query_text=user_label,
        user_id=user_id,
        session_id=out_session_id,
        search_kind="similar",
        results=results,
        spec=spec,
    )

    return SimilarResponse(
        session_id=out_session_id,
        assistant_message=msg,
        results=[_result_card_out(c) for c in reply.results],
        parsed_query=_parsed_with_notes(spec, notes),
        search_event_id=search_event_id,
    )


@router.post("/session/reset", response_model=SessionResetResponse)
def session_reset(body: SessionResetRequest) -> SessionResetResponse:
    session = reset_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionResetResponse(session_id=body.session_id)


@router.get("/images/{image_id}")
def get_image(image_id: str) -> FileResponse:
    record = metadata_db.get_record(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = resolve_image_file(record)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="image file not on disk")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/sources/{image_id}")
def get_source(image_id: str) -> FileResponse:
    record = metadata_db.get_record(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = resolve_source_file(record)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="source file not on disk")
    media = _SOURCE_MEDIA.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/corpus/catalog", response_model=CorpusCatalogResponse)
def corpus_catalog(limit: int = 40, sort: Optional[str] = None) -> CorpusCatalogResponse:
    """Recently ingested images with catalog metadata (name, tags, use cases)."""
    resolved = _resolve_sort_param(sort, is_search=False)
    limit = max(1, min(limit, 200))
    records = metadata_db.list_catalog_records(limit, sort=resolved)
    items: List[CatalogItemOut] = []
    for r in records:
        image_name, use_case, tags, recommended, theme, aliases = catalog_fields_from_record(r)
        prov = provenance_from_record(r)
        quality = (r.caption_quality or "ok").lower()
        created_at = r.created_at.isoformat() if r.created_at else None
        items.append(
            CatalogItemOut(
                image_id=r.image_id,
                image_url=f"/api/images/{r.image_id}",
                image_name=image_name,
                use_case=use_case,
                tags=tags,
                recommended_cases=recommended,
                theme=theme,
                aliases=aliases,
                caption=_display_caption(r),
                source_name=prov.source_name,
                source_file=r.source_file or "",
                created_at=created_at,
                caption_quality=quality,
                needs_regeneration=needs_regeneration(quality),
                asset_type=(r.asset_type or "").strip(),
            )
        )
    try:
        n = vector_store.count()
    except Exception:  # noqa: BLE001
        n = len(records)
    return CorpusCatalogResponse(items=items, indexed_count=n)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: List[UploadFile] = File(...),
    skip_caption: bool = Form(False),
    skip_ocr: bool = Form(False),
    force: bool = Form(False),
    workers: Optional[int] = Form(None),
    _: str = Depends(require_admin),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    saved, stage_errors = await save_uploads_from_files(files)
    if not saved:
        msg = "No supported files could be staged."
        if stage_errors:
            msg += "\n" + "\n".join(f"- {e}" for e in stage_errors)
        try:
            n = vector_store.count()
        except Exception:  # noqa: BLE001
            n = 0
        return IngestResponse(message=msg, indexed_count=n, stats={})

    try:
        ingest_workers = workers if workers is not None else SETTINGS.ingest_workers
        ingest_workers = max(1, min(int(ingest_workers), 32))
        stats = ingest_paths(
            saved,
            skip_caption=skip_caption,
            skip_ocr=skip_ocr,
            force=force,
            workers=ingest_workers,
        )
    except IngestInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    lines: list[str] = []
    if stage_errors:
        lines.append("Staging warnings:")
        lines.extend(f"- {e}" for e in stage_errors)
        lines.append("")
    elapsed = stats.get("elapsed_sec", 0)
    processed = stats.get("images_added", 0) + stats.get("images_updated", 0)
    rate = ""
    if elapsed > 0 and processed > 0:
        rate = f" ({processed / elapsed:.2f} images/s)"
    lines.append(
        f"Ingest complete in {elapsed}s{rate}: "
        f"files={stats.get('files', 0)}, "
        f"images_seen={stats.get('images_seen', 0)}, "
        f"added={stats.get('images_added', 0)}, "
        f"updated={stats.get('images_updated', 0)}, "
        f"duplicates={stats.get('skipped_duplicates', 0)}, "
        f"errors={stats.get('errors', 0)}."
    )
    from imagecb.repair import format_post_repair_summary

    repair_line = format_post_repair_summary(stats.get("post_repair") or {})
    if repair_line:
        lines.append(repair_line)
    try:
        n = vector_store.count()
    except Exception:  # noqa: BLE001
        n = 0
    return IngestResponse(message="\n".join(lines), indexed_count=n, stats=stats)


@router.post("/deck/suggest", response_model=DeckSuggestResponse)
async def deck_suggest(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    min_match_percent: int = Form(0),
    sort: Optional[str] = Form(None),
    user_id: str = Depends(resolve_user_id),  # noqa: ARG001
) -> DeckSuggestResponse:
    """Suggest corpus images per slide from an uploaded PowerPoint deck."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    if not file.filename.lower().endswith(".pptx"):
        raise HTTPException(status_code=400, detail="only .pptx files are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    k = max(1, min(int(top_k), 30))
    min_pct = max(0, min(int(min_match_percent), 100))
    _resolve_sort_param(sort, is_search=True)

    try:
        result = process_deck_upload(
            raw,
            file.filename,
            top_k=k,
            min_match_percent=min_pct,
            sort=sort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Deck suggest failed")
        raise HTTPException(status_code=500, detail=f"Deck suggest failed: {exc}") from exc

    return _deck_suggest_response(result)


@router.post("/deck/force", response_model=DeckForceResponse)
def deck_force(
    body: DeckForceRequest,
    user_id: str = Depends(resolve_user_id),  # noqa: ARG001
) -> DeckForceResponse:
    """Force image suggestion for a slide marked no_image_needed."""
    k = max(1, min(int(body.top_k), 30))
    min_pct = max(0, min(int(body.min_match_percent), 100))
    _resolve_sort_param(body.sort, is_search=True)

    try:
        slide = force_slide_image(
            body.deck_hash,
            body.slide_index,
            top_k=k,
            min_match_percent=min_pct,
            sort=body.sort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Deck force failed")
        raise HTTPException(status_code=500, detail=f"Deck force failed: {exc}") from exc

    return DeckForceResponse(slide=_slide_suggestion_out(slide))
