"""FastAPI route handlers."""

from __future__ import annotations

import io
import json
import logging
from typing import Iterator, List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
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
from imagecb.ingest import ingest_paths
from imagecb.paths import resolve_image_file, resolve_source_file
from imagecb.retrieval.image_query import SimilarityAxis, axis_label
from imagecb.retrieval.query_parser import QuerySpec
from imagecb.retrieval.session import AskResult
from imagecb.retrieval.similar import search_similar
from imagecb.storage import metadata_db, vector_store
from imagecb.suggestions import generate_suggestions
from imagecb.uploads import save_uploads_from_files

logger = logging.getLogger(__name__)

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
        source_url=d.get("source_url"),
        source_location=str(d.get("source_location", "")),
        source_path=d.get("source_path"),
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
        source_url=card.source_url,
        source_location=card.source_location,
        source_path=card.source_path,
    )


def _parsed_with_notes(spec: QuerySpec, notes: List[str]):
    return spec_to_parsed_query(spec, interpretation_notes=notes)


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    try:
        n = vector_store.count()
    except Exception:  # noqa: BLE001
        n = 0
    return StatusResponse(indexed_count=n)


@router.post("/suggestions", response_model=SuggestionsResponse)
def suggestions(body: SuggestionsRequest) -> SuggestionsResponse:
    result = generate_suggestions(
        body.recent_titles,
        limit=body.limit,
    )
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
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=format_query_error(exc)) from exc

    spec = ask_result.spec
    results = ask_result.results
    notes = build_interpretation_notes(
        spec,
        applied_refinement_pool=ask_result.applied_refinement_pool,
        pool_size=ask_result.pool_size,
        sticky_merged=ask_result.sticky_merged,
        min_match_percent=ask_result.min_match_percent,
        relaxed_min_score=ask_result.relaxed_min_score,
        dense_failed=ask_result.dense_failed,
        sparse_failed=ask_result.sparse_failed,
    )
    try:
        indexed_count = vector_store.count()
    except Exception:  # noqa: BLE001
        indexed_count = ask_result.indexed_count

    reply = build_conversational_reply(
        message,
        ask_result,
        notes,
        indexed_count=indexed_count,
    )
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
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=format_query_error(exc)) from exc

    spec = ask_result.spec
    notes = build_interpretation_notes(
        spec,
        applied_refinement_pool=ask_result.applied_refinement_pool,
        pool_size=ask_result.pool_size,
        sticky_merged=ask_result.sticky_merged,
        min_match_percent=ask_result.min_match_percent,
        relaxed_min_score=ask_result.relaxed_min_score,
        dense_failed=ask_result.dense_failed,
        sparse_failed=ask_result.sparse_failed,
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
        session.record_turn(message, assistant_message)
        yield _sse_event(
            {"type": "done", "assistant_message": assistant_message}
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
    body: Optional[SimilarRequest] = Body(None),
    file: Optional[UploadFile] = File(None),
    image_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    top_k: int = Form(10),
    min_match_percent: int = Form(0),
    similarity_axis: str = Form("balanced"),
    user_id: str = Depends(resolve_user_id),
) -> SimilarResponse:
    """Find visually similar images (JSON body or multipart with optional file)."""
    ref_id: Optional[str] = None
    upload_image: Optional[Image.Image] = None
    sid: Optional[str] = None
    k = top_k
    min_pct = min_match_percent
    axis = similarity_axis

    if body is not None:
        ref_id = body.image_id
        sid = body.session_id
        k = body.top_k
        min_pct = body.min_match_percent
        axis = body.similarity_axis
    if image_id:
        ref_id = image_id
    if session_id:
        sid = session_id

    try:
        SimilarityAxis.parse(axis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if file is not None and file.filename:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty image file")
        try:
            upload_image = Image.open(io.BytesIO(raw))
            upload_image.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    if not ref_id and upload_image is None:
        raise HTTPException(status_code=400, detail="image_id or image file is required")

    restrict_to: Optional[List[str]] = None
    if sid:
        session = get_session(sid)
        if session is not None and session.last_candidate_ids and session.last_spec is not None:
            if session.last_spec.is_refinement:
                restrict_to = list(session.last_candidate_ids)

    try:
        outcome = search_similar(
            image_id=ref_id,
            image=upload_image,
            top_k=k,
            exclude_image_id=ref_id,
            min_match_percent=min_pct,
            similarity_axis=axis,
            restrict_to=restrict_to,
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

    if upload_image is not None and file is not None and file.filename:
        user_label = f"[Image search] {file.filename}"
    elif ref_id:
        rec = metadata_db.get_record(ref_id)
        name = (rec.image_name if rec else None) or ref_id
        user_label = f"[Find similar] {name}"
    else:
        user_label = "[similar image search]"

    notes = [
        f"Blended visual similarity with text search ({axis_label(parsed_axis)}).",
        f"Generated query: {facets.search_query}" if facets.search_query else "",
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
def corpus_catalog(limit: int = 40) -> CorpusCatalogResponse:
    """Recently ingested images with catalog metadata (name, tags, use cases)."""
    limit = max(1, min(limit, 200))
    records = metadata_db.get_recent_records(limit)
    items: List[CatalogItemOut] = []
    for r in records:
        image_name, use_case, tags, recommended = catalog_fields_from_record(r)
        prov = provenance_from_record(r)
        items.append(
            CatalogItemOut(
                image_id=r.image_id,
                image_url=f"/api/images/{r.image_id}",
                image_name=image_name,
                use_case=use_case,
                tags=tags,
                recommended_cases=recommended,
                caption=_display_caption(r),
                source_name=prov.source_name,
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

    try:
        result = process_deck_upload(
            raw,
            file.filename,
            top_k=k,
            min_match_percent=min_pct,
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

    try:
        slide = force_slide_image(
            body.deck_hash,
            body.slide_index,
            top_k=k,
            min_match_percent=min_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Deck force failed")
        raise HTTPException(status_code=500, detail=f"Deck force failed: {exc}") from exc

    return DeckForceResponse(slide=_slide_suggestion_out(slide))
