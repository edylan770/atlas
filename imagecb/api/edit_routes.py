"""Public Nano Banana image-edit API (same audience as chat)."""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from imagecb.api.edit_sessions import (
    EditSession,
    EditTurn,
    create_edit_session,
    delete_edit_session,
    get_edit_session,
)
from imagecb.api.rate_limit import check_llm_rate_limit
from imagecb.config import SETTINGS
from imagecb.models.secrets import nano_banana_status
from imagecb.paths import image_fallbacks
from imagecb.pending_edits import create_pending_edit
from imagecb.storage import blob_store, metadata_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edit", tags=["edit"])


class CreateEditSessionRequest(BaseModel):
    image_id: str = Field(..., min_length=1)


class EditTurnRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


class CreateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


class ReviseTurnRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    base_turn_index: int = Field(..., ge=0)


def _require_nano_banana() -> None:
    status = nano_banana_status()
    if not status["available"]:
        reason = status.get("error") or "Gemini configuration could not be loaded"
        raise HTTPException(
            status_code=503,
            detail=f"Nano Banana editing is unavailable. {reason}",
        )


def _load_corpus_png(image_id: str) -> bytes:
    record = metadata_db.get_record(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    last_error: Optional[Exception] = None
    for candidate in (record.image_path, record.source_file):
        if not candidate:
            continue
        try:
            return blob_store.read_bytes(candidate, fallbacks=image_fallbacks(record))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise HTTPException(status_code=404, detail="image blob not found") from last_error


def _png_response(data: bytes, *, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
        },
    )


def _apply_image_error(exc: Exception, *, session_id: str, source_image_id: str) -> HTTPException:
    from imagecb.models.image_edit import ImageEditError

    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ImageEditError):
        logger.warning(
            "Nano Banana image failed session=%s source_image=%s error_code=%s",
            session_id,
            source_image_id,
            exc.code,
        )
        return HTTPException(
            status_code=502,
            detail=f"Image generation failed [{exc.code}]: {exc}",
        )
    logger.exception(
        "Nano Banana image failed session=%s source_image=%s",
        session_id,
        source_image_id,
    )
    return HTTPException(
        status_code=502,
        detail="Image generation failed [internal_error]. Check server logs.",
    )


def _session_payload(session_id: str, session) -> dict:
    return {
        "session_id": session_id,
        "source_image_id": session.source_image_id,
        "original_image_url": f"/api/edit/sessions/{session_id}/original",
        "image_url": f"/api/edit/sessions/{session_id}/image",
        "turn_count": len(session.turns),
        "last_prompt": session.last_prompt,
        "submitted": session.submitted,
        "turns": [
            {
                "prompt": t.prompt,
                "image_url": (
                    f"/api/edit/sessions/{session_id}/turns/{index}/image"
                ),
            }
            for index, t in enumerate(session.turns)
        ],
    }


@router.get("/status")
def edit_status():
    """Public availability probe. Includes safe diagnostics when unavailable."""
    return nano_banana_status()


@router.post("/sessions")
def create_session(
    body: CreateEditSessionRequest,
    _rl: None = Depends(check_llm_rate_limit),
):
    _require_nano_banana()
    png = _load_corpus_png(body.image_id)
    session_id, session = create_edit_session(
        source_image_id=body.image_id,
        working_image_png=png,
    )
    return _session_payload(session_id, session)


@router.post("/sessions/create")
def create_image_session(
    body: CreateImageRequest,
    _rl: None = Depends(check_llm_rate_limit),
):
    """Create a new image from a text prompt (no corpus source)."""
    _require_nano_banana()
    from imagecb.models.image_edit import generate_image

    prompt = body.prompt.strip()
    try:
        result = generate_image(prompt)
    except Exception as exc:  # noqa: BLE001
        raise _apply_image_error(exc, session_id="new", source_image_id="") from exc

    session_id, session = create_edit_session(
        source_image_id="",
        working_image_png=result,
    )
    session.last_prompt = prompt
    session.turns.append(EditTurn(prompt=prompt, result_image_png=result))
    return _session_payload(session_id, session)


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    return _session_payload(session_id, session)


@router.get("/sessions/{session_id}/image")
def get_session_image(session_id: str):
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    return _png_response(
        session.working_image_png,
        filename=f"edit-{session_id}.png",
    )


@router.get("/sessions/{session_id}/original")
def get_session_original(session_id: str):
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    return _png_response(
        session.original_image_png,
        filename=f"original-{session_id}.png",
    )


@router.get("/sessions/{session_id}/turns/{turn_index}/image")
def get_session_turn_image(session_id: str, turn_index: int):
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    if turn_index < 0 or turn_index >= len(session.turns):
        raise HTTPException(status_code=404, detail="edit turn not found")
    return _png_response(
        session.turns[turn_index].result_image_png,
        filename=f"edit-{session_id}-turn-{turn_index}.png",
    )


@contextmanager
def _locked_session(session_id: str) -> Iterator[EditSession]:
    """Yield an unsubmitted session while holding its lock (409 if busy)."""
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    if not session.lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="another edit is already running for this session",
        )
    try:
        if session.submitted:
            raise HTTPException(status_code=409, detail="edit session already submitted")
        yield session
    finally:
        session.lock.release()


def _check_turn_cap(turn_count_after: int) -> None:
    cap = SETTINGS.edit_session_max_turns
    if cap > 0 and turn_count_after > cap:
        raise HTTPException(
            status_code=409,
            detail=f"edit session reached the {cap}-turn limit; start a new edit",
        )


@router.post("/sessions/{session_id}/turn")
def edit_turn(
    session_id: str,
    body: EditTurnRequest,
    _rl: None = Depends(check_llm_rate_limit),
):
    _require_nano_banana()
    with _locked_session(session_id) as session:
        _check_turn_cap(len(session.turns) + 1)

        from imagecb.models.image_edit import edit_image

        try:
            result = edit_image(session.working_image_png, body.prompt)
        except Exception as exc:  # noqa: BLE001
            raise _apply_image_error(
                exc,
                session_id=session_id,
                source_image_id=session.source_image_id,
            ) from exc

        session.working_image_png = result
        session.last_prompt = body.prompt.strip()
        session.turns.append(
            EditTurn(prompt=session.last_prompt, result_image_png=result)
        )
        return _session_payload(session_id, session)


@router.post("/sessions/{session_id}/revise")
def revise_turn(
    session_id: str,
    body: ReviseTurnRequest,
    _rl: None = Depends(check_llm_rate_limit),
):
    """Replace later turns by editing the image from ``base_turn_index``."""
    _require_nano_banana()
    with _locked_session(session_id) as session:
        if body.base_turn_index >= len(session.turns):
            raise HTTPException(status_code=404, detail="edit turn not found")
        _check_turn_cap(body.base_turn_index + 2)

        from imagecb.models.image_edit import edit_image

        base = session.turns[body.base_turn_index]
        session.turns = session.turns[: body.base_turn_index + 1]
        session.working_image_png = base.result_image_png
        prompt = body.prompt.strip()
        try:
            result = edit_image(session.working_image_png, prompt)
        except Exception as exc:  # noqa: BLE001
            raise _apply_image_error(
                exc,
                session_id=session_id,
                source_image_id=session.source_image_id,
            ) from exc

        session.working_image_png = result
        session.last_prompt = prompt
        session.turns.append(EditTurn(prompt=prompt, result_image_png=result))
        return _session_payload(session_id, session)


@router.post("/sessions/{session_id}/submit")
def submit_session(
    session_id: str,
    _rl: None = Depends(check_llm_rate_limit),
):
    with _locked_session(session_id) as session:
        if not session.turns:
            raise HTTPException(
                status_code=400,
                detail="edit the image at least once before adding to the database",
            )

        pending = create_pending_edit(
            source_image_id=(session.source_image_id or "").strip(),
            image_bytes=session.working_image_png,
            last_prompt=session.last_prompt,
        )
        session.submitted = True
    delete_edit_session(session_id)
    return {"ok": True, "pending": pending}


@router.delete("/sessions/{session_id}")
def discard_session(session_id: str):
    """Free an abandoned edit session's image bytes (idempotent)."""
    delete_edit_session(session_id)
    return {"ok": True}


@router.get("/pending/{pending_id}/image")
def get_pending_image(pending_id: str):
    """UUID-gated preview for staged pending edits (no admin header required)."""
    from imagecb.pending_edits import read_pending_image_bytes

    try:
        data = read_pending_image_bytes(pending_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="pending-{pending_id}.png"',
        },
    )


@router.get("/pending/{pending_id}/thumb")
def get_pending_thumb(pending_id: str):
    from imagecb.pending_edits import read_pending_image_bytes, read_pending_thumb_bytes

    try:
        data = read_pending_thumb_bytes(pending_id)
        if data is None:
            data = read_pending_image_bytes(pending_id)
            media = "image/png"
        else:
            media = "image/jpeg"
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pending edit not found") from exc
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Cache-Control": "no-store"},
    )
