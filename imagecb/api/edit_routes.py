"""Public Nano Banana image-edit API (same audience as chat)."""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from imagecb.api.edit_sessions import (
    create_edit_session,
    delete_edit_session,
    get_edit_session,
)
from imagecb.api.rate_limit import check_llm_rate_limit
from imagecb.config import SETTINGS
from imagecb.models.secrets import is_nano_banana_available
from imagecb.paths import image_fallbacks
from imagecb.pending_edits import create_pending_edit
from imagecb.storage import blob_store, metadata_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edit", tags=["edit"])


class CreateEditSessionRequest(BaseModel):
    image_id: str = Field(..., min_length=1)


class EditTurnRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


def _require_nano_banana() -> None:
    if not is_nano_banana_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Nano Banana editing is unavailable. Configure GEMINI_API_KEY or "
                "Secrets Manager secret access."
            ),
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


def _session_payload(session_id: str, session) -> dict:
    return {
        "session_id": session_id,
        "source_image_id": session.source_image_id,
        "image_url": f"/api/edit/sessions/{session_id}/image",
        "turn_count": len(session.turns),
        "last_prompt": session.last_prompt,
        "submitted": session.submitted,
        "turns": [{"prompt": t.prompt} for t in session.turns],
    }


@router.get("/status")
def edit_status():
    return {
        "available": is_nano_banana_available(),
        "model": SETTINGS.nano_banana_model,
    }


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
    data = session.working_image_png
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="edit-{session_id}.png"',
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
        },
    )


@router.post("/sessions/{session_id}/turn")
def edit_turn(
    session_id: str,
    body: EditTurnRequest,
    _rl: None = Depends(check_llm_rate_limit),
):
    _require_nano_banana()
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    if session.submitted:
        raise HTTPException(status_code=409, detail="edit session already submitted")

    from imagecb.models.image_edit import edit_image

    try:
        result = edit_image(session.working_image_png, body.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Nano Banana edit failed for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail=f"image edit failed: {exc}",
        ) from exc

    session.working_image_png = result
    session.last_prompt = body.prompt.strip()
    from imagecb.api.edit_sessions import EditTurn

    session.turns.append(EditTurn(prompt=session.last_prompt))
    return _session_payload(session_id, session)


@router.post("/sessions/{session_id}/submit")
def submit_session(
    session_id: str,
    _rl: None = Depends(check_llm_rate_limit),
):
    session = get_edit_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="edit session not found")
    if session.submitted:
        raise HTTPException(status_code=409, detail="edit session already submitted")
    if not session.turns:
        raise HTTPException(
            status_code=400,
            detail="edit the image at least once before adding to the database",
        )

    pending = create_pending_edit(
        source_image_id=session.source_image_id,
        image_bytes=session.working_image_png,
        last_prompt=session.last_prompt,
    )
    session.submitted = True
    delete_edit_session(session_id)
    return {"ok": True, "pending": pending}


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
