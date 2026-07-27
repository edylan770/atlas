"""FastAPI routes for the isolated Pipeline Lab.

Exposes a standalone comparison page at ``GET /lab`` and JSON/SSE endpoints
under ``/api/lab``. Self-contained: delete this package and the lab_router
block in ``imagecb/api/server.py`` to fully remove the feature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from imagecb.api.auth import require_admin
from imagecb.experiments.variants import (
    iter_comparison,
    run_comparison,
    run_single_variant,
    variant_catalog,
)

logger = logging.getLogger(__name__)

# Every lab request runs multiple Bedrock calls (parse + search + rerank per
# variant); admin-gate the whole router so anonymous traffic can't spend or
# trigger the hubness rebuild.
lab_router = APIRouter(dependencies=[Depends(require_admin)])

_LAB_HTML = Path(__file__).resolve().parent / "lab.html"


class CompareRequest(BaseModel):
    query: str
    top_k: int = 10
    variant: Optional[str] = None


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@lab_router.get("/lab")
def lab_page() -> FileResponse:
    if not _LAB_HTML.is_file():
        raise HTTPException(status_code=404, detail="lab.html not found")
    return FileResponse(_LAB_HTML, media_type="text/html")


@lab_router.get("/api/lab/variants")
def lab_variants() -> dict:
    return {"variants": variant_catalog()}


@lab_router.post("/api/lab/compare")
def lab_compare(body: CompareRequest) -> dict:
    """Run all variants (or one if ``variant`` is set) and return JSON."""
    try:
        if body.variant:
            return run_single_variant(body.variant, body.query, body.top_k)
        return run_comparison(body.query, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lab compare failed")
        raise HTTPException(status_code=500, detail="Lab comparison failed") from exc


@lab_router.post("/api/lab/compare/stream")
def lab_compare_stream(body: CompareRequest) -> StreamingResponse:
    """Stream variants as Server-Sent Events so columns render progressively."""
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    def event_stream() -> Iterator[str]:
        try:
            for event in iter_comparison(query, body.top_k):
                yield _sse_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lab compare stream failed")
            yield _sse_event({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
