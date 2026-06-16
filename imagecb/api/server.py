"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from imagecb.api.routes import router
from imagecb.api.static_ui import resolve_static_dir
from imagecb.admin.routes import router as admin_router
from imagecb.config import SETTINGS
from imagecb.telemetry.schema import ensure_telemetry_schema

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from imagecb.repair import assess_index_health, reconcile_index_safe

    report = assess_index_health(include_weak=True)
    logger.info(
        "Startup index health: healthy=%s stores_in_sync=%s records=%s",
        report.is_healthy,
        report.stores_in_sync,
        report.total_records,
    )
    if SETTINGS.index_reconcile_on_startup:
        stats = reconcile_index_safe()
        logger.info("Startup index reconcile: %s", stats)
    yield


def create_app() -> FastAPI:
    ensure_telemetry_schema()
    app = FastAPI(title="Imagecb", version="1.0.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8080",
            "http://localhost:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.include_router(admin_router)

    # --- Pipeline Lab (experimental, remove this block + imagecb/experiments to uninstall) ---
    from imagecb.experiments.routes import lab_router

    app.include_router(lab_router)
    # --- end Pipeline Lab ---

    static, _kind = resolve_static_dir()
    if static is not None:
        app.mount("/", StaticFiles(directory=str(static), html=True), name="static")

    return app


def launch(*, host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(
        "imagecb.api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
    )
