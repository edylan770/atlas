"""Lightweight schema migration (create_all + ALTER for new columns)."""

from __future__ import annotations

import logging
from sqlalchemy import inspect, text

from imagecb.storage.metadata_db import Base, get_engine

logger = logging.getLogger(__name__)


def ensure_telemetry_schema() -> None:
    """Create telemetry tables and add soft-delete / timing columns if missing."""
    engine = get_engine()
    Base.metadata.create_all(engine)

    insp = inspect(engine)
    alters: list[str] = []

    if insp.has_table("images"):
        existing = {c["name"] for c in insp.get_columns("images")}
        if "deleted_at" not in existing:
            alters.append("ALTER TABLE images ADD COLUMN deleted_at DATETIME")
        if "deleted_by" not in existing:
            alters.append("ALTER TABLE images ADD COLUMN deleted_by VARCHAR")

    if insp.has_table("search_events"):
        search_cols = {c["name"] for c in insp.get_columns("search_events")}
        for col, ddl in (
            ("total_ms", "ALTER TABLE search_events ADD COLUMN total_ms FLOAT"),
            ("ask_ms", "ALTER TABLE search_events ADD COLUMN ask_ms FLOAT"),
            ("reply_ms", "ALTER TABLE search_events ADD COLUMN reply_ms FLOAT"),
            ("timings_json", "ALTER TABLE search_events ADD COLUMN timings_json TEXT"),
            ("timing_log", "ALTER TABLE search_events ADD COLUMN timing_log TEXT"),
        ):
            if col not in search_cols:
                alters.append(ddl)

    if not alters:
        return

    with engine.begin() as conn:
        for stmt in alters:
            logger.info("Applying schema migration: %s", stmt)
            conn.execute(text(stmt))
