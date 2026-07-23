"""Telemetry and audit SQLAlchemy models (same SQLite DB as images)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from imagecb.storage.metadata_db import Base


class SearchEvent(Base):
    __tablename__ = "search_events"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    query_text = Column(Text, nullable=False)
    user_id = Column(String, nullable=False, index=True, default="anonymous")
    session_id = Column(String, nullable=True, index=True)
    search_kind = Column(String, nullable=False, index=True)  # chat | similar
    served_image_ids_json = Column(Text, nullable=False, default="[]")
    result_count = Column(Integer, nullable=False, default=0)
    top_score = Column(Float, nullable=True)
    top_score_kind = Column(String, nullable=True)  # rerank | dense
    parsed_semantic_query = Column(Text, nullable=True)
    # End-to-end request latency (ms); prefer request_total from query timing.
    total_ms = Column(Float, nullable=True)
    # session.ask wall time until results are ready (ms).
    ask_ms = Column(Float, nullable=True)
    # Conversational reply generation / stream (ms).
    reply_ms = Column(Float, nullable=True)
    # Full {step: ms} map from QueryTimingSession.
    timings_json = Column(Text, nullable=True)
    # Blob URI/path of durable query timing .txt report.
    timing_log = Column(Text, nullable=True)


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    id = Column(String, primary_key=True)
    search_event_id = Column(
        String,
        ForeignKey("search_events.id"),
        nullable=False,
        index=True,
    )
    image_id = Column(String, nullable=False, index=True)
    interaction_type = Column(String, nullable=False)  # view | download | similar
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    user_id = Column(String, nullable=False, default="anonymous")
    rank = Column(Integer, nullable=True)


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_log"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    actor = Column(String, nullable=False)
    action = Column(String, nullable=False, index=True)
    target_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False, index=True)
    details_json = Column(Text, nullable=True)
