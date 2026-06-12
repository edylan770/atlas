"""SQLite-backed metadata store (the canonical provenance record).

Each ingested image gets one row keyed by a UUID `image_id`. The same id
is used in Chroma and in the BM25 index, so we can join the three by id
at query time.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from imagecb.config import SETTINGS


class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    __tablename__ = "images"

    image_id = Column(String, primary_key=True)
    content_hash = Column(String, index=True, unique=True)
    image_path = Column(String, nullable=False)

    source_file = Column(String, index=True, nullable=False)
    source_type = Column(String, index=True, nullable=False)  # pptx|pdf|image
    source_modified_at = Column(DateTime, nullable=True, index=True)
    source_created_at = Column(DateTime, nullable=True)
    author = Column(String, nullable=True, index=True)

    slide_index = Column(Integer, nullable=True)
    page_index = Column(Integer, nullable=True)
    slide_title = Column(Text, nullable=True)
    slide_notes = Column(Text, nullable=True)

    ocr_text = Column(Text, nullable=True)
    caption_short = Column(Text, nullable=True)
    caption_detailed = Column(Text, nullable=True)
    objects_json = Column(Text, nullable=True)
    tags_json = Column(Text, nullable=True)
    scene = Column(Text, nullable=True)
    text_overlay_summary = Column(Text, nullable=True)

    image_name = Column(Text, nullable=True)
    use_case = Column(Text, nullable=True)
    recommended_cases_json = Column(Text, nullable=True)

    theme = Column(Text, nullable=True)
    search_aliases_json = Column(Text, nullable=True)
    slide_body_text = Column(Text, nullable=True)
    caption_quality = Column(String, nullable=True)  # ok | weak | failed
    text_read_uncertain = Column(Integer, nullable=True)  # 0/1
    asset_type = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    deleted_at = Column(DateTime, nullable=True, index=True)
    deleted_by = Column(String, nullable=True)


_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _engine_url() -> str:
    return f"sqlite:///{SETTINGS.sqlite_path}"


def _migrate_schema(engine) -> None:
    """Add catalog columns to existing SQLite databases."""
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(images)")).fetchall()
        cols = {row[1] for row in rows}
        for col, ddl in (
            ("image_name", "ALTER TABLE images ADD COLUMN image_name TEXT"),
            ("use_case", "ALTER TABLE images ADD COLUMN use_case TEXT"),
            (
                "recommended_cases_json",
                "ALTER TABLE images ADD COLUMN recommended_cases_json TEXT",
            ),
            ("theme", "ALTER TABLE images ADD COLUMN theme TEXT"),
            ("search_aliases_json", "ALTER TABLE images ADD COLUMN search_aliases_json TEXT"),
            ("slide_body_text", "ALTER TABLE images ADD COLUMN slide_body_text TEXT"),
            ("caption_quality", "ALTER TABLE images ADD COLUMN caption_quality TEXT"),
            ("text_read_uncertain", "ALTER TABLE images ADD COLUMN text_read_uncertain INTEGER"),
            ("asset_type", "ALTER TABLE images ADD COLUMN asset_type TEXT"),
        ):
            if col not in cols:
                conn.execute(text(ddl))
        conn.commit()


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        from sqlalchemy import event

        _engine = create_engine(
            _engine_url(),
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        event.listen(_engine, "connect", _configure_sqlite_connection)
        Base.metadata.create_all(_engine)
        _migrate_schema(_engine)
        from imagecb.telemetry.schema import ensure_telemetry_schema

        ensure_telemetry_schema()
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def is_active(record: ImageRecord) -> bool:
    return record.deleted_at is None


@contextmanager
def session_scope() -> Session:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def new_image_id() -> str:
    return str(uuid.uuid4())


def existing_hashes() -> set[str]:
    with session_scope() as s:
        rows = s.execute(select(ImageRecord.content_hash)).all()
        return {r[0] for r in rows if r[0]}


def get_record_by_hash(content_hash: str) -> Optional[ImageRecord]:
    with session_scope() as s:
        row = s.execute(
            select(ImageRecord).where(ImageRecord.content_hash == content_hash)
        ).scalar_one_or_none()
        if row is not None:
            s.expunge(row)
        return row


def upsert_image(record: ImageRecord) -> None:
    with session_scope() as s:
        s.merge(record)


def get_record(image_id: str, *, include_deleted: bool = False) -> Optional[ImageRecord]:
    rows = get_records([image_id])
    if not rows:
        return None
    rec = rows[0]
    if not include_deleted and rec.deleted_at is not None:
        return None
    return rec


def get_records(
    image_ids: Sequence[str],
    *,
    include_deleted: bool = False,
) -> List[ImageRecord]:
    if not image_ids:
        return []
    with session_scope() as s:
        stmt = select(ImageRecord).where(ImageRecord.image_id.in_(list(image_ids)))
        if not include_deleted:
            stmt = stmt.where(ImageRecord.deleted_at.is_(None))
        rows = s.execute(stmt).scalars().all()
        # Detach so callers can access attributes after the session closes.
        for r in rows:
            s.expunge(r)
        return list(rows)


def get_all_records(*, include_deleted: bool = False) -> List[ImageRecord]:
    with session_scope() as s:
        stmt = select(ImageRecord)
        if not include_deleted:
            stmt = stmt.where(ImageRecord.deleted_at.is_(None))
        rows = s.execute(stmt).scalars().all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def get_active_image_ids() -> List[str]:
    with session_scope() as s:
        rows = s.execute(
            select(ImageRecord.image_id).where(ImageRecord.deleted_at.is_(None))
        ).all()
        return [r[0] for r in rows]


def get_deleted_records() -> List[ImageRecord]:
    with session_scope() as s:
        rows = s.execute(
            select(ImageRecord).where(ImageRecord.deleted_at.is_not(None))
        ).scalars().all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def list_catalog_records(limit: int = 50, sort: str = "newest") -> List[ImageRecord]:
    """Catalog browse listing with configurable sort order."""
    from imagecb.retrieval.sort import catalog_sort_for_sql, resolve_sort

    limit = max(1, min(limit, 200))
    resolved = resolve_sort(sort, is_search=False)
    order_clauses = catalog_sort_for_sql(resolved)
    with session_scope() as s:
        query = select(ImageRecord).where(ImageRecord.deleted_at.is_(None))
        for clause in order_clauses:
            query = query.order_by(clause)
        query = query.limit(limit)
        rows = s.execute(query).scalars().all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def get_recent_records(limit: int = 50) -> List[ImageRecord]:
    """Most recently ingested images (for corpus catalog UI)."""
    return list_catalog_records(limit, sort="newest")


def filter_image_ids(
    *,
    file_types: Optional[Iterable[str]] = None,
    asset_types: Optional[Iterable[str]] = None,
    filename_contains: Optional[Iterable[str]] = None,
    authors: Optional[Iterable[str]] = None,
    modified_after: Optional[datetime] = None,
    modified_before: Optional[datetime] = None,
) -> List[str]:
    """Return image_ids matching the provided metadata filters.

    Empty/None arguments are ignored. Author and filename matches are
    case-insensitive substring checks; file_types and asset_types are exact.
    When asset_types is set, rows with null/empty asset_type are excluded.
    """
    from imagecb.caption.asset_type import normalize_asset_types

    ft = [t.lower() for t in (file_types or []) if t]
    at = normalize_asset_types(list(asset_types or []))
    fc = [c.lower() for c in (filename_contains or []) if c]
    au = [a.lower() for a in (authors or []) if a]

    # We pull all rows and filter in Python: at prototype scale (hundreds-to-
    # low-thousands of images) the SQL gymnastics aren't worth it.
    with session_scope() as s:
        records = s.execute(
            select(ImageRecord).where(ImageRecord.deleted_at.is_(None))
        ).scalars().all()
        result: List[str] = []
        for r in records:
            if ft and (r.source_type or "").lower() not in ft:
                continue
            if at:
                row_type = (r.asset_type or "").strip().lower()
                if not row_type or row_type not in at:
                    continue
            if fc:
                src = Path(r.source_file or "").name.lower()
                if not any(c in src for c in fc):
                    continue
            if au:
                a = (r.author or "").lower()
                if not any(x in a for x in au):
                    continue
            if modified_after and (r.source_modified_at is None or r.source_modified_at < modified_after):
                continue
            if modified_before and (r.source_modified_at is None or r.source_modified_at > modified_before):
                continue
            result.append(r.image_id)
        return result


def record_from_dict(d: dict) -> ImageRecord:
    return ImageRecord(**d)


def serialize_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def deserialize_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
        if isinstance(loaded, list):
            return [str(x) for x in loaded]
    except json.JSONDecodeError:
        return []
    return []
