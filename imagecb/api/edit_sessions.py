"""In-memory Nano Banana edit sessions (TTL + LRU, like chat sessions)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

from imagecb.config import SETTINGS

_lock = Lock()


@dataclass
class EditTurn:
    prompt: str
    created_at: float = field(default_factory=time.time)


@dataclass
class EditSession:
    source_image_id: str
    working_image_png: bytes
    last_prompt: Optional[str] = None
    turns: List[EditTurn] = field(default_factory=list)
    submitted: bool = False


@dataclass
class _Entry:
    session: EditSession
    last_used: float = field(default_factory=time.monotonic)


_sessions: Dict[str, _Entry] = {}


def _evict_locked() -> None:
    now = time.monotonic()
    ttl = SETTINGS.session_ttl_sec
    if ttl > 0:
        expired = [sid for sid, e in _sessions.items() if now - e.last_used > ttl]
        for sid in expired:
            del _sessions[sid]
    cap = SETTINGS.session_max_count
    if cap > 0 and len(_sessions) > cap:
        by_age = sorted(_sessions.items(), key=lambda kv: kv[1].last_used)
        for sid, _ in by_age[: len(_sessions) - cap]:
            del _sessions[sid]


def create_edit_session(
    *,
    source_image_id: str,
    working_image_png: bytes,
) -> tuple[str, EditSession]:
    session_id = str(uuid.uuid4())
    session = EditSession(
        source_image_id=source_image_id,
        working_image_png=working_image_png,
    )
    with _lock:
        _evict_locked()
        _sessions[session_id] = _Entry(session=session)
    return session_id, session


def get_edit_session(session_id: str) -> EditSession | None:
    with _lock:
        entry = _sessions.get(session_id)
        if entry is None:
            return None
        entry.last_used = time.monotonic()
        return entry.session


def delete_edit_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def edit_session_count() -> int:
    with _lock:
        return len(_sessions)


def clear_edit_sessions() -> None:
    with _lock:
        _sessions.clear()
