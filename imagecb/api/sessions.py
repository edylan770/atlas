"""In-memory chat session store with TTL eviction and a size cap.

Sessions are created for every anonymous chat request, so without eviction
the store (and the result records each session pins) grows for the lifetime
of the process.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict

from imagecb.config import SETTINGS
from imagecb.retrieval.session import ChatSession

_lock = Lock()


@dataclass
class _Entry:
    session: ChatSession
    last_used: float = field(default_factory=time.monotonic)


_sessions: Dict[str, _Entry] = {}


def _evict_locked() -> None:
    """Drop expired sessions; if still over cap, drop least-recently-used."""
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


def create_session() -> tuple[str, ChatSession]:
    session_id = str(uuid.uuid4())
    session = ChatSession()
    with _lock:
        _evict_locked()
        _sessions[session_id] = _Entry(session=session)
    return session_id, session


def get_session(session_id: str) -> ChatSession | None:
    with _lock:
        entry = _sessions.get(session_id)
        if entry is None:
            return None
        entry.last_used = time.monotonic()
        return entry.session


def get_or_create_session(session_id: str | None) -> tuple[str, ChatSession]:
    if session_id:
        session = get_session(session_id)
        if session is not None:
            return session_id, session
    return create_session()


def reset_session(session_id: str) -> ChatSession | None:
    session = get_session(session_id)
    if session is None:
        return None
    session.reset()
    return session


def session_count() -> int:
    with _lock:
        return len(_sessions)
