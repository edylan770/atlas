"""Per-client sliding-window rate limiting for expensive endpoints.

Every request to an LLM-backed endpoint costs real money (Bedrock calls), so
anonymous traffic is capped per client IP. In-process state only — this
matches the app's single-process deployment; a multi-instance deployment
would need a shared store.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict

from fastapi import HTTPException, Request

from imagecb.config import SETTINGS

_lock = threading.Lock()
_hits: Dict[str, Deque[float]] = {}
_MAX_TRACKED_CLIENTS = 10_000
_WINDOW_SEC = 60.0


def _client_key(request: Request) -> str:
    # Behind a proxy/ALB the peer address is the proxy; prefer the original
    # client from X-Forwarded-For (first hop).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def check_llm_rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 when a client exceeds the per-minute budget."""
    limit = SETTINGS.llm_rate_limit_per_minute
    if limit <= 0:  # 0 disables limiting
        return
    key = _client_key(request)
    now = time.monotonic()
    with _lock:
        window = _hits.get(key)
        if window is None:
            if len(_hits) >= _MAX_TRACKED_CLIENTS:
                # Drop the stalest client rather than growing unbounded.
                stalest = min(_hits, key=lambda k: _hits[k][-1] if _hits[k] else 0.0)
                del _hits[stalest]
            window = deque()
            _hits[key] = window
        while window and now - window[0] > _WINDOW_SEC:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, int(_WINDOW_SEC - (now - window[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded; slow down and retry.",
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)


def reset() -> None:
    """Clear limiter state (tests)."""
    with _lock:
        _hits.clear()
