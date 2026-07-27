"""Admin API key authentication."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from imagecb.config import SETTINGS


def _extract_admin_key(
    authorization: Optional[str],
    x_admin_api_key: Optional[str],
) -> Optional[str]:
    if x_admin_api_key:
        return x_admin_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def require_admin(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-Api-Key"),
) -> str:
    """FastAPI dependency: valid admin API key required."""
    if not SETTINGS.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured (set ADMIN_API_KEY)",
        )
    key = _extract_admin_key(authorization, x_admin_api_key)
    if not key:
        raise HTTPException(status_code=401, detail="Admin API key required")
    if not secrets.compare_digest(key, SETTINGS.admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    return "admin-api"


def resolve_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> str:
    """Optional end-user id for telemetry (not authentication)."""
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()[:256]
    return "anonymous"
