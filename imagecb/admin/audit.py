"""Append-only admin audit log."""

from __future__ import annotations

import json
import uuid
from typing import Any, List, Optional

from sqlalchemy import desc, select

from imagecb.storage.metadata_db import session_scope
from imagecb.telemetry.models import AdminAuditLog
from imagecb.telemetry.schema import ensure_telemetry_schema


def append_audit(
    *,
    actor: str,
    action: str,
    target_type: str,
    target_id: str,
    details: Optional[dict[str, Any]] = None,
) -> str:
    ensure_telemetry_schema()
    entry_id = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            AdminAuditLog(
                id=entry_id,
                actor=actor,
                action=action,
                target_type=target_type,
                target_id=target_id,
                details_json=json.dumps(details or {}, ensure_ascii=False),
            )
        )
    return entry_id


def list_audit_entries(*, limit: int = 100, offset: int = 0) -> List[dict]:
    ensure_telemetry_schema()
    with session_scope() as s:
        rows = s.execute(
            select(AdminAuditLog)
            .order_by(desc(AdminAuditLog.created_at))
            .limit(limit)
            .offset(offset)
        ).scalars().all()
        out: List[dict] = []
        for r in rows:
            details = {}
            if r.details_json:
                try:
                    details = json.loads(r.details_json)
                except json.JSONDecodeError:
                    details = {}
            out.append(
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "actor": r.actor,
                    "action": r.action,
                    "target_type": r.target_type,
                    "target_id": r.target_id,
                    "details": details,
                }
            )
        return out
