"""Resolve which static UI bundle serve-web should mount."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

_PKG_WEB = Path(__file__).resolve().parent.parent / "web"
_SHIPPED_REACT = _PKG_WEB / "frontend_dist"
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class StaticUiKind(str, Enum):
    REACT = "react"
    NONE = "none"


def resolve_static_dir() -> tuple[Path | None, StaticUiKind]:
    """Return (directory, kind) for the best available UI bundle."""
    cwd_dist = (Path.cwd() / "frontend" / "dist").resolve()
    if cwd_dist.is_dir() and (cwd_dist / "index.html").is_file():
        return cwd_dist, StaticUiKind.REACT
    if _SHIPPED_REACT.is_dir() and (_SHIPPED_REACT / "index.html").is_file():
        return _SHIPPED_REACT, StaticUiKind.REACT
    if _FRONTEND_DIST.is_dir() and (_FRONTEND_DIST / "index.html").is_file():
        return _FRONTEND_DIST, StaticUiKind.REACT
    return None, StaticUiKind.NONE


def _react_bundle_label(static_dir: Path) -> str:
    """Human-readable label for the mounted React bundle (ATLAS vs legacy)."""
    try:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
    except OSError:
        return "React"
    if "ATLAS" in html.upper():
        return "ATLAS (React, imagecb/web/frontend_dist)"
    if static_dir == _SHIPPED_REACT:
        return "React (imagecb/web/frontend_dist)"
    return "React (frontend/dist)"


def react_bundle_has_deck_route(static_dir: Path) -> bool:
    """True if the mounted React bundle includes the /deck client route."""
    assets = static_dir / "assets"
    if not assets.is_dir():
        return False
    for path in assets.glob("*.js"):
        try:
            if "/deck" in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def warn_if_deck_route_missing() -> str | None:
    """Return a warning message when the React bundle lacks /deck, else None."""
    static, kind = resolve_static_dir()
    if kind != StaticUiKind.REACT or static is None:
        return None
    if react_bundle_has_deck_route(static):
        return None
    return (
        "WARNING: UI bundle is missing the /deck route. "
        "Run: cd frontend && npm run build && cd .. && python scripts/sync_frontend_dist.py"
    )


def format_serve_web_urls(*, host: str, port: int) -> list[str]:
    """Lines to print when starting serve-web."""
    static, kind = resolve_static_dir()
    base = f"http://{host}:{port}"
    lines = [f"Chat UI:   {base}/"]
    if kind == StaticUiKind.REACT and static is not None:
        lines.append(f"UI bundle: {_react_bundle_label(static)}")
        lines.append(f"Admin UI:  {base}/admin")
        lines.append(f"Deck suggest: {base}/deck")
        return lines
    lines.append(
        "Admin UI:  not available (missing bundled React build in imagecb/web/frontend_dist)"
    )
    return lines
