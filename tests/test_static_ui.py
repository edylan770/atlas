"""Static UI resolution for serve-web."""

from __future__ import annotations

from pathlib import Path

from imagecb.api.static_ui import StaticUiKind, react_bundle_has_deck_route, resolve_static_dir


def test_resolve_static_prefers_shipped_frontend_dist():
    static, kind = resolve_static_dir()
    shipped = Path(__file__).resolve().parents[1] / "imagecb" / "web" / "frontend_dist"
    assert shipped.is_dir()
    assert kind == StaticUiKind.REACT
    assert static == shipped


def test_shipped_frontend_dist_is_atlas_bundle():
    index = (
        Path(__file__).resolve().parents[1] / "imagecb" / "web" / "frontend_dist" / "index.html"
    )
    assert index.is_file()
    html = index.read_text(encoding="utf-8")
    assert "ATLAS" in html.upper()


def test_shipped_frontend_dist_includes_deck_route():
    shipped = Path(__file__).resolve().parents[1] / "imagecb" / "web" / "frontend_dist"
    assert react_bundle_has_deck_route(shipped), (
        "Shipped bundle missing /deck. Run: cd frontend && npm run build && "
        "python scripts/sync_frontend_dist.py"
    )


def test_spa_routes_serve_index_and_api_404s_stay_json():
    from fastapi.testclient import TestClient

    from imagecb.api.server import create_app

    client = TestClient(create_app())
    for route in ("/admin", "/deck", "/admin/ingestions"):
        r = client.get(route)
        assert r.status_code == 200, route
        assert "text/html" in r.headers["content-type"], route
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    # a missing asset must 404, not silently become index.html
    r = client.get("/assets/index-STALEHASH.js")
    assert r.status_code == 404
