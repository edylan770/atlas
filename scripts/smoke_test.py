"""Offline smoke tests — no Bedrock/API calls required."""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        FAILURES.append(f"{name}: {exc}")


def _make_test_png(path: Path, color: tuple[int, int, int], text: str) -> None:
    img = Image.new("RGB", (200, 120), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 50), text, fill=(255, 255, 255))
    img.save(path, format="PNG")


def test_image_extractor() -> None:
    from imagecb.extractors import image_file

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.png"
        _make_test_png(p, (30, 90, 200), "dashboard")
        items = list(image_file.extract(p))
        assert len(items) == 1
        assert items[0].provenance.source_type == "image"
        assert items[0].image.size == (200, 120)


def test_dispatch_iter_corpus() -> None:
    from imagecb.extractors.dispatch import iter_corpus

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_test_png(root / "a.png", (10, 10, 10), "a")
        (root / "skip.txt").write_text("nope", encoding="utf-8")
        found = {p.name for p in iter_corpus(root)}
        assert found == {"a.png"}


def test_bm25_roundtrip() -> None:
    from imagecb.storage.bm25_index import BM25Index, tokenize

    assert tokenize("Hello World-2024!") == ["hello", "world", "2024"]
    # BM25 needs a modest corpus for non-zero IDF; two docs often score 0.
    ids = [f"id{i}" for i in range(8)]
    texts = [
        "system architecture diagram internal deck",
        "bar chart quarterly revenue finance",
        "screenshot of monitoring dashboard grafana",
        "network topology map data center",
        "team photo offsite event",
        "table of contents document outline",
        "flowchart approval workflow process",
        "pie chart market share analysis",
    ]
    idx = BM25Index()
    idx.build(ids, texts)
    hits = idx.query("architecture diagram", top_k=5)
    assert hits, "expected non-empty BM25 hits on a multi-doc corpus"
    assert hits[0][0] == "id0"


def test_metadata_db_filter() -> None:
    import uuid

    from imagecb.storage import metadata_db
    from imagecb.storage.metadata_db import ImageRecord, new_image_id, session_scope

    rid = new_image_id()
    rec = ImageRecord(
        image_id=rid,
        content_hash=hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        image_path="/tmp/x.png",
        source_file="C:/docs/Q3_Review.pptx",
        source_type="pptx",
        source_modified_at=datetime(2026, 5, 8),
        author="Alice",
        caption_short="dashboard screenshot",
    )
    with session_scope() as s:
        s.merge(rec)
    ids = metadata_db.filter_image_ids(
        file_types=["pptx"],
        filename_contains=["q3_review"],
        authors=["alice"],
        modified_after=datetime(2026, 1, 1),
    )
    assert rid in ids


def test_vector_store_upsert_query() -> None:
    from imagecb.config import SETTINGS
    from imagecb.storage import vector_store

    dim = SETTINGS.embedding_dim
    iid = "smoke-test-" + hashlib.sha256(b"vec").hexdigest()[:12]
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    vector_store.upsert(
        image_ids=[iid],
        embeddings=emb.reshape(1, -1),
        metadatas=[{"image_id": iid, "source_type": "image", "source_file": "x.png", "author": ""}],
    )
    hits = vector_store.query(emb, top_k=3)
    assert any(h[0] == iid for h in hits)


def test_rrf_merge() -> None:
    from imagecb.retrieval.hybrid import rrf_merge

    merged = rrf_merge(
        [("a", 0.9), ("b", 0.8)],
        [("b", 1.0), ("c", 0.7)],
        k=60,
    )
    ids = [c.image_id for c in merged]
    assert ids[0] == "b"


def test_query_parser_fallback() -> None:
    from imagecb.retrieval.query_parser import _build_spec, parse_query

    spec = parse_query("")
    assert spec.semantic_query == ""
    spec2 = _build_spec(
        {
            "semantic_query": "dashboards",
            "must_have_keywords": ["chart"],
            "must_avoid_keywords": ["bar"],
            "source_filters": {"file_types": ["pptx"], "filename_contains": ["Q3"]},
            "time_filter": {"after": "2026-01-01"},
            "is_refinement": True,
            "top_k": 5,
        },
        "find dashboards",
    )
    assert spec2.semantic_query == "dashboards"
    assert spec2.source_filters.file_types == ["pptx"]
    assert spec2.is_refinement is True


def test_vlm_caption_json() -> None:
    from imagecb.models.vlm import CaptionJSON, _parse_json_lenient

    raw = '```json\n{"short_caption": "test", "objects": ["a"]}\n```'
    d = _parse_json_lenient(raw)
    assert d and d["short_caption"] == "test"
    cap = CaptionJSON.from_dict(d)
    assert cap.objects == ["a"]


def test_app_builds() -> None:
    from imagecb.app import build_ui

    demo = build_ui()
    assert demo is not None


def main() -> int:
    print("Running offline smoke tests...\n")
    check("image extractor", test_image_extractor)
    check("dispatch iter_corpus", test_dispatch_iter_corpus)
    check("BM25 index", test_bm25_roundtrip)
    check("metadata DB filter", test_metadata_db_filter)
    check("vector store upsert/query", test_vector_store_upsert_query)
    check("RRF merge", test_rrf_merge)
    check("query parser fallback", test_query_parser_fallback)
    check("VLM JSON parsing", test_vlm_caption_json)
    check("Gradio UI build", test_app_builds)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
