"""Tests for BM25 index text including asset_type and aliases."""

from __future__ import annotations

from datetime import datetime

import imagecb.storage.bm25_index as bm25_module
from imagecb.storage.bm25_index import BM25Index, rebuild_from_records
from imagecb.storage.metadata_db import ImageRecord, serialize_list


def test_rebuild_from_records_includes_asset_type_and_aliases(monkeypatch):
    monkeypatch.setattr(bm25_module, "_index", None)
    monkeypatch.setattr(BM25Index, "save", lambda self, path=None: None)

    record = ImageRecord(
        image_id="img-1",
        content_hash="hash-1",
        image_path="/tmp/img-1.png",
        source_file="/docs/test.pptx",
        source_type="pptx",
        asset_type="diagram",
        caption_short="System architecture overview",
        tags_json=serialize_list(["architecture", "cloud"]),
        search_aliases_json=serialize_list(["schematic", "flowchart"]),
        recommended_cases_json=serialize_list(["cloud architecture diagram"]),
        created_at=datetime.utcnow(),
    )

    rebuild_from_records([record])
    idx = bm25_module.get_index()
    assert idx._state is not None
    doc_tokens = idx._state.docs[0]
    joined = " ".join(doc_tokens)
    for fragment in ("diagram", "schematic", "flowchart", "architecture", "cloud"):
        assert fragment in joined
