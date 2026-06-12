"""Searchable caption document for an image record.

Shared by the BM25 sparse index and the caption-text dense lane so both
score against the same text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from imagecb.storage.metadata_db import ImageRecord


def caption_document_text(record: "ImageRecord") -> str:
    """Join grounded + interpretive caption fields into one document."""
    from imagecb.caption.asset_type import format_asset_type_label
    from imagecb.storage.metadata_db import deserialize_list

    grounded: List[str] = []
    asset_label = format_asset_type_label(record.asset_type)
    if asset_label:
        grounded.append(f"asset_type: {asset_label}")
    for v in (
        record.scene,
        record.text_overlay_summary,
        record.ocr_text,
        record.slide_title,
        record.slide_notes,
        record.slide_body_text,
    ):
        if v:
            grounded.append(v)
    grounded.extend(deserialize_list(record.objects_json))

    interpretive: List[str] = []
    for v in (
        record.image_name,
        record.caption_short,
        record.caption_detailed,
        record.theme,
        record.use_case,
    ):
        if v:
            interpretive.append(v)
    interpretive.extend(deserialize_list(record.tags_json))
    interpretive.extend(deserialize_list(record.recommended_cases_json))
    interpretive.extend(deserialize_list(record.search_aliases_json))

    return " \n ".join(grounded + interpretive)
