"""Caption quality assessment for auto-flagging weak outputs."""

from __future__ import annotations

import re
from typing import List, Literal, Tuple, TYPE_CHECKING

from imagecb.storage.metadata_db import ImageRecord, deserialize_list

if TYPE_CHECKING:
    from imagecb.models.vlm import CaptionJSON

CAPTION_FAILED = "[caption failed]"
_VLM_ERROR_PREFIX = "VLM error:"

_GENERIC_PATTERNS = [
    re.compile(r"^an?\s+(image|photo|picture|illustration|graphic)\s+(of|showing)\b", re.I),
    re.compile(r"^this\s+(image|photo|picture)\s+(shows|depicts)\b", re.I),
    re.compile(r"^various\s+objects\b", re.I),
    re.compile(r"^a\s+scene\s+with\b", re.I),
    re.compile(r"^a\s+(visual|graphic)\b", re.I),
    re.compile(r"^stock\s+", re.I),
    re.compile(r"^screenshot\b", re.I),
]


def caption_json_from_record(record: ImageRecord) -> "CaptionJSON":
    """Rebuild CaptionJSON from a persisted SQLite row (no VLM call)."""
    from imagecb.caption.asset_type import normalize_asset_type
    from imagecb.models.vlm import CaptionJSON, GroundedCaption, InterpretiveCaption, SearchTerms

    return CaptionJSON(
        image_name=(record.image_name or "").strip(),
        grounded=GroundedCaption(
            objects=deserialize_list(record.objects_json),
            scene=(record.scene or "").strip(),
            readable_text=(record.text_overlay_summary or "").strip(),
            text_read_uncertain=bool(record.text_read_uncertain),
            asset_type=normalize_asset_type(record.asset_type),
        ),
        interpretive=InterpretiveCaption(
            theme=(record.theme or "").strip(),
            use_case=(record.use_case or "").strip(),
            short_caption=(record.caption_short or "").strip(),
            detailed_description=(record.caption_detailed or "").strip(),
        ),
        search=SearchTerms(
            tags=deserialize_list(record.tags_json),
            recommended_cases=deserialize_list(record.recommended_cases_json),
            aliases=deserialize_list(record.search_aliases_json),
        ),
        caption_quality=(record.caption_quality or "ok").lower(),
    )


def needs_regeneration(quality: str | None) -> bool:
    q = (quality or "").lower()
    return q in ("weak", "failed")


def assess_caption_with_reasons(caption: "CaptionJSON") -> Tuple[Literal["ok", "weak", "failed"], List[str]]:
    short = (caption.short_caption or "").strip()
    detailed = (caption.detailed_description or "").strip()

    if short == CAPTION_FAILED or detailed.startswith(_VLM_ERROR_PREFIX):
        return "failed", ["caption_failed_marker"]

    reasons: List[str] = []
    name = (caption.image_name or "").strip()
    scene = (caption.scene or "").strip()
    objects = [o for o in (caption.objects or []) if (o or "").strip()]

    if not name:
        reasons.append("missing_image_name")
    if not short:
        reasons.append("missing_short_caption")
    elif len(short) < 15 or len(short.split()) < 4:
        reasons.append("short_caption_too_brief")
    if not detailed:
        reasons.append("missing_detailed_description")

    for pat in _GENERIC_PATTERNS:
        if short and pat.search(short):
            reasons.append("generic_caption_opener")
            break

    if not scene and not objects:
        reasons.append("missing_scene_and_objects")

    if caption.text_read_uncertain and (caption.readable_text or "").strip() and not objects:
        reasons.append("uncertain_text_without_objects")

    tags = [t for t in (caption.tags or []) if (t or "").strip()]
    recommended = [c for c in (caption.recommended_cases or []) if (c or "").strip()]
    aliases = [a for a in (caption.aliases or []) if (a or "").strip()]
    if len(tags) < 3:
        reasons.append("insufficient_tags")
    if len(recommended) < 3:
        reasons.append("insufficient_recommended_cases")
    if len(aliases) < 2:
        reasons.append("insufficient_aliases")

    if reasons:
        return "weak", reasons
    return "ok", []


def assess_caption(caption: "CaptionJSON") -> Literal["ok", "weak", "failed"]:
    quality, _ = assess_caption_with_reasons(caption)
    return quality


__all__ = [
    "CAPTION_FAILED",
    "assess_caption",
    "assess_caption_with_reasons",
    "caption_json_from_record",
    "needs_regeneration",
]
