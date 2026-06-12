"""JSON Schema for enforced VLM caption structured output."""

from __future__ import annotations

from typing import Any, Dict

from imagecb.caption.asset_type import ASSET_TYPES

CAPTION_TOOL_NAME = "emit_caption"

# JSON Schema for tool / structured output (all required fields populated).
CAPTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_name": {
            "type": "string",
            "description": "Short human-friendly title (<= 8 words).",
        },
        "grounded": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Salient visible objects/entities only.",
                },
                "scene": {
                    "type": "string",
                    "description": "Short phrase for visible scene/setting.",
                },
                "readable_text": {
                    "type": "string",
                    "description": "Text actually legible in the image; empty if none.",
                },
                "text_read_uncertain": {
                    "type": "boolean",
                    "description": "True when text is present but partially illegible.",
                },
                "asset_type": {
                    "type": "string",
                    "enum": list(ASSET_TYPES),
                    "description": "Visual format; exactly one value from the closed list.",
                },
            },
            "required": [
                "objects",
                "scene",
                "readable_text",
                "text_read_uncertain",
                "asset_type",
            ],
            "additionalProperties": False,
        },
        "interpretive": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "High-level subject/topic (inference allowed).",
                },
                "use_case": {
                    "type": "string",
                    "description": "Likely business or creative use case.",
                },
                "short_caption": {
                    "type": "string",
                    "description": "<= 20 words, single sentence.",
                },
                "detailed_description": {
                    "type": "string",
                    "description": "1-3 sentences.",
                },
            },
            "required": ["theme", "use_case", "short_caption", "detailed_description"],
            "additionalProperties": False,
        },
        "search": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 10,
                    "description": "3-10 lowercase tags; prefer corpus vocabulary on full literal match.",
                },
                "recommended_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 6,
                    "description": "3-6 natural-language queries a searcher would type (primary retrieval field).",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Alternate names, synonyms, or spelled-out acronyms a searcher might use.",
                },
            },
            "required": ["tags", "recommended_cases", "aliases"],
            "additionalProperties": False,
        },
    },
    "required": ["image_name", "grounded", "interpretive", "search"],
    "additionalProperties": False,
}


def validate_caption_dict(data: dict) -> bool:
    """Lightweight structural validation without external deps."""
    if not isinstance(data, dict):
        return False
    for key in ("image_name", "grounded", "interpretive", "search"):
        if key not in data:
            return False
    g = data.get("grounded")
    i = data.get("interpretive")
    s = data.get("search")
    if not isinstance(g, dict) or not isinstance(i, dict) or not isinstance(s, dict):
        return False
    for key in ("objects", "scene", "readable_text", "text_read_uncertain", "asset_type"):
        if key not in g:
            return False
    if not isinstance(g["objects"], list) or not isinstance(g["text_read_uncertain"], bool):
        return False
    if len(g["objects"]) < 1:
        return False
    if g.get("asset_type") not in ASSET_TYPES:
        return False
    for key in ("theme", "use_case", "short_caption", "detailed_description"):
        if key not in i:
            return False
    for key in ("tags", "recommended_cases", "aliases"):
        if key not in s or not isinstance(s[key], list):
            return False
    if not (3 <= len(s["tags"]) <= 10):
        return False
    if not (3 <= len(s["recommended_cases"]) <= 6):
        return False
    if len(s["aliases"]) < 2:
        return False
    return True
