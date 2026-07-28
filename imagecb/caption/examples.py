"""Curated few-shot caption examples for VLM prompt consistency."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# (label, example dict) — labels are one-line scene descriptions for the prompt.
CAPTION_FEW_SHOT_EXAMPLES: List[Tuple[str, Dict[str, Any]]] = [
    (
        "Standalone photo: team meeting in a modern office",
        {
            "image_name": "Team Meeting Photo",
            "grounded": {
                "objects": ["people", "conference table", "laptop"],
                "scene": "office meeting room",
                "readable_text": "",
                "text_read_uncertain": False,
                "asset_type": "photo",
            },
            "interpretive": {
                "theme": "workplace collaboration",
                "use_case": "internal communications or HR slide",
                "short_caption": "Group of colleagues seated around a table in an office",
                "detailed_description": (
                    "Five people converse around a conference table with laptops open. "
                    "Large windows and neutral decor suggest a corporate office setting."
                ),
            },
            "search": {
                "tags": ["people", "office", "meeting", "team", "collaboration"],
                "recommended_cases": [
                    "team meeting photo",
                    "office collaboration image",
                    "colleagues in conference room",
                ],
                "aliases": [
                    "workplace meeting",
                    "staff discussion",
                    "corporate team photo",
                ],
            },
        },
    ),
    (
        "Standalone illustration: flat vector process flowchart (PNG)",
        {
            "image_name": "Process Flow Diagram",
            "grounded": {
                "objects": ["flowchart", "arrows", "labeled boxes"],
                "scene": "white background infographic",
                "readable_text": "Start, Review, Approve",
                "text_read_uncertain": False,
                "asset_type": "diagram",
            },
            "interpretive": {
                "theme": "workflow process",
                "use_case": "operations or training material",
                "short_caption": "Three-step flowchart with arrows connecting labeled stages",
                "detailed_description": (
                    "A horizontal flowchart shows Start, Review, and Approve stages "
                    "connected by arrows on a plain white background."
                ),
            },
            "search": {
                "tags": ["diagram", "process", "vector", "illustration", "arrow"],
                "recommended_cases": [
                    "process flowchart diagram",
                    "workflow steps illustration",
                    "approval process graphic",
                ],
                "aliases": [
                    "flowchart",
                    "infographic",
                    "workflow",
                    "process map",
                    "procedure diagram",
                    "operational workflow chart",
                ],
            },
        },
    ),
    (
        "Presentation slide: quarterly sales bar chart",
        {
            "image_name": "Quarterly Sales Chart",
            "grounded": {
                "objects": ["bar chart", "axis labels", "legend"],
                "scene": "presentation slide",
                "readable_text": (
                    "Quarterly Sales by Region\nQ3 2024\nRevenue ($M)\n"
                    "0 20 40 60 80\nNorth America\nEMEA\nAPAC\nLATAM"
                ),
                "text_read_uncertain": False,
                "asset_type": "chart",
            },
            "interpretive": {
                "theme": "sales performance",
                "use_case": "quarterly business review",
                "short_caption": "Bar chart of quarterly sales by region",
                "detailed_description": (
                    "Colorful vertical bars compare sales across regions for each quarter. "
                    "A legend and axis labels frame the chart on a slide layout."
                ),
            },
            "search": {
                "tags": ["chart", "sales", "quarterly", "bar", "slide"],
                "recommended_cases": [
                    "quarterly sales chart",
                    "sales by region bar graph",
                    "Q3 performance slide",
                ],
                "aliases": [
                    "revenue",
                    "Q3 results",
                    "key performance indicator",
                ],
            },
        },
    ),
    (
        "Presentation slide: text-dense pricing table",
        {
            "image_name": "Support Plan Pricing Table",
            "grounded": {
                "objects": ["table", "column headers", "price figures"],
                "scene": "presentation slide",
                "readable_text": (
                    "Support Plans\nPlan | Monthly | Response SLA | Channels\n"
                    "Basic | $99 | 48 hours | Email\n"
                    "Business | $499 | 8 hours | Email, Phone\n"
                    "Enterprise | $1,999 | 1 hour | Email, Phone, Dedicated TAM\n"
                    "All prices in USD. Annual billing saves 15%."
                ),
                "text_read_uncertain": False,
                "asset_type": "table",
            },
            "interpretive": {
                "theme": "product pricing",
                "use_case": "sales proposal or pricing discussion",
                "short_caption": "Three-tier support plan pricing table with SLAs",
                "detailed_description": (
                    "A pricing table compares Basic, Business, and Enterprise "
                    "support plans by monthly cost, response SLA, and support "
                    "channels, with an annual billing discount note."
                ),
            },
            "search": {
                "tags": ["table", "pricing", "plan", "support", "cost"],
                "recommended_cases": [
                    "support plan pricing table",
                    "tiered pricing comparison",
                    "service level agreement table",
                ],
                "aliases": ["price list", "SLA tiers", "rate card"],
            },
        },
    )
]


def format_few_shot_for_prompt(
    examples: List[Tuple[str, Dict[str, Any]]] | None = None,
) -> str:
    """Render few-shot examples as compact JSON blocks for the VLM user prompt."""
    items = examples if examples is not None else CAPTION_FEW_SHOT_EXAMPLES
    parts: List[str] = []
    for i, (label, data) in enumerate(items, start=1):
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        parts.append(f"Example {i} ({label}):\n{payload}")
    return "\n\n".join(parts)


__all__ = ["CAPTION_FEW_SHOT_EXAMPLES", "format_few_shot_for_prompt"]
