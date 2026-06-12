"""Closed visual asset-type taxonomy for ingest classification and search filters."""

from __future__ import annotations

from typing import List, Sequence

from imagecb.caption.normalize import normalize_tag

ASSET_TYPES: tuple[str, ...] = (
    "photo",
    "diagram",
    "chart",
    "screenshot",
    "logo",
    "illustration",
    "icon",
    "table",
    "map",
    "other",
)

DEFAULT_ASSET_TYPE = "other"

# Bump only when ASSET_TYPES changes; pair with backfill --all.
ASSET_TYPE_TAXONOMY_VERSION = 1

# Audit / freeze thresholds.
OTHER_WARN_PCT = 15.0
UNCLASSIFIED_WARN_PCT = 5.0
CONFUSION_PAIR_MIN = 10

ASSET_TYPE_SET = frozenset(ASSET_TYPES)

# Colloquial / near-miss terms -> canonical value (query + post-VLM normalization).
_SYNONYM_MAP: dict[str, str] = {
    "photograph": "photo",
    "photography": "photo",
    "picture": "photo",
    "pictures": "photo",
    "photos": "photo",
    "image": "photo",
    "images": "photo",
    "flowchart": "diagram",
    "flow chart": "diagram",
    "infographic": "diagram",
    "infographics": "diagram",
    "schematic": "diagram",
    "architecture diagram": "diagram",
    "process map": "diagram",
    "graph": "chart",
    "graphs": "chart",
    "bar chart": "chart",
    "bar graph": "chart",
    "line chart": "chart",
    "pie chart": "chart",
    "data visualization": "chart",
    "data viz": "chart",
    "screenshots": "screenshot",
    "screen capture": "screenshot",
    "ui capture": "screenshot",
    "logos": "logo",
    "wordmark": "logo",
    "brand mark": "logo",
    "illustrations": "illustration",
    "vector art": "illustration",
    "drawing": "illustration",
    "icons": "icon",
    "glyph": "icon",
    "pictogram": "icon",
    "tables": "table",
    "spreadsheet": "table",
    "maps": "map",
    "geographic map": "map",
}

_ASSET_TYPE_GUIDANCE: tuple[tuple[str, str], ...] = (
    ("photo", "Real-world photograph (people, places, products)"),
    ("chart", "Quantitative data viz (bar/line/pie/scatter)"),
    ("diagram", "Process/structure visuals (flowchart, architecture, infographic without numeric axes)"),
    ("screenshot", "Software UI / browser / app capture"),
    ("logo", "Brand mark or wordmark"),
    ("illustration", "Drawn/vector art that is not a diagram, icon, or logo"),
    ("icon", "Small symbolic glyph or pictogram"),
    ("table", "Tabular rows/columns of data"),
    ("map", "Geographic map"),
    ("other", "None of the above"),
)


def format_taxonomy_for_prompt() -> str:
    """One-line definitions for VLM / query prompts."""
    lines = [f"- {name}: {desc}" for name, desc in _ASSET_TYPE_GUIDANCE]
    allowed = ", ".join(ASSET_TYPES)
    return f"Allowed values (exactly one): {allowed}\n" + "\n".join(lines)


def normalize_asset_type(raw: str | None) -> str:
    """Map a raw label to a canonical asset type; unknown -> other."""
    if not raw or not str(raw).strip():
        return DEFAULT_ASSET_TYPE
    key = str(raw).strip().lower().replace("_", " ")
    if key in ASSET_TYPE_SET:
        return key
    if key in _SYNONYM_MAP:
        return _SYNONYM_MAP[key]
    # Partial token match for compound phrases (e.g. "quarterly bar chart slide").
    for synonym, canonical in sorted(_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if synonym in key:
            return canonical
    return DEFAULT_ASSET_TYPE


def resolve_query_asset_type(token: str) -> str | None:
    """Map a query token to an asset type when unambiguous; else None."""
    key = normalize_tag(token)
    if not key:
        return None
    if key in ASSET_TYPE_SET:
        return key
    if key in _SYNONYM_MAP:
        return _SYNONYM_MAP[key]
    return None


def normalize_asset_types(values: Sequence[str] | None) -> List[str]:
    """Dedupe and keep only allowed canonical values (for query filters)."""
    if not values:
        return []
    out: List[str] = []
    for v in values:
        normalized = normalize_asset_type(v)
        if normalized in ASSET_TYPE_SET and normalized not in out:
            out.append(normalized)
    return out


def format_asset_type_label(asset_type: str | None) -> str:
    """Human-readable chip label (e.g. photo -> Photo)."""
    if not asset_type or not str(asset_type).strip():
        return ""
    canonical = normalize_asset_type(asset_type)
    if canonical == DEFAULT_ASSET_TYPE:
        return "Other"
    return canonical.replace("_", " ").title()


def taxonomy_snapshot_path(version: int | None = None) -> str:
    """Default path for a frozen taxonomy manifest under data/."""
    v = version if version is not None else ASSET_TYPE_TAXONOMY_VERSION
    return f"data/asset_type_taxonomy_v{v}.json"


__all__ = [
    "ASSET_TYPES",
    "ASSET_TYPE_SET",
    "ASSET_TYPE_TAXONOMY_VERSION",
    "CONFUSION_PAIR_MIN",
    "DEFAULT_ASSET_TYPE",
    "OTHER_WARN_PCT",
    "UNCLASSIFIED_WARN_PCT",
    "format_asset_type_label",
    "format_taxonomy_for_prompt",
    "normalize_asset_type",
    "normalize_asset_types",
    "resolve_query_asset_type",
    "taxonomy_snapshot_path",
]
