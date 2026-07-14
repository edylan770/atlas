"""Search and interaction telemetry for admin analytics."""

__all__ = ["record_interaction", "record_search_from_results"]


def __getattr__(name: str):
    """Load recorder helpers lazily to avoid import cycles during CLI startup."""
    if name in __all__:
        from imagecb.telemetry import recorder

        return getattr(recorder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
