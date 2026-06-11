"""Dynamic query suggestions (empty-state and follow-up)."""

from imagecb.suggestions.follow_up import generate_follow_up_suggestions
from imagecb.suggestions.generate import generate_suggestions

__all__ = ["generate_follow_up_suggestions", "generate_suggestions"]
