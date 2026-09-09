"""Utilities for creating explicit long-term memory candidates."""

from __future__ import annotations

from .events import Event


def user_preference(content: str, *, confidence: float = 0.9, source: str = "user") -> Event:
    """Create an explicit preference candidate for the memory pipeline."""
    if not content.strip():
        raise ValueError("content must not be empty")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return Event(
        type="user.preference",
        data={"content": content.strip(), "confidence": confidence},
        source=source,
    )


def user_fact(content: str, *, confidence: float = 0.9, source: str = "user") -> Event:
    """Create an explicit factual candidate for the memory pipeline."""
    if not content.strip():
        raise ValueError("content must not be empty")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return Event(
        type="user.fact",
        data={"content": content.strip(), "confidence": confidence},
        source=source,
    )
