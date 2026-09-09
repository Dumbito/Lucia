"""Deterministic policy for deciding which attended events become memory."""

from __future__ import annotations

from dataclasses import dataclass

from .events import Event


@dataclass(slots=True)
class MemoryGate:
    """Select memory-worthy events without storing transient system noise."""

    salience_threshold: float = 0.75
    rememberable_types: frozenset[str] = frozenset(
        {
            "memory.candidate",
            "user.fact",
            "user.preference",
            "project.milestone",
            "decision.made",
        }
    )

    def should_remember(self, event: Event, salience: float) -> bool:
        """Return whether an event is eligible for long-term memory."""
        if not 0.0 <= salience <= 1.0:
            raise ValueError("salience must be between 0 and 1")
        if event.type not in self.rememberable_types:
            return False
        return salience >= self.salience_threshold

    def to_memory_fields(self, event: Event, salience: float) -> dict:
        """Build conservative Memory fields from an approved event."""
        if not self.should_remember(event, salience):
            raise ValueError("event is not approved for memory")

        kind = {
            "user.fact": "semantic",
            "user.preference": "preference",
            "project.milestone": "project",
            "decision.made": "procedural",
            "memory.candidate": "episodic",
        }[event.type]
        return {
            "content": str(event.data.get("content", event.type)),
            "kind": kind,
            "importance": salience,
            "confidence": float(event.data.get("confidence", 0.8)),
            "metadata": {
                "source": event.source,
                "event_type": event.type,
                "event_timestamp": event.timestamp.isoformat(),
            },
        }
