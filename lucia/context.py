"""Working context for the current Lucía cycle."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Context:
    """Ephemeral state assembled for one or more reasoning cycles."""

    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_goal: str | None = None
    current_task: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.now = datetime.now(timezone.utc)
