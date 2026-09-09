"""Persistent-memory abstractions.

The memory contract stays independent from the retrieval implementation so
storage and embedding backends can evolve independently.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True)
class Memory:
    """A single persistent memory item."""

    content: str
    kind: str = "episodic"
    importance: float = 0.5
    confidence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    """Contract implemented by persistent memory backends."""

    def save(self, memory: Memory) -> None: ...

    def search(self, query: str, limit: int = 5) -> list[Memory]: ...

    def list_all(self) -> list[Memory]: ...
