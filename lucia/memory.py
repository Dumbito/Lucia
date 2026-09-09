"""Persistent-memory abstractions.

The memory contract stays independent from the retrieval implementation so
storage and embedding backends can evolve independently.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True)
class Memory:
    """A single persistent memory item.

    ``embedding`` is optional so the same memory object works with the
    dependency-free lexical backend and with persistent dense retrieval.
    """

    content: str
    kind: str = "episodic"
    importance: float = 0.5
    confidence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None


class MemoryStore(Protocol):
    """Contract implemented by persistent memory backends."""

    def save(self, memory: Memory) -> None: ...

    def search(self, query: str, limit: int = 5) -> list[Memory]: ...

    def list_all(self) -> list[Memory]: ...

    def get_embedding(self, memory: Memory, model_name: str) -> tuple[float, ...] | None: ...

    def save_embedding(self, memory: Memory, model_name: str, embedding: tuple[float, ...]) -> None: ...
