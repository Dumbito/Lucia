"""Memory retrieval for Lucía's working context."""

from __future__ import annotations

from dataclasses import dataclass

from .memory import Memory, MemoryStore


@dataclass(slots=True)
class MemoryRetriever:
    """Retrieve persistent memories relevant to the current task."""

    store: MemoryStore
    limit: int = 5

    def retrieve(self, *, goal: str | None = None, task: str | None = None) -> list[Memory]:
        """Search memory using the most specific available context."""
        query = " ".join(part.strip() for part in (task, goal) if part and part.strip())
        if not query:
            return []
        return self.store.search(query, limit=self.limit)

    def retrieve_for_context(self, context: "Context") -> list[Memory]:
        """Retrieve memories for a Lucía working context."""
        return self.retrieve(goal=context.active_goal, task=context.current_task)


from .context import Context
