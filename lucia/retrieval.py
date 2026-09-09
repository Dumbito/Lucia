"""Hybrid lexical/semantic retrieval for Lucía's working context."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import cosine_similarity, text_vector
from .memory import Memory, MemoryStore


@dataclass(slots=True)
class MemoryRetriever:
    """Retrieve persistent memories relevant to the current task.

    The first semantic layer is deliberately local and dependency-free:
    normalized sparse text vectors provide similarity without requiring an
    embedding model. A real local embedding backend can replace this later.
    """

    store: MemoryStore
    limit: int = 5
    min_similarity: float = 0.15

    def retrieve(self, *, goal: str | None = None, task: str | None = None) -> list[Memory]:
        """Rank memories by normalized token similarity and importance."""
        query = " ".join(part.strip() for part in (task, goal) if part and part.strip())
        if not query:
            return []

        # Request a broad lexical candidate set, then rank it semantically.
        candidates = self.store.search(query, limit=max(self.limit * 5, 25))
        query_vector = text_vector(query)
        ranked: list[tuple[float, Memory]] = []

        for memory in candidates:
            similarity = cosine_similarity(query_vector, text_vector(memory.content))
            if similarity < self.min_similarity:
                continue
            score = similarity * 0.75 + memory.importance * 0.25
            ranked.append((score, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in ranked[: self.limit]]

    def retrieve_for_context(self, context: "Context") -> list[Memory]:
        """Retrieve memories for a Lucía working context."""
        return self.retrieve(goal=context.active_goal, task=context.current_task)


from .context import Context
