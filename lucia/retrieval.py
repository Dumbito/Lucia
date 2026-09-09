"""Hybrid lexical/semantic retrieval for Lucía's working context."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    dense_cosine_similarity,
    text_vector,
)
from .memory import Memory, MemoryStore


@dataclass(slots=True)
class MemoryRetriever:
    """Retrieve persistent memories relevant to the current task.

    Dense embeddings are cached by the storage backend when possible. A cache
    miss computes the vector once and persists it, so later queries do not
    re-encode the same memory.
    """

    store: MemoryStore
    limit: int = 5
    min_similarity: float = 0.15
    embedding_provider: EmbeddingProvider | None = None
    embedding_model_name: str | None = None

    def retrieve(self, *, goal: str | None = None, task: str | None = None) -> list[Memory]:
        """Rank memories by semantic similarity and importance."""
        query = " ".join(part.strip() for part in (task, goal) if part and part.strip())
        if not query:
            return []

        if self.embedding_provider is None:
            return self._retrieve_lexical(query)

        query_vector = self.embedding_provider.embed(query)
        candidates = self.store.list_all()
        ranked: list[tuple[float, Memory]] = []
        model_name = self.embedding_model_name or self.embedding_provider.__class__.__name__

        for memory in candidates:
            embedding = self.store.get_embedding(memory, model_name)
            if embedding is None:
                embedding = self.embedding_provider.embed(memory.content)
                self.store.save_embedding(memory, model_name, embedding)

            similarity = dense_cosine_similarity(query_vector, embedding)
            if similarity < self.min_similarity:
                continue
            score = similarity * 0.75 + memory.importance * 0.25
            ranked.append((score, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in ranked[: self.limit]]

    def _retrieve_lexical(self, query: str) -> list[Memory]:
        """Dependency-free fallback used before a model is configured."""
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
