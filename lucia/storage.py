"""Local persistent storage for Lucía's memories."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

from .memory import Memory, MemoryStore


class SQLiteMemoryStore(MemoryStore):
    """Small SQLite backend; no external database dependency required."""

    def __init__(self, path: str | Path = "data/lucia.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    def save(self, memory: Memory) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memories
                   (content, kind, importance, confidence, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory.content,
                    memory.kind,
                    memory.importance,
                    memory.confidence,
                    memory.created_at.isoformat(),
                    json.dumps(memory.metadata),
                ),
            )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for forgiving local lexical retrieval."""
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Return memories matching meaningful query terms.

        Retrieval is intentionally lexical for now, but it is accent-insensitive
        and token-aware. This lets "Lucia" retrieve "Lucía" and avoids requiring
        the entire goal string to occur verbatim in a memory. Semantic retrieval
        can replace this implementation later without changing MemoryStore.
        """
        if limit <= 0:
            return []

        normalized_terms = {
            term
            for term in self._normalize(query).split()
            if len(term) >= 3
        }
        if not normalized_terms:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, content, kind, importance, confidence, created_at, metadata
                   FROM memories
                   ORDER BY importance DESC, id DESC"""
            ).fetchall()

        matches: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            normalized_content = self._normalize(row["content"])
            matched_terms = sum(term in normalized_content for term in normalized_terms)
            if matched_terms:
                matches.append((matched_terms, row))

        matches.sort(key=lambda item: (item[0], item[1]["importance"], item[1]["id"]), reverse=True)
        return [
            Memory(
                content=row["content"],
                kind=row["kind"],
                importance=row["importance"],
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"]),
            )
            for _, row in matches[:limit]
        ]
