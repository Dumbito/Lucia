"""Local persistent storage for Lucía's memories."""

from __future__ import annotations

import json
import sqlite3
import struct
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
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB,
                    embedding_model TEXT
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
            if "embedding" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            if "embedding_model" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN embedding_model TEXT")

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        embedding_blob = row["embedding"]
        embedding = None
        if embedding_blob:
            count = len(embedding_blob) // struct.calcsize("d")
            embedding = tuple(struct.unpack(f"{count}d", embedding_blob))
        return Memory(
            content=row["content"],
            kind=row["kind"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]),
            embedding=embedding,
        )

    def save(self, memory: Memory) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memories
                   (content, kind, importance, confidence, created_at, metadata, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.content,
                    memory.kind,
                    memory.importance,
                    memory.confidence,
                    memory.created_at.isoformat(),
                    json.dumps(memory.metadata),
                    self._pack_embedding(memory.embedding),
                ),
            )

    @staticmethod
    def _pack_embedding(embedding: tuple[float, ...] | None) -> bytes | None:
        if embedding is None:
            return None
        return struct.pack(f"{len(embedding)}d", *embedding)

    def list_all(self) -> list[Memory]:
        """Return all memories for model-based retrieval."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, content, kind, importance, confidence, created_at, metadata,
                          embedding, embedding_model
                   FROM memories
                   ORDER BY id DESC"""
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_embedding(self, memory: Memory, model_name: str) -> tuple[float, ...] | None:
        """Return a cached embedding only when it belongs to this model."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT embedding, embedding_model FROM memories
                   WHERE content = ? AND created_at = ?
                   ORDER BY id DESC LIMIT 1""",
                (memory.content, memory.created_at.isoformat()),
            ).fetchone()
        if row is None or row["embedding_model"] != model_name or not row["embedding"]:
            return None
        blob = row["embedding"]
        count = len(blob) // struct.calcsize("d")
        return tuple(struct.unpack(f"{count}d", blob))

    def save_embedding(self, memory: Memory, model_name: str, embedding: tuple[float, ...]) -> None:
        """Persist a dense embedding alongside its source memory."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE memories SET embedding = ?, embedding_model = ?
                   WHERE content = ? AND created_at = ?""",
                (
                    self._pack_embedding(embedding),
                    model_name,
                    memory.content,
                    memory.created_at.isoformat(),
                ),
            )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for forgiving local lexical retrieval."""
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Return memories matching meaningful query terms."""
        if limit <= 0:
            return []

        normalized_terms = {
            term
            for term in self._normalize(query).split()
            if len(term) >= 3
        }
        if not normalized_terms:
            return []

        matches: list[tuple[int, sqlite3.Row]] = []
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, content, kind, importance, confidence, created_at, metadata,
                          embedding, embedding_model
                   FROM memories
                   ORDER BY importance DESC, id DESC"""
            ).fetchall()

        for row in rows:
            normalized_content = self._normalize(row["content"])
            matched_terms = sum(term in normalized_content for term in normalized_terms)
            if matched_terms:
                matches.append((matched_terms, row))

        matches.sort(key=lambda item: (item[0], item[1]["importance"], item[1]["id"]), reverse=True)
        return [self._row_to_memory(row) for _, row in matches[:limit]]
