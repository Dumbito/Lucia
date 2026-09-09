"""Local persistent storage for Lucía's memories."""

import sqlite3
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
        import json

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

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        import json
        from datetime import datetime

        rows = []
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT content, kind, importance, confidence, created_at, metadata
                   FROM memories
                   WHERE content LIKE ?
                   ORDER BY importance DESC, id DESC
                   LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()

        return [
            Memory(
                content=row["content"],
                kind=row["kind"],
                importance=row["importance"],
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]
