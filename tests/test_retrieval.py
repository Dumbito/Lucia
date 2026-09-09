from pathlib import Path

from lucia.context import Context
from lucia.memory import Memory
from lucia.retrieval import MemoryRetriever
from lucia.storage import SQLiteMemoryStore


def test_retrieval_uses_current_context(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    store.save(Memory(content="Lucía uses a local SQLite memory store.", kind="semantic", importance=0.9))
    store.save(Memory(content="The avatar will be connected later.", kind="project", importance=0.8))

    context = Context(active_goal="Proyecto Lucia", current_task="SQLite memory")
    memories = MemoryRetriever(store).retrieve_for_context(context)

    assert len(memories) == 1
    assert memories[0].content == "Lucía uses a local SQLite memory store."


def test_retrieval_returns_empty_without_query(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    store.save(Memory(content="A memory that should not be returned without context."))

    context = Context()
    assert MemoryRetriever(store).retrieve_for_context(context) == []
