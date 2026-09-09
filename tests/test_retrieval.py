from pathlib import Path

from lucia.context import Context
from lucia.embeddings import normalize_text
from lucia.memory import Memory
from lucia.retrieval import MemoryRetriever
from lucia.storage import SQLiteMemoryStore


class FakeEmbeddingProvider:
    """Tiny deterministic provider for testing the dense retrieval path."""

    vectors = {
        "arquitectura procesamiento local": (1.0, 0.0, 0.0),
        "la computacion debe ejecutarse en el equipo del usuario": (0.98, 0.05, 0.0),
        "el avatar tendra una interfaz visual": (0.0, 1.0, 0.0),
    }

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors.get(normalize_text(text).strip(".?!"), (0.0, 0.0, 1.0))


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        return super().embed(text)


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


def test_retrieval_handles_accents_and_partial_context(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    store.save(
        Memory(
            content="Lucía debe ser principalmente local.",
            kind="preference",
            importance=0.72,
        )
    )

    context = Context(active_goal="Arquitectura Lucia", current_task="procesamiento local")
    memories = MemoryRetriever(store).retrieve_for_context(context)

    assert len(memories) == 1
    assert memories[0].content == "Lucía debe ser principalmente local."


def test_dense_retrieval_does_not_require_shared_words(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    store.save(
        Memory(
            content="La computación debe ejecutarse en el equipo del usuario.",
            kind="preference",
            importance=0.8,
        )
    )
    store.save(Memory(content="El avatar tendrá una interfaz visual.", kind="project", importance=0.95))

    provider = FakeEmbeddingProvider()
    retriever = MemoryRetriever(
        store,
        limit=1,
        min_similarity=0.5,
        embedding_provider=provider,
    )

    memories = retriever.retrieve(goal="arquitectura procesamiento local", task=None)

    assert len(memories) == 1
    assert memories[0].content == "La computación debe ejecutarse en el equipo del usuario."


def test_dense_embeddings_are_cached_in_sqlite(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    store.save(Memory(content="La computación debe ejecutarse en el equipo del usuario.", importance=0.8))
    provider = CountingEmbeddingProvider()
    retriever = MemoryRetriever(
        store,
        limit=1,
        min_similarity=0.5,
        embedding_provider=provider,
        embedding_model_name="fake-v1",
    )

    first = retriever.retrieve(goal="arquitectura procesamiento local")
    first_calls = provider.calls
    second = retriever.retrieve(goal="arquitectura procesamiento local")

    assert first and second
    assert first_calls == 2
    assert provider.calls == first_calls + 1
    assert store.list_all()[0].embedding is not None
