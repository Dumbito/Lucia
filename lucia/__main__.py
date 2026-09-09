"""Command-line entry point for Lucía."""

from .context import Context
from .core import LuciaCore
from .memory import Memory, MemoryStore


class InMemoryStore(MemoryStore):
    """Temporary backend used only by the bootstrap CLI."""

    def __init__(self) -> None:
        self.items: list[Memory] = []

    def save(self, memory: Memory) -> None:
        self.items.append(memory)

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        terms = query.lower().split()
        matches = [m for m in self.items if all(t in m.content.lower() for t in terms)]
        return matches[:limit]


def main() -> None:
    store = InMemoryStore()
    lucia = LuciaCore(memory=store)
    context: Context = lucia.observe({"type": "system.started"})

    print(f"{lucia.name} v0.1.0")
    print(f"Context events: {len(context.events)}")
    print("Core online.")


if __name__ == "__main__":
    main()
