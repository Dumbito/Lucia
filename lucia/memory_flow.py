"""End-to-end demonstration of Lucía's explicit memory flow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .attention_adapter import AttentionAdapter
from .context import Context
from .memory import Memory
from .memory_candidate import user_fact, user_preference
from .memory_gate import MemoryGate
from .retrieval import MemoryRetriever
from .storage import SQLiteMemoryStore


def run_flow(
    *,
    content: str,
    kind: str,
    goal: str,
    memory_db: str | Path,
) -> tuple[float, bool, list[Memory]]:
    """Run candidate -> attention -> gate -> SQLite -> retrieval."""
    if kind == "preference":
        event = user_preference(content)
    elif kind == "fact":
        event = user_fact(content)
    else:
        raise ValueError("kind must be 'preference' or 'fact'")

    context = Context(active_goal=goal)
    attention = AttentionAdapter(threshold=0.5)
    salience, process, _ = attention.evaluate(
        event,
        active_goal=context.active_goal,
        current_task=context.current_task,
    )
    if not process:
        return salience, False, []

    gate = MemoryGate(salience_threshold=0.75)
    if not gate.should_remember(event, salience):
        return salience, False, []

    store = SQLiteMemoryStore(memory_db)
    fields = gate.to_memory_fields(event, salience)
    store.save(Memory(**fields))

    retriever = MemoryRetriever(store, limit=5)
    retrieved = retriever.retrieve_for_context(context)
    context.retrieved_memories = [
        {
            "content": memory.content,
            "kind": memory.kind,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "metadata": memory.metadata,
        }
        for memory in retrieved
    ]
    return salience, True, retrieved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lucía's end-to-end memory flow")
    parser.add_argument("content", help="memory content to persist")
    parser.add_argument("--kind", choices=("preference", "fact"), default="preference")
    parser.add_argument("--goal", required=True, help="goal used by attention and retrieval")
    parser.add_argument("--memory-db", default="data/lucia-demo.db")
    args = parser.parse_args()

    salience, saved, retrieved = run_flow(
        content=args.content,
        kind=args.kind,
        goal=args.goal,
        memory_db=args.memory_db,
    )

    print("Lucía — Candidate → Attention → Memory Gate → SQLite → Retrieval")
    print(f"Contenido: {args.content}")
    print(f"Salience:  {salience:.2f}")
    print(f"Guardado:  {'sí' if saved else 'no'}")
    print(f"Recuperados: {len(retrieved)}")
    for memory in retrieved:
        print(f"  [{memory.kind}] {memory.content} (importance={memory.importance:.2f})")


if __name__ == "__main__":
    main()
