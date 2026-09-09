"""Runnable perception -> attention -> context -> memory pipeline for Lucía."""

from __future__ import annotations

import argparse
import time

from .attention_adapter import AttentionAdapter
from .context import Context
from .events import Event
from .memory import Memory
from .memory_gate import MemoryGate
from .perception import LinuxPerception, SystemSnapshot
from .retrieval import MemoryRetriever
from .storage import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lucía perception, attention, context and memory")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots")
    parser.add_argument("--threshold", type=float, default=0.5, help="attention threshold")
    parser.add_argument("--memory-threshold", type=float, default=0.75, help="minimum salience for long-term memory")
    parser.add_argument("--memory-db", default="data/lucia.db", help="SQLite memory database path")
    parser.add_argument("--memory-limit", type=int, default=5, help="maximum memories retrieved into context")
    parser.add_argument("--goal", default=None, help="current active goal")
    parser.add_argument("--task", default=None, help="current task")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if not 0.0 <= args.memory_threshold <= 1.0:
        parser.error("--memory-threshold must be between 0 and 1")
    if args.memory_limit <= 0:
        parser.error("--memory-limit must be greater than zero")

    perception = LinuxPerception()
    attention = AttentionAdapter(threshold=args.threshold)
    context = Context(active_goal=args.goal, current_task=args.task)
    memory_gate = MemoryGate(salience_threshold=args.memory_threshold)
    memory_store = SQLiteMemoryStore(args.memory_db)
    memory_retriever = MemoryRetriever(memory_store, limit=args.memory_limit)
    previous: SystemSnapshot | None = None
    memories_saved = 0

    print("Lucía — Perception → Attention → Context → Retrieval → Memory v0.4")
    print("Los eventos salientes entran al contexto; la memoria se recupera según el objetivo/tarea.")
    if args.goal or args.task:
        print(f"Objetivo: {args.goal or '—'}")
        print(f"Tarea:    {args.task or '—'}")
    print(f"Memoria:  {args.memory_db}")

    retrieved = memory_retriever.retrieve_for_context(context)
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
    print(f"Recuerdos recuperados: {len(retrieved)}")
    for memory in retrieved:
        print(f"  [{memory.kind}] {memory.content} (importance={memory.importance:.2f})")
    print("=" * 76)

    try:
        while True:
            current = perception.snapshot()
            events = perception.events(previous, current)

            for event in events:
                salience, process, components = attention.evaluate(
                    event,
                    active_goal=context.active_goal,
                    current_task=context.current_task,
                )
                decision = "PRESTAR ATENCIÓN" if process else "IGNORAR"

                print(f"[{event.timestamp.astimezone().strftime('%H:%M:%S')}] {event.type}")
                print(f"  {event.data}")
                print(
                    "  "
                    f"N={components['novelty']:.2f} "
                    f"R={components['relevance']:.2f} "
                    f"U={components['urgency']:.2f} "
                    f"G={components['goal_alignment']:.2f}"
                )
                print(f"  Salience: {salience:.2f} → {decision}")

                if process:
                    context.add_event(
                        {
                            **event.as_dict(),
                            "salience": salience,
                            "attention": components,
                        }
                    )
                    print(f"  CONTEXT: evento incorporado ({len(context.events)} total)")

                    if memory_gate.should_remember(event, salience):
                        fields = memory_gate.to_memory_fields(event, salience)
                        memory_store.save(Memory(**fields))
                        memories_saved += 1
                        print(
                            f"  MEMORY: guardado como {fields['kind']} "
                            f"(importance={fields['importance']:.2f})"
                        )
                    else:
                        print("  MEMORY: no guardar (evento transitorio o no elegible)")
                print("-" * 76)

            previous = current
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nPipeline detenido.")
        print(f"Eventos conservados en contexto: {len(context.events)}")
        print(f"Recuerdos en contexto: {len(context.retrieved_memories)}")
        print(f"Memorias guardadas: {memories_saved}")


if __name__ == "__main__":
    main()
