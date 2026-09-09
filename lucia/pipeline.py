"""Runnable perception -> attention -> context pipeline for Lucía."""

from __future__ import annotations

import argparse
import time

from .attention_adapter import AttentionAdapter
from .context import Context
from .perception import LinuxPerception, SystemSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lucía perception, attention and context")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots")
    parser.add_argument("--threshold", type=float, default=0.5, help="attention threshold")
    parser.add_argument("--goal", default=None, help="current active goal")
    parser.add_argument("--task", default=None, help="current task")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    perception = LinuxPerception()
    attention = AttentionAdapter(threshold=args.threshold)
    context = Context(active_goal=args.goal, current_task=args.task)
    previous: SystemSnapshot | None = None

    print("Lucía — Perception → Attention → Context v0.2")
    print("Los eventos salientes entran al contexto de trabajo; los demás se filtran.")
    if args.goal or args.task:
        print(f"Objetivo: {args.goal or '—'}")
        print(f"Tarea:    {args.task or '—'}")
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
                print("-" * 76)

            previous = current
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nPipeline detenido.")
        print(f"Eventos conservados en contexto: {len(context.events)}")


if __name__ == "__main__":
    main()
