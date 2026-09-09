"""Small runnable perception -> attention pipeline for Lucía."""

from __future__ import annotations

import argparse
import time

from .attention_adapter import AttentionAdapter
from .perception import LinuxPerception, SystemSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lucía perception and attention")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots")
    parser.add_argument("--threshold", type=float, default=0.5, help="attention threshold")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    perception = LinuxPerception()
    attention = AttentionAdapter(threshold=args.threshold)
    previous: SystemSnapshot | None = None

    print("Lucía — Perception → Attention v0.1")
    print("Observando eventos y filtrándolos antes de cognition. Ctrl+C para salir.")
    print("=" * 72)

    try:
        while True:
            current = perception.snapshot()
            events = perception.events(previous, current)

            for event in events:
                salience, process, components = attention.evaluate(event)
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
                print("-" * 72)

            previous = current
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nPipeline detenido.")


if __name__ == "__main__":
    main()
