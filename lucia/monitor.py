"""Continuous Linux perception monitor."""

from __future__ import annotations

import argparse
import time

from .events import Event
from .perception import LinuxPerception, SystemSnapshot


def _describe(snapshot: SystemSnapshot) -> str:
    return (
        f"window={snapshot.active_window!r} "
        f"media={snapshot.music_status!r} "
        f"cpu_load={snapshot.cpu_load!r} "
        f"memory={snapshot.memory_percent!r}%"
    )


class LinuxEventSource:
    """Stateful event source suitable for ``ProactiveLoop``."""

    def __init__(self, perception: LinuxPerception | None = None) -> None:
        self.perception = perception or LinuxPerception()
        self.previous: SystemSnapshot | None = None

    def __call__(self) -> list[Event]:
        current = self.perception.snapshot()
        events = self.perception.events(self.previous, current)
        self.previous = current
        return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Linux state for Lucía")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    source = LinuxEventSource()

    print("Lucía — Perception Monitor v0.1")
    print("Observando cambios del sistema. Ctrl+C para salir.")
    print("=" * 56)

    try:
        while True:
            for event in source():
                print(f"[{event.timestamp.astimezone().strftime('%H:%M:%S')}] {event.type}")
                print(f"  {event.data}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor detenido.")


if __name__ == "__main__":
    main()
