"""Minimal event loop: observe, evaluate, remember selected events."""

from .core import LuciaCore
from .events import Event


class CognitiveLoop:
    """First executable form of Lucía's perception-to-memory cycle."""

    def __init__(self, core: LuciaCore) -> None:
        self.core = core

    def process(self, event: Event) -> None:
        context = self.core.observe(event.as_dict())

        # v0.1 records only explicit high-value events. Salience will become
        # a separate module before external perception is connected.
        importance = float(event.data.get("importance", 0.0))
        if importance >= 0.7:
            self.core.remember(
                f"[{event.type}] {event.data}",
                kind="episodic",
                importance=importance,
            )

        print(
            f"[cycle] {event.type} | "
            f"events={len(context.events)} | importance={importance:.2f}"
        )
