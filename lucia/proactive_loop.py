"""Controlled event loop for Lucía's proactive behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from .core import LuciaCore
from .events import Event


EventSource = Callable[[], Iterable[Event]]


class ProactiveLoop:
    """Poll an event source and route events through Lucía's proactive cycle.

    The loop is intentionally source-agnostic. A source only needs to return
    normalized ``Event`` objects; attention and initiative remain responsible
    for deciding whether an event deserves processing or action.
    """

    def __init__(
        self,
        core: LuciaCore,
        source: EventSource,
        *,
        interval: float = 1.0,
        goal: str | None = None,
        task: str | None = None,
        max_iterations: int = 4,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.core = core
        self.source = source
        self.interval = interval
        self.goal = goal
        self.task = task
        self.max_iterations = max_iterations
        self.running = False

    def tick(self) -> list:
        """Poll once and process every event returned by the source."""
        results = []
        for event in self.source():
            results.append(
                self.core.run_proactive(
                    event.as_dict(),
                    goal=self.goal,
                    task=self.task,
                    max_iterations=self.max_iterations,
                )
            )
        return results

    def run(self, *, max_ticks: int | None = None) -> None:
        """Run until stopped or until ``max_ticks`` have been completed."""
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max_ticks must be at least 1 when provided")

        self.running = True
        ticks = 0
        try:
            while self.running and (max_ticks is None or ticks < max_ticks):
                self.tick()
                ticks += 1
                if self.running and (max_ticks is None or ticks < max_ticks):
                    time.sleep(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False

    def stop(self) -> None:
        """Request that a running loop stop after its current tick."""
        self.running = False
