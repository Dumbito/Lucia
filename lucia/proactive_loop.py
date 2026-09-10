"""Controlled event loop for Lucía's proactive behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from .core import LuciaCore
from .events import Event


EventSource = Callable[[], Iterable[Event]]


class ProactiveLoop:
    """Poll an event source and route worthwhile events through Lucía.

    The loop stays deliberately conservative: identical events can be put on
    cooldown, and an idle source progressively increases its polling interval.
    This prevents a persistent runtime from repeatedly waking expensive
    cognitive components when the environment is stable.
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
        cooldown: float = 0.0,
        max_interval: float | None = None,
        idle_backoff: float = 1.0,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if cooldown < 0:
            raise ValueError("cooldown must be zero or greater")
        if max_interval is not None and max_interval < interval:
            raise ValueError("max_interval must be greater than or equal to interval")
        if idle_backoff < 1.0:
            raise ValueError("idle_backoff must be at least 1")

        self.core = core
        self.source = source
        self.interval = interval
        self.goal = goal
        self.task = task
        self.max_iterations = max_iterations
        self.cooldown = cooldown
        self.max_interval = max_interval if max_interval is not None else interval
        self.idle_backoff = idle_backoff
        self.running = False
        self._next_allowed: dict[tuple[str, str], float] = {}

    @staticmethod
    def _signature(event: Event) -> tuple[str, str]:
        return event.type, repr(event.data)

    def tick(self) -> list:
        """Poll once and process events not currently on cooldown."""
        now = time.monotonic()
        results = []
        for event in self.source():
            signature = self._signature(event)
            if self._next_allowed.get(signature, 0.0) > now:
                continue
            results.append(
                self.core.run_proactive(
                    event.as_dict(),
                    goal=self.goal,
                    task=self.task,
                    max_iterations=self.max_iterations,
                )
            )
            if self.cooldown > 0:
                self._next_allowed[signature] = now + self.cooldown
        return results

    def run(self, *, max_ticks: int | None = None) -> None:
        """Run until stopped or until ``max_ticks`` have been completed."""
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max_ticks must be at least 1 when provided")

        self.running = True
        ticks = 0
        current_interval = self.interval
        try:
            while self.running and (max_ticks is None or ticks < max_ticks):
                results = self.tick()
                ticks += 1
                if results:
                    current_interval = self.interval
                else:
                    current_interval = min(
                        self.max_interval,
                        current_interval * self.idle_backoff,
                    )
                if self.running and (max_ticks is None or ticks < max_ticks):
                    time.sleep(current_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False

    def stop(self) -> None:
        """Request that a running loop stop after its current tick."""
        self.running = False
