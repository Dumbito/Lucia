"""Deterministic event-to-attention heuristics for Lucía v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field

from .attention import AttentionEngine
from .events import Event


@dataclass(slots=True)
class AttentionAdapter:
    """Translate normalized events into interpretable attention inputs.

    These are deliberately simple heuristics. They do not claim semantic
    understanding; later cognition can replace or refine them.
    """

    engine: AttentionEngine = field(default_factory=AttentionEngine)
    threshold: float = 0.5
    _seen: set[tuple[str, str]] = field(default_factory=set)

    def evaluate(
        self,
        event: Event,
        *,
        active_goal: str | None = None,
        current_task: str | None = None,
    ) -> tuple[float, bool, dict[str, float]]:
        """Return (salience, process, component scores) for an event."""
        signature = (event.type, repr(event.data))
        novelty = 1.0 if signature not in self._seen else 0.2
        self._seen.add(signature)

        relevance = self._relevance(event, active_goal, current_task)
        urgency = self._urgency(event)
        goal_alignment = self._goal_alignment(event, active_goal, current_task)

        salience = self.engine.score(
            event,
            novelty=novelty,
            relevance=relevance,
            urgency=urgency,
            goal_alignment=goal_alignment,
        )
        return salience, self.engine.should_process(salience, self.threshold), {
            "novelty": novelty,
            "relevance": relevance,
            "urgency": urgency,
            "goal_alignment": goal_alignment,
        }

    @staticmethod
    def _relevance(
        event: Event,
        active_goal: str | None,
        current_task: str | None,
    ) -> float:
        if not active_goal and not current_task:
            return 0.25

        text = " ".join(
            [
                str(event.type),
                *(str(value) for value in event.data.values()),
            ]
        ).lower()
        context = " ".join(filter(None, [active_goal, current_task])).lower()
        tokens = {token for token in context.split() if len(token) >= 4}
        return 0.75 if tokens and any(token in text for token in tokens) else 0.30

    @staticmethod
    def _goal_alignment(
        event: Event,
        active_goal: str | None,
        current_task: str | None,
    ) -> float:
        if not active_goal and not current_task:
            return 0.0
        return AttentionAdapter._relevance(event, active_goal, current_task)

    @staticmethod
    def _urgency(event: Event) -> float:
        return {
            "system.snapshot": 0.0,
            "window.changed": 0.05,
            "media.changed": 0.05,
        }.get(event.type, 0.20)
