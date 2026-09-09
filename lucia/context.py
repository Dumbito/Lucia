"""Working context for the current Lucía cycle."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Context:
    """Ephemeral state assembled for one or more reasoning cycles."""

    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_goal: str | None = None
    current_task: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    cognitive_results: list[dict[str, Any]] = field(default_factory=list)
    action_results: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.now = datetime.now(timezone.utc)

    def add_cognitive_result(self, result: dict[str, Any]) -> None:
        """Add a normalized cognitive result to working context."""
        self.cognitive_results.append(result)
        self.add_event({"type": "cognition.result", "data": result, "source": "cognitive_engine"})

    def add_action_result(self, result: dict[str, Any]) -> None:
        """Add a normalized action result to working context."""
        self.action_results.append(result)
        self.add_event({"type": "action.result", "data": result, "source": "action_executor"})

    def add_evaluation(self, evaluation: dict[str, Any]) -> None:
        """Add an evaluation to working context."""
        self.evaluations.append(evaluation)
        self.add_event({"type": "action.evaluation", "data": evaluation, "source": "evaluator"})
