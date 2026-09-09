"""Task planning primitives for Lucía.

The planner is intentionally independent from any language model. A later
cognitive engine can replace the rule-based implementation without changing
how plans are represented or executed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .context import Context


@dataclass(slots=True, frozen=True)
class PlanStep:
    """One atomic intention that can later be handed to an executor."""

    action: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Plan:
    """A transient plan produced for the current task."""

    goal: str | None
    task: str
    steps: tuple[PlanStep, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Planner(Protocol):
    """Interface implemented by any Lucía planning strategy."""

    def plan(self, context: Context) -> Plan: ...


@dataclass(slots=True)
class RuleBasedPlanner:
    """Small deterministic planner used until an LLM planner is connected."""

    def plan(self, context: Context) -> Plan:
        task = (context.current_task or "").strip()
        if not task:
            return Plan(goal=context.active_goal, task="", steps=())

        step = PlanStep(
            action="reason",
            description=task,
            parameters={
                "goal": context.active_goal,
                "memory_count": len(context.retrieved_memories),
            },
        )
        return Plan(
            goal=context.active_goal,
            task=task,
            steps=(step,),
        )
