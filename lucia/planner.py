"""Task planning primitives for Lucía."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Protocol

from .context import Context
from .cognitive import CognitiveEngine, build_cognitive_request


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
    """Small deterministic planner used as a safe baseline."""

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
        return Plan(goal=context.active_goal, task=task, steps=(step,))


@dataclass(slots=True)
class CognitivePlanner:
    """Build structured plans using a replaceable cognitive engine.

    ``allowed_actions`` is an explicit capability boundary. When provided,
    the model may only emit ``reason`` or actions present in that set.
    """

    engine: CognitiveEngine
    max_steps: int = 8
    allowed_actions: frozenset[str] | None = None

    def plan(self, context: Context) -> Plan:
        task = (context.current_task or "").strip()
        if not task:
            return Plan(goal=context.active_goal, task="", steps=())

        request = build_cognitive_request(context)
        available = "reason" if self.allowed_actions is None else ", ".join(
            ["reason", *sorted(self.allowed_actions)]
        )
        request = type(request)(
            task=(
                "Create a JSON plan for the task. Return only a JSON object with "
                'a "steps" array. Each step must contain "action", "description", '
                'and optional "parameters". Use action="reason" when no tool is needed. '
                "If the task has already been satisfied by a successful prior action, "
                "return an empty steps array instead of repeating that action. "
                f"Allowed actions: {available}. Never invent an action.\n\n"
                f"Task: {request.task}"
            ),
            goal=request.goal,
            context=request.context,
            output_format="json",
        )
        result = self.engine.reason(request)
        context.add_cognitive_result(result.as_dict())
        if not result.success or not isinstance(result.output, str):
            return RuleBasedPlanner().plan(context)

        payload = self._parse_json(result.output)
        if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
            return RuleBasedPlanner().plan(context)

        raw_steps = payload["steps"]
        steps: list[PlanStep] = []
        rejected_action = False
        for raw_step in raw_steps[: self.max_steps]:
            if not isinstance(raw_step, dict):
                continue
            action = raw_step.get("action")
            description = raw_step.get("description")
            parameters = raw_step.get("parameters", {})
            if not isinstance(action, str) or not action.strip():
                continue
            if not isinstance(description, str) or not description.strip():
                continue
            action = action.strip()
            if action != "reason" and self.allowed_actions is not None and action not in self.allowed_actions:
                rejected_action = True
                continue
            if not isinstance(parameters, dict):
                parameters = {}
            steps.append(
                PlanStep(
                    action=action,
                    description=description.strip(),
                    parameters=dict(parameters),
                )
            )

        if rejected_action and not steps:
            return RuleBasedPlanner().plan(context)

        if not steps:
            return Plan(goal=context.active_goal, task=task, steps=())
        return Plan(goal=context.active_goal, task=task, steps=tuple(steps))

    @staticmethod
    def _parse_json(output: str) -> Any:
        text = output.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
