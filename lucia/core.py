"""Central coordinator for the first Lucía prototype."""

from dataclasses import dataclass, field
from typing import Any

from .actions import Action, ActionExecutor
from .context import Context
from .evaluation import Evaluation, RuleBasedEvaluator
from .memory import Memory, MemoryStore
from .planner import Plan, Planner, RuleBasedPlanner


@dataclass(slots=True)
class LuciaCore:
    """Owns identity-independent state and coordinates cognitive cycles."""

    memory: MemoryStore
    name: str = "Lucía"
    planner: Planner = field(default_factory=RuleBasedPlanner)
    action_executor: ActionExecutor | None = None
    evaluator: RuleBasedEvaluator = field(default_factory=RuleBasedEvaluator)

    def observe(self, event: dict[str, Any]) -> Context:
        """Add an observed event to working context."""
        context = Context()
        context.add_event(event)
        return context

    def remember(self, content: str, *, kind: str = "episodic", importance: float = 0.5) -> Memory:
        """Persist a selected experience."""
        memory = Memory(content=content, kind=kind, importance=importance)
        self.memory.save(memory)
        return memory

    def plan(self, context: Context) -> Plan:
        """Produce a transient plan without executing any action."""
        return self.planner.plan(context)

    def execute(self, action: Action) -> Any:
        """Execute one action through the configured executor."""
        if self.action_executor is None:
            raise RuntimeError("No action executor configured")
        return self.action_executor.execute(action)

    def execute_plan(self, plan: Plan, context: Context) -> list[dict[str, Any]]:
        """Execute plan steps and evaluate each action result."""
        if self.action_executor is None:
            raise RuntimeError("No action executor configured")

        results: list[dict[str, Any]] = []
        for step in plan.steps:
            if step.action == "reason":
                continue

            result = self.action_executor.execute(
                Action(name=step.action, parameters=dict(step.parameters))
            )
            normalized = result.as_dict()
            evaluation: Evaluation = self.evaluator.evaluate(result)
            normalized_evaluation = evaluation.as_dict()

            context.add_action_result(normalized)
            context.add_evaluation(normalized_evaluation)
            results.append({"result": normalized, "evaluation": normalized_evaluation})

            if evaluation.should_remember:
                summary = (
                    f"Action {step.action}: {evaluation.summary} "
                    f"success={evaluation.success} score={evaluation.score:.2f}"
                )
                self.remember(summary, kind="episodic", importance=max(evaluation.score, 0.7))

        return results
