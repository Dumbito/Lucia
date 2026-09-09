"""Central coordinator for the first Lucía prototype."""

from dataclasses import dataclass, field
from typing import Any

from .actions import Action, ActionExecutor
from .context import Context
from .cognitive import CognitiveEngine, RuleBasedCognitiveEngine, build_cognitive_request
from .cycle import CycleResult
from .evaluation import Evaluation, RuleBasedEvaluator
from .memory import Memory, MemoryStore
from .planner import Plan, Planner, RuleBasedPlanner


@dataclass(slots=True)
class LuciaCore:
    """Owns identity-independent state and coordinates cognitive cycles."""

    memory: MemoryStore
    name: str = "Lucía"
    planner: Planner = field(default_factory=RuleBasedPlanner)
    cognitive_engine: CognitiveEngine = field(default_factory=RuleBasedCognitiveEngine)
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

    def reason(self, context: Context, *, description: str | None = None) -> dict[str, Any]:
        """Run the configured cognitive engine and store its result in context."""
        request = build_cognitive_request(context, description=description)
        result = self.cognitive_engine.reason(request)
        normalized = result.as_dict()
        context.add_cognitive_result(normalized)
        return normalized

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

    def run_cycle(
        self,
        event: dict[str, Any],
        *,
        goal: str | None = None,
        task: str | None = None,
    ) -> CycleResult:
        """Run one complete cognitive cycle."""
        context = self.observe(event)
        context.active_goal = goal
        context.current_task = task

        plan = self.plan(context)

        for step in plan.steps:
            if step.action == "reason":
                self.reason(context, description=step.description)

        action_results = tuple(self.execute_plan(plan, context))
        return CycleResult(
            context=context,
            plan=plan,
            action_results=action_results,
        )
