"""Central coordinator for the first Lucía prototype."""

from dataclasses import dataclass, field
from typing import Any

from .actions import Action, ActionExecutor
from .attention_adapter import AttentionAdapter
from .context import Context
from .cognitive import CognitiveEngine, RuleBasedCognitiveEngine, build_cognitive_request
from .cycle import CycleResult
from .evaluation import Evaluation, RuleBasedEvaluator
from .events import Event
from .initiative import InitiativeEngine
from .memory import Memory, MemoryStore
from .model_router import ModelRouter
from .planner import CognitivePlanner, Plan, PlanStep, Planner, RuleBasedPlanner
from .retrieval import MemoryRetriever


@dataclass(slots=True)
class LuciaCore:
    """Owns identity-independent state and coordinates cognitive cycles."""

    memory: MemoryStore
    name: str = "Lucía"
    planner: Planner = field(default_factory=RuleBasedPlanner)
    cognitive_engine: CognitiveEngine = field(default_factory=RuleBasedCognitiveEngine)
    action_executor: ActionExecutor | None = None
    evaluator: RuleBasedEvaluator = field(default_factory=RuleBasedEvaluator)
    retriever: MemoryRetriever | None = None
    attention_adapter: AttentionAdapter | None = None
    initiative_engine: InitiativeEngine | None = None

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

    def retrieve_memories(self, context: Context) -> list[Memory]:
        """Retrieve memories relevant to the current working context."""
        retriever = self.retriever or MemoryRetriever(self.memory)
        return retriever.retrieve_for_context(context)

    @staticmethod
    def _memory_as_dict(memory: Memory) -> dict[str, Any]:
        """Expose only useful memory fields to the cognitive context."""
        return {
            "content": memory.content,
            "kind": memory.kind,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "metadata": dict(memory.metadata),
        }

    def _planner(self) -> Planner:
        """Return a planner configured with the executor's capabilities."""
        if not isinstance(self.planner, CognitivePlanner):
            return self.planner
        if self.action_executor is None:
            return self.planner
        return CognitivePlanner(
            engine=self.planner.engine,
            max_steps=self.planner.max_steps,
            allowed_actions=frozenset(self.action_executor.registry.names()),
        )

    def plan(self, context: Context) -> Plan:
        """Produce a transient plan without executing any action."""
        return self._planner().plan(context)

    def reason(self, context: Context, *, description: str | None = None) -> dict[str, Any]:
        """Run the configured cognitive engine and store its result in context."""
        request = build_cognitive_request(context, description=description)
        result = self.cognitive_engine.reason(request)
        normalized = result.as_dict()
        context.add_cognitive_result(normalized)
        return normalized

    def configure_model_router(self, router: ModelRouter) -> None:
        """Use a model router as Lucía's cognitive engine."""
        self.cognitive_engine = router
        if isinstance(self.planner, CognitivePlanner):
            self.planner = CognitivePlanner(
                engine=router,
                max_steps=self.planner.max_steps,
                allowed_actions=self.planner.allowed_actions,
            )

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

    @staticmethod
    def _already_succeeded(step: PlanStep, context: Context) -> bool:
        """Detect an exact action/parameter repeat that already succeeded."""
        for action_result, evaluation in zip(context.action_results, context.evaluations):
            if not evaluation.get("success"):
                continue
            if action_result.get("action") != step.action:
                continue
            parameters = action_result.get("parameters", {})
            if parameters == dict(step.parameters):
                return True
        return False

    def run_cycle(
        self,
        event: dict[str, Any],
        *,
        goal: str | None = None,
        task: str | None = None,
    ) -> CycleResult:
        """Run one complete cognitive cycle."""
        return self.run_agent(
            event,
            goal=goal,
            task=task,
            max_iterations=1,
        )

    def run_proactive(
        self,
        event: dict[str, Any],
        *,
        goal: str | None = None,
        task: str | None = None,
        max_iterations: int = 4,
    ) -> CycleResult:
        """Run a cycle in proactive mode, gated by attention and initiative."""
        return self.run_agent(
            event,
            goal=goal,
            task=task,
            max_iterations=max_iterations,
            proactive=True,
        )

    def run_agent(
        self,
        event: dict[str, Any],
        *,
        goal: str | None = None,
        task: str | None = None,
        max_iterations: int = 4,
        proactive: bool = False,
    ) -> CycleResult:
        """Iterate plan → action → evaluation → reasoning until completion."""
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if proactive and self.initiative_engine is None:
            raise RuntimeError("No initiative engine configured")
        if proactive and self.attention_adapter is None:
            raise RuntimeError("Proactive mode requires an attention adapter")

        context = self.observe(event)
        context.active_goal = goal
        context.current_task = task
        if self.attention_adapter is not None:
            event_type = str(event.get("type", "observation"))
            event_data = event.get("data", {})
            if not isinstance(event_data, dict):
                event_data = {"content": event_data}
            normalized_event = Event(
                type=event_type,
                data=dict(event_data),
                source=str(event.get("source", "system")),
            )
            salience, process, components = self.attention_adapter.evaluate(
                normalized_event,
                active_goal=goal,
                current_task=task,
            )
            context.add_event(
                {
                    "type": "attention.result",
                    "data": {
                        "salience": salience,
                        "process": process,
                        "components": components,
                    },
                    "source": "attention",
                }
            )
            if not process:
                return CycleResult(
                    context=context,
                    plan=Plan(goal=goal, task=(task or "").strip(), steps=()),
                    action_results=(),
                )

            if proactive:
                decision = self.initiative_engine.decide(
                    salience=salience,
                    **components,
                )
                context.add_event(
                    {
                        "type": "initiative.result",
                        "data": {
                            "act": decision.act,
                            "score": decision.score,
                            "reason": decision.reason,
                        },
                        "source": "initiative",
                    }
                )
                if not decision.act:
                    return CycleResult(
                        context=context,
                        plan=Plan(goal=goal, task=(task or "").strip(), steps=()),
                        action_results=(),
                    )

        all_action_results: list[dict[str, Any]] = []
        last_plan = Plan(goal=goal, task=(task or "").strip(), steps=())

        for iteration in range(max_iterations):
            context.retrieved_memories = [
                self._memory_as_dict(memory) for memory in self.retrieve_memories(context)
            ]
            last_plan = self.plan(context)

            executable_steps = tuple(step for step in last_plan.steps if step.action != "reason")
            if not executable_steps:
                if iteration == 0:
                    for step in last_plan.steps:
                        if step.action == "reason":
                            self.reason(context, description=step.description)
                break

            fresh_steps = tuple(
                step for step in executable_steps if not self._already_succeeded(step, context)
            )
            if not fresh_steps:
                break

            for step in last_plan.steps:
                if step.action == "reason":
                    self.reason(context, description=step.description)

            iteration_results = self.execute_plan(
                Plan(goal=last_plan.goal, task=last_plan.task, steps=fresh_steps),
                context,
            )
            all_action_results.extend(iteration_results)

        return CycleResult(
            context=context,
            plan=last_plan,
            action_results=tuple(all_action_results),
        )
