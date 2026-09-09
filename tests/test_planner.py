from datetime import timezone

from lucia.context import Context
from lucia.cognitive import CognitiveRequest, CognitiveResult
from lucia.planner import CognitivePlanner, Plan, RuleBasedPlanner


class FakeEngine:
    name = "fake"

    def __init__(self, output: str, success: bool = True) -> None:
        self.output = output
        self.success = success
        self.requests: list[CognitiveRequest] = []

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        self.requests.append(request)
        return CognitiveResult(
            success=self.success,
            output=self.output if self.success else None,
            error=None if self.success else "failed",
            engine=self.name,
        )


def test_rule_based_planner_creates_reasoning_step() -> None:
    context = Context(
        active_goal="Construir Lucia",
        current_task="Diseñar la memoria persistente",
        retrieved_memories=[{"content": "Memoria local"}],
    )

    plan = RuleBasedPlanner().plan(context)

    assert isinstance(plan, Plan)
    assert plan.goal == "Construir Lucia"
    assert plan.task == "Diseñar la memoria persistente"
    assert len(plan.steps) == 1
    assert plan.steps[0].action == "reason"
    assert plan.steps[0].description == "Diseñar la memoria persistente"
    assert plan.steps[0].parameters == {
        "goal": "Construir Lucia",
        "memory_count": 1,
    }
    assert plan.created_at.tzinfo == timezone.utc


def test_rule_based_planner_returns_empty_plan_without_task() -> None:
    context = Context(active_goal="Construir Lucia")

    plan = RuleBasedPlanner().plan(context)

    assert plan.task == ""
    assert plan.steps == ()
    assert plan.goal == "Construir Lucia"


def test_cognitive_planner_creates_structured_plan() -> None:
    engine = FakeEngine(
        '{"steps": ['
        '{"action": "get_time", "description": "Obtener la hora actual"},'
        '{"action": "reason", "description": "Interpretar el resultado"}'
        ']}'
    )
    planner = CognitivePlanner(engine, allowed_actions=frozenset({"get_time"}))
    context = Context(active_goal="Ayudar", current_task="¿Qué hora es?")

    plan = planner.plan(context)

    assert [step.action for step in plan.steps] == ["get_time", "reason"]
    assert "Allowed actions: reason, get_time" in engine.requests[0].task


def test_cognitive_planner_rejects_unknown_actions() -> None:
    engine = FakeEngine(
        '{"steps": [{"action": "delete_everything", "description": "bad"}]}'
    )
    planner = CognitivePlanner(engine, allowed_actions=frozenset({"get_time"}))
    context = Context(current_task="Haz algo")

    plan = planner.plan(context)

    assert plan.steps[0].action == "reason"
    assert plan.steps[0].description == "Haz algo"


def test_cognitive_planner_preserves_terminal_empty_plan() -> None:
    engine = FakeEngine('{"steps": []}')
    planner = CognitivePlanner(engine)
    context = Context(current_task="Tarea ya completada")

    plan = planner.plan(context)

    assert plan.steps == ()


def test_cognitive_planner_falls_back_on_invalid_output() -> None:
    engine = FakeEngine("not json")
    planner = CognitivePlanner(engine)
    context = Context(current_task="Pensar")

    plan = planner.plan(context)

    assert plan.steps[0].action == "reason"


def test_cognitive_planner_limits_steps() -> None:
    engine = FakeEngine(
        '{"steps": ['
        '{"action": "reason", "description": "one"},'
        '{"action": "reason", "description": "two"},'
        '{"action": "reason", "description": "three"}'
        ']}'
    )
    planner = CognitivePlanner(engine, max_steps=2)
    context = Context(current_task="Planificar")

    plan = planner.plan(context)

    assert len(plan.steps) == 2
