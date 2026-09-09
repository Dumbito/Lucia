from datetime import timezone

from lucia.context import Context
from lucia.planner import Plan, RuleBasedPlanner


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
