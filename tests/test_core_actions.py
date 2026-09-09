import pytest

from lucia.actions import Action, ActionExecutor, ToolRegistry
from lucia.context import Context
from lucia.core import LuciaCore
from lucia.planner import Plan, PlanStep
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetTimeTool


def make_core(tmp_path):
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    return LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        action_executor=ActionExecutor(registry),
    )


def test_core_executes_action(tmp_path) -> None:
    core = make_core(tmp_path)
    result = core.execute(Action(name="get_time"))

    assert result.success is True
    assert result.output["timezone"] == "UTC"


def test_execute_plan_appends_action_result_to_context(tmp_path) -> None:
    core = make_core(tmp_path)
    context = Context(active_goal="test", current_task="check time")
    plan = Plan(
        goal="test",
        task="check time",
        steps=(PlanStep(action="get_time", description="Get current time"),),
    )

    results = core.execute_plan(plan, context)

    assert len(results) == 1
    assert results[0]["success"] is True
    assert len(context.events) == 1
    assert context.events[0]["type"] == "action.result"
    assert context.events[0]["data"]["action"] == "get_time"


def test_execute_plan_skips_reason_steps(tmp_path) -> None:
    core = make_core(tmp_path)
    context = Context(current_task="think")
    plan = Plan(
        goal=None,
        task="think",
        steps=(PlanStep(action="reason", description="Think"),),
    )

    assert core.execute_plan(plan, context) == []
    assert context.events == []


def test_core_requires_executor_for_actions(tmp_path) -> None:
    core = LuciaCore(SQLiteMemoryStore(tmp_path / "lucia.db"))

    with pytest.raises(RuntimeError, match="No action executor configured"):
        core.execute(Action(name="get_time"))
