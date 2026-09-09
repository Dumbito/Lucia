from lucia.actions import ActionExecutor, ToolRegistry
from lucia.core import LuciaCore
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetTimeTool


def make_core(tmp_path):
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    return LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        action_executor=ActionExecutor(registry),
    )


def test_run_cycle_orchestrates_observe_plan_and_cognition(tmp_path) -> None:
    core = make_core(tmp_path)

    result = core.run_cycle(
        {"type": "user.message", "data": "check time"},
        goal="help user",
        task="check time",
    )

    assert result.context.active_goal == "help user"
    assert result.context.current_task == "check time"
    assert result.plan.task == "check time"
    assert result.plan.steps[0].action == "reason"
    assert result.action_results == ()
    assert result.evaluations == ()
    assert len(result.context.cognitive_results) == 1
    assert result.context.cognitive_results[0]["success"] is True
    assert result.context.cognitive_results[0]["engine"] == "rule_based"
    assert result.context.events[0]["type"] == "user.message"
    assert result.context.events[1]["type"] == "cognition.result"


def test_cycle_executes_non_reason_steps(tmp_path) -> None:
    core = make_core(tmp_path)

    class ActionPlanner:
        def plan(self, context):
            from lucia.planner import Plan, PlanStep

            return Plan(
                goal=context.active_goal,
                task=context.current_task or "",
                steps=(PlanStep(action="get_time", description="Get time"),),
            )

    core.planner = ActionPlanner()
    result = core.run_cycle(
        {"type": "user.message", "data": "what time is it"},
        task="what time is it",
    )

    assert len(result.action_results) == 1
    assert result.action_results[0]["result"]["success"] is True
    assert result.evaluations[0]["success"] is True
    assert len(result.context.action_results) == 1
    assert len(result.context.evaluations) == 1


def test_cycle_can_use_custom_cognitive_engine(tmp_path) -> None:
    core = make_core(tmp_path)

    class StubEngine:
        name = "stub"

        def reason(self, request):
            from lucia.cognitive import CognitiveResult

            return CognitiveResult(
                success=True,
                output={"decision": "test"},
                engine=self.name,
            )

    core.cognitive_engine = StubEngine()
    result = core.run_cycle(
        {"type": "user.message", "data": "do something"},
        task="do something",
    )

    assert result.context.cognitive_results[0]["engine"] == "stub"
    assert result.context.cognitive_results[0]["output"] == {"decision": "test"}
