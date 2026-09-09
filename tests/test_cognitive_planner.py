from lucia.context import Context
from lucia.cognitive import CognitiveResult
from lucia.planner import CognitivePlanner


class StubEngine:
    name = "stub"

    def __init__(self, output, success=True):
        self.output = output
        self.success = success

    def reason(self, request):
        return CognitiveResult(
            success=self.success,
            output=self.output,
            engine=self.name,
        )


def test_cognitive_planner_parses_structured_plan() -> None:
    engine = StubEngine(
        '{"steps":[{"action":"get_time","description":"Get the current time","parameters":{}}]}'
    )
    context = Context(active_goal="help user", current_task="check the time")

    plan = CognitivePlanner(engine).plan(context)

    assert plan.task == "check the time"
    assert len(plan.steps) == 1
    assert plan.steps[0].action == "get_time"
    assert plan.steps[0].description == "Get the current time"


def test_cognitive_planner_falls_back_on_invalid_output() -> None:
    context = Context(current_task="check the time")
    plan = CognitivePlanner(StubEngine("not json")).plan(context)

    assert plan.steps[0].action == "reason"
    assert plan.steps[0].description == "check the time"


def test_cognitive_planner_limits_steps() -> None:
    steps = [
        {"action": f"tool_{index}", "description": f"step {index}"}
        for index in range(20)
    ]
    engine = StubEngine(__import__("json").dumps({"steps": steps}))
    plan = CognitivePlanner(engine, max_steps=3).plan(Context(current_task="do task"))

    assert len(plan.steps) == 3
