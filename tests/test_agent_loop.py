from pathlib import Path

from lucia.actions import ActionExecutor, ToolRegistry
from lucia.context import Context
from lucia.core import LuciaCore
from lucia.cognitive import CognitiveRequest, CognitiveResult
from lucia.planner import CognitivePlanner
from lucia.storage import SQLiteMemoryStore


class SequenceEngine:
    name = "sequence"

    def __init__(self) -> None:
        self.calls = 0

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        self.calls += 1
        if self.calls == 1:
            output = '{"steps": [{"action": "get_time", "description": "Obtener hora"}]}'
        else:
            output = '{"steps": [{"action": "reason", "description": "Interpretar resultado"}]}'
        return CognitiveResult(success=True, output=output, engine=self.name)


class TimeTool:
    name = "get_time"

    def execute(self, parameters: dict) -> dict:
        return {"iso": "2026-09-08T00:00:00+00:00", "timezone": "UTC"}


def test_run_agent_iterates_until_reason_only(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(TimeTool())
    engine = SequenceEngine()
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        planner=CognitivePlanner(engine),
        cognitive_engine=engine,
        action_executor=ActionExecutor(registry),
    )

    result = core.run_agent(
        {"type": "user.message", "content": "¿Qué hora es?"},
        task="¿Qué hora es?",
        max_iterations=4,
    )

    assert len(result.action_results) == 1
    assert result.action_results[0]["result"]["action"] == "get_time"
    assert result.action_results[0]["evaluation"]["success"] is True
    assert engine.calls == 2
    assert result.plan.steps[0].action == "reason"
