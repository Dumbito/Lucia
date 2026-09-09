from pathlib import Path

from lucia.actions import ActionExecutor, ToolRegistry
from lucia.context import Context
from lucia.core import LuciaCore
from lucia.cognitive import CognitiveRequest, CognitiveResult
from lucia.planner import CognitivePlanner
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetSystemInfoTool, GetTimeTool


class FakeEngine:
    name = "fake"

    def __init__(self) -> None:
        self.request: CognitiveRequest | None = None

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        self.request = request
        return CognitiveResult(
            success=True,
            output='{"steps": [{"action": "get_time", "description": "Obtener hora"}]}',
            engine=self.name,
        )


def test_core_derives_allowed_actions_from_registry(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    executor = ActionExecutor(registry)
    engine = FakeEngine()
    planner = CognitivePlanner(engine)
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        planner=planner,
        cognitive_engine=engine,
        action_executor=executor,
    )

    plan = core.plan(Context(current_task="¿Qué hora es?"))

    assert plan.steps[0].action == "get_time"
    assert engine.request is not None
    assert "get_time" in engine.request.task
    assert "get_system_info" in engine.request.task
