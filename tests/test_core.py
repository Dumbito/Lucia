from pathlib import Path

from lucia.context import Context
from lucia.core import LuciaCore
from lucia.model_router import ModelRouter
from lucia.storage import SQLiteMemoryStore


class FakeEngine:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def reason(self, request):
        from lucia.cognitive import CognitiveResult

        self.calls += 1
        return CognitiveResult(success=True, output={"task": request.task}, engine=self.name)


def test_core_exposes_planning_without_executing_actions(tmp_path: Path) -> None:
    core = LuciaCore(SQLiteMemoryStore(tmp_path / "lucia.db"))
    context = Context(
        active_goal="Construir Lucia",
        current_task="Evaluar memoria recuperada",
    )

    plan = core.plan(context)

    assert plan.goal == "Construir Lucia"
    assert plan.task == "Evaluar memoria recuperada"
    assert plan.steps[0].action == "reason"


def test_core_can_configure_model_router_as_cognitive_engine(tmp_path: Path) -> None:
    engine = FakeEngine()
    router = ModelRouter()
    router.register(
        "fake-model",
        engine,
        capabilities=("general",),
        context_window=32768,
        speed=5,
        priority=1,
    )
    core = LuciaCore(SQLiteMemoryStore(tmp_path / "lucia.db"))
    core.configure_model_router(router)

    context = Context(active_goal="Test", current_task="Analizar contexto")
    result = core.reason(context)

    assert core.cognitive_engine is router
    assert engine.calls == 1
    assert result["success"] is True
    assert result["engine"] == "fake"
    assert context.cognitive_results[-1]["output"] == {"task": "Analizar contexto"}
