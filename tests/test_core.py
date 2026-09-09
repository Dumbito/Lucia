from pathlib import Path

from lucia.attention_adapter import AttentionAdapter
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


def test_core_attention_allows_relevant_events(tmp_path: Path) -> None:
    engine = FakeEngine()
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        cognitive_engine=engine,
        attention_adapter=AttentionAdapter(),
    )

    result = core.run_agent(
        {"type": "user_message", "data": {"content": "Analizar memoria"}},
        goal="Construir Lucia",
        task="Analizar memoria",
        max_iterations=1,
    )

    assert result.context.events[-1]["type"] == "cognition.result"
    assert result.context.cognitive_results
    assert any(event["type"] == "attention.result" for event in result.context.events)


def test_core_attention_filters_low_salience_events(tmp_path: Path) -> None:
    engine = FakeEngine()
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        cognitive_engine=engine,
        attention_adapter=AttentionAdapter(threshold=0.9),
    )

    result = core.run_agent(
        {"type": "system.snapshot", "data": {"content": "background noise"}},
        goal="Construir Lucia",
        task="Analizar memoria",
        max_iterations=1,
    )

    assert engine.calls == 0
    assert not result.context.cognitive_results
    attention_event = next(
        event for event in result.context.events if event["type"] == "attention.result"
    )
    assert attention_event["data"]["process"] is False
