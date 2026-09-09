"""Real smoke test for Attention -> Initiative -> Ollama -> action."""

from lucia.actions import ActionExecutor
from lucia.attention_adapter import AttentionAdapter
from lucia.core import LuciaCore
from lucia.initiative import InitiativeEngine
from lucia.model_router import build_ollama_router
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetTimeTool, ToolRegistry
from lucia.planner import CognitivePlanner


if __name__ == "__main__":
    store = SQLiteMemoryStore(":memory:")
    router = build_ollama_router()
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    core = LuciaCore(
        store,
        planner=CognitivePlanner(engine=router, max_steps=4),
        cognitive_engine=router,
        action_executor=ActionExecutor(registry),
        attention_adapter=AttentionAdapter(),
        initiative_engine=InitiativeEngine(),
    )

    result = core.run_proactive(
        {
            "type": "system.alert",
            "data": {"content": "Necesito obtener la hora actual para responder al usuario."},
            "source": "smoke_test",
        },
        goal="Responder correctamente al usuario",
        task="Obtener la hora actual usando una herramienta disponible",
        max_iterations=3,
    )

    attention = next(
        event for event in result.context.events if event["type"] == "attention.result"
    )
    initiative = next(
        event for event in result.context.events if event["type"] == "initiative.result"
    )

    print("=== LUCÍA PROACTIVE CYCLE → ATTENTION → INITIATIVE → OLLAMA ===")
    print(f"Atención: {attention['data']}")
    print(f"Iniciativa: {initiative['data']}")
    print(f"Plan final: {result.plan}")
    print(f"Acciones ejecutadas: {len(result.action_results)}")
    for index, item in enumerate(result.action_results, start=1):
        print(f"Acción #{index}: {item}")
    print(f"Resultados cognitivos: {len(result.context.cognitive_results)}")
    for index, item in enumerate(result.context.cognitive_results, start=1):
        print(
            f"Cognición #{index}: engine={item.get('engine')} "
            f"success={item.get('success')} output={item.get('output')}"
        )
