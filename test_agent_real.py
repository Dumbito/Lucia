"""Real end-to-end agent cycle using LucíaCore, ModelRouter and Ollama.

Run manually with Ollama running. This is intentionally a smoke test rather
than a pytest test because it depends on a locally installed Ollama model.
"""

from pathlib import Path

from lucia.actions import ActionExecutor, ToolRegistry
from lucia.core import LuciaCore
from lucia.model_router import build_ollama_router
from lucia.planner import CognitivePlanner
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetTimeTool


def main() -> None:
    database = Path(".lucia-agent-real-test.db")
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    router = build_ollama_router()
    core = LuciaCore(
        SQLiteMemoryStore(database),
        planner=CognitivePlanner(
            engine=router,
            allowed_actions=frozenset(registry.names()),
        ),
        action_executor=ActionExecutor(registry),
    )
    core.configure_model_router(router)

    core.remember(
        "Lucía debe usar herramientas disponibles cuando una tarea requiere un dato del entorno, y evaluar el resultado después de ejecutar la acción.",
        kind="procedural",
        importance=0.9,
    )

    result = core.run_agent(
        {
            "type": "user_message",
            "content": "Necesito la hora actual. Usa la herramienta disponible para obtenerla y termina cuando tengas el resultado.",
        },
        goal="Responder correctamente al usuario",
        task="Obtener la hora actual usando una herramienta disponible",
        max_iterations=3,
    )

    print("=== LUCÍA AGENT CYCLE → OLLAMA ===")
    print(f"Plan final: {result.plan}")
    print(f"Acciones ejecutadas: {len(result.action_results)}")
    print(f"Memorias recuperadas: {len(result.context.retrieved_memories)}")
    print(f"Resultados cognitivos: {len(result.context.cognitive_results)}")
    print(f"Evaluaciones: {len(result.context.evaluations)}")

    for index, action_result in enumerate(result.action_results, start=1):
        print(f"Acción #{index}: {action_result}")

    if result.context.cognitive_results:
        for index, cognitive in enumerate(result.context.cognitive_results, start=1):
            print(
                f"Cognición #{index}: engine={cognitive.get('engine')} "
                f"success={cognitive.get('success')} "
                f"output={str(cognitive.get('output')).replace(chr(10), ' ')[:800]}"
            )

    if result.context.evaluations:
        for index, evaluation in enumerate(result.context.evaluations, start=1):
            print(f"Evaluación #{index}: {evaluation}")

    if database.exists():
        database.unlink()


if __name__ == "__main__":
    main()
