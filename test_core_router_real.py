"""Real end-to-end smoke test for LuciaCore + ModelRouter + Ollama.

Run manually with Ollama running. The test uses the installed candidate model
set discovered by the router and keeps the request context inside LuciaCore.
"""

from pathlib import Path

from lucia.core import LuciaCore
from lucia.cognitive import build_cognitive_request
from lucia.model_router import build_ollama_router
from lucia.storage import SQLiteMemoryStore


def main() -> None:
    database = Path(".lucia-real-test.db")
    core = LuciaCore(SQLiteMemoryStore(database))
    router = build_ollama_router()
    core.configure_model_router(router)

    core.remember(
        "La arquitectura de Lucía usa un ModelRouter para seleccionar dinámicamente el modelo adecuado según la tarea, complejidad, contexto y capacidades disponibles.",
        kind="semantic",
        importance=0.9,
    )

    event = {
        "type": "user_message",
        "content": "Analiza la arquitectura actual de Lucía usando el contexto disponible y explica qué componente selecciona el modelo.",
    }
    context = core.observe(event)
    context.active_goal = "Construir Lucía"
    context.current_task = "Analizar la arquitectura actual de Lucía y explicar el papel del ModelRouter"
    context.retrieved_memories = [
        core._memory_as_dict(memory) for memory in core.retrieve_memories(context)
    ]

    request = build_cognitive_request(context)
    selection = router.select(request)
    result = core.reason(context)

    print("=== LUCÍA CORE → MEMORY → ROUTER → OLLAMA ===")
    print(f"Modelo seleccionado: {selection.model_id}")
    print(f"Score: {selection.score}")
    print(f"Razón: {selection.reason}")
    print(f"Engine real: {result['engine']}")
    print(f"Success: {result['success']}")
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Output: {str(result['output']).replace(chr(10), ' ')[:1200]}")
    print(f"Memorias recuperadas: {len(context.retrieved_memories)}")
    print(f"Resultados cognitivos: {len(context.cognitive_results)}")

    if database.exists():
        database.unlink()


if __name__ == "__main__":
    main()
