"""Real smoke test for Linux perception -> proactive loop -> Lucía."""

from tempfile import TemporaryDirectory

from lucia.actions import ActionExecutor, ToolRegistry
from lucia.attention_adapter import AttentionAdapter
from lucia.core import LuciaCore
from lucia.initiative import InitiativeEngine
from lucia.model_router import build_ollama_router
from lucia.monitor import LinuxEventSource
from lucia.planner import CognitivePlanner
from lucia.proactive_loop import ProactiveLoop
from lucia.storage import SQLiteMemoryStore
from lucia.tools import GetTimeTool


if __name__ == "__main__":
    with TemporaryDirectory(prefix="lucia-proactive-") as directory:
        store = SQLiteMemoryStore(f"{directory}/memories.db")
        router = build_ollama_router()
        registry = ToolRegistry()
        registry.register(GetTimeTool())

        core = LuciaCore(
            store,
            planner=CognitivePlanner(engine=router, max_steps=4),
            cognitive_engine=router,
            action_executor=ActionExecutor(registry),
            attention_adapter=AttentionAdapter(),
            initiative_engine=InitiativeEngine(threshold=0.5),
        )

        source = LinuxEventSource()
        loop = ProactiveLoop(
            core,
            source,
            interval=0.01,
            goal="Responder correctamente al usuario",
            task="Evaluar el estado actual del sistema y actuar si es necesario",
            max_iterations=2,
        )

        results = loop.tick()

        print("=== LUCÍA REAL: LINUX → PROACTIVE LOOP → ATTENTION → INITIATIVE ===")
        print(f"Eventos procesados: {len(results)}")
        for index, result in enumerate(results, start=1):
            attention = next(
                event for event in result.context.events
                if event["type"] == "attention.result"
            )
            initiative = next(
                event for event in result.context.events
                if event["type"] == "initiative.result"
            )
            print(f"\n--- Evento #{index} ---")
            print(f"Atención: {attention['data']}")
            print(f"Iniciativa: {initiative['data']}")
            print(f"Plan final: {result.plan}")
            print(f"Acciones ejecutadas: {len(result.action_results)}")
            print(f"Resultados cognitivos: {len(result.context.cognitive_results)}")
            for cognitive_index, item in enumerate(
                result.context.cognitive_results, start=1
            ):
                print(
                    f"Cognición #{cognitive_index}: engine={item.get('engine')} "
                    f"success={item.get('success')} output={item.get('output')}"
                )
