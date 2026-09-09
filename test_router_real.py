"""Real Ollama smoke test for Lucía's multi-model router.

Run manually with Ollama running. Models are invoked sequentially, so they do
not need to stay loaded in VRAM at the same time.
"""

from lucia.cognitive import CognitiveRequest
from lucia.model_router import build_ollama_router


CASES = (
    ("fast", "What time is it?"),
    ("coding", "Implement a Python function to calculate Fibonacci numbers and add tests."),
    ("reasoning", "Analyze the architecture of Lucía and compare two strategies for model selection."),
)


def main() -> None:
    router = build_ollama_router()

    print("=== LUCÍA MULTI-MODEL ROUTER ===")
    for label, task in CASES:
        request = CognitiveRequest(task=task)
        selection = router.select(request)
        print(f"\n[{label}]")
        print(f"Seleccionado: {selection.model_id}")
        print(f"Score: {selection.score}")
        print(f"Razón: {selection.reason}")
        print("Ejecutando...")
        result = router.reason(request)
        print(f"Engine real: {result.engine}")
        print(f"Success: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            output = str(result.output).replace("\n", " ")
            print(f"Output: {output[:500]}")


if __name__ == "__main__":
    main()
