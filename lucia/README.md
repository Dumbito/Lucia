# Lucía package

The `lucia` package contains the first executable cognitive components of Proyecto Lucía.

## Current components

- `events.py` — normalized observations and system events.
- `attention.py` — deterministic salience scoring and filtering.
- `context.py` — current working state.
- `memory.py` — memory contracts and data model.
- `core.py` — central coordinator.
- `loop.py` — cognitive-cycle orchestration.

The implementation intentionally starts with deterministic, dependency-free mechanisms. Local LLM runtimes, perception providers, and richer storage backends will be integrated behind explicit interfaces later.
