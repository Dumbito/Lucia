"""Replaceable cognitive-engine interfaces for Lucía."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .context import Context


@dataclass(slots=True, frozen=True)
class CognitiveRequest:
    """Input presented to a cognitive engine for one reasoning step."""

    task: str
    goal: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CognitiveResult:
    """Normalized result returned by a cognitive engine."""

    success: bool
    output: Any = None
    error: str | None = None
    engine: str = "unknown"


class CognitiveEngine(Protocol):
    """Interface for local or remote reasoning engines."""

    @property
    def name(self) -> str: ...

    def reason(self, request: CognitiveRequest) -> CognitiveResult: ...


@dataclass(slots=True)
class RuleBasedCognitiveEngine:
    """Minimal deterministic engine used before connecting an LLM."""

    name: str = "rule_based"

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        task = request.task.strip()
        if not task:
            return CognitiveResult(
                success=False,
                error="No task provided",
                engine=self.name,
            )

        return CognitiveResult(
            success=True,
            output={
                "task": task,
                "goal": request.goal,
                "context_keys": tuple(sorted(request.context)),
            },
            engine=self.name,
        )


def build_cognitive_request(context: Context, *, description: str | None = None) -> CognitiveRequest:
    """Build a cognitive request from the current working context."""
    return CognitiveRequest(
        task=(description or context.current_task or "").strip(),
        goal=context.active_goal,
        context={
            "events": tuple(context.events),
            "retrieved_memories": tuple(context.retrieved_memories),
            "action_results": tuple(context.action_results),
            "evaluations": tuple(context.evaluations),
        },
    )
