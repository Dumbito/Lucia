"""Deterministic model selection for Lucía's local cognitive engines."""

from dataclasses import dataclass, field
import json
from typing import Iterable

from .cognitive import CognitiveEngine, CognitiveRequest, CognitiveResult


@dataclass(slots=True, frozen=True)
class ModelProfile:
    """Capabilities and resource characteristics of one candidate model."""

    model_id: str
    capabilities: frozenset[str] = frozenset()
    context_window: int = 8192
    speed: int = 1
    reasoning: int = 1
    coding: int = 1
    priority: int = 0
    enabled: bool = True


@dataclass(slots=True, frozen=True)
class ModelSelection:
    """Decision made by the router before delegation."""

    model_id: str
    score: int
    reason: str


@dataclass(slots=True)
class ModelRouter:
    """Select and delegate to the most suitable registered cognitive engine.

    Routing is deliberately deterministic in this first implementation. It
    uses task signals, context size, and model profiles rather than another
    LLM. Ollama remains responsible for loading the selected model.
    """

    engines: dict[str, CognitiveEngine] = field(default_factory=dict)
    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    default_model: str = "qwen3:8b"

    def register(
        self,
        model_id: str,
        engine: CognitiveEngine,
        *,
        capabilities: Iterable[str] = (),
        context_window: int = 8192,
        speed: int = 1,
        reasoning: int = 1,
        coding: int = 1,
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        self.engines[model_id] = engine
        self.profiles[model_id] = ModelProfile(
            model_id=model_id,
            capabilities=frozenset(capabilities),
            context_window=context_window,
            speed=speed,
            reasoning=reasoning,
            coding=coding,
            priority=priority,
            enabled=enabled,
        )

    def select(self, request: CognitiveRequest) -> ModelSelection:
        """Choose a model without executing it."""
        candidates = [
            profile for model_id, profile in self.profiles.items()
            if model_id in self.engines and profile.enabled
        ]
        if not candidates:
            raise RuntimeError("No enabled cognitive models are registered")

        signals = self._signals(request)
        context_tokens = self._estimate_context_tokens(request)
        scored = [
            (self._score(profile, signals, context_tokens), profile)
            for profile in candidates
        ]
        scored.sort(key=lambda item: (item[0], item[1].priority, item[1].speed), reverse=True)
        score, profile = scored[0]
        return ModelSelection(
            model_id=profile.model_id,
            score=score,
            reason=self._reason(profile, signals, context_tokens),
        )

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        """Route one cognitive request to the selected engine."""
        selection = self.select(request)
        result = self.engines[selection.model_id].reason(request)
        if result.engine == "unknown":
            return CognitiveResult(
                success=result.success,
                output=result.output,
                error=result.error,
                engine=selection.model_id,
            )
        return result

    def _score(self, profile: ModelProfile, signals: set[str], context_tokens: int) -> int:
        if context_tokens > profile.context_window:
            return -10_000

        score = profile.priority + profile.speed
        if "coding" in signals:
            score += profile.coding * 4
            score += 4 if "coding" in profile.capabilities else 0
        if "reasoning" in signals:
            score += profile.reasoning * 4
            score += 4 if "reasoning" in profile.capabilities else 0
        if "agent" in signals:
            score += 3 if "agent" in profile.capabilities else 0
        if "long_context" in signals:
            score += 6 if "long_context" in profile.capabilities else 0
        if context_tokens > profile.context_window * 0.60:
            score += 8 if "long_context" in profile.capabilities else 0
            score += min(profile.context_window // 16384, 8)
        if "fast" in signals:
            score += profile.speed * 3
        if "general" in signals:
            score += 2 if "general" in profile.capabilities else 0
        return score

    @staticmethod
    def _signals(request: CognitiveRequest) -> set[str]:
        text = f"{request.task} {request.goal or ''}".lower()
        signals: set[str] = set()
        coding_terms = (
            "code", "coding", "program", "programming", "python", "bug",
            "debug", "repository", "repo", "git", "function", "class",
            "implement", "refactor", "test", "api", "script",
        )
        reasoning_terms = (
            "analyze", "analysis", "reason", "reasoning", "compare", "design",
            "architecture", "research", "hypothesis", "evaluate", "explain",
            "complex", "strategy", "plan",
        )
        agent_terms = ("agent", "tool", "tools", "execute", "action", "workflow")
        long_terms = (
            "long context", "large context", "entire repository",
            "whole repository", "many documents",
        )
        fast_terms = (
            "quick", "fast", "simple", "brief", "short", "just",
            "what time", "current time",
        )
        if any(term in text for term in coding_terms):
            signals.add("coding")
        if any(term in text for term in reasoning_terms):
            signals.add("reasoning")
        if any(term in text for term in agent_terms):
            signals.add("agent")
        if any(term in text for term in long_terms):
            signals.add("long_context")
        if any(term in text for term in fast_terms):
            signals.add("fast")
        if not signals:
            signals.add("general")
        return signals

    @staticmethod
    def _estimate_context_tokens(request: CognitiveRequest) -> int:
        """Estimate tokens conservatively without requiring a tokenizer."""
        payload = {
            "task": request.task,
            "goal": request.goal,
            "context": request.context,
        }
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        return max(1, len(serialized) // 4)

    @staticmethod
    def _reason(profile: ModelProfile, signals: set[str], context_tokens: int) -> str:
        matched = [
            signal for signal in
            ("coding", "reasoning", "agent", "long_context", "fast", "general")
            if signal in signals
        ]
        if context_tokens > profile.context_window * 0.60:
            matched.append("context_fit")
        return f"Selected {profile.model_id} for {', '.join(matched)}"


def build_default_router(engines: dict[str, CognitiveEngine]) -> ModelRouter:
    """Build Lucía's initial local model registry."""
    router = ModelRouter()
    profiles = {
        "gpt-oss:20b": dict(
            capabilities=("reasoning", "agent", "long_context", "general"),
            context_window=131072, speed=2, reasoning=5, coding=3, priority=4,
        ),
        "qwen3-coder:30b": dict(
            capabilities=("coding", "agent", "long_context"),
            context_window=262144, speed=2, reasoning=4, coding=5, priority=5,
        ),
        "qwen2.5-coder:14b": dict(
            capabilities=("coding",),
            context_window=32768, speed=3, reasoning=3, coding=4, priority=3,
        ),
        "qwen3:14b": dict(
            capabilities=("general", "reasoning", "agent"),
            context_window=32768, speed=3, reasoning=4, coding=2, priority=2,
        ),
        "qwen3:8b": dict(
            capabilities=("general", "fast", "reasoning", "agent"),
            context_window=32768, speed=5, reasoning=3, coding=2, priority=1,
        ),
    }
    for model_id, kwargs in profiles.items():
        engine = engines.get(model_id)
        if engine is not None:
            router.register(model_id, engine, **kwargs)
    if not router.engines:
        raise ValueError("At least one cognitive engine must be supplied")
    if router.default_model not in router.engines:
        router.default_model = next(iter(router.engines))
    return router


def build_ollama_router(
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
    models: Iterable[str] | None = None,
) -> ModelRouter:
    """Build an Ollama-backed router without downloading models."""
    from .ollama import OllamaCognitiveEngine

    default_models = (
        "gpt-oss:20b",
        "qwen3-coder:30b",
        "qwen2.5-coder:14b",
        "qwen3:14b",
        "qwen3:8b",
    )
    selected_models = tuple(models or default_models)
    engines = {
        model_id: OllamaCognitiveEngine(
            model=model_id, base_url=base_url, timeout=timeout
        )
        for model_id in selected_models
    }
    return build_default_router(engines)
