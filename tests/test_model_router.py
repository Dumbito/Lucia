from dataclasses import dataclass, replace

import pytest

from lucia.cognitive import CognitiveRequest, CognitiveResult
from lucia.model_router import ModelRouter, build_default_router


@dataclass
class FakeEngine:
    engine_name: str
    calls: int = 0

    @property
    def name(self) -> str:
        return self.engine_name

    def reason(self, request: CognitiveRequest) -> CognitiveResult:
        self.calls += 1
        return CognitiveResult(success=True, output=request.task, engine=self.engine_name)


def make_router() -> tuple[ModelRouter, dict[str, FakeEngine]]:
    engines = {
        "gpt-oss:20b": FakeEngine("gpt-oss:20b"),
        "qwen3-coder:30b": FakeEngine("qwen3-coder:30b"),
        "qwen2.5-coder:14b": FakeEngine("qwen2.5-coder:14b"),
        "qwen3:14b": FakeEngine("qwen3:14b"),
        "qwen3:8b": FakeEngine("qwen3:8b"),
    }
    return build_default_router(engines), engines


def test_router_prefers_fast_model_for_simple_request() -> None:
    router, _ = make_router()
    selection = router.select(CognitiveRequest(task="What time is it?"))
    assert selection.model_id == "qwen3:8b"
    assert "fast" in selection.reason


def test_router_prefers_coder_for_coding_request() -> None:
    router, _ = make_router()
    selection = router.select(CognitiveRequest(task="Implement a Python API and add tests"))
    assert selection.model_id == "qwen3-coder:30b"
    assert "coding" in selection.reason


def test_router_prefers_reasoning_model_for_complex_analysis() -> None:
    router, _ = make_router()
    selection = router.select(CognitiveRequest(task="Analyze the architecture and compare two strategies"))
    assert selection.model_id == "gpt-oss:20b"
    assert "reasoning" in selection.reason


def test_router_prefers_long_context_model_for_large_context() -> None:
    router, _ = make_router()
    selection = router.select(CognitiveRequest(task="Analyze the entire repository with long context"))
    assert selection.model_id == "qwen3-coder:30b"
    assert "long_context" in selection.reason


def test_router_delegates_only_to_selected_engine() -> None:
    router, engines = make_router()
    result = router.reason(CognitiveRequest(task="Fix this Python bug"))
    assert result.success is True
    assert result.engine == "qwen3-coder:30b"
    assert engines["qwen3-coder:30b"].calls == 1
    assert engines["gpt-oss:20b"].calls == 0


def test_router_skips_disabled_models() -> None:
    router, engines = make_router()
    profile = router.profiles["qwen3-coder:30b"]
    router.profiles["qwen3-coder:30b"] = replace(profile, enabled=False)
    selection = router.select(CognitiveRequest(task="Implement a Python API"))
    assert selection.model_id != "qwen3-coder:30b"
    assert engines["qwen3-coder:30b"].calls == 0


def test_router_respects_context_window() -> None:
    router, _ = make_router()
    huge_context = {"payload": "x" * 140_000}
    selection = router.select(CognitiveRequest(task="Analyze", context=huge_context))
    assert selection.model_id == "qwen3-coder:30b"
    assert selection.model_id != "qwen3:8b"


def test_router_requires_an_enabled_model() -> None:
    router = ModelRouter()
    with pytest.raises(RuntimeError, match="No enabled cognitive models"):
        router.select(CognitiveRequest(task="hello"))
