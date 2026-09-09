from unittest.mock import patch

from lucia.cognitive import CognitiveRequest
from lucia.ollama import OllamaCognitiveEngine


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b'{"message": {"content": "Hello from local model"}}'


def test_ollama_engine_normalizes_response() -> None:
    engine = OllamaCognitiveEngine(model="test-model")

    with patch("lucia.ollama.request.urlopen", return_value=FakeResponse()):
        result = engine.reason(CognitiveRequest(task="say hello"))

    assert result.success is True
    assert result.output == "Hello from local model"
    assert result.engine == "ollama:test-model"


def test_ollama_engine_reports_transport_failure() -> None:
    engine = OllamaCognitiveEngine()

    with patch("lucia.ollama.request.urlopen", side_effect=TimeoutError("timed out")):
        result = engine.reason(CognitiveRequest(task="say hello"))

    assert result.success is False
    assert result.output is None
    assert "Ollama request failed" in result.error


def test_ollama_request_contains_goal_task_and_context() -> None:
    engine = OllamaCognitiveEngine()
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["body"] = http_request.data
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("lucia.ollama.request.urlopen", side_effect=fake_urlopen):
        engine.reason(
            CognitiveRequest(
                task="answer",
                goal="help",
                context={"events": ({"type": "user.message"},)},
            )
        )

    body = captured["body"].decode("utf-8")
    assert '"model": "qwen3:4b"' in body
    assert "Goal: help" in body
    assert "Task: answer" in body
    assert "user.message" in body


def test_ollama_request_uses_json_format_when_requested() -> None:
    engine = OllamaCognitiveEngine()
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["body"] = http_request.data
        return FakeResponse()

    with patch("lucia.ollama.request.urlopen", side_effect=fake_urlopen):
        engine.reason(CognitiveRequest(task="plan", output_format="json"))

    body = captured["body"].decode("utf-8")
    assert '"format": "json"' in body
