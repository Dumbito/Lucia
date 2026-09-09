from lucia.cognitive import (
    CognitiveRequest,
    RuleBasedCognitiveEngine,
    build_cognitive_request,
)
from lucia.context import Context


def test_rule_based_engine_returns_normalized_result() -> None:
    engine = RuleBasedCognitiveEngine()
    result = engine.reason(CognitiveRequest(task="check time", goal="help user"))

    assert engine.name == "rule_based"
    assert result.success is True
    assert result.engine == "rule_based"
    assert result.output["task"] == "check time"
    assert result.output["goal"] == "help user"


def test_rule_based_engine_rejects_empty_task() -> None:
    result = RuleBasedCognitiveEngine().reason(CognitiveRequest(task=""))

    assert result.success is False
    assert result.output is None
    assert result.error == "No task provided"


def test_build_cognitive_request_captures_working_context() -> None:
    context = Context(
        active_goal="help user",
        current_task="answer question",
        events=[{"type": "user.message"}],
        retrieved_memories=[{"content": "previous context"}],
    )

    request = build_cognitive_request(context)

    assert request.task == "answer question"
    assert request.goal == "help user"
    assert request.context["events"] == ({"type": "user.message"},)
    assert request.context["retrieved_memories"] == ({"content": "previous context"},)
