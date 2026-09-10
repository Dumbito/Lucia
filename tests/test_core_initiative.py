import pytest

from lucia.attention_adapter import AttentionAdapter
from lucia.core import LuciaCore
from lucia.initiative import InitiativeEngine
from lucia.storage import SQLiteMemoryStore


def test_core_proactive_mode_reaches_planning_after_attention_and_initiative(tmp_path):
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        attention_adapter=AttentionAdapter(),
        # This test explicitly exercises the proactive path; use a lower
        # policy threshold rather than coupling the integration test to the
        # conservative production default.
        initiative_engine=InitiativeEngine(threshold=0.5),
    )

    result = core.run_proactive(
        {"type": "user_message", "data": {"content": "Analizar memoria"}},
        goal="Construir Lucia",
        task="Analizar memoria",
        max_iterations=1,
    )

    attention = next(
        event for event in result.context.events if event["type"] == "attention.result"
    )
    initiative = next(
        event for event in result.context.events if event["type"] == "initiative.result"
    )

    assert attention["data"]["process"] is True
    assert initiative["data"]["act"] is True
    assert result.context.cognitive_results


def test_core_proactive_mode_stops_when_initiative_rejects(tmp_path):
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        attention_adapter=AttentionAdapter(),
        initiative_engine=InitiativeEngine(threshold=0.95),
    )

    result = core.run_proactive(
        {"type": "user_message", "data": {"content": "Analizar memoria"}},
        goal="Construir Lucia",
        task="Analizar memoria",
        max_iterations=1,
    )

    initiative = next(
        event for event in result.context.events if event["type"] == "initiative.result"
    )

    assert initiative["data"]["act"] is False
    assert not result.context.cognitive_results


def test_core_proactive_mode_stops_before_initiative_when_attention_filters(tmp_path):
    core = LuciaCore(
        SQLiteMemoryStore(tmp_path / "lucia.db"),
        attention_adapter=AttentionAdapter(threshold=0.9),
        initiative_engine=InitiativeEngine(),
    )

    result = core.run_proactive(
        {"type": "system.snapshot", "data": {"content": "background noise"}},
        goal="Construir Lucia",
        task="Analizar memoria",
        max_iterations=1,
    )

    attention = next(
        event for event in result.context.events if event["type"] == "attention.result"
    )

    assert attention["data"]["process"] is False
    assert not any(event["type"] == "initiative.result" for event in result.context.events)
    assert not result.context.cognitive_results


def test_core_proactive_mode_requires_attention_and_initiative(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "lucia.db")

    with pytest.raises(RuntimeError, match="initiative engine"):
        LuciaCore(store, attention_adapter=AttentionAdapter()).run_proactive({"type": "event"})

    with pytest.raises(RuntimeError, match="attention adapter"):
        LuciaCore(store, initiative_engine=InitiativeEngine()).run_proactive({"type": "event"})
