from lucia.attention_adapter import AttentionAdapter
from lucia.events import Event


def test_first_event_is_novel_and_repeated_event_less_novel() -> None:
    adapter = AttentionAdapter()
    event = Event(type="window.changed", data={"current": "VS Code"})

    first, _, first_parts = adapter.evaluate(event)
    second, _, second_parts = adapter.evaluate(event)

    assert first > second
    assert first_parts["novelty"] == 1.0
    assert second_parts["novelty"] == 0.2


def test_context_can_increase_relevance_and_goal_alignment() -> None:
    adapter = AttentionAdapter()
    event = Event(type="window.changed", data={"current": "Lucia - VS Code"})

    _, _, parts = adapter.evaluate(
        event,
        active_goal="Proyecto Lucia",
        current_task="VS Code",
    )

    assert parts["relevance"] == 0.75
    assert parts["goal_alignment"] == 0.75


def test_unknown_event_gets_small_default_urgency() -> None:
    adapter = AttentionAdapter()
    event = Event(type="external.alert", data={})

    _, _, parts = adapter.evaluate(event)

    assert parts["urgency"] == 0.20
