from lucia.attention_adapter import AttentionAdapter
from lucia.context import Context
from lucia.events import Event


def test_salient_event_is_added_to_working_context() -> None:
    context = Context(active_goal="Proyecto Lucia", current_task="VS Code")
    adapter = AttentionAdapter(threshold=0.5)
    event = Event(type="window.changed", data={"current": "Lucia - VS Code"})

    salience, process, components = adapter.evaluate(
        event,
        active_goal=context.active_goal,
        current_task=context.current_task,
    )

    assert process is True
    context.add_event(
        {
            **event.as_dict(),
            "salience": salience,
            "attention": components,
        }
    )

    assert len(context.events) == 1
    assert context.events[0]["type"] == "window.changed"
    assert context.events[0]["salience"] == salience


def test_non_salient_event_is_not_added() -> None:
    context = Context()
    adapter = AttentionAdapter(threshold=0.5)
    event = Event(type="window.changed", data={"current": "Brave"})

    _, process, _ = adapter.evaluate(event)

    assert process is False
    if process:
        context.add_event(event.as_dict())

    assert context.events == []
