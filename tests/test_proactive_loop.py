from unittest.mock import patch

from lucia.core import LuciaCore
from lucia.events import Event
from lucia.initiative import InitiativeEngine
from lucia.proactive_loop import ProactiveLoop
from lucia.storage import SQLiteMemoryStore
from lucia.attention_adapter import AttentionAdapter


class StubCore(LuciaCore):
    def __init__(self):
        super().__init__(
            SQLiteMemoryStore(":memory:"),
            attention_adapter=AttentionAdapter(),
            initiative_engine=InitiativeEngine(),
        )
        self.proactive_calls = []

    def run_proactive(self, event, *, goal=None, task=None, max_iterations=4):
        self.proactive_calls.append((event, goal, task, max_iterations))
        return "result"


def test_tick_processes_all_source_events():
    core = StubCore()
    events = [Event("a", {"x": 1}), Event("b", {"x": 2})]
    loop = ProactiveLoop(core, lambda: events, goal="g", task="t", max_iterations=2)

    assert loop.tick() == ["result", "result"]
    assert len(core.proactive_calls) == 2
    assert core.proactive_calls[0][0]["type"] == "a"
    assert core.proactive_calls[1][0]["type"] == "b"
    assert core.proactive_calls[0][1:] == ("g", "t", 2)


def test_cooldown_skips_duplicate_events():
    core = StubCore()
    loop = ProactiveLoop(core, lambda: [Event("tick", {"x": 1})], cooldown=60.0)

    assert loop.tick() == ["result"]
    assert loop.tick() == []
    assert len(core.proactive_calls) == 1


def test_run_can_be_bounded_without_sleeping_after_last_tick():
    core = StubCore()
    loop = ProactiveLoop(core, lambda: [Event("tick", {})], interval=0.001)

    loop.run(max_ticks=2)

    assert len(core.proactive_calls) == 2
    assert loop.running is False


def test_run_backoff_increases_idle_interval():
    core = StubCore()
    source = iter([[], [], []])
    loop = ProactiveLoop(
        core,
        lambda: next(source),
        interval=1.0,
        max_interval=4.0,
        idle_backoff=2.0,
    )

    with patch("lucia.proactive_loop.time.sleep") as sleep:
        loop.run(max_ticks=3)

    assert [call.args[0] for call in sleep.call_args_list] == [2.0, 4.0]


def test_stop_sets_running_false():
    core = StubCore()
    loop = ProactiveLoop(core, lambda: [])

    loop.running = True
    loop.stop()

    assert loop.running is False


def test_invalid_configuration_is_rejected():
    core = StubCore()

    try:
        ProactiveLoop(core, lambda: [], interval=0)
        assert False
    except ValueError as exc:
        assert "interval" in str(exc)

    try:
        ProactiveLoop(core, lambda: [], max_iterations=0)
        assert False
    except ValueError as exc:
        assert "max_iterations" in str(exc)

    try:
        ProactiveLoop(core, lambda: [], cooldown=-1)
        assert False
    except ValueError as exc:
        assert "cooldown" in str(exc)

    try:
        ProactiveLoop(core, lambda: [], interval=2, max_interval=1)
        assert False
    except ValueError as exc:
        assert "max_interval" in str(exc)

    try:
        ProactiveLoop(core, lambda: [], idle_backoff=0.5)
        assert False
    except ValueError as exc:
        assert "idle_backoff" in str(exc)

    loop = ProactiveLoop(core, lambda: [])
    try:
        loop.run(max_ticks=0)
        assert False
    except ValueError as exc:
        assert "max_ticks" in str(exc)
