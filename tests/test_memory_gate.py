from lucia.events import Event
from lucia.memory import Memory
from lucia.memory_gate import MemoryGate
from lucia.storage import SQLiteMemoryStore


def test_transient_events_are_not_memories():
    gate = MemoryGate()
    event = Event(type="window.changed", data={"current": "Visual Studio Code"})

    assert gate.should_remember(event, 1.0) is False


def test_explicit_memory_candidate_is_accepted():
    gate = MemoryGate()
    event = Event(
        type="user.preference",
        data={"content": "User prefers local-first AI systems.", "confidence": 0.95},
    )

    assert gate.should_remember(event, 0.8) is True
    fields = gate.to_memory_fields(event, 0.8)

    assert fields["kind"] == "preference"
    assert fields["content"] == "User prefers local-first AI systems."
    assert fields["confidence"] == 0.95


def test_memory_candidate_persists(tmp_path):
    gate = MemoryGate()
    store = SQLiteMemoryStore(tmp_path / "lucia.db")
    event = Event(
        type="project.milestone",
        data={"content": "Lucía completed the perception-to-context pipeline."},
        source="pipeline",
    )

    fields = gate.to_memory_fields(event, 0.9)
    store.save(Memory(**fields))

    matches = store.search("perception-to-context")
    assert len(matches) == 1
    assert matches[0].kind == "project"
    assert matches[0].importance == 0.9
