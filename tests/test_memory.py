from lucia.memory import Memory
from lucia.storage import SQLiteMemoryStore


def test_memory_persists(tmp_path):
    database = tmp_path / "lucia.db"
    store = SQLiteMemoryStore(database)

    store.save(Memory(content="Lucía started her first persistent memory.", importance=0.9))

    matches = store.search("persistent")
    assert len(matches) == 1
    assert matches[0].importance == 0.9
