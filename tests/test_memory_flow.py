from lucia.memory_flow import run_flow


def test_memory_flow_persists_and_retrieves(tmp_path):
    db = tmp_path / "lucia.db"

    salience, saved, retrieved = run_flow(
        content="Lucía debe ser principalmente local.",
        kind="preference",
        goal="Proyecto Lucia local",
        memory_db=db,
    )

    # The calibrated memory gate accepts salience >= 0.70.
    assert salience >= 0.70
    assert saved is True
    assert len(retrieved) == 1
    assert retrieved[0].content == "Lucía debe ser principalmente local."
    assert retrieved[0].kind == "preference"
