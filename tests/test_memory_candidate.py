from lucia.memory_candidate import user_fact, user_preference


def test_user_preference_creates_memory_eligible_event() -> None:
    event = user_preference("Lucía should be local-first.", confidence=0.95)
    assert event.type == "user.preference"
    assert event.data["content"] == "Lucía should be local-first."
    assert event.data["confidence"] == 0.95


def test_user_fact_rejects_invalid_confidence() -> None:
    try:
        user_fact("Lucía is a project.", confidence=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid confidence should raise ValueError")
