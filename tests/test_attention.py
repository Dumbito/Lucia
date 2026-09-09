from lucia.attention import AttentionEngine
from lucia.events import Event


def test_salience_is_weighted_and_bounded() -> None:
    engine = AttentionEngine()
    event = Event(type="test")

    score = engine.score(
        event,
        novelty=1.0,
        relevance=1.0,
        urgency=1.0,
        goal_alignment=1.0,
    )

    assert score == 1.0
    assert engine.should_process(score)


def test_low_salience_is_filtered() -> None:
    engine = AttentionEngine()
    event = Event(type="test")

    score = engine.score(event, novelty=0.1, relevance=0.1)

    assert score < 0.5
    assert not engine.should_process(score)


def test_invalid_salience_input_is_rejected() -> None:
    engine = AttentionEngine()
    event = Event(type="test")

    try:
        engine.score(event, novelty=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid salience input should raise ValueError")
