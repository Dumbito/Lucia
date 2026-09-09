import pytest

from lucia.initiative import InitiativeEngine


def test_initiative_reaches_threshold_for_urgent_relevant_event():
    engine = InitiativeEngine()

    decision = engine.decide(
        salience=0.9,
        novelty=1.0,
        relevance=1.0,
        urgency=1.0,
        goal_alignment=1.0,
        uncertainty=0.5,
    )

    assert decision.act is True
    assert decision.score >= engine.threshold
    assert "threshold" in decision.reason


def test_initiative_rejects_low_value_event():
    engine = InitiativeEngine()

    decision = engine.decide(
        salience=0.2,
        novelty=0.2,
        relevance=0.3,
        urgency=0.0,
        goal_alignment=0.0,
        uncertainty=0.0,
    )

    assert decision.act is False
    assert decision.score < engine.threshold


def test_initiative_penalizes_interruption_cost():
    engine = InitiativeEngine()
    common = dict(
        salience=0.8,
        novelty=0.8,
        relevance=0.8,
        urgency=0.8,
        goal_alignment=0.8,
        uncertainty=0.8,
    )

    low_cost = engine.decide(**common, interruption_cost=0.0)
    high_cost = engine.decide(**common, interruption_cost=1.0)

    assert high_cost.score < low_cost.score


def test_initiative_rejects_invalid_inputs():
    engine = InitiativeEngine()

    with pytest.raises(ValueError):
        engine.decide(
            salience=1.1,
            novelty=0.0,
            relevance=0.0,
            urgency=0.0,
            goal_alignment=0.0,
        )

    with pytest.raises(ValueError):
        InitiativeEngine(threshold=1.5).decide(
            salience=0.0,
            novelty=0.0,
            relevance=0.0,
            urgency=0.0,
            goal_alignment=0.0,
        )
