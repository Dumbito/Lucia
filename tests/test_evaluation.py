from lucia.actions import Action, ActionResult
from lucia.evaluation import RuleBasedEvaluator


def test_successful_action_with_output_is_useful() -> None:
    result = ActionResult(action=Action(name="test"), success=True, output={"value": 1})
    evaluation = RuleBasedEvaluator().evaluate(result)

    assert evaluation.success is True
    assert evaluation.useful is True
    assert evaluation.score == 1.0
    assert evaluation.should_remember is False


def test_successful_action_without_output_gets_neutral_score() -> None:
    result = ActionResult(action=Action(name="test"), success=True)
    evaluation = RuleBasedEvaluator().evaluate(result)

    assert evaluation.success is True
    assert evaluation.useful is False
    assert evaluation.score == 0.5
    assert evaluation.should_remember is False


def test_failed_action_should_be_remembered() -> None:
    result = ActionResult(
        action=Action(name="test"),
        success=False,
        error="RuntimeError: failed",
    )
    evaluation = RuleBasedEvaluator().evaluate(result)

    assert evaluation.success is False
    assert evaluation.useful is False
    assert evaluation.score == 0.0
    assert evaluation.should_remember is True
    assert "RuntimeError: failed" in evaluation.summary
