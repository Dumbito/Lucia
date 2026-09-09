"""Evaluation primitives for Lucía's action outcomes."""

from dataclasses import dataclass
from typing import Any

from .actions import ActionResult


@dataclass(slots=True, frozen=True)
class Evaluation:
    """Normalized assessment of one action result."""

    success: bool
    useful: bool
    score: float
    summary: str
    should_remember: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "useful": self.useful,
            "score": self.score,
            "summary": self.summary,
            "should_remember": self.should_remember,
        }


class RuleBasedEvaluator:
    """Deterministic evaluator used before an LLM-based metacognitive layer."""

    def evaluate(self, result: ActionResult) -> Evaluation:
        if not result.success:
            return Evaluation(
                success=False,
                useful=False,
                score=0.0,
                summary=f"Action failed: {result.error}",
                should_remember=True,
            )

        useful = result.output is not None
        score = 1.0 if useful else 0.5
        summary = "Action completed with a result." if useful else "Action completed without output."
        return Evaluation(
            success=True,
            useful=useful,
            score=score,
            summary=summary,
            should_remember=False,
        )
