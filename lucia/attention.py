"""Salience and attention filtering for observed events."""

from dataclasses import dataclass

from .events import Event


@dataclass(slots=True)
class SalienceWeights:
    """Weights for the first deterministic salience model."""

    novelty: float = 0.30
    relevance: float = 0.35
    urgency: float = 0.20
    goal_alignment: float = 0.15


class AttentionEngine:
    """Assigns an interpretable salience score before expensive cognition."""

    def __init__(self, weights: SalienceWeights | None = None) -> None:
        self.weights = weights or SalienceWeights()

    def score(
        self,
        event: Event,
        *,
        novelty: float = 0.0,
        relevance: float = 0.0,
        urgency: float = 0.0,
        goal_alignment: float = 0.0,
    ) -> float:
        """Return a score in [0, 1]. Inputs are expected in [0, 1]."""
        values = (novelty, relevance, urgency, goal_alignment)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("salience inputs must be between 0 and 1")

        score = (
            novelty * self.weights.novelty
            + relevance * self.weights.relevance
            + urgency * self.weights.urgency
            + goal_alignment * self.weights.goal_alignment
        )
        return min(1.0, max(0.0, score))

    def should_process(self, score: float, threshold: float = 0.5) -> bool:
        """Decide whether an event deserves downstream processing."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return score >= threshold
