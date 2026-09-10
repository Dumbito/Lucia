"""Deterministic initiative decisions for Lucía.

Initiative is intentionally separate from attention: attention answers
"should this event be processed?", while initiative answers "should Lucía
proactively create an intention from it?".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    """Decision about whether an attended event deserves proactive action."""

    act: bool
    score: float
    reason: str


@dataclass(slots=True)
class InitiativeEngine:
    """Interpretable heuristic for proactive behavior.

    The weights form a true convex combination: without an interruption-cost
    penalty, the score stays in the same 0..1 scale as its inputs. This keeps
    the threshold meaningful and prevents accidental saturation.
    """

    threshold: float = 0.65
    salience_weight: float = 0.30 / 1.30
    novelty_weight: float = 0.15 / 1.30
    relevance_weight: float = 0.30 / 1.30
    urgency_weight: float = 0.25 / 1.30
    goal_alignment_weight: float = 0.20 / 1.30
    uncertainty_weight: float = 0.10 / 1.30
    interruption_cost_weight: float = 0.15

    def decide(
        self,
        *,
        salience: float,
        novelty: float,
        relevance: float,
        urgency: float,
        goal_alignment: float,
        uncertainty: float = 0.0,
        interruption_cost: float = 0.0,
    ) -> InitiativeDecision:
        values = {
            "salience": salience,
            "novelty": novelty,
            "relevance": relevance,
            "urgency": urgency,
            "goal_alignment": goal_alignment,
            "uncertainty": uncertainty,
            "interruption_cost": interruption_cost,
        }
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("initiative inputs must be between 0 and 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        score = (
            self.salience_weight * salience
            + self.novelty_weight * novelty
            + self.relevance_weight * relevance
            + self.urgency_weight * urgency
            + self.goal_alignment_weight * goal_alignment
            + self.uncertainty_weight * uncertainty
            - self.interruption_cost_weight * interruption_cost
        )
        score = max(0.0, min(1.0, score))

        if score >= self.threshold:
            return InitiativeDecision(
                act=True,
                score=score,
                reason="initiative threshold reached",
            )
        return InitiativeDecision(
            act=False,
            score=score,
            reason="initiative threshold not reached",
        )
