"""Cognitive cycle result primitives for Lucia."""

from dataclasses import dataclass
from typing import Any

from .context import Context
from .planner import Plan


@dataclass(slots=True, frozen=True)
class CycleResult:
    """Complete outcome of one cognitive cycle."""

    context: Context
    plan: Plan
    action_results: tuple[dict[str, Any], ...]
