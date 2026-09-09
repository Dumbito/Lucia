"""Action and tool execution primitives for Lucía.

The planner describes intentions; this module provides the controlled boundary
where those intentions can be resolved into concrete tool calls.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class Action:
    """A concrete action requested by a plan step."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None


@dataclass(slots=True, frozen=True)
class ActionResult:
    """Normalized outcome of an action execution."""

    action: Action
    success: bool
    output: Any = None
    error: str | None = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Tool(Protocol):
    """Interface implemented by executable Lucía tools."""

    @property
    def name(self) -> str: ...

    def execute(self, parameters: dict[str, Any]) -> Any: ...


@dataclass(slots=True)
class ToolRegistry:
    """Registry mapping stable tool names to executable implementations."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        name = tool.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


@dataclass(slots=True)
class ActionExecutor:
    """Safely resolves actions through an explicit tool registry."""

    registry: ToolRegistry

    def execute(self, action: Action) -> ActionResult:
        try:
            tool = self.registry.get(action.name)
            output = tool.execute(dict(action.parameters))
            return ActionResult(action=action, success=True, output=output)
        except Exception as exc:  # noqa: BLE001 - boundary normalizes tool errors
            return ActionResult(
                action=action,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
