"""Central coordinator for the first Lucía prototype."""

from dataclasses import dataclass, field

from .context import Context
from .memory import Memory, MemoryStore
from .planner import Plan, Planner, RuleBasedPlanner


@dataclass(slots=True)
class LuciaCore:
    """Owns identity-independent system state and coordinates a cycle."""

    memory: MemoryStore
    name: str = "Lucía"
    planner: Planner = field(default_factory=RuleBasedPlanner)

    def observe(self, event: dict) -> Context:
        """Add an observed event to working context."""
        context = Context()
        context.add_event(event)
        return context

    def remember(self, content: str, *, kind: str = "episodic", importance: float = 0.5) -> Memory:
        """Persist a selected experience."""
        memory = Memory(content=content, kind=kind, importance=importance)
        self.memory.save(memory)
        return memory

    def plan(self, context: Context) -> Plan:
        """Produce a transient plan without executing any action."""
        return self.planner.plan(context)
