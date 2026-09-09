from pathlib import Path

from lucia.context import Context
from lucia.core import LuciaCore
from lucia.storage import SQLiteMemoryStore


def test_core_exposes_planning_without_executing_actions(tmp_path: Path) -> None:
    core = LuciaCore(SQLiteMemoryStore(tmp_path / "lucia.db"))
    context = Context(
        active_goal="Construir Lucia",
        current_task="Evaluar memoria recuperada",
    )

    plan = core.plan(context)

    assert plan.goal == "Construir Lucia"
    assert plan.task == "Evaluar memoria recuperada"
    assert plan.steps[0].action == "reason"
