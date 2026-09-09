from lucia.actions import Action, ActionExecutor, ToolRegistry
from lucia.tools import GetSystemInfoTool, GetTimeTool


def test_registry_registers_and_lists_tools() -> None:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())

    assert registry.names() == ("get_system_info", "get_time")
    assert registry.get("get_time").name == "get_time"


def test_executor_runs_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    executor = ActionExecutor(registry)

    result = executor.execute(Action(name="get_time"))

    assert result.success is True
    assert result.error is None
    assert result.output["timezone"] == "UTC"
    assert "iso" in result.output


def test_executor_normalizes_unknown_tool_error() -> None:
    executor = ActionExecutor(ToolRegistry())

    result = executor.execute(Action(name="missing_tool"))

    assert result.success is False
    assert result.output is None
    assert result.error == "KeyError: 'Unknown tool: missing_tool'"


def test_registry_rejects_duplicate_tools() -> None:
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    try:
        registry.register(GetTimeTool())
    except ValueError as exc:
        assert str(exc) == "Tool already registered: get_time"
    else:
        raise AssertionError("Expected duplicate registration to fail")
