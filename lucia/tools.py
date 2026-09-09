"""Built-in deterministic tools for Lucía."""

from datetime import datetime, timezone
from platform import machine, platform, processor


class GetTimeTool:
    """Return the current UTC time in a structured form."""

    @property
    def name(self) -> str:
        return "get_time"

    def execute(self, parameters: dict) -> dict:
        now = datetime.now(timezone.utc)
        return {"iso": now.isoformat(), "timezone": "UTC"}


class GetSystemInfoTool:
    """Return basic non-sensitive runtime platform information."""

    @property
    def name(self) -> str:
        return "get_system_info"

    def execute(self, parameters: dict) -> dict:
        return {
            "platform": platform(),
            "machine": machine(),
            "processor": processor(),
        }
