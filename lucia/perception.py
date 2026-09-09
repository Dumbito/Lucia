"""Operating-system perception adapters for Lucía v0.1.

The first adapter reads inexpensive local state from Linux rather than
capturing the screen. This keeps perception observable and low-cost.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .events import Event


@dataclass(slots=True)
class SystemSnapshot:
    """Small, privacy-conscious snapshot of local system state."""

    active_window: str | None
    music_status: str | None
    cpu_load: float | None
    memory_percent: float | None


class LinuxPerception:
    """Collect basic desktop state using commands available on Linux."""

    def snapshot(self) -> SystemSnapshot:
        return SystemSnapshot(
            active_window=self._active_window(),
            music_status=self._media_status(),
            cpu_load=self._load_average(),
            memory_percent=self._memory_percent(),
        )

    def events(self, previous: SystemSnapshot | None, current: SystemSnapshot) -> list[Event]:
        """Convert changes in the snapshot into normalized events."""
        if previous is None:
            return [Event(type="system.snapshot", data=current.__dict__ if hasattr(current, "__dict__") else {
                "active_window": current.active_window,
                "music_status": current.music_status,
                "cpu_load": current.cpu_load,
                "memory_percent": current.memory_percent,
            }, source="linux")]

        events: list[Event] = []
        if current.active_window != previous.active_window:
            events.append(Event(
                type="window.changed",
                data={"previous": previous.active_window, "current": current.active_window},
                source="linux",
            ))
        if current.music_status != previous.music_status:
            events.append(Event(
                type="media.changed",
                data={"previous": previous.music_status, "current": current.music_status},
                source="linux",
            ))
        return events

    @staticmethod
    def _active_window() -> str | None:
        if shutil.which("hyprctl"):
            result = subprocess.run(
                ["hyprctl", "activewindow", "-j"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                import json
                try:
                    data = json.loads(result.stdout)
                    title = data.get("title")
                    return str(title) if title else None
                except json.JSONDecodeError:
                    return None
        return os.environ.get("XDG_CURRENT_DESKTOP")

    @staticmethod
    def _media_status() -> str | None:
        if not shutil.which("playerctl"):
            return None
        result = subprocess.run(
            ["playerctl", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = result.stdout.strip()
        return status or None

    @staticmethod
    def _load_average() -> float | None:
        try:
            return os.getloadavg()[0]
        except OSError:
            return None

    @staticmethod
    def _memory_percent() -> float | None:
        try:
            total = available = None
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, value, *_ = line.split()
                    if key == "MemTotal:":
                        total = float(value)
                    elif key == "MemAvailable:":
                        available = float(value)
                    if total is not None and available is not None:
                        break
            if not total:
                return None
            return round((1 - available / total) * 100, 2)
        except (OSError, ValueError):
            return None
