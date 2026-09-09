"""Ollama-backed cognitive engine for Lucía.

The adapter keeps the core independent from Ollama's client library by using
its HTTP API directly. No Ollama dependency is required to import Lucía.
"""

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request

from .cognitive import CognitiveRequest, CognitiveResult


@dataclass(slots=True)
class OllamaCognitiveEngine:
    """Use a local Ollama model as Lucía's cognitive engine."""

    model: str = "qwen3:4b"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120.0
    name: str = "ollama"

    def reason(self, request_data: CognitiveRequest) -> CognitiveResult:
        """Send one reasoning request to Ollama and normalize the response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Lucía's cognitive engine. Return a concise, useful answer to the task.",
                },
                {
                    "role": "user",
                    "content": self._format_request(request_data),
                },
            ],
        }
        if request_data.output_format == "json":
            payload["format"] = "json"

        body = json.dumps(payload).encode("utf-8")
        endpoint = self.base_url.rstrip("/") + "/api/chat"
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return CognitiveResult(
                success=False,
                error=f"Ollama request failed: {type(exc).__name__}: {exc}",
                engine=self.name,
            )

        message = data.get("message")
        if not isinstance(message, dict):
            return CognitiveResult(
                success=False,
                error="Ollama response did not contain a message",
                engine=self.name,
            )

        output = message.get("content")
        if not isinstance(output, str):
            return CognitiveResult(
                success=False,
                error="Ollama response message did not contain text content",
                engine=self.name,
            )

        return CognitiveResult(
            success=True,
            output=output,
            engine=f"ollama:{self.model}",
        )

    @staticmethod
    def _format_request(request_data: CognitiveRequest) -> str:
        context = json.dumps(request_data.context, ensure_ascii=False, default=str)
        return (
            f"Goal: {request_data.goal or 'none'}\n"
            f"Task: {request_data.task}\n"
            f"Working context: {context}"
        )
