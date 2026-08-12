"""Application-level event subscription contract."""

from typing import Protocol

from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class EventSubscriber(Protocol):
    def __call__(self, event: ExecutionTrace) -> None: ...
