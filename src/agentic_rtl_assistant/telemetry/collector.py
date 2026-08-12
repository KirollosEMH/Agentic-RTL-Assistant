"""In-process event collector, independent of logging and UI."""

from __future__ import annotations

from collections.abc import Callable

from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class TelemetryCollector:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._events: list[ExecutionTrace] = []
        self._subscribers: list[Callable[[ExecutionTrace], None]] = []

    @property
    def events(self) -> tuple[ExecutionTrace, ...]:
        return tuple(self._events)

    def subscribe(self, callback: Callable[[ExecutionTrace], None]) -> None:
        self._subscribers.append(callback)

    def record(self, event: ExecutionTrace) -> None:
        if not self.enabled:
            return
        self._events.append(event)
        for subscriber in self._subscribers:
            subscriber(event)

    def for_request(self, request_id: str) -> tuple[ExecutionTrace, ...]:
        return tuple(event for event in self._events if event.request_id == request_id)
