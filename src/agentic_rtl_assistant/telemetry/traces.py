from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    REQUEST_STARTED = "request_started"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_COMPLETED = "model_call_completed"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    request_id: str
    component: str
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data
