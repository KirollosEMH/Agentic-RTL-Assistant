"""Common request/result contract shared by all architecture families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.rtl.types import ValidationResult
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


@dataclass(frozen=True, slots=True)
class UserRequest:
    text: str
    request_id: str

    @classmethod
    def create(cls, text: str) -> UserRequest:
        return cls(text=text, request_id=str(uuid4()))


@dataclass(frozen=True, slots=True)
class RunContext:
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    request_id: str
    approach: str
    answer: str | None = None
    generated_code: str | None = None
    evidence: EvidencePack = EvidencePack()
    usage: TokenUsage = TokenUsage()
    timing: TimingMetrics = TimingMetrics()
    traces: tuple[ExecutionTrace, ...] = ()
    validation: ValidationResult | None = None
    provider: str | None = None
    model: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and (self.validation is None or self.validation.valid)


class AssistantApproach(Protocol):
    name: str

    async def run(self, request: UserRequest, context: RunContext) -> RunResult: ...
