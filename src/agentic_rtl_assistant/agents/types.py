"""Typed contracts for narrowly scoped agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class Intent(StrEnum):
    LIST = "list"
    EXPLAIN = "explain"
    GENERATE = "generate"
    MODIFY = "modify"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IntentInput:
    user_request: str


@dataclass(frozen=True, slots=True)
class IntentOutput:
    intent: Intent


@dataclass(frozen=True, slots=True)
class ExplanationInput:
    user_request: str
    evidence: EvidencePack


@dataclass(frozen=True, slots=True)
class ExplanationOutput:
    answer: str


@dataclass(frozen=True, slots=True)
class CodeGenerationInput:
    requirement: str
    evidence: EvidencePack


@dataclass(frozen=True, slots=True)
class CodeGenerationOutput:
    code: str


@dataclass(frozen=True, slots=True)
class RepairInput:
    requirement: str
    generated_code: str
    validation_errors: tuple[str, ...]
    evidence: EvidencePack


@dataclass(frozen=True, slots=True)
class AgentResult[T]:
    output: T
    usage: TokenUsage
    traces: tuple[ExecutionTrace, ...] = ()
