"""Typed per-request LangGraph state, separate from sessions and application state."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agentic_rtl_assistant.agents.types import Intent
from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.rtl.types import ValidationResult
from agentic_rtl_assistant.session.models import ConversationMessage
from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class AgentState(TypedDict, total=False):
    request_id: str
    user_request: str
    session_id: str
    conversation_history: tuple[ConversationMessage, ...]
    resolved_entities: tuple[str, ...]
    intent: Intent
    entities: list[str]
    evidence: EvidencePack
    answer: str
    generated_code: str
    validation_result: ValidationResult
    repair_attempts: int
    usage_events: Annotated[list[TokenUsage], operator.add]
    traces: Annotated[list[ExecutionTrace], operator.add]
    error: str
