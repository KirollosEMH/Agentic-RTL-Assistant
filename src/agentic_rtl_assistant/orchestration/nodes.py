"""LangGraph nodes; retrieval and validation remain deterministic services."""

from __future__ import annotations

import re
import time
from typing import Protocol

from agentic_rtl_assistant.agents import (
    IntentClassifierAgent,
    RTLCodeAgent,
    RTLExplanationAgent,
    RTLRepairAgent,
)
from agentic_rtl_assistant.agents.types import (
    CodeGenerationInput,
    ExplanationInput,
    IntentInput,
    RepairInput,
)
from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.orchestration.state import AgentState
from agentic_rtl_assistant.rtl.validator import RTLValidator
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


class RetrievalService(Protocol):
    def prepare(self) -> bool: ...

    def retrieve(self, question: str) -> EvidencePack: ...


class WorkflowNodes:
    def __init__(
        self,
        *,
        intent_agent: IntentClassifierAgent,
        explanation_agent: RTLExplanationAgent,
        code_agent: RTLCodeAgent,
        repair_agent: RTLRepairAgent,
        retrieval: RetrievalService,
        validator: RTLValidator,
        telemetry: TelemetryCollector,
    ) -> None:
        self.intent_agent = intent_agent
        self.explanation_agent = explanation_agent
        self.code_agent = code_agent
        self.repair_agent = repair_agent
        self.retrieval = retrieval
        self.validator = validator
        self.telemetry = telemetry

    def _trace(
        self,
        state: AgentState,
        component: str,
        event_type: EventType,
        *,
        duration: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ExecutionTrace:
        trace = ExecutionTrace(
            request_id=state["request_id"],
            component=component,
            event_type=event_type,
            duration_seconds=duration,
            metadata=metadata or {},
        )
        self.telemetry.record(trace)
        return trace

    async def classify_intent(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, self.intent_agent.name, EventType.AGENT_STARTED)
        result = await self.intent_agent.execute(IntentInput(state["user_request"]))
        end = self._trace(
            state,
            self.intent_agent.name,
            EventType.AGENT_COMPLETED,
            duration=time.perf_counter() - started,
            metadata={"intent": result.output.intent.value},
        )
        return {
            "intent": result.output.intent,
            "usage_events": [result.usage],
            "traces": [start, end],
        }

    def graph_retrieve(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, "knowledge_graph", EventType.RETRIEVAL_STARTED)
        rebuilt = self.retrieval.prepare()
        end = self._trace(
            state,
            "knowledge_graph",
            EventType.RETRIEVAL_COMPLETED,
            duration=time.perf_counter() - started,
            metadata={"rebuilt": rebuilt},
        )
        return {"traces": [start, end]}

    def source_retrieve(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        evidence = self.retrieval.retrieve(state["user_request"])
        trace = self._trace(
            state,
            "evidence_retrieval",
            EventType.RETRIEVAL_COMPLETED,
            duration=time.perf_counter() - started,
            metadata={
                "entities": len(evidence.entities),
                "source_chunks": len(evidence.source_evidence),
                "graph_nodes": evidence.metrics.graph_nodes_visited,
            },
        )
        return {"evidence": evidence, "entities": list(evidence.entities), "traces": [trace]}

    async def explain(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, self.explanation_agent.name, EventType.AGENT_STARTED)
        result = await self.explanation_agent.execute(
            ExplanationInput(state["user_request"], state.get("evidence", EvidencePack()))
        )
        end = self._trace(
            state,
            self.explanation_agent.name,
            EventType.AGENT_COMPLETED,
            duration=time.perf_counter() - started,
        )
        return {
            "answer": result.output.answer,
            "usage_events": [result.usage],
            "traces": [start, end],
        }

    async def generate(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, self.code_agent.name, EventType.AGENT_STARTED)
        result = await self.code_agent.execute(
            CodeGenerationInput(state["user_request"], state.get("evidence", EvidencePack()))
        )
        end = self._trace(
            state,
            self.code_agent.name,
            EventType.AGENT_COMPLETED,
            duration=time.perf_counter() - started,
        )
        return {
            "generated_code": result.output.code,
            "usage_events": [result.usage],
            "traces": [start, end],
        }

    def validate(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, "rtl_validator", EventType.VALIDATION_STARTED)
        match = re.search(r"module\s+(?:named\s+)?([A-Za-z_][A-Za-z0-9_]*)", state["user_request"])
        expected_module = match.group(1) if match else None
        result = self.validator.validate_verilog(
            state.get("generated_code", ""), expected_module=expected_module
        )
        end = self._trace(
            state,
            "rtl_validator",
            EventType.VALIDATION_COMPLETED,
            duration=time.perf_counter() - started,
            metadata={"valid": result.valid, "errors": len(result.errors)},
        )
        return {"validation_result": result, "traces": [start, end]}

    async def repair(self, state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        start = self._trace(state, self.repair_agent.name, EventType.AGENT_STARTED)
        validation = state["validation_result"]
        result = await self.repair_agent.execute(
            RepairInput(
                requirement=state["user_request"],
                generated_code=state.get("generated_code", ""),
                validation_errors=validation.errors,
                evidence=state.get("evidence", EvidencePack()),
            )
        )
        attempts = state.get("repair_attempts", 0) + 1
        end = self._trace(
            state,
            self.repair_agent.name,
            EventType.AGENT_COMPLETED,
            duration=time.perf_counter() - started,
            metadata={"repair_attempt": attempts},
        )
        return {
            "generated_code": result.output.code,
            "repair_attempts": attempts,
            "usage_events": [result.usage],
            "traces": [start, end],
        }

    def finalize(self, state: AgentState) -> dict[str, object]:
        if state.get("intent") is None:
            return {"error": "request could not be classified"}
        validation = state.get("validation_result")
        if validation is not None and not validation.valid:
            return {"error": "RTL validation failed after configured repair attempts"}
        return {}
