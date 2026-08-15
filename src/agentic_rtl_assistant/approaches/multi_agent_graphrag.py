from __future__ import annotations

import time

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.orchestration.state import AgentState
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.tokens import aggregate_usage
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


class MultiAgentGraphRAGApproach:
    name = "multi_agent_graphrag"

    def __init__(
        self,
        workflow,
        *,
        telemetry: TelemetryCollector,
        max_steps: int,
        provider: str,
        model: str,
    ) -> None:
        self.workflow = workflow
        self.telemetry = telemetry
        self.max_steps = max_steps
        self.provider = provider
        self.model = model

    async def run(self, request: UserRequest, context: RunContext) -> RunResult:
        started = time.perf_counter()
        start = ExecutionTrace(request.request_id, self.name, EventType.REQUEST_STARTED)
        self.telemetry.record(start)
        initial: AgentState = {
            "request_id": request.request_id,
            "user_request": request.text,
            "conversation_history": context.recent_messages,
            "resolved_entities": context.resolved_entities,
            "repair_attempts": 0,
            "usage_events": [],
            "traces": [start],
        }
        if context.session_id:
            initial["session_id"] = context.session_id
        if context.write_confirmation is not None:
            initial["write_confirmation"] = context.write_confirmation
        try:
            state = await self.workflow.ainvoke(
                initial, config={"recursion_limit": self.max_steps}
            )
        except Exception as exc:  # graph boundary converts failures into RunResult
            duration = time.perf_counter() - started
            failed = ExecutionTrace(
                request.request_id,
                self.name,
                EventType.REQUEST_FAILED,
                duration_seconds=duration,
                metadata={"error_type": type(exc).__name__},
            )
            self.telemetry.record(failed)
            return RunResult(
                request_id=request.request_id,
                approach=self.name,
                usage=TokenUsage(),
                timing=TimingMetrics(total_seconds=duration),
                traces=tuple(self.telemetry.for_request(request.request_id)),
                provider=self.provider,
                model=self.model,
                error=str(exc),
            )

        duration = time.perf_counter() - started
        completed = ExecutionTrace(
            request.request_id,
            self.name,
            EventType.REQUEST_COMPLETED,
            duration_seconds=duration,
            metadata={"success": "error" not in state},
        )
        self.telemetry.record(completed)
        evidence = state.get("evidence", EvidencePack())
        return RunResult(
            request_id=request.request_id,
            approach=self.name,
            answer=state.get("answer"),
            generated_code=state.get("generated_code"),
            written_files=tuple(state.get("written_files", [])),
            write_error=state.get("write_error"),
            evidence=evidence,
            usage=aggregate_usage(state.get("usage_events", [])),
            timing=TimingMetrics(
                total_seconds=duration,
                retrieval_seconds=evidence.metrics.retrieval_latency_seconds,
            ),
            traces=tuple(self.telemetry.for_request(request.request_id)),
            validation=state.get("validation_result"),
            provider=self.provider,
            model=self.model,
            error=state.get("error"),
        )
