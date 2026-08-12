"""A0: direct model baseline without retrieval or an agent loop."""

from __future__ import annotations

import time

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.models.base import ModelProvider
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.session.models import as_model_messages
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


class DirectLLMApproach:
    name = "direct_llm"

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        provider_name: str,
        prompt: str,
        telemetry: TelemetryCollector,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.provider_name = provider_name
        self.prompt = prompt
        self.telemetry = telemetry
        self.temperature = temperature

    async def run(self, request: UserRequest, context: RunContext) -> RunResult:
        started = time.perf_counter()
        start = ExecutionTrace(request.request_id, self.name, EventType.REQUEST_STARTED)
        self.telemetry.record(start)
        response = await self.provider.generate(
            ModelRequest(
                model=self.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    *as_model_messages(context.recent_messages),
                    ModelMessage("user", request.text),
                ),
                temperature=self.temperature,
                metadata=context.model_metadata,
            )
        )
        duration = time.perf_counter() - started
        end = ExecutionTrace(
            request.request_id,
            self.name,
            EventType.REQUEST_COMPLETED,
            duration_seconds=duration,
        )
        self.telemetry.record(end)
        return RunResult(
            request_id=request.request_id,
            approach=self.name,
            answer=response.content,
            usage=response.usage,
            timing=TimingMetrics(total_seconds=duration, model_seconds=duration),
            traces=(start, end),
            provider=self.provider_name,
            model=self.model,
        )
