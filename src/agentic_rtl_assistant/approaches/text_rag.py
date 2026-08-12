"""A1: direct model plus deterministic RTL-aware text retrieval."""

from __future__ import annotations

import time

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.knowledge.rag.text_retriever import TextRetriever
from agentic_rtl_assistant.models.base import ModelProvider
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


class TextRAGApproach:
    name = "text_rag"

    def __init__(
        self,
        provider: ModelProvider,
        retriever: TextRetriever,
        *,
        model: str,
        provider_name: str,
        prompt: str,
        telemetry: TelemetryCollector,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.retriever = retriever
        self.model = model
        self.provider_name = provider_name
        self.prompt = prompt
        self.telemetry = telemetry
        self.temperature = temperature

    async def run(self, request: UserRequest, context: RunContext) -> RunResult:
        del context
        started = time.perf_counter()
        start = ExecutionTrace(request.request_id, self.name, EventType.REQUEST_STARTED)
        self.telemetry.record(start)
        retrieval_started = time.perf_counter()
        evidence = self.retriever.retrieve(request.text)
        retrieval_seconds = time.perf_counter() - retrieval_started
        model_started = time.perf_counter()
        response = await self.provider.generate(
            ModelRequest(
                model=self.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    ModelMessage(
                        "user", f"Question:\n{request.text}\n\nEvidence:\n{evidence.to_prompt()}"
                    ),
                ),
                temperature=self.temperature,
            )
        )
        model_seconds = time.perf_counter() - model_started
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
            evidence=evidence,
            usage=response.usage,
            timing=TimingMetrics(
                total_seconds=duration,
                model_seconds=model_seconds,
                retrieval_seconds=retrieval_seconds,
            ),
            traces=(start, end),
            provider=self.provider_name,
            model=self.model,
        )
