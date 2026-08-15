from __future__ import annotations

import time

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    RetrievalMetrics,
    SourceEvidence,
)
from agentic_rtl_assistant.models.base import ModelProvider
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.session.models import as_model_messages
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.context import ContextWindowMetrics
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
        repository: RTLRepository,
        telemetry: TelemetryCollector,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.provider_name = provider_name
        self.prompt = prompt
        self.repository = repository
        self.telemetry = telemetry
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _project_evidence(self) -> EvidencePack:
        started = time.perf_counter()
        sources = []
        source_tokens = 0
        for path in self.repository.list_verilog_files():
            content = self.repository.read_source(path)
            relative_path = path.relative_to(self.repository.root).as_posix()
            lines = content.splitlines()
            sources.append(
                SourceEvidence(
                    path=relative_path,
                    start_line=1,
                    end_line=max(1, len(lines)),
                    content=content,
                    retrieval_type="full_project_context",
                )
            )
            source_tokens += len(content.split())
        return EvidencePack(
            source_evidence=tuple(sources),
            metrics=RetrievalMetrics(
                source_chunks_retrieved=len(sources),
                source_tokens_retrieved=source_tokens,
                retrieval_latency_seconds=time.perf_counter() - started,
            ),
        )

    async def run(self, request: UserRequest, context: RunContext) -> RunResult:
        started = time.perf_counter()
        start = ExecutionTrace(request.request_id, self.name, EventType.REQUEST_STARTED)
        self.telemetry.record(start)
        evidence = self._project_evidence()
        response = await self.provider.generate(
            ModelRequest(
                model=self.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    *as_model_messages(context.recent_messages),
                    ModelMessage(
                        "user",
                        f"Question:\n{request.text}\n\n"
                        f"Complete project RTL source:\n{evidence.to_prompt()}",
                    ),
                ),
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
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
            evidence=evidence,
            usage=response.usage,
            context_window=ContextWindowMetrics.from_usage_events(
                [response.usage], history_messages=len(context.recent_messages)
            ),
            timing=TimingMetrics(
                total_seconds=duration,
                model_seconds=max(
                    0.0, duration - evidence.metrics.retrieval_latency_seconds
                ),
                retrieval_seconds=evidence.metrics.retrieval_latency_seconds,
            ),
            traces=(start, end),
            provider=self.provider_name,
            model=self.model,
        )
