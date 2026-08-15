from __future__ import annotations

import asyncio
from uuid import uuid4

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.approaches.factory import ApproachFactory
from agentic_rtl_assistant.config.models import AppConfig
from agentic_rtl_assistant.rtl.tools import WriteConfirmation
from agentic_rtl_assistant.session.store import InMemorySessionStore
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


class ApplicationService:
    def __init__(
        self,
        config: AppConfig,
        approach_factory: ApproachFactory | None = None,
        telemetry: TelemetryCollector | None = None,
        session_store: InMemorySessionStore | None = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry or TelemetryCollector(config.telemetry.enabled)
        self.approach_factory = approach_factory or ApproachFactory(config, self.telemetry)
        self.approach = self.approach_factory.create()
        self.sessions = session_store or InMemorySessionStore()

    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions.get_or_create(session_id)
        return session_id

    async def ask(
        self,
        text: str,
        *,
        session_id: str | None = None,
        write_confirmation: WriteConfirmation | None = None,
    ) -> RunResult:
        request_text = text.strip()
        if not request_text:
            raise ValueError("request cannot be empty")

        session = self.sessions.get_or_create(session_id) if session_id else None
        context = RunContext(
            session_id=session_id,
            recent_messages=tuple(session.recent_messages) if session else (),
            resolved_entities=tuple(session.resolved_entities) if session else (),
            write_confirmation=write_confirmation,
        )
        request = UserRequest.create(request_text)
        timeout = self.config.orchestration.timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                result = await self.approach.run(request, context)
        except TimeoutError:
            trace = ExecutionTrace(
                request.request_id,
                "application_service",
                EventType.REQUEST_FAILED,
                metadata={"reason": "timeout", "timeout_seconds": timeout},
            )
            self.telemetry.record(trace)
            result = RunResult(
                request_id=request.request_id,
                approach=self.approach.name,
                timing=TimingMetrics(total_seconds=timeout),
                traces=tuple(self.telemetry.for_request(request.request_id)),
                error=f"request timed out after {timeout:g} seconds",
            )
        response_text = result.generated_code or result.answer
        if session is not None and result.succeeded and response_text:
            session.record_turn(
                request_text,
                response_text,
                limit=self.config.context.max_conversation_messages,
            )
            for entity in result.evidence.entities:
                if entity not in session.resolved_entities:
                    session.resolved_entities.append(entity)
            entity_limit = self.config.context.max_resolved_entities
            if len(session.resolved_entities) > entity_limit:
                del session.resolved_entities[:-entity_limit]
        return result
