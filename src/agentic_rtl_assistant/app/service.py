"""UI-independent application service."""

from __future__ import annotations

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.approaches.factory import ApproachFactory
from agentic_rtl_assistant.config.models import AppConfig
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector


class ApplicationService:
    def __init__(
        self,
        config: AppConfig,
        approach_factory: ApproachFactory | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry or TelemetryCollector(config.telemetry.enabled)
        self.approach_factory = approach_factory or ApproachFactory(config, self.telemetry)
        self.approach = self.approach_factory.create()

    async def ask(self, text: str, *, session_id: str | None = None) -> RunResult:
        if not text.strip():
            raise ValueError("request cannot be empty")
        return await self.approach.run(UserRequest.create(text.strip()), RunContext(session_id))
