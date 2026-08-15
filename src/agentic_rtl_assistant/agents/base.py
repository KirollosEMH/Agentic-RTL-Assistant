from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agentic_rtl_assistant.config.models import AgentSettings, ModelProfile
from agentic_rtl_assistant.models.base import ModelProvider


class Agent[InputT, OutputT](ABC):
    name: str

    def __init__(
        self,
        model_client: ModelProvider,
        config: AgentSettings,
        profile: ModelProfile,
    ) -> None:
        self.model_client = model_client
        self.config = config
        self.profile = profile
        self.prompt = Path(config.prompt).read_text(encoding="utf-8").strip()

    @abstractmethod
    async def execute(self, context: InputT) -> OutputT:
        """Perform exactly this agent's one language task."""
        ...
