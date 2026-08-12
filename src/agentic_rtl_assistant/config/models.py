"""Typed application configuration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ApproachType(StrEnum):
    DIRECT_LLM = "direct_llm"
    TEXT_RAG = "text_rag"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_RAG = "multi_agent_rag"
    MULTI_AGENT_GRAPHRAG = "multi_agent_graphrag"


class AppSettings(StrictModel):
    name: str = "Agentic RTL Assistant"
    environment: str = "local"
    log_level: str = "INFO"


class ProjectSettings(StrictModel):
    root: Path
    verilog_extensions: list[str] = Field(default_factory=lambda: [".v", ".sv"])
    ignored_paths: list[str] = Field(default_factory=lambda: [".git", ".venv", "build"])


class ApproachSettings(StrictModel):
    type: ApproachType = ApproachType.MULTI_AGENT_GRAPHRAG


class AgentSettings(StrictModel):
    enabled: bool = True
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    prompt: Path


class OrchestrationSettings(StrictModel):
    max_steps: int = Field(default=12, ge=1)
    max_repair_attempts: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=120, gt=0)
    agents: dict[str, AgentSettings]


class ModelProfile(StrictModel):
    provider: str
    model: str
    timeout_seconds: float = Field(default=60, gt=0)
    retries: int = Field(default=1, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


class ProviderSettings(StrictModel):
    base_url: str | None = None
    api_key_env: str | None = None
    account_id_env: str | None = None


class ModelsSettings(StrictModel):
    profiles: dict[str, ModelProfile]
    providers: dict[str, ProviderSettings]


class GraphSettings(StrictModel):
    enabled: bool = True
    backend: Literal["memory"] = "memory"
    build_mode: Literal["lazy", "eager"] = "lazy"
    extracted_node_types: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


class GraphRAGSettings(StrictModel):
    enabled: bool = True
    max_hops: int = Field(default=2, ge=0)
    max_nodes: int = Field(default=24, ge=1)
    max_source_chunks: int = Field(default=6, ge=1)
    max_context_tokens: int = Field(default=4000, ge=1)
    include_graph_facts: bool = True
    include_source_evidence: bool = True


class TextRAGSettings(StrictModel):
    enabled: bool = True
    chunking_strategy: Literal["rtl_semantic", "module"] = "rtl_semantic"
    max_chunks: int = Field(default=6, ge=1)
    max_chunk_tokens: int = Field(default=1200, ge=1)


class KnowledgeSettings(StrictModel):
    graph: GraphSettings
    graphrag: GraphRAGSettings
    text_rag: TextRAGSettings


class ValidationSettings(StrictModel):
    parser_check: bool = True
    compile_check: bool = False
    simulator: str | None = None


class FilesystemSettings(StrictModel):
    allow_reads: bool = True
    allow_writes: bool = False
    require_confirmation_for_write: bool = True


class RTLSettings(StrictModel):
    parser_backend: Literal["pyverilog"] = "pyverilog"
    validation: ValidationSettings
    filesystem: FilesystemSettings


class TokenBudgetSettings(StrictModel):
    max_request_tokens: int = Field(default=8000, ge=1)
    max_evidence_tokens: int = Field(default=4000, ge=1)


class ContextSettings(StrictModel):
    max_conversation_messages: int = Field(default=12, ge=0)
    max_resolved_entities: int = Field(default=24, ge=1)
    max_evidence_items: int = Field(default=12, ge=1)
    token_budget: TokenBudgetSettings


class TelemetrySettings(StrictModel):
    enabled: bool = True
    track_input_tokens: bool = True
    track_output_tokens: bool = True
    track_cached_tokens: bool = True
    track_timing: bool = True
    track_tools: bool = True
    track_agents: bool = True
    track_retrieval: bool = True
    persistence_path: Path = Path("runs")


class EvaluationSettings(StrictModel):
    datasets: list[Path] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    repetitions: int = Field(default=1, ge=1)
    approaches: list[ApproachType] = Field(default_factory=list)
    model_profiles: list[str] = Field(default_factory=list)


class UISettings(StrictModel):
    type: Literal["textual"] = "textual"
    show_execution_panel: bool = True
    show_token_panel: bool = True
    show_metrics_panel: bool = True
    show_agent_activity: bool = True
    show_tool_activity: bool = True


class AppConfig(StrictModel):
    app: AppSettings
    project: ProjectSettings
    approach: ApproachSettings
    orchestration: OrchestrationSettings
    models: ModelsSettings
    knowledge: KnowledgeSettings
    rtl: RTLSettings
    context: ContextSettings
    telemetry: TelemetrySettings
    evaluation: EvaluationSettings
    ui: UISettings

    @model_validator(mode="after")
    def validate_architecture(self) -> AppConfig:
        approach = self.approach.type
        if approach is ApproachType.MULTI_AGENT_GRAPHRAG:
            if not self.knowledge.graph.enabled:
                raise ValueError("multi_agent_graphrag requires knowledge.graph.enabled=true")
            if not self.knowledge.graphrag.enabled:
                raise ValueError("multi_agent_graphrag requires knowledge.graphrag.enabled=true")
        if (
            approach in {ApproachType.TEXT_RAG, ApproachType.MULTI_AGENT_RAG}
            and not self.knowledge.text_rag.enabled
        ):
            raise ValueError(f"{approach.value} requires knowledge.text_rag.enabled=true")

        missing_profiles = {
            agent.model
            for agent in self.orchestration.agents.values()
            if agent.enabled and agent.model not in self.models.profiles
        }
        missing_profiles.update(
            profile
            for profile in self.evaluation.model_profiles
            if profile not in self.models.profiles
        )
        if missing_profiles:
            names = ", ".join(sorted(missing_profiles))
            raise ValueError(f"unknown model profile(s): {names}")

        missing_providers = {
            profile.provider
            for profile in self.models.profiles.values()
            if profile.provider not in self.models.providers
        }
        if missing_providers:
            names = ", ".join(sorted(missing_providers))
            raise ValueError(f"model profiles reference unknown provider(s): {names}")

        if (
            self.context.token_budget.max_evidence_tokens
            > self.context.token_budget.max_request_tokens
        ):
            raise ValueError("max_evidence_tokens cannot exceed max_request_tokens")
        return self
