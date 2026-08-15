"""Composition root for interchangeable assistant approaches."""

from __future__ import annotations

from pathlib import Path

from agentic_rtl_assistant.agents import (
    IntentClassifierAgent,
    RTLCodeAgent,
    RTLExplanationAgent,
    RTLRepairAgent,
)
from agentic_rtl_assistant.approaches.base import AssistantApproach
from agentic_rtl_assistant.approaches.direct import DirectLLMApproach
from agentic_rtl_assistant.approaches.multi_agent_graphrag import MultiAgentGraphRAGApproach
from agentic_rtl_assistant.approaches.multi_agent_rag import MultiAgentRAGApproach
from agentic_rtl_assistant.approaches.single_agent import SingleAgentApproach
from agentic_rtl_assistant.approaches.text_rag import TextRAGApproach
from agentic_rtl_assistant.config.models import AppConfig, ApproachType
from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.knowledge.graph import (
    GraphQuery,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)
from agentic_rtl_assistant.knowledge.rag import GraphRetriever, TextRetriever
from agentic_rtl_assistant.knowledge.service import KnowledgeService
from agentic_rtl_assistant.models.factory import ModelProviderFactory
from agentic_rtl_assistant.orchestration.graph import build_workflow
from agentic_rtl_assistant.orchestration.nodes import WorkflowNodes
from agentic_rtl_assistant.rtl import PyVerilogParser, RTLRepository, RTLValidator
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector


class TextRetrievalService:
    def __init__(self, retriever: TextRetriever) -> None:
        self.retriever = retriever

    def prepare(self) -> bool:
        return False

    def retrieve(self, question: str) -> EvidencePack:
        return self.retriever.retrieve(question)


class ApproachFactory:
    def __init__(
        self,
        config: AppConfig,
        telemetry: TelemetryCollector | None = None,
        model_factory: ModelProviderFactory | None = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry or TelemetryCollector(config.telemetry.enabled)
        self.model_factory = model_factory or ModelProviderFactory(config.models)
        self.repository = RTLRepository(
            config.project.root,
            extensions=tuple(config.project.verilog_extensions),
            ignored_paths=tuple(config.project.ignored_paths),
            allow_reads=config.rtl.filesystem.allow_reads,
        )
        self.parser = PyVerilogParser(self.repository)

    def create(self, approach_type: ApproachType | None = None) -> AssistantApproach:
        selected = approach_type or self.config.approach.type
        if selected is ApproachType.DIRECT_LLM:
            return self._direct()
        if selected is ApproachType.TEXT_RAG:
            return self._text_rag()
        if selected is ApproachType.SINGLE_AGENT:
            return self._single_agent()
        if selected is ApproachType.MULTI_AGENT_RAG:
            return self._multi_agent(graphrag=False)
        if selected is ApproachType.MULTI_AGENT_GRAPHRAG:
            return self._multi_agent(graphrag=True)
        raise ValueError(f"unsupported approach: {selected}")

    def _agent_config(self, name: str):
        try:
            return self.config.orchestration.agents[name]
        except KeyError as exc:
            raise ValueError(f"missing required agent configuration: {name}") from exc

    def _direct(self) -> DirectLLMApproach:
        agent = self._agent_config("rtl_explanation")
        provider, profile = self.model_factory.create_for_profile(agent.model)
        return DirectLLMApproach(
            provider,
            model=profile.model,
            provider_name=profile.provider,
            prompt=Path(agent.prompt).read_text(encoding="utf-8"),
            repository=self.repository,
            telemetry=self.telemetry,
            temperature=agent.temperature,
            max_output_tokens=profile.max_output_tokens,
        )

    def _single_agent(self) -> SingleAgentApproach:
        agent = self._agent_config("single_agent")
        provider, profile = self.model_factory.create_for_profile(agent.model)
        return SingleAgentApproach(
            provider,
            self.repository,
            model=profile.model,
            provider_name=profile.provider,
            prompt=Path(agent.prompt).read_text(encoding="utf-8"),
            telemetry=self.telemetry,
            temperature=agent.temperature,
            max_steps=self.config.orchestration.max_steps,
            max_evidence_items=self.config.context.max_evidence_items,
            max_evidence_tokens=self.config.context.token_budget.max_evidence_tokens,
            max_output_tokens=profile.max_output_tokens,
        )

    def _text_retriever(self) -> TextRetriever:
        settings = self.config.knowledge.text_rag
        return TextRetriever(
            self.repository,
            self.parser,
            max_chunks=settings.max_chunks,
            max_chunk_tokens=settings.max_chunk_tokens,
        )

    def _text_rag(self) -> TextRAGApproach:
        agent = self._agent_config("rtl_explanation")
        provider, profile = self.model_factory.create_for_profile(agent.model)
        return TextRAGApproach(
            provider,
            self._text_retriever(),
            model=profile.model,
            provider_name=profile.provider,
            prompt=Path(agent.prompt).read_text(encoding="utf-8"),
            telemetry=self.telemetry,
            temperature=agent.temperature,
        )

    def _build_agent(self, name: str, agent_class):
        settings = self._agent_config(name)
        provider, profile = self.model_factory.create_for_profile(settings.model)
        return agent_class(provider, settings, profile)

    def _graph_retrieval_service(self) -> KnowledgeService:
        store = InMemoryGraphStore()
        builder = KnowledgeGraphBuilder(store)
        graph_settings = self.config.knowledge.graphrag
        retriever = GraphRetriever(
            self.repository,
            GraphQuery(store),
            max_hops=graph_settings.max_hops,
            max_nodes=graph_settings.max_nodes,
            max_source_chunks=graph_settings.max_source_chunks,
            max_context_tokens=graph_settings.max_context_tokens,
            include_graph_facts=graph_settings.include_graph_facts,
            include_source_evidence=graph_settings.include_source_evidence,
        )
        service = KnowledgeService(self.repository, self.parser, builder, retriever)
        if self.config.knowledge.graph.build_mode == "eager":
            service.prepare()
        return service

    def _multi_agent(self, *, graphrag: bool) -> MultiAgentGraphRAGApproach:
        intent = self._build_agent("intent_classifier", IntentClassifierAgent)
        explanation = self._build_agent("rtl_explanation", RTLExplanationAgent)
        code = self._build_agent("rtl_codegen", RTLCodeAgent)
        repair = self._build_agent("rtl_repair", RTLRepairAgent)
        retrieval = (
            self._graph_retrieval_service()
            if graphrag
            else TextRetrievalService(self._text_retriever())
        )
        nodes = WorkflowNodes(
            intent_agent=intent,
            explanation_agent=explanation,
            code_agent=code,
            repair_agent=repair,
            retrieval=retrieval,
            validator=RTLValidator(self.parser),
            telemetry=self.telemetry,
        )
        workflow = build_workflow(
            nodes, max_repair_attempts=self.config.orchestration.max_repair_attempts
        )
        profile = self.config.models.profiles[self._agent_config("rtl_explanation").model]
        approach_class = MultiAgentGraphRAGApproach if graphrag else MultiAgentRAGApproach
        return approach_class(
            workflow,
            telemetry=self.telemetry,
            max_steps=self.config.orchestration.max_steps,
            provider=profile.provider,
            model=profile.model,
        )
