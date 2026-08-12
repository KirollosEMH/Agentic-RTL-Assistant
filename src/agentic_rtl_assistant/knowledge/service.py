"""Derived-index lifecycle; source files remain authoritative."""

from __future__ import annotations

from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.knowledge.graph.builder import KnowledgeGraphBuilder
from agentic_rtl_assistant.knowledge.rag.graph_retriever import GraphRetriever
from agentic_rtl_assistant.rtl.parser import RTLParser
from agentic_rtl_assistant.rtl.repository import RTLRepository


class KnowledgeService:
    def __init__(
        self,
        repository: RTLRepository,
        parser: RTLParser,
        builder: KnowledgeGraphBuilder,
        graph_retriever: GraphRetriever,
    ) -> None:
        self.repository = repository
        self.parser = parser
        self.builder = builder
        self.graph_retriever = graph_retriever
        self._source_hashes: dict[str, str] = {}

    def prepare(self) -> bool:
        """Refresh the derived graph only when source hashes changed."""

        return self.refresh_if_stale()

    def refresh_if_stale(self) -> bool:
        files = self.repository.list_verilog_files()
        snapshots = {
            snapshot.path: snapshot.content_hash
            for snapshot in map(self.repository.snapshot, files)
        }
        if snapshots == self._source_hashes:
            return False
        modules = self.parser.parse_files(files)
        self.builder.rebuild(modules)
        self._source_hashes = snapshots
        return True

    def retrieve(self, question: str) -> EvidencePack:
        self.refresh_if_stale()
        return self.graph_retriever.retrieve(question)
