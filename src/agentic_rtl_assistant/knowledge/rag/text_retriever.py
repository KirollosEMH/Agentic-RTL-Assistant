from __future__ import annotations

import re
import time

from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    RetrievalMetrics,
    SourceEvidence,
)
from agentic_rtl_assistant.knowledge.rag.chunker import RTLSemanticChunker
from agentic_rtl_assistant.rtl.parser import RTLParser
from agentic_rtl_assistant.rtl.repository import RTLRepository


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z_][a-z0-9_]*", text.casefold()))


class TextRetriever:
    def __init__(
        self,
        repository: RTLRepository,
        parser: RTLParser,
        *,
        max_chunks: int = 6,
        max_chunk_tokens: int = 1200,
    ) -> None:
        self.repository = repository
        self.parser = parser
        self.max_chunks = max_chunks
        self.max_chunk_tokens = max_chunk_tokens
        self.chunker = RTLSemanticChunker(repository)

    def retrieve(self, question: str) -> EvidencePack:
        started = time.perf_counter()
        modules = self.parser.parse_files(self.repository.list_verilog_files())
        query_terms = _terms(question)
        chunks = self.chunker.chunk_modules(modules)
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                len(query_terms & _terms(chunk.entity + " " + chunk.content)),
                chunk.entity,
            ),
            reverse=True,
        )[: self.max_chunks]
        evidence: list[SourceEvidence] = []
        token_count = 0
        for chunk in ranked:
            content = " ".join(chunk.content.split()[: self.max_chunk_tokens])
            words = len(content.split())
            evidence.append(
                SourceEvidence(
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    content,
                    "text_rag",
                    chunk.entity,
                )
            )
            token_count += words
        return EvidencePack(
            entities=tuple(chunk.entity for chunk in ranked),
            source_evidence=tuple(evidence),
            metrics=RetrievalMetrics(
                source_chunks_retrieved=len(evidence),
                source_tokens_retrieved=token_count,
                retrieval_latency_seconds=time.perf_counter() - started,
            ),
        )
