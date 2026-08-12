from typing import Protocol

from agentic_rtl_assistant.knowledge.evidence import EvidencePack


class EvidenceRetriever(Protocol):
    def retrieve(self, question: str) -> EvidencePack: ...
