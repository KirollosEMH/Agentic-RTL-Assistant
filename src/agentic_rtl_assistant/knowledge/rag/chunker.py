from dataclasses import dataclass

from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.rtl.types import ModuleInfo


@dataclass(frozen=True, slots=True)
class RTLChunk:
    entity: str
    path: str
    start_line: int
    end_line: int
    content: str


class RTLSemanticChunker:
    def __init__(self, repository: RTLRepository) -> None:
        self.repository = repository

    def chunk_modules(self, modules: tuple[ModuleInfo, ...]) -> tuple[RTLChunk, ...]:
        return tuple(
            RTLChunk(
                entity=module.name,
                path=module.location.path,
                start_line=module.location.start_line,
                end_line=module.location.end_line,
                content=self.repository.read_lines(
                    module.location.path,
                    module.location.start_line,
                    module.location.end_line,
                ),
            )
            for module in modules
        )
