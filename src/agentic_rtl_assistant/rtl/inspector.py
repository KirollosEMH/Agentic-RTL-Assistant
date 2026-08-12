"""Deterministic query surface over parsed RTL metadata."""

from agentic_rtl_assistant.rtl.parser import RTLParser
from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.rtl.types import InstanceInfo, ModuleInfo


class RTLInspector:
    def __init__(self, repository: RTLRepository, parser: RTLParser) -> None:
        self.repository = repository
        self.parser = parser

    def parse_project_or_files(self) -> tuple[ModuleInfo, ...]:
        return self.parser.parse_files(self.repository.list_verilog_files())

    def list_modules(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.parse_project_or_files())

    def inspect_module(self, name: str) -> ModuleInfo | None:
        return next(
            (module for module in self.parse_project_or_files() if module.name == name),
            None,
        )

    def find_instantiations(self, module_name: str) -> tuple[tuple[str, InstanceInfo], ...]:
        matches: list[tuple[str, InstanceInfo]] = []
        for parent in self.parse_project_or_files():
            matches.extend(
                (parent.name, instance)
                for instance in parent.instances
                if instance.module == module_name
            )
        return tuple(matches)
