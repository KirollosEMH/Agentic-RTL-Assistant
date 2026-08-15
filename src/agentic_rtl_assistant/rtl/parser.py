from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Protocol

from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser import ast as vast
from pyverilog.vparser.parser import ParseError, VerilogParser

from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.rtl.types import InstanceInfo, ModuleInfo, PortInfo, SourceLocation


class RTLParseError(RuntimeError):
    """Normalized Verilog parse failure."""


class RTLParser(Protocol):
    def parse_files(self, files: tuple[Path, ...]) -> tuple[ModuleInfo, ...]: ...

    def parse_text(self, source: str, *, path: str = "<generated>") -> tuple[ModuleInfo, ...]: ...


class PyVerilogParser:
    def __init__(self, repository: RTLRepository | None = None) -> None:
        self.repository = repository
        self._codegen = ASTCodeGenerator()

    def parse_project(self) -> tuple[ModuleInfo, ...]:
        if self.repository is None:
            raise RTLParseError("parse_project requires an RTLRepository")
        return self.parse_files(self.repository.list_verilog_files())

    def parse_files(self, files: tuple[Path, ...]) -> tuple[ModuleInfo, ...]:
        modules: list[ModuleInfo] = []
        for path in files:
            source = (
                self.repository.read_source(path)
                if self.repository is not None
                else path.read_text(encoding="utf-8")
            )
            display_path = (
                path.resolve().relative_to(self.repository.root).as_posix()
                if self.repository is not None
                else path.as_posix()
            )
            modules.extend(self.parse_text(source, path=display_path))
        return tuple(modules)

    def parse_text(self, source: str, *, path: str = "<generated>") -> tuple[ModuleInfo, ...]:
        try:
            with tempfile.TemporaryDirectory(prefix="rtl-parser-") as output_dir:
                ast = VerilogParser(outputdir=output_dir, debug=False).parse(source, debug=0)
        except (ParseError, OSError, SyntaxError) as exc:
            raise RTLParseError(f"failed to parse {path}: {exc}") from exc

        source_lines = source.splitlines()
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        descriptions = getattr(ast, "description", None)
        definitions = getattr(descriptions, "definitions", ())
        return tuple(
            self._module_info(definition, path, source_lines, source_hash)
            for definition in definitions
            if isinstance(definition, vast.ModuleDef)
        )

    def _module_info(
        self,
        module: vast.ModuleDef,
        path: str,
        source_lines: list[str],
        source_hash: str,
    ) -> ModuleInfo:
        start = max(1, module.lineno)
        end = self._find_endmodule(source_lines, start)
        return ModuleInfo(
            name=module.name,
            location=SourceLocation(path=path, start_line=start, end_line=end),
            ports=self._ports(module),
            instances=self._instances(module),
            source_hash=source_hash,
        )

    def _ports(self, module: vast.ModuleDef) -> tuple[PortInfo, ...]:
        ports: list[PortInfo] = []
        declarations: dict[str, vast.Node] = {}
        for item in module.items or ():
            if isinstance(item, vast.Decl):
                for declaration in item.list:
                    if isinstance(declaration, (vast.Input, vast.Output, vast.Inout)):
                        declarations[declaration.name] = declaration

        for port in module.portlist.ports if module.portlist else ():
            declaration = (
                port.first if isinstance(port, vast.Ioport) else declarations.get(port.name)
            )
            if not isinstance(declaration, (vast.Input, vast.Output, vast.Inout)):
                continue
            direction = declaration.__class__.__name__.lower()
            second = port.second if isinstance(port, vast.Ioport) else None
            data_type = "reg" if isinstance(second, vast.Reg) else "wire"
            width = self._codegen.visit(declaration.width).strip() if declaration.width else None
            ports.append(PortInfo(declaration.name, direction, data_type, width))
        return tuple(ports)

    def _instances(self, module: vast.ModuleDef) -> tuple[InstanceInfo, ...]:
        instances: list[InstanceInfo] = []
        for item in module.items or ():
            if not isinstance(item, vast.InstanceList):
                continue
            for instance in item.instances:
                connections = tuple(
                    (
                        connection.portname or "",
                        self._codegen.visit(connection.argname).strip()
                        if connection.argname is not None
                        else "",
                    )
                    for connection in instance.portlist
                )
                instances.append(InstanceInfo(instance.name, item.module, connections))
        return tuple(instances)

    @staticmethod
    def _find_endmodule(lines: list[str], start_line: int) -> int:
        for number in range(start_line, len(lines) + 1):
            if lines[number - 1].strip().startswith("endmodule"):
                return number
        return len(lines)
