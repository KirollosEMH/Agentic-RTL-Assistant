"""Compact structural RTL types derived from source ASTs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceLocation:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class PortInfo:
    name: str
    direction: str
    data_type: str
    width: str | None = None


@dataclass(frozen=True, slots=True)
class InstanceInfo:
    name: str
    module: str
    connections: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    name: str
    location: SourceLocation
    ports: tuple[PortInfo, ...]
    instances: tuple[InstanceInfo, ...]
    source_hash: str


@dataclass(frozen=True, slots=True)
class SourceMatch:
    path: str
    line: int
    content: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    stages_run: tuple[str, ...] = ()
