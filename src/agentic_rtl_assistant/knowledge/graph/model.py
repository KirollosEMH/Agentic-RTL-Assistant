"""Extensible in-memory knowledge-graph domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agentic_rtl_assistant.rtl.types import SourceLocation


class NodeType(StrEnum):
    FILE = "file"
    MODULE = "module"
    PORT = "port"
    INSTANCE = "instance"


class RelationType(StrEnum):
    DEFINES = "DEFINES"
    HAS_PORT = "HAS_PORT"
    HAS_INSTANCE = "HAS_INSTANCE"
    INSTANTIATES = "INSTANTIATES"
    INSTANCE_OF = "INSTANCE_OF"


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    type: NodeType
    label: str
    location: SourceLocation | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    relation: RelationType
    target: str
