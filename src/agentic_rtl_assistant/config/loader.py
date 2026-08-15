"""YAML loading with deterministic precedence and typed validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from agentic_rtl_assistant.config.models import AppConfig


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be loaded."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"could not load configuration {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    return loaded


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _environment_overrides(environment: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if root := environment.get("RTL_ASSISTANT_PROJECT_ROOT"):
        overrides.setdefault("project", {})["root"] = root
    if approach := environment.get("RTL_ASSISTANT_APPROACH"):
        overrides.setdefault("approach", {})["type"] = approach
    return overrides


def _resolve_paths(config: AppConfig, repository_root: Path) -> AppConfig:
    updates: dict[str, Any] = {}
    if not config.project.root.is_absolute():
        updates["project"] = config.project.model_copy(
            update={"root": (repository_root / config.project.root).resolve()}
        )

    resolved_agents = {
        name: settings.model_copy(
            update={
                "prompt": settings.prompt
                if settings.prompt.is_absolute()
                else (repository_root / settings.prompt).resolve()
            }
        )
        for name, settings in config.orchestration.agents.items()
    }
    updates["orchestration"] = config.orchestration.model_copy(
        update={"agents": resolved_agents}
    )
    if not config.telemetry.persistence_path.is_absolute():
        updates["telemetry"] = config.telemetry.model_copy(
            update={
                "persistence_path": (repository_root / config.telemetry.persistence_path).resolve()
            }
        )
    updates["evaluation"] = config.evaluation.model_copy(
        update={
            "datasets": [
                path if path.is_absolute() else (repository_root / path).resolve()
                for path in config.evaluation.datasets
            ]
        }
    )
    return config.model_copy(update=updates)


def load_config(
    selected_path: str | Path = "config/default.yaml",
    *,
    default_path: str | Path = "config/default.yaml",
    environment: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load default + selected YAML + supported environment overrides."""

    selected = Path(selected_path).resolve()
    default = Path(default_path).resolve()
    if not default.is_file():
        raise ConfigurationError(f"default configuration does not exist: {default}")
    data = _read_yaml(default)
    if selected != default:
        if not selected.is_file():
            raise ConfigurationError(f"selected configuration does not exist: {selected}")
        data = _deep_merge(data, _read_yaml(selected))
    data = _deep_merge(data, _environment_overrides(environment or os.environ))
    repository_root = default.parent.parent
    return _resolve_paths(AppConfig.model_validate(data), repository_root)
