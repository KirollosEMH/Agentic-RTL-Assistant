from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rtl_assistant.config import ApproachType, load_config


def test_default_configuration_selects_primary_architecture(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})

    assert config.approach.type is ApproachType.MULTI_AGENT_GRAPHRAG
    assert config.knowledge.graph.build_mode == "lazy"
    assert config.project.root == repository_root / "descriptive_verilog_design"


def test_graphrag_requires_enabled_graph(repository_root: Path, tmp_path: Path) -> None:
    override = tmp_path / "invalid.yaml"
    override.write_text("knowledge:\n  graph:\n    enabled: false\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="requires knowledge.graph.enabled"):
        load_config(
            override,
            default_path=repository_root / "config" / "default.yaml",
            environment={},
        )


def test_agent_model_profile_must_exist(repository_root: Path, tmp_path: Path) -> None:
    override = tmp_path / "invalid-profile.yaml"
    override.write_text(
        "orchestration:\n  agents:\n    rtl_codegen:\n      model: missing\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="unknown model profile"):
        load_config(
            override,
            default_path=repository_root / "config" / "default.yaml",
            environment={},
        )
