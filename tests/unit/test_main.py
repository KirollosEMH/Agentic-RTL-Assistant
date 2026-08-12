import importlib
import sys
from pathlib import Path

import pytest

from agentic_rtl_assistant.config import load_config

main_module = importlib.import_module("agentic_rtl_assistant.main")


def test_interactive_startup_defers_project_validation(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    missing_root = repository_root / "choose-project-in-ui"
    config = config.model_copy(
        update={
            "project": config.project.model_copy(update={"root": missing_root})
        }
    )
    started_with = []

    class StubApp:
        def __init__(self, selected_config) -> None:
            started_with.append(selected_config)

        def run(self) -> None:
            pass

    def unexpected_validation(_config) -> None:
        pytest.fail("interactive startup must validate the project after UI selection")

    monkeypatch.setattr(sys, "argv", ["rtl-assistant"])
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "validate_runtime_paths", unexpected_validation)
    monkeypatch.setattr(main_module, "RTLAssistantTUI", StubApp)

    main_module.main()

    assert started_with == [config]
