from pathlib import Path

import pytest

from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.ui.app import RTLAssistantTUI


class StubTelemetry:
    def __init__(self) -> None:
        self.subscribers = []

    def subscribe(self, callback) -> None:
        self.subscribers.append(callback)


class StubService:
    def __init__(self, config) -> None:
        self.config = config
        self.telemetry = StubTelemetry()
        self.created_sessions = 0

    def create_session(self) -> str:
        self.created_sessions += 1
        return f"session-{self.created_sessions}"


@pytest.mark.asyncio
async def test_project_is_selected_before_service_creation(
    repository_root: Path, rtl_root: Path
) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    created = []

    def create_service(selected_config):
        service = StubService(selected_config)
        created.append(service)
        return service

    app = RTLAssistantTUI(config, service_factory=create_service)
    async with app.run_test() as pilot:
        assert app.service is None
        path_input = app.query_one("#project-path")
        path_input.value = str(rtl_root)
        await pilot.click("#open-project")
        await pilot.pause()

        assert app.service is created[0]
        assert app.service.config.project.root == rtl_root.resolve()
        assert app.session_id == "session-1"
        assert "session-" in str(app.query_one("#session").render())
        assert app.query_one("#project-setup").has_class("hidden")
        assert not app.query_one("#workspace").has_class("hidden")

        await pilot.click("#new-session")
        await pilot.pause()

        assert app.session_id == "session-2"

        await pilot.click("#choose-project")
        await pilot.pause()

        assert not app.query_one("#project-setup").has_class("hidden")
        assert app.query_one("#workspace").has_class("hidden")
        assert app.query_one("#project-path").value == str(rtl_root.resolve())


@pytest.mark.asyncio
async def test_invalid_project_keeps_setup_visible(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    app = RTLAssistantTUI(config, service_factory=StubService)

    async with app.run_test() as pilot:
        path_input = app.query_one("#project-path")
        path_input.value = str(repository_root / "missing-project")
        await pilot.click("#open-project")
        await pilot.pause()

        assert app.service is None
        assert "does not exist" in str(app.query_one("#project-error").render())
        assert not app.query_one("#project-setup").has_class("hidden")
