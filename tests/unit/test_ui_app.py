from pathlib import Path

import pytest

from agentic_rtl_assistant.approaches.base import RunResult
from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.rtl.tools import WriteRequest
from agentic_rtl_assistant.telemetry.context import ContextWindowMetrics
from agentic_rtl_assistant.ui.app import RTLAssistantTUI, WriteConfirmationScreen


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


class FailingService(StubService):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ask_calls = 0

    async def ask(self, text, *, session_id=None, write_confirmation=None):
        del text, session_id, write_confirmation
        self.ask_calls += 1
        raise RuntimeError("provider failed")


class ApprovalService(StubService):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.approved: bool | None = None

    async def ask(self, text, *, session_id=None, write_confirmation=None):
        del text, session_id
        assert write_confirmation is not None
        source = "module NewModule; endmodule\n"
        self.approved = await write_confirmation(
            WriteRequest("new_module.v", source)
        )
        return RunResult(
            request_id="request",
            approach="test",
            generated_code=source,
            written_files=("new_module.v",) if self.approved else (),
        )


def test_context_summary_replaces_cached_token_display() -> None:
    result = RunResult(
        request_id="request",
        approach="test",
        usage=TokenUsage(900, 100, 400, 3),
        context_window=ContextWindowMetrics(350, 500, 4),
    )

    summary = RTLAssistantTUI._context_summary(result)

    assert summary == (
        "context latest=350 peak=500 history_messages=4 "
        "tokens total_in=900 out=100 calls=3"
    )
    assert "cached" not in summary


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


@pytest.mark.asyncio
async def test_request_worker_restores_controls_after_exception(
    repository_root: Path, rtl_root: Path
) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    app = RTLAssistantTUI(config, service_factory=FailingService)

    async with app.run_test() as pilot:
        app.query_one("#project-path").value = str(rtl_root)
        await pilot.click("#open-project")
        question = app.query_one("#question")
        question.value = "trigger an error"
        question.focus()
        await pilot.press("enter")
        assert app._request_worker is not None
        await app._request_worker.wait()

        assert isinstance(app.service, FailingService)
        assert app.service.ask_calls == 1
        assert not question.disabled
        assert not app.query_one("#new-session").disabled
        assert not app.query_one("#choose-project").disabled


@pytest.mark.asyncio
async def test_request_worker_waits_for_write_confirmation(
    repository_root: Path, rtl_root: Path
) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    app = RTLAssistantTUI(config, service_factory=ApprovalService)

    async with app.run_test() as pilot:
        app.query_one("#project-path").value = str(rtl_root)
        await pilot.click("#open-project")
        question = app.query_one("#question")
        question.value = "create a module"
        question.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, WriteConfirmationScreen)
        await pilot.click("#approve-write")
        assert app._request_worker is not None
        await app._request_worker.wait()

        assert isinstance(app.service, ApprovalService)
        assert app.service.approved is True
        assert not question.disabled
