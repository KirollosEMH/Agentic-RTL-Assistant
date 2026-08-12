"""Responsive Textual shell consuming application events."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, RichLog

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.config.models import AppConfig
from agentic_rtl_assistant.config.validation import validate_runtime_paths
from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class RTLAssistantTUI(App[None]):
    TITLE = "Agentic RTL Assistant"
    CSS = """
    #project-setup { height: 1fr; padding: 1 2; }
    #project-setup-title { height: 2; text-style: bold; }
    #project-path-row { height: 3; }
    #project-path { width: 1fr; }
    #open-project { width: auto; margin-left: 1; }
    #project-error { height: 2; color: $error; }
    #project-browser { height: 1fr; border: round $secondary; }
    #workspace { height: 1fr; }
    #project-header { height: 3; }
    #project { width: 1fr; padding: 1 2; }
    #session { width: auto; padding: 1 2; color: $text-muted; }
    #new-session { width: auto; margin-right: 1; }
    #choose-project { width: auto; margin-right: 2; }
    #panels { height: 1fr; }
    #chat { width: 2fr; border: round $accent; }
    #execution { width: 1fr; border: round $secondary; }
    #question { dock: bottom; }
    .hidden { display: none; }
    """

    def __init__(
        self,
        config: AppConfig,
        service_factory: Callable[[AppConfig], ApplicationService] = ApplicationService,
    ) -> None:
        super().__init__()
        self.config = config
        self.service: ApplicationService | None = None
        self.session_id: str | None = None
        self._service_factory = service_factory

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="project-setup"):
            yield Label("Select an RTL project directory", id="project-setup-title")
            with Horizontal(id="project-path-row"):
                yield Input(value=str(self.config.project.root), id="project-path")
                yield Button("Open project", id="open-project", variant="primary")
            yield Label("Paste a path or select a folder below.", id="project-error")
            yield DirectoryTree(self._browser_root(), id="project-browser")
        with Vertical(id="workspace", classes="hidden"):
            with Horizontal(id="project-header"):
                yield Label("Project:", id="project")
                yield Label("Session: not started", id="session")
                yield Button("New session", id="new-session")
                yield Button("Choose project", id="choose-project")
            with Horizontal(id="panels"):
                yield RichLog(id="chat", wrap=True, markup=True)
                yield RichLog(id="execution", wrap=True, markup=True)
            yield Input(placeholder="Ask about the RTL project...", id="question")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#project-path", Input).focus()

    def _browser_root(self) -> Path:
        configured = self.config.project.root
        if configured.is_dir():
            return configured.parent
        return Path.cwd()

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self.query_one("#project-path", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-project":
            self._open_project(self.query_one("#project-path", Input).value)
        elif event.button.id == "choose-project":
            self._show_project_setup()
        elif event.button.id == "new-session":
            self._start_new_session()

    def _start_new_session(self) -> None:
        if self.service is None:
            return
        self.session_id = self.service.create_session()
        self.query_one("#session", Label).update(f"Session: {self.session_id[:8]}")
        self.query_one("#chat", RichLog).clear()
        self.query_one("#execution", RichLog).clear()
        self.query_one("#chat", RichLog).write(
            "[dim]New session started. Conversation context is empty.[/]"
        )
        self.query_one("#question", Input).focus()

    def _show_project_setup(self) -> None:
        self.query_one("#project-path", Input).value = str(self.config.project.root)
        self.query_one("#project-error", Label).update(
            "Paste a path or select a folder below."
        )
        self.query_one("#project-browser", DirectoryTree).path = self._browser_root()
        self.query_one("#workspace", Vertical).add_class("hidden")
        self.query_one("#project-setup", Vertical).remove_class("hidden")
        self.query_one("#project-path", Input).focus()

    def _open_project(self, raw_path: str) -> None:
        value = raw_path.strip().strip('"')
        status = self.query_one("#project-error", Label)
        if not value:
            status.update("Enter or select a project directory.")
            return

        project_root = Path(value).expanduser().resolve()
        selected_config = self.config.model_copy(
            update={
                "project": self.config.project.model_copy(update={"root": project_root})
            }
        )
        try:
            validate_runtime_paths(selected_config)
            service = self._service_factory(selected_config)
        except (OSError, ValueError) as exc:
            status.update(str(exc))
            return

        self.config = selected_config
        self.service = service
        service.telemetry.subscribe(self._on_trace)
        self._start_new_session()
        self.query_one("#project", Label).update(f"Project: {project_root}")
        self.query_one("#project-setup", Vertical).add_class("hidden")
        self.query_one("#workspace", Vertical).remove_class("hidden")
        self.query_one("#question", Input).focus()

    def _on_trace(self, event: ExecutionTrace) -> None:
        if not self.is_running:
            return
        self.query_one("#execution", RichLog).write(
            f"{event.event_type.value}: {event.component}"
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "project-path":
            self._open_project(event.value)
            return
        if event.input.id != "question" or self.service is None:
            return

        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        event.input.disabled = True
        chat = self.query_one("#chat", RichLog)
        chat.write(f"[bold cyan]You:[/] {question}")
        result = await self.service.ask(question, session_id=self.session_id)
        if result.error:
            chat.write(f"[bold red]Error:[/] {result.error}")
        elif result.generated_code:
            chat.write(f"[bold green]Assistant:[/]\n{result.generated_code}")
        else:
            chat.write(f"[bold green]Assistant:[/] {result.answer}")
        cached = result.usage.cached_input_tokens
        self.query_one("#execution", RichLog).write(
            f"tokens in={result.usage.input_tokens} cached={cached} "
            f"out={result.usage.output_tokens} calls={result.usage.llm_calls}"
        )
        event.input.disabled = False
        event.input.focus()
