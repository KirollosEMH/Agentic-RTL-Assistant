"""Responsive Textual shell consuming application events."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Label, RichLog

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.telemetry.traces import ExecutionTrace


class RTLAssistantTUI(App[None]):
    TITLE = "Agentic RTL Assistant"
    CSS = """
    #project { height: 3; padding: 1 2; }
    #panels { height: 1fr; }
    #chat { width: 2fr; border: round $accent; }
    #execution { width: 1fr; border: round $secondary; }
    #question { dock: bottom; }
    """

    def __init__(self, service: ApplicationService) -> None:
        super().__init__()
        self.service = service
        self.service.telemetry.subscribe(self._on_trace)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Project: {self.service.config.project.root}", id="project")
        with Horizontal(id="panels"):
            yield RichLog(id="chat", wrap=True, markup=True)
            yield RichLog(id="execution", wrap=True, markup=True)
        yield Input(placeholder="Ask about the RTL project...", id="question")
        yield Footer()

    def _on_trace(self, event: ExecutionTrace) -> None:
        if not self.is_running:
            return
        self.query_one("#execution", RichLog).write(
            f"{event.event_type.value}: {event.component}"
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        event.input.disabled = True
        chat = self.query_one("#chat", RichLog)
        chat.write(f"[bold cyan]You:[/] {question}")
        result = await self.service.ask(question)
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
