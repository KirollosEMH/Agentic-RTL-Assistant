from pathlib import Path

from agentic_rtl_assistant.config.models import AppConfig


def validate_runtime_paths(config: AppConfig) -> None:
    """Validate paths needed for a runnable request and fail early."""

    if not config.project.root.is_dir():
        raise ValueError(f"project directory does not exist: {config.project.root}")
    missing_prompts = [
        settings.prompt
        for settings in config.orchestration.agents.values()
        if not Path(settings.prompt).is_file()
    ]
    if missing_prompts:
        joined = ", ".join(str(path) for path in missing_prompts)
        raise ValueError(f"agent prompt file(s) do not exist: {joined}")
