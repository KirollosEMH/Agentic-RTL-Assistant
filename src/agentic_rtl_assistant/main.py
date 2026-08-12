"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.config import load_config, validate_runtime_paths
from agentic_rtl_assistant.ui.app import RTLAssistantTUI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtl-assistant")
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration")
    parser.add_argument("--ask", help="run one non-interactive request")
    subparsers = parser.add_subparsers(dest="command")
    evaluation = subparsers.add_parser("eval", help="run configured evaluation cases")
    evaluation.add_argument("--config", dest="eval_config", help="experiment YAML")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(getattr(args, "eval_config", None) or args.config)
    config = load_config(config_path)
    validate_runtime_paths(config)
    if args.command == "eval":
        from agentic_rtl_assistant.evaluation.runner import EvaluationRunner

        run_directory = asyncio.run(EvaluationRunner(config).run())
        print(f"Evaluation results: {run_directory}")
        return
    service = ApplicationService(config)
    if args.ask:
        result = asyncio.run(service.ask(args.ask))
        if result.error:
            raise SystemExit(result.error)
        print(result.generated_code or result.answer or "")
        return
    RTLAssistantTUI(service).run()


if __name__ == "__main__":
    main()
